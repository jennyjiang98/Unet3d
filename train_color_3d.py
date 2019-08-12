import sys
import os
from optparse import OptionParser
import numpy as np
from tensorboardX import SummaryWriter
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from data_loader import CECTDataset
from eval import eval_net
from unet_model import VNet
import glob
from torch.utils.data import DataLoader
import loss as bioloss
from utils import get_ids,get_ids_grey, split_ids, split_train_val, get_imgs_and_masks_grey_val,get_imgs_and_masks_grey, batch

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255.0 if normalized else cmap##正常255的最大
    return cmap

class VOCColorize(object):
    def __init__(self, n=22, normalized = True):
        self.normalized = normalized
        self.cmap = color_map(22,normalized=normalized)
        self.cmap = torch.from_numpy(self.cmap[:n])
        self.n = n
    def __call__(self, gray_image):
        gray_image.squeeze()
        size = gray_image.shape
        if not self.normalized:
            color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)
        else:
            color_image = np.zeros((3, size[0], size[1]), dtype=np.float32)

        for label in range(0, self.n):
            mask = (label == gray_image)###每一种class 上色
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void  全黑  预测结果里不一定有
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image



def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    if label.shape[0]==0:
        return 0.0
    label = Variable(label.long())
    criterion = CrossEntropy2d()
    if gpu == True:
        label = label.cuda()
        criterion = criterion.cuda()

    return criterion(pred, label)

class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        print(predict.shape)
        assert predict.dim() == 5
        print(target.shape)
        target=torch.squeeze(target, dim = 1)
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(2))
        assert predict.size(4) == target.size(3), "{0} vs {1} ".format(predict.size(4), target.size(3))
        n, c,t, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous()
        predict = predict[target_mask.view(n,t, h, w, 1).repeat(1, 1,1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict[target_mask]
        loss = F.binary_cross_entropy_with_logits(predict, target, weight=weight, size_average=self.size_average)
        return loss

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5,
              coeff = 1,
              ratio = 1):
    # dir_img = '/ldap_shared/home/v_ying_jiang/denoised/noisy'  ##noisy1/0bed37d7-3cf3-4cdd-808f-cce3b3e9675c
    # dir_mask = '/ldap_shared/home/v_ying_jiang/mask'
    dir_checkpoint = 'checkpoints3d_color_'+str(ratio)+'datalr'+str(lr)+'k'+str(coeff)+'/'
    if not os.path.exists(dir_checkpoint):
        os.mkdir(dir_checkpoint)
    writer = SummaryWriter(logdir='run3d_color_' + str(ratio) + 'datalr' + str(lr) + 'k' + str(coeff) + '/',
                           comment='Linear')
    colorize = VOCColorize()
    try:
        os.mkdir(dir_checkpoint)
    except FileExistsError:
        pass

    # ids = get_ids_grey(dir_img, dir_mask)
    # ids = split_ids(ids)##默认重复2遍
    #
    # iddataset = split_train_val(ids, val_percent)##分割有重复的
    # drive = ['0027']

    # dir11 = glob.glob(dir_img + "*/*/*.png", )
    # # dir22 = glob.glob(dir + "mask*/", )
    # d = [f[45:] for f in dir11]
    with open('/tmp/jenny/allfiles.txt', 'r') as f:
        d = [line.strip() for line in f]

    n = len(d)
    d_val = d[:int(0.1*n)]
    d_train = d[int(0.1*n):]
    n = len(d_train)
    n_labeled = int(ratio * n)
    n_unlabeled = n - int(ratio * n)
    train_set = CECTDataset(d_train, ratio)
    train_loader = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)
                            #pin_memory=True)
    n = len(train_set)

    valid_set = CECTDataset(d_val,ratio)
    val_loader = DataLoader(valid_set, batch_size=batch_size,
                            shuffle=True, num_workers=1)
                            #pin_memory=True)

    ##tensor already

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_set),
               len(valid_set), str(save_cp), str(gpu)))

    N_train = len(train_set)

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.0005)###1e-8
    total_steps=0
    total_val_steps = 0
    #criterion = CrossEntropy2d()#nn.BCELoss()###############

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators


        n_val = len(valid_set)

        epoch_loss = 0
        epoch_loss_reconstruct = 0
        for i, b in enumerate(train_loader):

            imgs, true_masks, weights = b

            if gpu == True:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred, model_y1 = net(imgs)
            #print(masks_pred.shape)
            # print(weights)
            #weights = np.array(weights, dtype = np.uint8)


            # print(weights)
            mk = weights.squeeze()#torch.Tensor([a[0] for a in weights])
            mk = mk.type(torch.uint8)
            print(mk)
            print(mk.sum())

            #print(masks_pred)
            print(masks_pred.shape)
            masks_probs_flat = masks_pred[mk]  # .view(-1)
            #print(masks_probs_flat)
            # print(masks_probs_flat.shape)
            true_masks_flat = true_masks[mk]  # .view(-1)

            # loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            print(true_masks.shape)
            # loss = F.nll_loss(masks_probs_flat, true_masks_flat)
            # loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            #
            #
            #
            # try:
            #     epoch_loss += loss.item()#####main
            #
            #     out = masks_probs_flat.permute(0, 2, 3, 4, 1).contiguous()
            #     # flatten
            #     out = out.view(-1, 2)
            #     out = F.log_softmax(out)
            #     target = true_masks_flat.view(true_masks_flat.numel())
            #
            #     dice_loss = bioloss.dice_error(out, target)
            #     pred = out.data.max(1)[1]  # get the index of the max log-probability
            #     print(out.data.max(1))
            #     print(pred.shape)
            #     incorrect = pred.ne(target.data.type(torch.long)).cpu().sum()
            #     err = 100. * incorrect / target.numel()
            #     writer.add_scalar('Seg', loss, total_steps)
            #     writer.add_scalar('Seg_err', err, total_steps)
            #     writer.add_scalar('Seg_dice', dice_loss, total_steps)
            #     loss *= coeff
            #     print('{0:.4f} --- seg loss: {1:.6f}'.format(i * batch_size / n, loss.item()/int(mk.sum())))  ###应该很多0
            #     optimizer.zero_grad()
            #     loss.backward(retain_graph=True)
            #     optimizer.step()
            #
            # except AttributeError:
            #     pass




            # if total_steps % 250 ==0:
            #
            #     out = masks_pred[0].cpu()
            #
            #     out = out.detach().numpy().transpose(1,2,0)
            #     out = np.asarray(np.argmax(out, axis=2), dtype=np.int)
            #     print(out[out > 0])
            #     out = colorize(out)
            #     writer.add_image('seg_imresult', out,total_steps)
            #
            #     m = true_masks[0].cpu().detach().numpy()
            #     m = m.squeeze()
            #     # print(m)
            #     # print(m[m>0])
            #     m = colorize(m)
            #
            #     # print(m)
            #     # print(m[m>0])
            #     writer.add_image('seg_gt', m, total_steps)
            # ##########seg over

            interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
            crit = nn.MSELoss()
            if gpu == True:
                crit = crit.cuda()
            loss_reconstruct = crit(model_y1, imgs)
            # loss_reconstruct += crit(interp(model_y2), imgs)
            epoch_loss_reconstruct += loss_reconstruct.item()
            print('{0:.4f} --- labeled reconstruct loss: {1:.6f}'.format(i * batch_size / n, loss_reconstruct.item()))

            optimizer.zero_grad()
            loss_reconstruct.backward(retain_graph=True)
            optimizer.step()
            writer.add_scalar('Color', loss_reconstruct, total_steps)


            #
            # if total_steps % 250 ==0:
            #     writer.add_image('color_imresult_y1', model_y1[0].cpu(), total_steps)
            #     writer.add_image('color_imresult_y2', model_y2[0].cpu(), total_steps)
            #     writer.add_image('color_im', imgs[0].cpu(), total_steps)
            #


            ###color over
            total_steps += 1


        print('Epoch labeled finished ! seg Loss: {}'.format(epoch_loss / int(ratio*n)))
        print('Epoch labeled finished ! labeled reconstruct Loss: {}'.format(epoch_loss_reconstruct / n))
        '''研究unlabeled的时机'''
        net.eval()
        epoch_loss = 0
        epoch_loss_reconstruct = 0
        for i, b in enumerate(val_loader):
            imgs, true_masks, weights = b
            if gpu == True:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred, model_y1 = net(imgs)
            # print(masks_pred.shape)
            masks_probs_flat = masks_pred  # .view(-1)

            true_masks_flat = true_masks  # .view(-1)
            # loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            # loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            #
            # try:
            #     epoch_loss += loss.item()  #####main
            #
            #     out = masks_probs_flat.permute(0, 2, 3, 4, 1).contiguous()
            #     # flatten
            #     out = out.view(-1, 2)
            #     out = F.log_softmax(out)
            #     target = true_masks_flat.view(true_masks_flat.numel())
            #
            #     dice_loss = bioloss.dice_error(out, target)
            #     pred = out.data.max(1)[1]  # get the index of the max log-probability
            #     print(out.data.max(1))
            #     print(pred.shape)
            #     incorrect = pred.ne(target.data.type(torch.long)).cpu().sum()
            #     err = 100. * incorrect / target.numel()
            #     writer.add_scalar('Seg_val', loss, total_val_steps)
            #     writer.add_scalar('Seg_val_err', err, total_steps)
            #     writer.add_scalar('Seg_val_dice', dice_loss, total_steps)
            #     loss *= coeff
            #     print('{0:.4f} --- seg val loss: {1:.6f}'.format(i * batch_size / n, loss.item() / int(mk.sum())))  ###应该很多0
            #
            # except AttributeError:
            #     pass
            # if total_val_steps % 100 == 0:
            #     out = masks_pred[0].cpu()
            #
            #     out = out.detach().numpy().transpose(1, 2, 0)
            #     out = np.asarray(np.argmax(out, axis=2), dtype=np.int)
            #     out = out.squeeze()
            #     print(out[out > 0])
            #
            #     out = colorize(out)
            #
            #     writer.add_image('seg_imresult_val', out, total_val_steps)
            #
            #     m = true_masks[0].cpu().detach().numpy()
            #     m = m.squeeze()
            #     # print(m)
            #     # print(m[m>0])
            #     m = colorize(m)
            #
            #     # print(m)
            #     # print(m[m>0])
            #     writer.add_image('seg_gt_val', m, total_val_steps)
            # ##########seg over

            # interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
            crit = nn.MSELoss()
            if gpu == True:
                crit = crit.cuda()
            loss_reconstruct = crit(model_y1, imgs)
            #loss_reconstruct += crit(interp(model_y2), imgs)
            epoch_loss_reconstruct += loss_reconstruct.item()
            print('{0:.4f} --- labeled reconstruct loss: {1:.6f}'.format(i * batch_size / n_val,
                                                                         loss_reconstruct.item()))

            # optimizer.zero_grad()
            # loss_reconstruct.backward(retain_graph=True)
            # optimizer.step()
            writer.add_scalar('Color_val', loss_reconstruct, total_val_steps)



            # if total_val_steps % 250 == 0:
            #     writer.add_image('color_imresult_y1_val', model_y1[0].cpu(), total_val_steps)
            #     writer.add_image('color_imresult_y2_val', model_y2[0].cpu(), total_val_steps)
            #     writer.add_image('color_im_val', imgs[0].cpu(), total_val_steps)
            total_val_steps += 1

        print('Epoch val finished ! seg Loss: {}'.format(epoch_loss / n_val))
        print('Epoch val finished ! labeled reconstruct Loss: {}'.format(epoch_loss_reconstruct / n_val))

        # if 1:
        #     val_dice = eval_net(net, val, gpu)
        #     print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
    writer.close()


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu',  dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-k', '--k', dest='coefficient', type='float',
                      default=50, help='seg:color loss')
    parser.add_option('-r', '--r', dest='rat', type='float',
                      default=0.1, help='lab:nonlab')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = VNet(elu=False, nll=True)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu == True:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale,
                  coeff = args.coefficient,
                  ratio= args.rat
                  )
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
