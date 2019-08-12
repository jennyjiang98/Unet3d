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
from data_loaderold import KittiLoader
from eval import eval_net
from unet import UNet
import glob
from torch.utils.data import DataLoader
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

writer=SummaryWriter(logdir='run_both_grey',comment = 'Linear')
colorize = VOCColorize()

def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long())
    criterion = CrossEntropy2d()
    if gpu == True:
        label = label.cuda()
        criterion = CrossEntropy2d().cuda()

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
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
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
              img_scale=0.5):
    dir_img = '/ldap_shared/home/v_ying_jiang/denoised/noisy'  ##noisy1/0bed37d7-3cf3-4cdd-808f-cce3b3e9675c
    dir_mask = '/ldap_shared/home/v_ying_jiang/mask'
    dir_checkpoint = 'checkpoints_grey_both/'
    try:
        os.mkdir(dir_checkpoint)
    except FileExistsError:
        pass

    # ids = get_ids_grey(dir_img, dir_mask)
    # ids = split_ids(ids)##默认重复2遍
    #
    # iddataset = split_train_val(ids, val_percent)##分割有重复的
    # drive = ['0027']

    dir11 = glob.glob(dir_img + "*/*/*.png", )
    # dir22 = glob.glob(dir + "mask*/", )
    d = (f[45:] for f in dir11)
    n = len(d)
    d_val = d[:0.1*n]
    d_train = d[0.1*n:]
    train_set = KittiLoader(dir_img, dir_mask, d_train)
    train_dataset = DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=2,
                            pin_memory=True)
    n_img = len(train_set)

    valid_set = KittiLoader(dir_img, dir_mask, d_val)
    val_dataset = DataLoader(valid_set, batch_size=batch_size,
                            shuffle=True, num_workers=1,
                            pin_memory=True)

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
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])

    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    total_steps=0
    total_val_steps = 0
    #criterion = CrossEntropy2d()#nn.BCELoss()###############

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
-------
for data in self.loader:
                # Load data
                data = to_device(data, self.device)
                left = data['left_image']
                right = data['right_image']
                laser = data['laser_image']
                gt = data['gt_image']
                feed = data['feed']
                # One optimization iteration
                self.optimizer.zero_grad()
                disps = self.model([left, laser])
                loss = self.loss_function(disps, [left, right, gt])
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())


-------
        # reset the generators
        n = len(iddataset['train'])
        n_labeled = int(0.1 * n)
        n_unlabeled = n - int(0.1 * n)
        train = get_imgs_and_masks_grey(iddataset['train'], dir_img, dir_mask, img_scale)
        #  idx = ids[:,0]
        # weights = np.array([ a%10==0 for a in idx])#, dtype = np.uint8
        # #weights = np.sign(weights)

        val = get_imgs_and_masks_grey_val(iddataset['val'], dir_img, dir_mask, img_scale)
        n_val = len(iddataset['val'])

        epoch_loss = 0
        epoch_loss_reconstruct = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)#######
            true_masks = np.array([i[1] for i in b])####
            weights = [i[2] for i in b]

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            msk = true_masks > 0  ###true_masks
            true_masks[msk] = 1


            if gpu == True:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred, model_y1, model_y2 = net(imgs)
            #print(masks_pred.shape)
            masks_probs_flat = masks_pred[weights]  # .view(-1)

            true_masks_flat = true_masks[weights]  # .view(-1)



            loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            epoch_loss += loss.item()

            print('{0:.4f} --- seg loss: {1:.6f}'.format(i * batch_size / n, loss.item()))###应该很多0

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            writer.add_scalar('Seg', loss, total_steps)
            if total_steps % 250 ==0:

                out = masks_pred[0].cpu()

                out = out.detach().numpy().transpose(1,2,0)
                out = np.asarray(np.argmax(out, axis=2), dtype=np.int)
                print(out[out > 0])
                out = colorize(out)
                writer.add_image('seg_imresult', out,total_steps)

                m = true_masks[0].cpu().detach().numpy()
                # print(m)
                # print(m[m>0])
                m = colorize(m)

                # print(m)
                # print(m[m>0])
                writer.add_image('seg_gt', m, total_steps)
            ##########seg over

            interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
            crit = nn.MSELoss()
            if gpu == True:
                crit = crit.cuda()
            loss_reconstruct = crit(interp(model_y1), imgs)
            loss_reconstruct += crit(interp(model_y2), imgs)
            epoch_loss_reconstruct += loss_reconstruct.item()
            print('{0:.4f} --- labeled reconstruct loss: {1:.6f}'.format(i * batch_size / n_labeled, loss_reconstruct.item()))

            optimizer.zero_grad()
            loss_reconstruct.backward(retain_graph=True)
            optimizer.step()
            writer.add_scalar('Color', loss_reconstruct, total_steps)



            if total_steps % 250 ==0:
                writer.add_image('color_imresult_y1', model_y1[0].cpu(), total_steps)
                writer.add_image('color_imresult_y2', model_y2[0].cpu(), total_steps)
                writer.add_image('color_im', imgs[0].cpu(), total_steps)
            total_steps += 1


        print('Epoch labeled finished ! seg Loss: {}'.format(epoch_loss / int(0.1*n)))
        print('Epoch labeled finished ! labeled reconstruct Loss: {}'.format(epoch_loss_reconstruct / i))
        '''研究unlabeled的时机'''
        net.eval()
        epoch_loss = 0
        epoch_loss_reconstruct = 0
        for i, b in enumerate(batch(val, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)  #######
            true_masks = np.array([i[1] for i in b])  ####
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            msk = true_masks > 0###true_masks
            true_masks[msk]= 1
            if gpu == True:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred, model_y1, model_y2 = net(imgs)
            # print(masks_pred.shape)
            masks_probs_flat = masks_pred  # .view(-1)

            true_masks_flat = true_masks  # .view(-1)

            loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            epoch_loss += loss.item()

            print('{0:.4f} --- seg loss: {1:.6f}'.format(i * batch_size / n, loss.item()))  ###应该很多0

            # optimizer.zero_grad()
            # loss.backward(retain_graph=True)
            # optimizer.step()
            writer.add_scalar('Seg_val', loss, total_val_steps)
            if total_val_steps % 100 == 0:
                out = masks_pred[0].cpu()

                out = out.detach().numpy().transpose(1, 2, 0)
                out = np.asarray(np.argmax(out, axis=2), dtype=np.int)
                print(out[out > 0])
                out = colorize(out)
                writer.add_image('seg_imresult_val', out, total_val_steps)

                m = true_masks[0].cpu().detach().numpy()
                # print(m)
                # print(m[m>0])
                m = colorize(m)

                # print(m)
                # print(m[m>0])
                writer.add_image('seg_gt_val', m, total_val_steps)
            ##########seg over

            interp = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
            crit = nn.MSELoss()
            if gpu == True:
                crit = crit.cuda()
            loss_reconstruct = crit(interp(model_y1), imgs)
            loss_reconstruct += crit(interp(model_y2), imgs)
            epoch_loss_reconstruct += loss_reconstruct.item()
            print('{0:.4f} --- labeled reconstruct loss: {1:.6f}'.format(i * batch_size / n_labeled,
                                                                         loss_reconstruct.item()))

            # optimizer.zero_grad()
            # loss_reconstruct.backward(retain_graph=True)
            # optimizer.step()
            writer.add_scalar('Color_val', loss_reconstruct, total_val_steps)



            if total_val_steps % 250 == 0:
                writer.add_image('color_imresult_y1_val', model_y1[0].cpu(), total_val_steps)
                writer.add_image('color_imresult_y2_val', model_y2[0].cpu(), total_val_steps)
                writer.add_image('color_im_val', imgs[0].cpu(), total_val_steps)
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
    parser.add_option('-e', '--epochs', dest='epochs', default=15, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=6,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu',  dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net = UNet(n_channels=3, n_classes=2)

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
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
