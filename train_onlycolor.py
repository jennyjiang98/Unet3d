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

from eval import eval_net
from unet import UNet
from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

writer=SummaryWriter(logdir='run_onlycolor',comment = 'Linear')



def loss_calc(pred, label, gpu):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long())
    criterion = CrossEntropy2d()
    if gpu:
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

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):

    dir_mask = '../AdvSemiSeg/dataset/VOC2012/SegmentationClassAug/'
    dir_img = '../AdvSemiSeg/dataset/VOC2012/JPEGImages/'
    dir_checkpoint = 'checkpoints_onlycolor/'

    ids = get_ids(dir_img, dir_mask)
    ids = split_ids(ids)##默认重复2遍

    iddataset = split_train_val(ids, val_percent)##分割有重复的

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
    n = len(iddataset['train'])
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)

    #criterion = CrossEntropy2d()#nn.BCELoss()###############
    total_steps=0

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)

        epoch_loss = 0
        epoch_loss_reconstruct = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)#######
            true_masks = np.array([i[1] for i in b])####

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)

            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred, model_y1, model_y2 = net(imgs)
            print(masks_pred.shape)
            masks_probs_flat = masks_pred#.view(-1)

            true_masks_flat = true_masks#.view(-1)

            # loss = loss_calc(masks_probs_flat, true_masks_flat, gpu)
            # epoch_loss += loss.item()

            interp = nn.Upsample(size=(320, 320), mode='bilinear', align_corners=True)
            crit = nn.MSELoss()
            if gpu:
                crit = crit.cuda()
            loss_reconstruct = crit(interp(model_y1), imgs)
            loss_reconstruct += crit(interp(model_y2), imgs)
            epoch_loss_reconstruct += loss_reconstruct.item()

            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            #
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            # writer.add_scalar('Train', loss, i)
            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss_reconstruct.item()))

            optimizer.zero_grad()
            loss_reconstruct.backward()
            optimizer.step()
            writer.add_scalar('Color', loss_reconstruct, total_steps)
            total_steps+=1
            if total_steps % 500 ==0:
                writer.add_image('color_imresult_y1', model_y1[0].cpu(), total_steps)
                writer.add_image('color_imresult_y2', model_y2[0].cpu(), total_steps)
                writer.add_image('color_im', imgs[0].cpu(), total_steps)


        # print('Epoch finished ! Loss: {}'.format(epoch_loss / i))
        print('Epoch finished ! Loss: {}'.format(epoch_loss_reconstruct / i))

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
    parser.add_option('-b', '--batch-size', dest='batchsize', default=8,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=21)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
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
