#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os
import glob
import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir, dir2):
    """Returns a list of the ids in the directory"""
    #return (f[:-4] for f in os.listdir(dir) if f[:-4]+'.png' in os.listdir(dir2))
    return (f[:-4] for f in os.listdir(dir2) if f.startswith('2'))

def get_ids_grey(dir, dir2):
    """Returns a list of the ids in the directory"""
    #return (f[:-4] for f in os.listdir(dir) if f[:-4]+'.png' in os.listdir(dir2))
    dir11 = glob.glob(dir+"*/*/*.png",)
    #dir22 = glob.glob(dir + "mask*/", )
    d = (f[45:-4] for f in dir11)
    return d


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id, pos in ids:

        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield get_square(im, pos)

def get_imgs_and_masks_new(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png', scale)

    idx = list(range(len(ids)))
    weights = np.array([ a%10==0 for a in idx])#, dtype = np.uint8
    #weights = np.sign(weights)
    return zip(imgs_normalized, masks, weights)

def get_imgs_and_masks_grey(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png', scale)

    idx = list(range(len(ids)))
    weights = np.array([ a%10==0 for a in idx])#, dtype = np.uint8
    #weights = np.sign(weights)
    return zip(imgs_normalized, masks, weights)

def get_imgs_and_masks_grey_val(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png', scale)

    return zip(imgs_normalized, masks)

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""

    imgs = to_cropped_imgs(ids, dir_img, '.jpg', scale)

    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png', scale)


    return zip(imgs_normalized, masks)


def get_full_img_and_mask(id, dir_img, dir_mask):
    im = Image.open(dir_img + id + '.jpg')
    mask = Image.open(dir_mask + id + '.png')
    return np.array(im), np.array(mask)
