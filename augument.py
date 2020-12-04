import cv2
import numpy as np
import pandas as pd


def get_more_images_v1(imgs):
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images

def get_more_images_v2(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    cent_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        ac = cv2.flip(a, -1)

        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        bc = cv2.flip(b, -1)

        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)
        cc = cv2.flip(c, -1)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
        cent_flip_imgs.append(np.dstack((ac, bc, cc)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    c = np.array(cent_flip_imgs)

    more_images = np.concatenate((imgs, v, h, c))

    return more_images

import imgaug as ia
from imgaug import augmenters as iaa

def get_more_images_v3(imgs, n_generate_times=2, n_ops=4):
    # define the augmentations

    augs = [
        iaa.Sequential([iaa.Affine(scale={"x": (0.75, 1.0), "y": (0.75, 1.0)})]),
        iaa.Sequential([iaa.Dropout(0.075)]),
        iaa.Sequential([iaa.GaussianBlur(sigma=(0.0, 2.0))]),
        iaa.Sequential([iaa.Fliplr(0.5)]),
        iaa.Sequential([iaa.Affine(scale=(0.75, 1.0))]),
        iaa.Sequential([iaa.Affine(translate_px={"x": (-20, 20), "y": (-20, 20)})]),
        iaa.Sequential([iaa.Rotate((-15, 15))]),
        iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.05))]),
        iaa.Sequential([iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5)))]), # New update
        iaa.Sequential([iaa.ContrastNormalization((0.75, 1.5))]),
        iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)]),
        iaa.Sequential([iaa.CropAndPad(percent=(-0.05, 0.05), pad_mode=ia.ALL, pad_cval=(0, 255))]),
        iaa.Sequential([iaa.Fliplr(1.0)]),
        iaa.Sequential([iaa.Flipud(1.0)]),
        iaa.Sequential([iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
                                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
                                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                                    shear=(-16, 16),  # shear by -16 to +16 degrees
                                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                                    cval=(0, 255))])
    ]
    img_augs = []

    result = imgs
    for i in range(1, n_generate_times):
        rand_augs = np.random.choice(np.arange(len(augs)), n_ops)
        for idx in rand_augs:
            aug = augs[idx]
            img_augs.append(aug)
        print(rand_augs)
        seqs = iaa.Sequential(img_augs, random_order=True)
        result = np.concatenate((result, seqs.augment_images(imgs)))
    return result

def get_more_images_v4(imgs):
    # define the augmentations

    seq1 = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
        iaa.ContrastNormalization((0.75, 1.5)),
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5)
    ], random_order=True)  # apply the augmenters in random order

    seq2 = iaa.Sequential([
        iaa.CropAndPad(
            percent=(-0.05, 0.05),
            pad_mode=ia.ALL,
            pad_cval=(0, 255))])

    seq3 = iaa.Sequential([
        iaa.Fliplr(1.0)])  # horizontally flip the images

    seq4 = iaa.Sequential([
        iaa.Flipud(1.0)])

    seq5 = iaa.Sequential([
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-45, 45),  # rotate by -45 to +45 degrees
            shear=(-16, 16),  # shear by -16 to +16 degrees
            order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
            cval=(0, 255))])  # if mode is constant, use a cval between 0 and 255)

    seqs = [seq1, seq2, seq3, seq4, seq5]
    imgaug_sequential_list = []
    for seq in seqs:
      imgaug_sequential_list.append(seq.imgaug_operation)
    imgaug_sequential = iaa.Sequential(
        imgaug_sequential_list, random_order=True)
    return np.concatenate((imgs,
                           imgaug_sequential.augment_images(imgs),
                           imgaug_sequential.augment_images(imgs),
                           imgaug_sequential.augment_images(imgs)
                           ))
