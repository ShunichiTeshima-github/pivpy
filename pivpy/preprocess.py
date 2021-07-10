#!/usr/bin/env python3
import numpy as np
import cv2


def remove_background(img, background):
    """remove_background(img, background)
    Return image removed background

    Parameters
    ----------
    img : ndarray
        8bit input image, only 1Ch(gray) iamge
    background : ndarray
        8bit background image, only 1Ch(gray) iamge

    Returns
    -------
    out : ndarray
        8bit output image, only 1Ch(gray) iamge
    """
    img_removed = np.where(img > background, img - background, 0)
    img_removed = img_removed.astype(np.uint8)
    return img_removed


def high_pass_filter(img, ksize=9):
    """high_pass_filter(img, ksize)
    Return image removed low frequency infomation

    Parameters
    ----------
    img : ndarray
        8bit input image, only 1Ch(gray) iamge
    ksize : int
        kernel size for threshold of filter

    Returns
    -------
    out : ndarray
        8bit output image, only 1Ch(gray) iamge
    """
    img_low_hz = cv2.medianBlur(img, ksize=ksize)
    img_removed = np.where(img > img_low_hz, img - img_low_hz, 0)
    img_removed = img_removed.astype(np.uint8)
    return img_removed


def clahe(img, tile_size=(10, 10)):
    clahe_conf = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tile_size)
    return clahe_conf.apply(img)


def extract_particle(img):
    thresh, _ = cv2.threshold(
        img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img[img <= thresh] = 0
    return img


def superposition(img0, img1):
    img_super = img0 + img1
    img_super[img_super > 255] = 255
    return img_super.astype(np.uint8)
