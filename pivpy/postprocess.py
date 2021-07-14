#!/usr/bin/env python3
import copy
import cv2
import numpy as np
from scipy import ndimage


def _interpolate_nan(value):
    value_2d = copy.deepcopy(value)
    print('Number of NAN value : ', end='')
    print('%d / %d' % (np.count_nonzero(np.isnan(value_2d)), value_2d.size))

    W = 1 / (2**0.5)
    for j in range(1, value_2d.shape[0]-1):
        for i in range(1, value_2d.shape[1]-1):
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i+1] + value_2d[j, i-1]) / (4 + 4*W) + (
                                 value_2d[j+1, i+1] + value_2d[j+1, i-1] + value_2d[j-1, i+1] + value_2d[j-1, i-1]) * W / (4 + 4*W)

    for j in range(1, value_2d.shape[0]-1):
        i = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i+1]) / (3 + 3*W) + (
                             value_2d[j+1, i+1] + value_2d[j-1, i+1]) * W / (2 + 2*W)
        i = value_2d.shape[1]-1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i-1]) / (3 + 3*W) + (
                             value_2d[j+1, i-1] + value_2d[j-1, i-1]) * W / (2 + 2*W)

    for i in range(1, value_2d.shape[1]-1):
        j = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j, i-1] + value_2d[j, i+1]) / (3 + 3*W) + (
                             value_2d[j+1, i+1] + value_2d[j+1, i-1]) * W / (2 + 2*W)
        j = value_2d.shape[0]-1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j, i+1] + value_2d[j-1, i] + value_2d[j, i-1]) / (3 + 3*W) + (
                             value_2d[j-1, i+1] + value_2d[j-1, i-1]) * W / (2 + 2*W)

    if np.isnan(value_2d[0, 0]):
        value_2d[0, 0] = (value_2d[1, 0] + value_2d[0, 1]) / (2 + 2*W) + value_2d[1, 1] * W / (1 + 1*W)
    if np.isnan(value_2d[-1, 0]):
        value_2d[-1, 0] = (value_2d[-2, 0] + value_2d[-1, 1]) / (2 + 2*W) + value_2d[-2, 1] * W / (1 + 1*W)
    if np.isnan(value_2d[0, -1]):
        value_2d[0, -1] = (value_2d[0, -2] + value_2d[1, -1]) / (2 + 2*W) + value_2d[1, -2] * W / (1 + 1*W)
    if np.isnan(value_2d[-1, -1]):
        value_2d[-1, -1] = (value_2d[-1, -2] + value_2d[-2, -1]) / (2 + 2*W) + value_2d[-2, -2] * W / (1 + 1*W)

    print('Number of NAN value : ', end='')
    print('%d / %d' % (np.count_nonzero(np.isnan(value_2d)), value_2d.size))

    last_nan_count = np.count_nonzero(np.isnan(value_2d))

    while True:
        for j in range(1, value_2d.shape[0]-1):
            for i in range(1, value_2d.shape[1]-1):
                if np.isnan(value_2d[j, i]):
                    value_2d[j, i] = np.nanmean([value_2d[j+1, i], value_2d[j-1, i], value_2d[j, i+1], value_2d[j, i-1],
                                                value_2d[j+1, i+1], value_2d[j+1, i-1], value_2d[j-1, i+1], value_2d[j-1, i-1]])
        for i in range(1, value_2d.shape[1]-1):
            j = 0
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean([value_2d[j+1, i], value_2d[j, i+1], value_2d[j, i-1], value_2d[j+1, i+1], value_2d[j+1, i-1]])
            j = value_2d.shape[0]-1
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean([value_2d[j-1, i], value_2d[j, i+1], value_2d[j, i-1], value_2d[j-1, i+1], value_2d[j-1, i-1]])

        for j in range(1, value_2d.shape[0]-1):
            i = 0
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean([value_2d[j+1, i], value_2d[j-1, i], value_2d[j, i+1], value_2d[j+1, i+1], value_2d[j-1, i+1]])
            i = value_2d.shape[1]-1
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = np.nanmean([value_2d[j+1, i], value_2d[j-1, i], value_2d[j, i-1], value_2d[j+1, i-1], value_2d[j-1, i-1]])

        if np.isnan(value_2d[0, 0]):
            value_2d[0, 0] = np.nanmean([value_2d[1, 0], value_2d[0, 1], value_2d[1, 1]])
        if np.isnan(value_2d[-1, 0]):
            value_2d[-1, 0] = np.nanmean([value_2d[-2, 0], value_2d[-1, 1], value_2d[-2, 1]])
        if np.isnan(value_2d[0, -1]):
            value_2d[0, -1] = np.nanmean([value_2d[0, -2], value_2d[1, -1], value_2d[1, -2]])
        if np.isnan(value_2d[-1, -1]):
            value_2d[-1, -1] = np.nanmean([value_2d[-1, -2], value_2d[-2, -1], value_2d[-2, -2]])

        if np.count_nonzero(np.isnan(value_2d)) == 0 or np.count_nonzero(np.isnan(value_2d)) == last_nan_count:
            break
        else:
            last_nan_count = np.count_nonzero(np.isnan(value_2d))
    return value_2d


def _median_test(value_2d, eps, thresh):
    check = np.zeros((value_2d.shape[0] + 2, value_2d.shape[1] + 2))
    check[:, :] = np.nan
    check[1: value_2d.shape[0] + 1, 1: value_2d.shape[1] + 1] = value_2d
    value_return = np.zeros(value_2d.shape)

    for j in range(1, value_2d.shape[0] + 1):
        for i in range(1, value_2d.shape[1] + 1):
            value_list = np.array([check[j-1, i-1], check[j-1, i], check[j-1, i+1], check[j, i-1],
                                  check[j, i+1], check[j+1, i-1], check[j+1, i], check[j+1, i+1]])
            value_list = value_list[~np.isnan(value_list)]
            value_median = np.median(value_list)
            value_rm = np.median(np.abs(value_list - value_median))

            if np.abs(value_2d[j - 1, i - 1] - value_median) / (value_rm + eps) > thresh:
                value_return[j - 1, i - 1] = np.abs(value_2d[j - 1, i - 1] - value_median) / (value_rm + eps)
    return value_return


def _interpolation(value, filter):
    value_2d = copy.deepcopy(value)
    W = 1 / (2**0.5)
    for j in range(1, value_2d.shape[0]-1):
        for i in range(1, value_2d.shape[1]-1):
            if np.isnan(value_2d[j, i]):
                value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i+1] + value_2d[j, i-1]) / (4 + 4*W) + (
                                 value_2d[j+1, i+1] + value_2d[j+1, i-1] + value_2d[j-1, i+1] + value_2d[j-1, i-1]) * W / (4 + 4*W)

    for j in range(1, value_2d.shape[0]-1):
        i = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i+1]) / (3 + 3*W) + (
                             value_2d[j+1, i+1] + value_2d[j-1, i+1]) * W / (2 + 2*W)
        i = value_2d.shape[1]-1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j-1, i] + value_2d[j, i-1]) / (3 + 3*W) + (
                             value_2d[j+1, i-1] + value_2d[j-1, i-1]) * W / (2 + 2*W)

    for i in range(1, value_2d.shape[1]-1):
        j = 0
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j+1, i] + value_2d[j, i-1] + value_2d[j, i+1]) / (3 + 3*W) + (
                             value_2d[j+1, i+1] + value_2d[j+1, i-1]) * W / (2 + 2*W)
        j = value_2d.shape[0]-1
        if np.isnan(value_2d[j, i]):
            value_2d[j, i] = (value_2d[j, i+1] + value_2d[j-1, i] + value_2d[j, i-1]) / (3 + 3*W) + (
                             value_2d[j-1, i+1] + value_2d[j-1, i-1]) * W / (2 + 2*W)

    if np.isnan(value_2d[0, 0]):
        value_2d[0, 0] = (value_2d[1, 0] + value_2d[0, 1]) / (2 + 2*W) + value_2d[1, 1] * W / (1 + 1*W)
    if np.isnan(value_2d[-1, 0]):
        value_2d[-1, 0] = (value_2d[-2, 0] + value_2d[-1, 1]) / (2 + 2*W) + value_2d[-2, 1] * W / (1 + 1*W)
    if np.isnan(value_2d[0, -1]):
        value_2d[0, -1] = (value_2d[0, -2] + value_2d[1, -1]) / (2 + 2*W) + value_2d[1, -2] * W / (1 + 1*W)
    if np.isnan(value_2d[-1, -1]):
        value_2d[-1, -1] = (value_2d[-1, -2] + value_2d[-2, -1]) / (2 + 2*W) + value_2d[-2, -2] * W / (1 + 1*W)

    return value_2d


def error_vector_interp_2d(vector_x, vector_y, eps=0.3, thresh=5):
    vector_x = _interpolate_nan(vector_x)
    vector_y = _interpolate_nan(vector_y)
    filter_x = _median_test(vector_x, eps=eps, thresh=thresh)
    filter_y = _median_test(vector_y, eps=eps, thresh=thresh)
    filter_error = filter_x + filter_y
    vector_x = _interpolation(vector_x, filter_error)
    vector_y = _interpolation(vector_y, filter_error)
    return vector_x, vector_y, filter_error


def error_vector_interp_3d(vector_x, vector_y, vector_z, eps=0.3, thresh=5):
    vector_x = _interpolate_nan(vector_x.copy())
    vector_y = _interpolate_nan(vector_y.copy())
    vector_z = _interpolate_nan(vector_z.copy())
    filter_x = _median_test(vector_x, eps=eps, thresh=thresh)
    filter_y = _median_test(vector_y, eps=eps, thresh=thresh)
    filter_z = _median_test(vector_z, eps=eps, thresh=thresh)
    filter_error = filter_x + filter_y + filter_z
    vector_x = _interpolation(vector_x.copy(), filter_error)
    vector_y = _interpolation(vector_y.copy(), filter_error)
    vector_z = _interpolation(vector_z.copy(), filter_error)
    return vector_x, vector_y, vector_z, filter_error


def smoothing(src, mode='gauss', ksize=21):
    src_save = copy.deepcopy(src)
    src_save[src_save == 0.0] = np.nan
    if mode == 'gauss':
        ret = cv2.GaussianBlur(src, ksize=(ksize, ksize), sigmaX=3)
    elif mode == 'median':
        ret = ndimage.median_filter(src, size=ksize)
    else:
        print('Error')
        return src
    src_expand = np.zeros((src.shape[0] + ksize*2, src.shape[1] + ksize*2))
    src_expand[:, :] = np.nan
    src_expand[ksize:-ksize, ksize:-ksize] = src_save
    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            if np.isnan(src_save[j, i]):
                ret[j, i] = np.nan
            elif np.isnan(ret[j, i]) and ~np.isnan(src_save[j, i]):
                window = src_expand[j-int(ksize//2)+ksize:j-int(ksize//2)+2*ksize, i-int(ksize//2)+ksize:i-int(ksize//2)+2*ksize]
                ret[j, i] = np.nanmean(window)
    return ret
