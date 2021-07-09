#!/usr/bin/env python3
import numpy as np


def _interpolate_nan(value_2d):
    for j in range(value_2d.shape[0]):
        for i in range(value_2d.shape[1]):
            if j > 1 and j < value_2d.shape[0]-2 and i > 1 and i < value_2d.shape[1]-2:
                if np.isnan(value_2d[j, i]):
                    value_2d[j, i] = np.nanmean((
                        value_2d[j+1, i+1], value_2d[j+1, i], value_2d[j+1, i-1], value_2d[j, i+1],
                        value_2d[j, i-1], value_2d[j-1, i+1], value_2d[j-1, i], value_2d[j-1, i-1]))
            else:
                if np.isnan(value_2d[j, i]):
                    value_2d[j, i] = 0.0
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


def _interpolation(value_2d, filter):
    value_2d[filter != 0.0] = np.nan
    check = np.zeros((value_2d.shape[0] + 2, value_2d.shape[1] + 2))
    check[:, :] = np.nan
    check[1: value_2d.shape[0] + 1, 1: value_2d.shape[1] + 1] = value_2d
    value_return = np.zeros(value_2d.shape)

    for j in range(1, value_2d.shape[0] + 1):
        for i in range(1, value_2d.shape[1] + 1):
            if filter[j-1, i-1] != 0.0:
                value_list = np.array([check[j-1, i-1], check[j-1, i], check[j-1, i+1], check[j, i-1],
                                      check[j, i+1], check[j+1, i-1], check[j+1, i], check[j+1, i+1]])
                value_list = value_list[~np.isnan(value_list)]
                value_return[j-1, i-1] = np.mean(value_list)
            else:
                value_return[j-1, i-1] = value_2d[j-1, i-1]

    return value_return


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
