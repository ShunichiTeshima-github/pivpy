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
