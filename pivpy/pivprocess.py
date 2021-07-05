#!/usr/bin/env python3
import numpy as np


def _correlation_list(
        img0,
        img1,
        xgrid,
        ygrid,
        interrogation_window,
        search_window):

    correlation_list = np.zeros((
                                 int(search_window[1] - search_window[0] + 1),
                                 int(search_window[3] - search_window[2] + 1),
                                 int(xgrid.shape[0] * xgrid.shape[1])))

    for j in range(xgrid.shape[0]):
        for i in range(xgrid.shape[1]):
            index_j = xgrid[j, i]
            index_i = ygrid[j, i]
            window_img0 = img0[
                               int(index_j - interrogation_window[0] // 2):
                               int(index_j + interrogation_window[0] // 2),
                               int(index_i - interrogation_window[1] // 2):
                               int(index_i + interrogation_window[1] // 2)]

            correlation_list[:, :, j * xgrid.shape[0] + i
                             ] = _correlation_map(
                                window_img0, img1,
                                interrogation_window, search_window)
    return correlation_list


def _correlation_map(window_img0, img1, interrogation_window, search_window):
    return 0


def normal_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window):

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = _correlation_list(
        img0, img1, x_grid, y_grid, interrogation_window, search_window)

    return vector_u, vector_v, correlation_list
