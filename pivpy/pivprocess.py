#!/usr/bin/env python3
import numpy as np


def _correlation_list(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window):

    correlation_list = np.zeros((
        int(search_window[1] - search_window[0] + 1), int(search_window[3] - search_window[2] + 1), int(x_grid.shape[0] * x_grid.shape[1])))

    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            index_j = x_grid[j, i]
            index_i = y_grid[j, i]
            window_img0 = img0[int(index_j - interrogation_window[0] // 2):int(index_j + interrogation_window[0] // 2),
                               int(index_i - interrogation_window[1] // 2):int(index_i + interrogation_window[1] // 2)]

            correlation_list[:, :, j * x_grid.shape[0] + i] = _correlation_map(
                window_img0, img1, interrogation_window, search_window, index_j, index_i)
    return correlation_list


def _correlation_map(window_img0, img1, interrogation_window, search_window, index_j, index_i):
    """operate correctly"""
    correlation_map = np.zeros((search_window[1] - search_window[0] + 1, search_window[3] - search_window[2] + 1))

    for dj in range(search_window[0], search_window[1] + 1):
        for di in range(search_window[2], search_window[3] + 1):
            index_j2 = index_j + dj
            index_i2 = index_i + di
            if (0 <= index_j2 - interrogation_window[0] // 2 and img1.shape[0] > index_j2 + interrogation_window[0] // 2 and
               0 <= index_i2 - interrogation_window[1] // 2 and img1.shape[1] > index_i2 + interrogation_window[1] // 2):
                window_img1 = img1[int(index_j2 - interrogation_window[0] // 2):int(index_j2 + interrogation_window[0] // 2),
                                   int(index_i2 - interrogation_window[1] // 2):int(index_i2 + interrogation_window[1] // 2)]
                if window_img0.shape == window_img1.shape:
                    correlation_map[abs(search_window[0]) + dj, abs(search_window[2]) + di
                                    ] = np.corrcoef(window_img0.flatten(), window_img1.flatten())[0, 1]

    return correlation_map


def _detect_peak(correlation_map):
    correlation_map[np.isnan(correlation_map)] = 0
    if np.sum(np.abs(correlation_map)) == 0:
        return np.nan, np.nan
    else:
        peak_j, peak_i = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
    return peak_j, peak_i


def normal_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window):

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = _correlation_list(img0, img1, x_grid, y_grid, interrogation_window, search_window)
    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            peak_j, peak_i = _detect_peak(correlation_list[:, :, j * x_grid.shape[0] + i])
            vector_v[j, i] = peak_j + search_window[0]
            vector_u[j, i] = peak_i + search_window[2]

    return vector_u, vector_v
