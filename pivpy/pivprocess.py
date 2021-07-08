#!/usr/bin/env python3
import numpy as np
from scipy.ndimage import maximum_filter


def _correlation_list(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset,
        cal_area):

    correlation_list = np.zeros((
        int(search_window[1] - search_window[0] + 1), int(search_window[3] - search_window[2] + 1), int(x_grid.shape[0] * x_grid.shape[1])))

    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            if cal_area[j, i] == 1:
                index_j = int(y_grid[j, i])
                index_i = int(x_grid[j, i])
                window_img0 = img0[int(index_j - interrogation_window[0] // 2):int(index_j + interrogation_window[0] // 2),
                                   int(index_i - interrogation_window[1] // 2):int(index_i + interrogation_window[1] // 2)]

                correlation_list[:, :, j * x_grid.shape[1] + i] = _correlation_map(
                    window_img0, img1, interrogation_window, search_window, index_j + yoffset[j, i], index_i + xoffset[j, i])
    return correlation_list


def _correlation_map(window_img0, img1, interrogation_window, search_window, index_j, index_i):
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
    def _sub_pixel_interpolation():
        try:
            value0 = max(correlation_map[peak_j, peak_i], 0.01)
            value1 = max(correlation_map[peak_j - 1, peak_i], 0.01)
            value2 = max(correlation_map[peak_j + 1, peak_i], 0.01)
            value3 = max(correlation_map[peak_j, peak_i - 1], 0.01)
            value4 = max(correlation_map[peak_j, peak_i + 1], 0.01)
        except IndexError:
            return peak_j, peak_i

        delta_j = peak_j + 0.5 * (np.log(value1) - np.log(value2)) / (np.log(value1) + np.log(value2) - 2*np.log(value0))
        delta_i = peak_i + 0.5 * (np.log(value3) - np.log(value4)) / (np.log(value3) + np.log(value4) - 2*np.log(value0))
        if np.isnan(delta_j):
            delta_j = peak_j
        if np.isnan(delta_i):
            delta_i = peak_i
        return delta_j, delta_i

    correlation_map[np.isnan(correlation_map)] = 0

    local_max = maximum_filter(correlation_map, footprint=np.ones((5, 5)), mode='constant')
    detected_peaks = np.ma.array(correlation_map, mask=~(correlation_map == local_max))
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * 0.3))
    peaks_index = np.where((temp.mask == 0))

    if len((peaks_index[0])) == 1:
        peak_j, peak_i = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
        return _sub_pixel_interpolation()
    elif len((peaks_index[0])) > 1:
        peak_j, peak_i = np.unravel_index(correlation_map.argmax(), correlation_map.shape)
        value_lis = []
        for num in range(len((peaks_index[0]))):
            value_lis.append(correlation_map[peaks_index[0][num], peaks_index[1][num]])
        rs1 = sorted(value_lis)[-1]
        rs2 = sorted(value_lis)[-2]
        if rs1 / rs2 >= 1.75:
            return _sub_pixel_interpolation()
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan


def normal_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset):

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = _correlation_list(img0, img1, x_grid, y_grid, interrogation_window, search_window, xoffset, yoffset)
    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            peak_j, peak_i = _detect_peak(correlation_list[:, :, j * x_grid.shape[1] + i])
            vector_v[j, i] = peak_j + search_window[0] + yoffset[j, i]
            vector_u[j, i] = peak_i + search_window[2] + xoffset[j, i]

    return vector_u, vector_v


def emsemble_piv(
        img0,
        img1,
        x_grid,
        y_grid,
        interrogation_window,
        search_window,
        xoffset,
        yoffset,
        cal_area):

    vector_u = np.zeros(x_grid.shape)
    vector_v = np.zeros(x_grid.shape)

    correlation_list = np.zeros((
        int(search_window[1] - search_window[0] + 1), int(search_window[3] - search_window[2] + 1), int(x_grid.shape[0] * x_grid.shape[1])))
    for pair_num in range(img0.shape[2]):
        correlation_list = correlation_list + _correlation_list(img0[:, :, pair_num], img1[:, :, pair_num],
                                                                x_grid, y_grid, interrogation_window, search_window, xoffset, yoffset, cal_area)
    for j in range(x_grid.shape[0]):
        for i in range(x_grid.shape[1]):
            peak_j, peak_i = _detect_peak(correlation_list[:, :, j * x_grid.shape[1] + i])
            vector_v[j, i] = peak_j + search_window[0] + yoffset[j, i]
            vector_u[j, i] = peak_i + search_window[2] + xoffset[j, i]

    return vector_u, vector_v
