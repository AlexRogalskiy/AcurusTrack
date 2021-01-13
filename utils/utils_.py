"""
 This file is part of AcurusTrack.

    AcurusTrack is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    AcurusTrack is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with AcurusTrack.  If not, see <https://www.gnu.org/licenses/>.
"""

import json
import math
import os
import re

import numpy as np

from config import FilesProcessingParams, SystemParams


def save_pd(df, save_dir, name):
    df.to_csv(os.path.join(save_dir,
                           '{}.csv'.format(name)))


def iterations_number_dependent_on_track(number_of_tracks, number_of_moves):
    return (number_of_tracks + 1) * number_of_moves


def fill_and_format(observations, start_frame_segment, end_frame_segment):
    w = {}
    if observations == {}:
        return w
    key_from_meta = list(observations.keys())[0]
    change_from_str = False
    if isinstance(key_from_meta, str):
        change_from_str = True

    for i in range(int(start_frame_segment), int(end_frame_segment) + 1):
        index_to_find = i
        if change_from_str:
            index_to_find = str(i)
        if index_to_find in observations:
            w[i] = observations[index_to_find]
        else:
            w[i] = []  # fill in empty frames in json
    return w


def load_and_process(path_to_json):
    with open(path_to_json, 'r') as curr_json:
        obs = json.load(curr_json)
        obs = {int(k): v for k, v in obs.items()}
    return obs


def sort_meta_by_key(meta):
    new_dict = {}
    assert type(list(meta.keys())[0]) == np.int
    for key in sorted(meta.keys()):
        assert type(key) == np.int
        new_dict[key] = meta[key]
    return new_dict


def count_log_lkl_by_list(flatten_likelihoods):
    if not flatten_likelihoods:
        return 1
    a = list(map(math.log, flatten_likelihoods))
    prod = np.sum(a)
    divi = prod / len(flatten_likelihoods)
    likelihood = math.exp(divi)
    return likelihood


def separate(x):
    if x == 0:
        return x
    mul_clean = math.floor(math.log10(abs(x)))
    mul = pow(10, mul_clean)
    if mul == 0:
        return 1, mul_clean
    return round(x / mul), mul_clean


def get_powers(data):
    number_powers = {k: 0 for k in range(1, 11)}
    for lkl in data:
        number, power = separate(lkl)
        number_powers[number] += power
    return number_powers


def choose_csv_from_dir(path_to_folder_to_choose_from):
    folders_ = os.listdir(path_to_folder_to_choose_from)
    folders_cleaned = [f for f in folders_ if os.path.isdir(os.path.join(path_to_folder_to_choose_from, f))]

    folders_cleaned.sort(
        key=lambda x: int(re.findall(r'_{}_{}'.format(SystemParams.Pattern, SystemParams.Pattern), x)[0][0]))
    files = []
    for folder in folders_cleaned:
        files_ = os.listdir(
            os.path.join(
                path_to_folder_to_choose_from,
                folder))
        hat_csv = [name for name in files_ if
                   name.endswith('{}.csv'.format(FilesProcessingParams.key_1_for_choose_from_dir))]

        if not hat_csv:
            by_ratio = [name for name in files_ if
                        name.endswith('{}.csv'.format(FilesProcessingParams.key_2_for_choose_from_dir))]
            assert by_ratio
            choice = by_ratio[0]
        else:
            choice = hat_csv[0]
        files.append(os.path.join(os.path.join(path_to_folder_to_choose_from, folder), choice))
    return files


def choose_best_csv_final_last(path_meta):
    folders_ = os.listdir(path_meta)
    folders = [f for f in folders_ if
               not (f.endswith('.png') or f.endswith('.сsv') or f.endswith('.txt')) and f != 'utils']
    folder = folders[0]
    final_folder_path = os.path.join(path_meta, folder)
    jsones = os.listdir(final_folder_path)
    best_json = [f for f in jsones if f.endswith("LAST_TRUE.csv")]
    if not best_json:
        ratio_json = [f for f in jsones if f.endswith("BEST_RATIO.csv")]
        choice = ratio_json[0]
    else:
        choice = best_json[0]
    path_to_best_json = os.path.join(final_folder_path, choice)
    return path_to_best_json


def list_split_in_consecutive_frames(list_of_values):
    splitted_segments = []
    curr_segment = []
    for i in range(len(list_of_values)):
        if i == 0:
            continue
        if list_of_values[i - 1] + 1 != list_of_values[i]:
            # if not curr_segment:
            curr_segment.append(list_of_values[i - 1])
            splitted_segments.append(curr_segment)
            curr_segment = []
        else:
            curr_segment.append(list_of_values[i - 1])
        if i == len(list_of_values) - 1:
            curr_segment.append(list_of_values[i])
            splitted_segments.append(curr_segment)
    return splitted_segments


def tracks_position_analyis(id_1_frames, id_2_frames, id_1, id_2):
    id_1_segments = list_split_in_consecutive_frames(id_1_frames)
    id_2_segments = list_split_in_consecutive_frames(id_2_frames)
    # assume they do not intersect
    segments_union = id_1_segments + id_2_segments
    # segments_ids = [id_1] * sum([len(i) for i in id_1_segments]) + [id_2] * sum([len(i) for i in id_2_segments])
    segments_ids = [id_1] * len(id_1_segments) + [id_2] * len(id_2_segments)
    segments_ids_sorted = sort_list(segments_union, segments_ids)
    return segments_ids_sorted


def sort_list(list1, list2):
    zipped_pairs = zip(list1, list2)

    z = [(x, y) for (y, x) in sorted(zipped_pairs)]

    return z
