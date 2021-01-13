from __future__ import absolute_import, division

import numpy as np

import initialisation.face as facu
from config import LogicParams


def remove_ids_from_meta(meta, replace_rule):
    cleaned_meta = {}
    for frame_no, frame_meta in meta.items():
        curr_frame_new_meta = []
        for single_elem in frame_meta:
            if single_elem['index'] not in replace_rule:
                continue
            else:
                single_elem['index'] = replace_rule[single_elem['index']]
                curr_frame_new_meta.append(single_elem)
        cleaned_meta[frame_no] = curr_frame_new_meta
    return cleaned_meta


def meta_preparation(info, start_frame, frame_no_k, min_track_len, last_init_frame_no):
    meta_for_initialisation = {frame_no: value for (frame_no, value) in info.items() if
                               frame_no <= start_frame + frame_no_k}
    meta_left = {frame_no: value for (frame_no, value) in info.items() if frame_no > start_frame + frame_no_k}
    for frame_no, value in meta_left.items():
        for single_element in value:
            single_element['index'] = np.nan
    initialise_ = facu.FaceMetaInitialisation(meta_for_initialisation)
    initialised_meta = initialise_.initialisation()
    initialised_tracks = facu.get_tracks(initialised_meta)
    ids_to_remove = []
    final_tracks = {}
    for track_id, track_value in initialised_tracks.items():
        if len(track_value) < min_track_len:
            ids_to_remove.append(track_id)
            continue
        curr_track_value = track_value
        diff_frame_no = last_init_frame_no - track_value[-1]['frame_no']
        if diff_frame_no:
            for i in range(diff_frame_no):
                curr_track_value.append({LogicParams.parts_.keys_to_use_for_estimation_pairs[0][0]: np.nan,
                                         LogicParams.parts_.keys_to_use_for_estimation_pairs[0][1]: np.nan,
                                         'index': track_id, 'frame_no': track_value[-1]['frame_no'] + i + 1})
        final_tracks[track_id] = curr_track_value

    left_ids = list(final_tracks.keys())
    replace_rule = {}
    for index, id in enumerate(left_ids, 1):
        replace_rule[id] = index

    final_tracks = {replace_rule[id]: final_tracks[id] for id in final_tracks}

    initialised_meta = remove_ids_from_meta(initialised_meta, replace_rule)
    return final_tracks, meta_left, initialised_meta
