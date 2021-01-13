import numpy as np
from scipy.optimize import linear_sum_assignment

import initialisation.face as facu
from config import LogicParams
from filtering import Filter


def get_lkl_matrix(filters, current_id_filters, frame_no_values, keys_for_estimation):
    lkls_matr = []
    for single_track_id in current_id_filters:
        assert filters[single_track_id].set_initial_state
        filters[single_track_id].filter.predict_filter()
        curr_frame_track_lkls = filters[single_track_id].filter.process_filter(frame_no_values, keys_for_estimation)
        lkls_matr.append(curr_frame_track_lkls)
    lkls_matr_arr = np.array(lkls_matr)
    return lkls_matr_arr


def find_assigment(lkls_matr_arr, current_id_filters):
    cost = -lkls_matr_arr
    correspondence = {}
    row_ind, col_ind = linear_sum_assignment(cost)  # так как минимизируем
    assert len(list(set(row_ind))) == len(row_ind)
    assert len(list(set(col_ind))) == len(col_ind)
    for row_i, col_j in zip(row_ind, col_ind):
        track_id_index = current_id_filters[row_i]
        correspondence[track_id_index] = [col_j, lkls_matr_arr[row_i][col_j]]  # нумерация треков идет с 1
    return correspondence


def curr_frame_lkls_analysis(curr_frame_lkls):
    final_correspondence = {}
    for track_id, ind_and_lkl in curr_frame_lkls.items():
        if ind_and_lkl[0] not in final_correspondence:
            final_correspondence[ind_and_lkl[0]] = [track_id, ind_and_lkl[1]]
        else:
            if ind_and_lkl[1] >= final_correspondence[ind_and_lkl[0]][1]:
                final_correspondence[ind_and_lkl[0]] = [track_id, ind_and_lkl[1]]
    return final_correspondence




def get_track_likelihoods(info, track_number):
    info = facu.add_centers_to_meta(info)
    tracks = facu.get_tracks(info)
    track_values_we_need = tracks[track_number]
    first_frame = track_values_we_need[0]['frame_no']

    track_states = []
    for single_segment in track_values_we_need:
        track_states.append([single_segment[[LogicParams.parts_.keys_to_use_for_estimation_pairs[0]]],
                             single_segment[[LogicParams.parts_.keys_to_use_for_estimation_pairs[1]]]])
    filter__ = Filter()
    filter__.initialise_filter()
    mu, cov, likelihoods = filter__.get_likelihoods_with_kalman_filter(track_states)
    return likelihoods

