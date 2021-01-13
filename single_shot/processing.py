from __future__ import absolute_import, division

import os

import numpy as np

import initialisation.face as facu
from FCS import fixed_coordinate_system as fixu
from config import LogicParams
from filtering import Filter


def filters_init(tracks_ids):
    filters = {}
    for id in tracks_ids:
        filters[id] = Filter()
    return filters


def filter_set_init(filter, curr_track):
    curr_track_init_state_0 = [curr_track[0][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][0]],
                               curr_track[0][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][1]]]
    curr_track_init_state_1 = [curr_track[1][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][0]],
                               curr_track[1][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][1]]]
    curr_track_first_frame = curr_track[0]['frame_no']
    filter.set_filter_initial_state(curr_track_init_state_0, curr_track_init_state_1, curr_track_first_frame)


def filter_already_processed_states_pass(filter, curr_track_elems):
    curr_track_states = [[i[LogicParams.parts_.keys_to_use_for_estimation_pairs[0][0]],
                          i[LogicParams.parts_.keys_to_use_for_estimation_pairs[0][1]]] for i in curr_track_elems]
    filter.filter.batch_filter(curr_track_states)


class SingleShotProcessing:
    def __init__(self, info, start_frame, path_to_homography_dict=None, frame_no_k=5, K=10, max_k=13, delta=1e-200):
        self.info = info
        info = facu.add_centers_to_meta(info)
        self.homography_dict = None
        if path_to_homography_dict is not None:
            self.homography_dict = fixu.reformat_homography_dict(
                path_to_homography_dict)
            self.info = fixu.to_fixed_coordinate_system(
                info, self.homography_dict, int(os.environ.get('fixed_coordinate_resize_h')),
                int(os.environ.get('fixed_coordinate_resize_w')), int(os.environ.get('img_h')),
                int(os.environ.get('img_w'))

            )
        if start_frame is None:
            start_frame = min(list(info.keys()))
        self.start_frame = start_frame
        self.frame_no_k = frame_no_k
        self.delta = delta
        self.max_k = max_k
        self.min_track_len = 3
        self.last_init_frame_no = start_frame + frame_no_k
        self.filters, self.initialised_tracks, self.meta_left, self.initialised_meta, self.K = self.prepare_and_init_filters()
        self.free_rectangles = {}
        self.free_rect_for_new_track_min = 5
        self.new_meta = {}

    def prepare_and_init_filters(self):
        initialised_tracks, meta_left, initialised_meta = meta_preparation(self.info, self.start_frame, self.frame_no_k,
                                                                           self.min_track_len, self.last_init_frame_no)
        tracks_ids = list(initialised_tracks.keys())
        K = len(tracks_ids)
        filters = filters_init(tracks_ids)
        return filters, initialised_tracks, meta_left, initialised_meta, K

    def free_rects_iou_analysis(self, frame_no):
        initialise_ = facu.FaceMetaInitialisation(self.free_rectangles)
        initialised_meta = initialise_.initialisation()
        appr_meta = facu.fill_in_gaps(initialised_meta, 0.3, 4, [-1, -10],
                                      2)
        initialise_ = facu.FaceMetaInitialisation(appr_meta)  # repeat initialisation after approximation
        initialised_meta = initialise_.initialisation()
        tracks_new_ = facu.get_tracks(initialised_meta)
        for track_id, track_value in tracks_new_.items():
            if len(track_value) >= self.free_rect_for_new_track_min and track_value[-1]['frame_no'] == frame_no:
                new_id = self.K + 1
                self.filters[new_id] = Filter()
                filter_set_init(self.filters[new_id], track_value)
                track_values_state = track_value[2:]
                filter_already_processed_states_pass(self.filters[new_id], track_values_state)
                self.K += 1
                assert self.K <= self.max_k

    def filter_rectangles(self, frame_no):
        filtered_rects = {}
        for key, value in self.free_rectangles.items():
            if frame_no - self.free_rect_for_new_track_min <= key <= frame_no:
                filtered_rects[key] = value
        self.free_rectangles = filtered_rects

    def get_free_rectangles(self, frame_no, frame_no_values, final_correspondence):
        rects_left_inds = list(
            set([i for i in range(len(frame_no_values))]).difference(
                set(list(final_correspondence.keys()))))  # free тать только те что со всем оч плохо например
        for ind in rects_left_inds:
            self.free_rectangles[frame_no].append(frame_no_values[ind])
        if len(list(self.free_rectangles.keys())) >= self.free_rect_for_new_track_min:
            self.free_rects_iou_analysis(frame_no)

    def get_final_ids_correspondence(self, current_id_filters, frame_no_values):
        lkls_matr_arr = get_lkl_matrix(self.filters, current_id_filters, frame_no_values,
                                       LogicParams.parts_.keys_to_use_for_estimation_pairs)
        curr_frame_lkls = find_assigment(lkls_matr_arr, current_id_filters)
        final_correspondence = curr_frame_lkls_analysis(curr_frame_lkls)
        return final_correspondence

    def single_frame_processing(self, frame_no, frame_no_values):
        self.free_rectangles[frame_no] = []
        self.new_meta[frame_no] = []
        current_id_filters = list(self.filters.keys())
        updated_filters_id = []
        final_correspondence = self.get_final_ids_correspondence(current_id_filters, frame_no_values)

        for meas_index, id_and_lkl in final_correspondence.items():
            if id_and_lkl[1] >= self.delta:
                value_to_update = [
                    frame_no_values[meas_index][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][0]],
                    frame_no_values[meas_index][LogicParams.parts_.keys_to_use_for_estimation_pairs[0][1]]]
                new_meta_item = frame_no_values[meas_index]
                new_meta_item['index'] = np.float64(id_and_lkl[0])
                self.new_meta[frame_no].append(new_meta_item)
            else:
                value_to_update = [np.nan, np.nan]
            self.filters[id_and_lkl[0]].filter.update(value_to_update)
            updated_filters_id.append(id_and_lkl[0])
        left_filters_ids = list(set(current_id_filters).difference(set(updated_filters_id)))
        if left_filters_ids:
            for id_not_updated in left_filters_ids:
                self.filters[id_not_updated].filter.update([np.nan, np.nan])

        self.get_free_rectangles(frame_no, frame_no_values, final_correspondence)
        self.filter_rectangles(frame_no)

    def processing_info(self):
        for track_id, curr_track in self.initialised_tracks.items():
            filter_set_init(self.filters[track_id], curr_track)
            curr_track_elems = curr_track[2:]
            filter_already_processed_states_pass(self.filters[track_id], curr_track_elems)

        for frame_no, frame_no_values in self.meta_left.items():
            self.single_frame_processing(frame_no, frame_no_values)

        self.new_meta.update(self.initialised_meta)

        if self.homography_dict is not None:
            self.new_meta = fixu.fixed_to_original_coordinate_system(
                self.new_meta, self.homography_dict, int(os.environ.get('fixed_coordinate_resize_h')),
                int(os.environ.get('fixed_coordinate_resize_w')), int(os.environ.get('img_h')),
                int(os.environ.get('img_w'))
            )

        return self.new_meta
