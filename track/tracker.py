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
import copy
import logging
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import os

import initialisation.pose_utils as posu
import utils.utils_ as util
import utils.utils_pandas_df as pdu
import visualization.visualization as visu
from additional.kalman_filter import KalmanFilter
from config import AcceptanceParams, LogicParams, DrawingParams, MetaProcessingParams
from config import KalmanParams
import utils.utils_math as utils_math


class AbstractTracker(ABC):
    @abstractmethod
    def __init__(self, data_processing, meta_partition, files_work):
        self.files_ = files_work
        self.final_merge = None
        self.p_d = AcceptanceParams.p_d
        self.p_z = AcceptanceParams.p_z
        self.lambda_b = AcceptanceParams.lambda_b
        self.lambda_f = AcceptanceParams.lambda_f
        self.change_track = {}
        self.chosen_move = 3
        self.acc_list = []
        self.u_list = []
        self.accepted = False
        self.u_random_curr_iter = None
        self.curr_acceptance = None
        self.ratio = None
        self.priors_parameters = {}
        self.acceptance = None
        self.__current_state = None
        self.meta_partition = meta_partition
        self.data_processing = data_processing
        self.cur_iter_name = 'initial'
        self.likelihood = None
        self.likelihoods = None
        self.priors = None
        self.first_iteration_done = False
        self.returned_state = False
        self.proposed_partition = None
        self.df_grouped_ids_proposed = None
        self.acc_obj = Acceptance()
        self.iteration = 0
        self.complete_iter_number = 0
        self.accepted = False

    def algo_iteration(self):
        """ Single interation of the algorithm"""
        if self.final_merge is None:
            self.internal_loop()
        else:
            self.final_merge_loop()

    def acc_update(self):
        self.accepted = True
        self.meta_partition.data_df = self.proposed_partition
        self.data_processing.dataframe = self.proposed_partition
        self.proposed_partition = None
        iter_info = round(self.acc_obj.u_random_curr_iter,
                          2) if self.acc_obj.u_random_curr_iter is not None else self.cur_iter_name
        self.meta_partition.info_name = str(self.complete_iter_number) + '_' + str(self.chosen_move) + '_' + str(round(
            max(list(self.acc_obj.curr_acceptance)),
            2)) + '_' + str(iter_info) + '_' + str(round(
            max(list(self.acc_obj.ratio.values())),
            2))

    @abstractmethod
    def internal_loop(self):
        raise NotImplementedError("Must override internal_loop")

    @abstractmethod
    def choose_move(self):
        raise NotImplementedError("Must override choose_move")

    @abstractmethod
    def propose(self):
        raise NotImplementedError("Must override propose")

    def get_current_id_(self, segments_ids, frame_no):
        i = [i for i in range(len(segments_ids) - 1) if
             segments_ids[i][1][0] <= frame_no < segments_ids[i + 1][1][0]]
        if i:
            assert len(i) == 1
            i = i[0]
            curr_id_to_continue = segments_ids[i][0]
        else:
            curr_id_to_continue = segments_ids[-1][0]
        return curr_id_to_continue

    def states_analysis(self, frame_numbers_indexes_current, frame_numbers_indexes_proposed, states,
                        frame_numbers_states):  # gaps will be filled in in Kalman filter
        new_track_id = self.change_track['new'][0]
        id_1 = self.change_track['current'][0]
        id_2 = self.change_track['current'][1]
        states_tracks = {id_1: {}, id_2: {}}
        segments_ids = util.tracks_position_analyis(self.data_processing.frames_no[id_1],
                                                    self.data_processing.frames_no[id_2], id_1,
                                                    id_2)
        frames_numbers_to_remove_likelihoods = {id_1: {}, id_2: {}}
        frames_numbers_states_current = {id_1: {}, id_2: {}}
        for pair_name, states_values in states[new_track_id].items():
            for key, value in states_tracks.items():
                states_tracks[key][pair_name] = []
                frames_numbers_to_remove_likelihoods[key][pair_name] = []
                frames_numbers_states_current[key][pair_name] = []
            states_new_track = np.stack(states_values)
            for index, (frame_no, state) in enumerate(
                    zip(frame_numbers_states[new_track_id][pair_name], states_new_track)):
                curr_id_to_continue = self.get_current_id_(segments_ids, frame_no)
                second_id = id_1 if curr_id_to_continue == id_2 else id_2
                next_is_nan = False
                if index + 1 < len(states_new_track) and np.all(np.isnan(states_new_track[index + 1])):
                    next_is_nan = True
                if (np.all(np.isnan(state)) or next_is_nan) and not states_tracks[curr_id_to_continue][
                    pair_name]:  # if this track begin from nan or second is nan - there will no be initiaal state in kalman
                    old_sec_id = second_id
                    second_id = curr_id_to_continue
                    curr_id_to_continue = old_sec_id
                states_tracks[curr_id_to_continue][pair_name].append(state.tolist())
                frames_numbers_states_current[curr_id_to_continue][pair_name].append(frame_no)

                if states_tracks[second_id][pair_name] != [] and second_id in [j[0] for j in segments_ids if
                                                                               j[1][0] > frame_no]:
                    states_tracks[second_id][pair_name].append([np.nan, np.nan])
                    frames_numbers_states_current[second_id][pair_name].append(frame_no)
                    frames_numbers_to_remove_likelihoods[second_id][pair_name].append(frame_no)
        return states_tracks, frame_numbers_indexes_current, frame_numbers_indexes_proposed, frames_numbers_to_remove_likelihoods, frames_numbers_states_current

    def frames_numbers_analysis(self, frame_numbers_indexes):
        # frames_numbers, counts = np.unique(frame_numbers_indexes[:, 0], return_counts=True)
        indexes_for_every_frame = {}
        for element in frame_numbers_indexes:
            if element[0] not in indexes_for_every_frame:
                indexes_for_every_frame[element[0]] = []
            indexes_for_every_frame[element[0]].append(element[1])
        return indexes_for_every_frame

    def final_merge_loop(self):
        flag = True
        check_ = 0
        counter = 0
        if not self.data_processing.pairs_to_consider:
            return
        while flag:
            counter += 1
            break_to_while = False
            if not self.data_processing.pairs_to_consider:
                break

            for index, pair in enumerate(self.data_processing.pairs_to_consider):
                self.cur_iter_name = str(counter) + '_' + str(pair)
                logging.debug('pair {} '.format(pair))

                if break_to_while:
                    check_ = 0
                    break
                self.accepted = False
                self.merge_move(final_merge=True,
                                pair=pair)
                logging.debug('self.change_track {} '.format(self.change_track))
                if self.proposed_partition is None:
                    if check_ >= len(self.data_processing.pairs_to_considercleaned) ** 2:
                        break_to_while = False
                        flag = False
                        break
                    else:
                        check_ += 1
                        continue
                states_proposed, frame_numbers_states = pdu.get_particular_states(self.proposed_partition,
                                                                                  self.change_track[
                                                                                      'new'])
                states_proposed = pdu.states_proposed_cleaning(states_proposed)
                frame_numbers_indexes_proposed = self.proposed_partition[['frame_no', 'id']].values
                frame_numbers_indexes_current = self.data_processing.dataframe[['frame_no', 'id']].values
                indexes_for_every_frame_current = self.frames_numbers_analysis(frame_numbers_indexes_current)
                indexes_for_every_frame_proposed = self.frames_numbers_analysis(frame_numbers_indexes_proposed)
                states_current_with_gap, indexes_for_every_frame_gap_current, indexes_for_every_frame_gap_proposed, \
                frames_numbers_to_remove_likelihoods, frames_numbers_states_current = self.states_analysis(
                    indexes_for_every_frame_current, indexes_for_every_frame_proposed, states_proposed,
                    frame_numbers_states)
                states_proposed, states_current_with_gap = pdu.clean_states(states_proposed, states_current_with_gap)
                self.acc_obj.current_tracks_probs_initialisation(indexes_for_every_frame_gap_current,
                                                                 states_current_with_gap, frames_numbers_states_current,
                                                                 frames_numbers_to_remove_likelihoods)
                self.acc_obj.propose(indexes_for_every_frame_gap_proposed, states_proposed, frame_numbers_states)

                accepted_count = self.acc_obj.analyse_acceptance(self.change_track)
                ratio = max(list(
                    self.acc_obj.ratio.values()))
                if DrawingParams.draw_every_iteration:
                    current = pdu.from_dataframe_to_dict(self.meta_partition.data_df)
                    visu.draw_partition(current, int(os.environ.get('img_w')),
                                        'partitions_iteration_{}_{}_'.format(self.cur_iter_name,
                                                                             "current_"),
                                        self.files_.curr_window_dir)

                if accepted_count >= AcceptanceParams.number_of_acc_for_acc:
                    self.acc_obj.accepted_(change_track=self.change_track)
                    self.meta_partition.data_df = self.proposed_partition
                    self.acc_update()
                    break_to_while = True
                if break_to_while:
                    check_ = 0
                    break
                check_ += 1
                if not break_to_while and check_ >= len(self.data_processing.pairs_to_consider):
                    flag = False

    def merge_move(self, final_merge=None, pair=None):
        if final_merge is None:
            if not self.data_processing.pairs_to_consider:
                self.returned_state = True
                return

            pair_selection = np.random.random_integers(
                0, len(self.data_processing.pairs_to_consider) - 1)
            self.create_new_partition_merge(self.data_processing.pairs_to_consider[pair_selection])
            del self.data_processing.pairs_to_consider[pair_selection]
        else:
            self.create_new_partition_merge(pair)

    def create_new_partition_merge(self, pair_chosen):
        self.change_track['current'] = [pair_chosen[0], pair_chosen[1]]
        new_df = self.data_processing.dataframe.copy(deep=True)
        new_index = max(self.data_processing.current_meta_indexes) + 1
        new_df = pdu.change_index_in_df(new_df, pair_chosen[0], new_index)
        new_df = pdu.change_index_in_df(new_df, pair_chosen[1], new_index)

        self.proposed_partition = new_df
        self.df_grouped_ids_proposed = new_df.groupby([new_df.id])
        self.change_track['new'] = [new_index]


class Acceptance:
    def __init__(self):
        """

        :param frame_no_ind: list of lists in the form [[frame_no, ind], [frame_no, ind], ...] - information for priors computation
        :param states_: dict in the form {id : {('body_part_x','body_part_y'):[[x_1, x_2, ...],[y_1, y_2, ...]]}, {...}, ...} - information for likelihoods computation
        """
        self.first_iteration_done = False
        self.u_random_curr_iter = np.random.random()
        self.curr_acceptance = 0
        self.ratio = {}
        self.curr_priors_obj = None
        self.curr_liks_obj = None
        self.acceptance = {}
        self.proposed_priors_obj = None
        self.proposed_liks_obj = None

    def current_tracks_probs_initialisation(self, frame_no_ind, states_, states_frames_numbers,
                                            frames_numbers_do_not_consider):
        self.curr_priors_obj = Priors(frame_no_ind)
        self.curr_liks_obj = Likelihoods(states_, states_frames_numbers, frames_numbers_do_not_consider)

    def propose(self, frame_no_ind, states, states_frames_numbers, frames_numbers_do_not_consider=None):
        assert frames_numbers_do_not_consider is None
        self.u_random_curr_iter = np.random.random()
        self.proposed_priors_obj = Priors(frame_no_ind)
        self.proposed_liks_obj = Likelihoods(states, states_frames_numbers, frames_numbers_do_not_consider)

    def analyse_acceptance(self, change_track):
        """ Analyzing ratio and acceptance. """

        self.curr_acceptance = self.get_acceptance(change_track)
        logging.debug('acc {} '.format(self.curr_acceptance))
        logging.debug(
            'self.curr_acceptance {} '.format(
                self.curr_acceptance))
        self.curr_acceptance = list(self.curr_acceptance.values())
        if not self.curr_acceptance:
            return None
        if not any(np.isfinite(list(self.ratio.values()))):
            raise ValueError('nan ratio')
        if max(self.curr_acceptance) < AcceptanceParams.acc:  # sometimes want to filter too low acc
            self.curr_acceptance = 0

        if AcceptanceParams.use_random_u:
            accepted_count = np.count_nonzero(
                np.array(self.curr_acceptance) > self.u_random_curr_iter)
        else:
            accepted_count = np.count_nonzero(
                np.array(self.curr_acceptance) > AcceptanceParams.acc)
        return accepted_count

    @staticmethod
    def get_posterior(liks_obj, priors_obj):
        """ Compute posterior for some partition"""
        liks_obj.compute_likelihood()
        priors_obj.compute_priors()

    def get_acceptance(self, change_track):
        self.get_posterior(self.curr_liks_obj, self.curr_priors_obj)
        self.get_posterior(self.proposed_liks_obj, self.proposed_priors_obj)
        self.ratio = {}

        all_keys_list = LogicParams.parts_.keys_to_use_for_estimation_pairs
        for pair in all_keys_list:
            ratio = self.compute_ratio(pair, change_track)
            self.ratio[pair] = ratio
            self.analyse_ratio(pair)
        logging.info('ratio {} '.format(self.ratio))
        return self.acceptance

    def analyse_ratio(self, pair):
        if np.isfinite(self.ratio[pair]):
            if self.ratio[pair] == 1:
                self.acceptance[pair] = 0
            else:
                self.acceptance[pair] = min(1, self.ratio[pair])
        else:
            self.acceptance[pair] = 0

    def likelihoods_preparation(self, likelihoods, name):
        logging.debug(' {} : {}'.format(name, likelihoods))
        likelihoods_rounded = [utils_math.roundFirst(x) for x in likelihoods]

        return likelihoods_rounded

    def likelihoods_multiplication(self, new_diff_curr,
                                   curr_diff_new):  # cannot kist multiply, on long time segment product will be zero
        numbers_powers_new_d_c = util.get_powers(new_diff_curr)
        numbers_powers_curr_d_new = util.get_powers(curr_diff_new)
        values = 0
        for key, value in numbers_powers_curr_d_new.items():
            values += numbers_powers_new_d_c[key] - value
        power = values / len(curr_diff_new)
        lkls_ratio = 10 ** power
        return lkls_ratio

    def compute_ratio(self, pair, change_track):
        priors_curr_diff_new = list((Counter(self.curr_priors_obj.priors_numbers) - (
            Counter(self.proposed_priors_obj.priors_numbers))).elements())
        priors_new_diff_curr = list((Counter(self.proposed_priors_obj.priors_numbers) - Counter(
            self.curr_priors_obj.priors_numbers)).elements())
        priors_d = (np.prod(np.array(priors_new_diff_curr)) / np.prod(
            np.array(priors_curr_diff_new)))

        logging.debug('priors_new_diff_curr {} '.format(priors_new_diff_curr))
        logging.debug('priors_curr_diff_new {} '.format(priors_curr_diff_new))

        proposed_likelihoods_for_consideration = self.proposed_liks_obj.sort_by_pairs()
        current_likelihoods_for_consideration = \
            self.curr_liks_obj.sort_by_pairs(particular_ids=change_track['current'])
        try:
            proposed_likelihoods_for_consideration_curr_pair = proposed_likelihoods_for_consideration[pair]
            current_likelihoods_for_consideration_curr_pair = current_likelihoods_for_consideration[pair]
        except KeyError:
            logging.info('no states for {} pair'.format(pair))
            return 0
        assert len(proposed_likelihoods_for_consideration_curr_pair) == len(
            current_likelihoods_for_consideration_curr_pair) + 2
        proposed_likelihoods_for_consideration_curr_pair_prepared = self.likelihoods_preparation(
            proposed_likelihoods_for_consideration_curr_pair, 'proposed')
        current_likelihoods_for_consideration_curr_pair_prepared = self.likelihoods_preparation(
            current_likelihoods_for_consideration_curr_pair, 'current')
        lkls_new_diff_curr = list((Counter(proposed_likelihoods_for_consideration_curr_pair_prepared) - (
            Counter(
                current_likelihoods_for_consideration_curr_pair_prepared))).elements())  # for precision and performance, consider only difference
        lkls_curr_diff_new = list((Counter(current_likelihoods_for_consideration_curr_pair_prepared) - Counter(
            proposed_likelihoods_for_consideration_curr_pair_prepared)).elements())
        lkls_ratio = self.likelihoods_multiplication(lkls_new_diff_curr, lkls_curr_diff_new)
        # lkls_ratio = (
        #         util.count_log_lkl_by_list(lkls_new_diff_curr) /
        #         util.count_log_lkl_by_list(lkls_curr_diff_new))
        ratio = priors_d * lkls_ratio
        return ratio

    def choose_likelihoods_of_difference(self, pair, change_track):
        likelihoods_we_need = []
        for id in change_track['current']:
            likelihoods_we_need.append(
                self.curr_liks_obj.likelihoods_numbers[id][pair])
        likelihoods_we_need = [
            i for subl in likelihoods_we_need for i in subl]
        return likelihoods_we_need

    def accepted_(self, change_track=None):
        self.accepted = True
        self.curr_priors_obj = self.proposed_priors_obj

        for id_ in change_track['current']:
            self.curr_liks_obj.delete_likelihoods_by_id(id_)
        self.curr_liks_obj.add_likelihoods(self.proposed_liks_obj.likelihoods_numbers)


class Likelihoods:
    def __init__(self, states_likelihoods_need, states_frames_numbers, frames_numbers_do_not_consider=None):
        self.filter = Filter()
        self.__states = states_likelihoods_need
        self.__states_frames_numbers = states_frames_numbers
        self.__frames_numbers_do_not_consider = frames_numbers_do_not_consider
        self.__likelihoods = {}
        self.likelihood = {}

    @property
    def likelihoods_numbers(self):
        return self.__likelihoods

    def delete_likelihoods_by_id(self, id):
        try:
            del self.__likelihoods[id]
        except:
            raise ValueError('cannot delete such id')

    def add_likelihoods(self, new_liks_id):
        self.__likelihoods.update(new_liks_id)

    def compute_likelihood(self):

        for track_index, track_state in self.__states.items():
            assert track_index not in MetaProcessingParams.false_indexes
            self.__likelihoods[track_index] = {}
            for pair_name, pair_state in track_state.items():
                if pair_name not in LogicParams.parts_.keys_to_use_for_estimation_pairs:
                    continue
                # final_states = np.stack(pair_state, axis=1)
                final_states = pair_state
                assert len(final_states) > 2  # for kalman
                self.find_single_likelihood(
                    final_states, pair_name, track_index)
            if 'person' in track_state:
                self.similarities_pose, pose_2 = posu.compute_pose_similarity_score(
                    track_state['person'])
                pose_states = np.stack(
                    [self.similarities_pose, pose_2], axis=1)
                self.find_single_likelihood(
                    pose_states, 'person', track_index)

    def find_single_likelihood(
            self, final_states, pair_name, track_index):
        mu, cov, likelihoods = self.filter.get_likelihoods_with_kalman_filter(
            final_states)
        if likelihoods:
            likelihoods_ready = likelihoods
            if self.__frames_numbers_do_not_consider is not None:
                if self.__frames_numbers_do_not_consider[track_index][pair_name]:
                    likelihoods_ready = self.clean_likelihoods(likelihoods, track_index, pair_name)
            self.__likelihoods[
                track_index][
                pair_name] = likelihoods_ready

    def clean_likelihoods(self, likelihoods, track_index, pair_name):
        cleaned_likelihoods = []
        for frame_no, lkl in zip(self.__states_frames_numbers[track_index][pair_name], likelihoods):
            if frame_no not in self.__frames_numbers_do_not_consider[track_index][pair_name]:
                cleaned_likelihoods.append(lkl)
        return cleaned_likelihoods

    def sort_by_pairs(self, particular_ids=None):
        new_likelihoods = {}
        for track_id, likelihoods_pairs in self.likelihoods_numbers.items():
            if particular_ids:
                if track_id not in particular_ids:
                    continue
            for pair_name, curr_pair_likelihoods in likelihoods_pairs.items():
                if pair_name not in new_likelihoods:
                    new_likelihoods[pair_name] = []
                new_likelihoods[pair_name].append(curr_pair_likelihoods)
        for pair, pair_liks in new_likelihoods.items():
            new_likelihoods[pair] = [
                i for subl in pair_liks for i in subl]
        return new_likelihoods


class Priors:
    def __init__(self, indexes_for_every_frame):
        self.indexes_for_every_frame = indexes_for_every_frame
        self.indexes_for_every_frame_values = list(self.indexes_for_every_frame.values())
        self._priors = None
        self.e_t_factrs = None
        self.a_t = None
        self.z_t = None
        self.c_t = None
        self.d_t = None
        self.f_t = None
        self.g_t = None
        self.e_t_1 = None
        self.tracks_numbers_at_curr_frame = None
        self.tracks_numbers_at_prev_frame = None
        self.det_falses = None
        self.__priors = None
        self.process_meta()

    @staticmethod
    def compute_single_prior(e_t, z_t, c_t, d_t, g_t, a_t, f_t):
        curr_prior = e_t * (AcceptanceParams.p_z ** z_t) * ((1 - AcceptanceParams.p_z) ** c_t) * \
                     (AcceptanceParams.p_d ** d_t) * ((1 - AcceptanceParams.p_d) ** g_t) * (
                             AcceptanceParams.lambda_b ** a_t) * (AcceptanceParams.lambda_f ** f_t)
        return curr_prior

    def process_meta(self):
        # indexes_for_every_frame_ = np.split(self.arr[:, 1], np.cumsum(
        #     np.unique(self.arr[:, 0], return_counts=True)[1])[:-1])
        # indexes_for_every_frame = [list(index)
        #                            for index in indexes_for_every_frame_]
        self.indexes_for_every_frame_values = list(
            map(pdu.remove_str_from_indexes, self.indexes_for_every_frame_values))
        len_indexes_for_every_frame = list(
            map(pdu.get_len_single, self.indexes_for_every_frame_values))
        self.det_falses = list(
            map(pdu.get_false_inds_and_detections, self.indexes_for_every_frame_values[1:]))
        self.e_t_1 = len_indexes_for_every_frame[:-1]
        self.tracks_numbers_at_curr_frame = self.indexes_for_every_frame_values[1:]
        self.tracks_numbers_at_prev_frame = self.indexes_for_every_frame_values[:-1]

    def get_characteristics_priors(self):
        """ Compute characteristics according to the article."""

        self.e_t_factrs = list(
            map(pdu.get_len_single_fact, self.indexes_for_every_frame_values[1:]))
        self.a_t = list(map(pdu.diff_consecutive_frames,
                            self.tracks_numbers_at_curr_frame,
                            self.tracks_numbers_at_prev_frame))
        self.z_t = list(map(pdu.diff_consecutive_frames, self.tracks_numbers_at_prev_frame,
                            self.tracks_numbers_at_curr_frame))
        self.c_t = [a - b for a, b in zip(self.e_t_1, self.z_t)]
        self.d_t = list(np.array(self.det_falses)[:, 0:1].T[0])
        self.f_t = list(np.array(self.det_falses)[:, 1:2].T[0])
        self.g_t = [
            a - b + c - d for a,
                              b,
                              c,
                              d in zip(
                self.e_t_1,
                self.z_t,
                self.a_t,
                self.d_t)]

    def compute_priors(self):
        """ Compute priors."""
        self.get_characteristics_priors()
        curr_priors = list(map(self.compute_single_prior,
                               self.e_t_factrs, self.z_t, self.c_t, self.d_t, self.g_t, self.a_t, self.f_t))
        self.__priors = curr_priors

    @property
    def priors_numbers(self):
        return self.__priors


class Filter:
    def __init__(self):
        dt = 1
        self.matrix_a = np.array([[1, 0, dt, 0],
                                  [0, 1, 0, dt],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])

        self.matrix_g = np.array([[(dt ** 2) / 2, 0],
                                  [0, (dt ** 2) / 2],
                                  [dt, 0],
                                  [0, dt]])

        self.matrix_c = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.r = np.zeros((2, 2), int)
        np.fill_diagonal(self.r, KalmanParams.r)
        self.q = np.zeros((4, 4), int)
        np.fill_diagonal(self.q, KalmanParams.q)
        self.filter = self.initialise_filter()
        logging.debug('filter initialised')

    def initialise_filter(self):
        filter_ = KalmanFilter(dim_x=4,
                               dim_z=2)  # need to instantiate every time to reset all fields
        filter_.F = self.matrix_a
        filter_.H = self.matrix_c
        filter_.B = self.matrix_g

        if KalmanParams.use_noise_in_kalman:
            u = np.random.normal(loc=0, scale=KalmanParams.var_kalman, size=2)
            filter_.u = u
        # u = Q_discrete_white_noise(dim=2, var=1)

        filter_.Q = self.q
        filter_.R = self.r
        return filter_

    def get_likelihoods_with_kalman_filter(self, states_info):
        self.initialise_filter()
        initial_state = [states_info[1][0], states_info[1][1], states_info[1][0] - states_info[0][0],
                         states_info[1][1] - states_info[0][1]]
        assert not np.all(np.isnan(initial_state))
        assert states_info[1][0] != 0 and states_info[1][1] != 0

        self.filter.x = np.array([initial_state[0], initial_state[1], initial_state[2],
                                  initial_state[3]]).T
        states_info = states_info[2:]
        mu = []
        cov = []

        likelihoods, xs, xu, means, covariances, means_p, covariances_p = self.filter.batch_filter(np.array(
            states_info))
        return mu, cov, likelihoods


class ExtendedPartition:
    def __init__(self, partition, grouped_ids, states):
        self.partition = partition
        self.grouped_ids = grouped_ids
        self.states = states

    class Memento(object):
        def __init__(self, mstate):
            self.mstate = mstate

        def rollback_state(self):
            return self.mstate

    def set_state(self, state):
        self.__current_state = state

    @property
    def curr_st(self):
        return self.__current_state

    def save_state(self):
        return self.Memento(copy.deepcopy(self))

    def rollback_state(self, memento):
        self = memento.rollback_state()
        print('rollback to state {} '.format(self.curr_st))
