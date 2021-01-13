import logging

import numpy as np

from KalmanFilter import KalmanFilter
from config import KalmanParams


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
        self.first_frame = None
        self.set_initial_state = False

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

    def set_filter_initial_state(self, states_1, states_2, first_frame):
        self.first_frame = first_frame
        initial_state = [states_2[0], states_2[1], states_2[0] - states_1[0],
                         states_2[1] - states_1[1]]
        assert not np.all(np.isnan(initial_state))
        assert states_2[0] != 0 and states_2[1] != 0

        self.filter.x = np.array([initial_state[0], initial_state[1], initial_state[2],
                                  initial_state[3]]).T
        self.set_initial_state = True
        # states_info = states_info[2:]

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
