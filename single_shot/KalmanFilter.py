"""

FilterPy library.
http://github.com/rlabbe/filterpy

Documentation at:
https://filterpy.readthedocs.org

Supporting book at:
https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python

This is licensed under an MIT license. See the readme.MD file
for more information.

Copyright 2014-2018 Roger R Labbe Jr.
"""

import sys
from copy import deepcopy
from math import log, exp

import numpy as np
import numpy.linalg as linalg
from filterpy.common import reshape_z
from filterpy.stats import logpdf
from numpy import dot, zeros, eye, isscalar


class KalmanFilter(object):

    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or greater')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or greater')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or greater')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.velocity_x = []
        self.velocity_y = []
        self.x = zeros((dim_x, 1))  # state
        self.P = eye(dim_x)  # uncertainty covariance
        self.Q = eye(dim_x)  # process uncertainty
        self.B = None  # control transition matrix
        self.u = None
        self.F = eye(dim_x)  # state transition matrix
        self.H = zeros((dim_z, dim_x))  # Measurement function
        self.R = eye(dim_z)  # state uncertainty
        self._alpha_sq = 1.  # fading memory control
        self.M = np.zeros((dim_z, dim_z))  # process-measurement cross correlation
        self.z = np.array([[None] * self.dim_z]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((dim_x, dim_z))  # kalman gain
        self.y = zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))  # system uncertainty
        self.SI = np.zeros((dim_z, dim_z))  # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(dim_x)

        # these will always be a copy of x,P after predict() is called
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

        # these will always be a copy of x,P after update() is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv

        self.zs = []

    def predict(self, u=None, B=None, F=None, Q=None):
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x) * Q

        # x = Fx + Bu
        if B is not None and u is not None:
            self.x = dot(F, self.x) + dot(B, u)
        else:
            self.x = dot(F, self.x)
        self.velocity_x.append(self.x[2])
        self.velocity_y.append(self.x[3])

        # P = FPF' + Q
        self.P = self._alpha_sq * dot(dot(F, self.P), F.T) + Q

        # save prior
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()

    def find_likelihood(self, z, R=None, H=None):
        if np.all(np.isnan(z)):
            x_mean_velocity = np.mean(self.velocity_x[-5:])
            y_mean_velocity = np.mean(self.velocity_y[-5:])

            z_noise = np.random.normal(loc=0, scale=0, size=2)
            z = list(
                self.x[0:2] - [self.velocity_x[-1], self.velocity_y[-1]] + [x_mean_velocity, y_mean_velocity] + z_noise)
        z = reshape_z(z, self.dim_z, self.x.ndim)
        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R
        if H is None:
            H = self.H
        y = z - dot(H, self.x)
        PHT = dot(self.P, H.T)
        S = dot(H, PHT) + R
        _log_likelihood = logpdf(x=y, cov=S)

        _likelihood = exp(_log_likelihood)
        if _likelihood == 0:
            _likelihood = sys.float_info.min
        return _likelihood

    def update(self, z, R=None, H=None):

        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if np.all(np.isnan(z)):
            x_mean_velocity = np.mean(self.velocity_x[-5:])
            y_mean_velocity = np.mean(self.velocity_y[-5:])

            z_noise = np.random.normal(loc=0, scale=0, size=2)
            self.z = list(
                self.x[0:2] - [self.velocity_x[-1], self.velocity_y[-1]] + [x_mean_velocity, y_mean_velocity] + z_noise)
            z = list(
                self.x[0:2] - [self.velocity_x[-1], self.velocity_y[-1]] + [x_mean_velocity, y_mean_velocity] + z_noise)

        z = reshape_z(z, self.dim_z, self.x.ndim)
        self.zs.append(z)

        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H

        # y = z - Hx
        # error (residual) between measurement and prediction
        self.y = z - dot(H, self.x)

        # common subexpression for speed
        PHT = dot(self.P, H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        self.S = dot(H, PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)

        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        # self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)
        self.P = dot(I_KH, self.P)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

    def predict_filter(self, Fs=None, Qs=None, Bs=None, us=None):
        # if Bs is None:
        #     Bs = [self.B] * n
        # if us is None:
        #     us = [self.u] * n
        # if Fs is None:
        #     Fs = [self.F] * n
        # if Qs is None:
        #     Qs = [self.Q] * n
        self.predict(u=self.u, B=self.B, F=self.F, Q=self.Q)

    def process_filter(self, t_measurements, keys_for_estimation, Hs=None,
                       Rs=None):
        liks = []
        xs = []
        xu = []
        index_to_return = None
        xs.append(self.x)
        likelihoods = []
        for index, single_meas in enumerate(t_measurements):
            likelihoods.append(
                self.find_likelihood([single_meas[keys_for_estimation[0][0]], single_meas[keys_for_estimation[0][1]]]))
        return likelihoods

    def batch_filter(self, zs, Fs=None, Qs=None, Hs=None,
                     Rs=None, Bs=None, us=None, update_first=False,
                     saver=None):
        # pylint: disable=too-many-statements
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [self.u] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))
        liks = []
        xs = []
        xu = []
        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P
                xs.append(self.x)

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P
                xu.append(self.x)
                if z is not None:
                    liks.append(self.likelihood)

                if saver is not None:
                    saver.save()

        return (liks, xs, xu, means, covariances, means_p, covariances_p)

    @property
    def log_likelihood(self):
        """
        log-likelihood of the last measurement.
        """
        if self._log_likelihood is None:
            self._log_likelihood = logpdf(x=self.y, cov=self.S)
        return self._log_likelihood

    @property
    def likelihood(self):
        """
        Computed from the log-likelihood. The log-likelihood can be very
        small,  meaning a large negative value such as -28000. Taking the
        exp() of that results in 0.0, which can break typical algorithms
        which multiply by this value, so by default we always return a
        number >= sys.float_info.min.
        """
        if self._likelihood is None:
            self._likelihood = exp(self.log_likelihood)
            if self._likelihood == 0:
                self._likelihood = sys.float_info.min
        return self._likelihood
