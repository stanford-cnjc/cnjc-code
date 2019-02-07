"""Fitting linear dynamical systems with EM

Based off of "Parameter Estimation for Linear Dynamical Systems"
by Gharamani and Hinton 1994

"""

__author__ = 'Blue Sheffer'

import numpy as np
import numpy.linalg as LA
from sklearn.decomposition import FactorAnalysis
from sklearn import model_selection

class gLDS(object):

    def __init__(self):
        self.is_fit = False
        self.A = None
        self.C = None
        self.Q = None
        self.R = None
        self.pi_0 = None
        self.V_0 = None


    def fit(self, y, d_latent, num_iterations=100):
        """ Fits linear dynamical system to observations y with expectation-maximization

        Args:
            y (np.ndarray): (T, N, d_obs) or (T, d_obs), observed data
            d_latent (int): dimensionality of latent space
            num_iterations (int, optional): number of iterations for EM
            TODO: add with tolerance condition based on LL

        Returns:
            x_s (np.ndarray): (T, N, d_latent) or (T, d_latent), estimated latent trajectory
            likelihood (list of float): loglikelihood at each iteration of EM
        """
        if y.ndim == 2:
            y = np.expand_dims(y, 1)
        A, C, Q, R, pi_0, V_0 = self._initialize(y, d_latent)
        T = y.shape[0]
        N = y.shape[1]
        likelihood = []
        YY = np.sum(y * y, axis=(0, 1)).T / (T * N)
        x_s, V_s, Vtsum, YX, A1, A2, A3, lik = self._e_step(y, A, C, Q, R, pi_0, V_0)
        likelihood.append(lik)
        for _ in range(num_iterations):
            A, C, Q, R, pi_0, V_0 = self._m_step(YY, x_s, V_s, Vtsum, YX, A1, A2, A3)
            x_s, V_s, Vtsum, YX, A1, A2, A3, lik = self._e_step(y, A, C, Q, R, pi_0, V_0)
            likelihood.append(lik)

        self.is_fit = True
        x_s = np.squeeze(x_s)

        C, sorted_args = sort_by_column_norm(C, rowargs=[A])
        self.C = C
        self.A = sorted_args[0]
        self.Q = Q
        self.R = R

        self.pi_0 = pi_0
        self.V_0 = V_0
        return x_s, likelihood

    def transform(self, y):
        assert self.is_fit, "LDS has not been fit"
        x_s, V_s, Vtsum, YX, A1, A2, A3, lik = self._e_step(y, self.A, self.C, self.Q, self.R, self.pi_0, self.V_0)
        return x_s

    @staticmethod
    def _initialize(y, d_latent):
        """ Initialize models parameters and initial latents with Factor Analysis

        Args:
            y (np.ndarray): (T, N, d_obs), observed data
            d_latent (int): dimensionality of latent space

        Returns:
            A (np.ndarray): State dynamics matrix (d_latent, d_latent)
            C (np.ndarray): Observation matrix (d_obs, d_latent)
            Q (np.ndarray): Covariance matrix of state noise (d_latent, d_latent)
            R (np.ndarray): Covariance "matrix" of observation noise; only diagonal elements (d_obs,)
            pi_0 (np.ndarray): Initial state estimate (d_latent, )
            V_0 (np.ndarray): Initial state covariance estimate (d_latent, )
        """
        if y.ndim == 2:
            y = np.expand_dims(y, 1)
        assert y.shape[2] >= d_latent, 'FA only works for d_obs > d_latent; will implement AR'
        T = y.shape[0]
        N = y.shape[1]
        y = y.reshape(-1, y.shape[-1])
        fa = FactorAnalysis(n_components=d_latent)
        fa.fit(y)
        C = fa.components_.T
        R = fa.noise_variance_

        # Woodbury matrix identity for y(R + CC')C
        Phi = np.diag(1. / R)
        temp1 = Phi.dot(C)
        temp2 = Phi - temp1.dot(LA.pinv(np.eye(d_latent) + C.T.dot(temp1))).dot(temp1.T)
        temp1 = y.dot(temp2).dot(C)
        pi_0 = np.mean(temp1, axis=0)
        Q = np.cov(temp1.T)
        V_0 = Q

        t1 = temp1[:N * T - 1]
        t2 = temp1[1:N * T, :]

        A = LA.pinv(t1.T.dot(t1) + Q).dot(t1.T.dot(t2))

        return A, C, Q, R, pi_0, V_0

    @staticmethod
    def _e_step(y, A, C, Q, R, pi_0, V_0):
        """ Expectation step in Expectation-Maximization algorithm

        Args:
            y (np.ndarray): (T, N, d_obs) or (T, d_obs), observed data
            A (np.ndarray): State dynamics matrix (d_latent, d_latent)
            C (np.ndarray): Observation matrix (d_obs, d_latent)
            Q (np.ndarray): Covariance matrix of state noise (d_latent, d_latent)
            R (np.ndarray): Covariance matrix of observation noise; only diagonal elements (d_obs, )
            pi_0 (np.ndarray): Initial state estimate (d_latent, )
            V_0 (np.ndarray): Initial state covariance estimate (d_latent, )

        Returns:
            x_s (np.ndarray): (T, N, d_latent) or (T, d_latent), smoothed latent state
            V_s (np.ndarray): (T, N, d_latent, d_latent) or (T, d_latent, d_latent), smoothed variance of latent state
            Vtsum (np.ndarray): (N, d_latent, d_latent) or (d_latent, d_latent), sum of V_s across time
            YX (np.ndarray): (d_obs, d_latent) sum of covariance between Ys and Xs across time
            A1, A2, A3: other quantities necessary in M step
            likelihood (float): log likelihood of data given params and latent state
        """
        #  prediction
        if y.ndim == 2:
            y = np.expand_dims(y, 1)
        T, N, d_obs = y.shape
        d_latent = pi_0.shape[0]

        x_p = np.zeros((N, d_latent))
        V_p = np.zeros((T, d_latent, d_latent))

        # filter
        x_f = np.zeros((T, N, d_latent))
        V_f = np.zeros((T, d_latent, d_latent))

        # smoother
        J = np.zeros((T, d_latent, d_latent))
        x_s = np.zeros((T, N, d_latent))
        V_s = np.zeros((T, d_latent, d_latent))

        K = np.zeros((d_latent, d_obs))
        I = np.eye(d_latent)
        likelihood = 0
        tiny = np.exp(-100)
        R = R + tiny * np.isclose(R, 0)
        # prediction/filter
        for t in range(T):
            if t == 0:  # initialize
                x_p = np.tile(pi_0, (N, 1))
                V_p[t] = V_0
            else:
                x_p = x_f[t - 1].dot(A.T)
                V_p[t] = A.dot(V_f[t - 1]).dot(A.T) + Q
            if d_latent < d_obs:  # Woodbury matrix identity
                temp1 = (C.T / R).T
                temp2 = temp1.dot(V_p[t])
                temp3 = C.T.dot(temp2)
                temp4 = LA.pinv(I + temp3).dot(temp1.T)
                invR = np.diag(1 / R)
                invtemp1 = invR - temp2.dot(temp4)
                CP = temp1.T - temp3.dot(temp4)
            else:
                temp = np.diag(R)
                temp1 = (C.dot(V_p[t]).dot(C.T) + temp)
                invtemp1 = LA.pinv(temp1)
                CP = C.T.dot(invtemp1)
            K = V_p[t].dot(CP)
            Ydiff = y[t] - x_p.dot(C.T)
            LA.det(invtemp1)
            detiP = np.sqrt(LA.det(invtemp1))
            likelihood = likelihood + (N * np.log(detiP) - 0.5 * np.sum(Ydiff * (Ydiff.dot(invtemp1))))

            x_f[t] = x_p + K.dot(y[t].T - C.dot(x_p.T)).T
            V_f[t] = V_p[t] - K.dot(C).dot(V_p[t])
        likelihood = likelihood + (N * T * np.log(np.power(2 * np.pi, -d_obs / 2.)))
        # smoother
        x_s[-1] = x_f[-1]
        V_s[-1] = V_f[-1]
        Vt = V_s[-1] + x_s[-1].T.dot(x_s[-1]) / N
        A2 = -Vt
        Vtsum = Vt
        YX = y[-1].T.dot(x_s[-1])

        for t in range(T - 1, 0, -1):
            J[t - 1] = V_f[t - 1].dot(A.T).dot(LA.pinv(V_p[t]))
            x_s[t - 1] = x_f[t - 1] + J[t - 1].dot(x_s[t].T - A.dot(x_f[t - 1].T)).T
            V_s[t - 1] = V_f[t - 1] + J[t - 1].dot((V_s[t] - V_p[t])).dot(J[t - 1].T)
            Vt = V_s[t - 1] + x_s[t - 1].T.dot(x_s[t - 1]) / N
            Vtsum += Vt
            YX += y[t - 1].T.dot(x_s[t - 1])
        A3 = Vtsum - Vt
        A2 = Vtsum + A2

        Pcov = (I - K.dot(C)).dot(A).dot(V_f[-2])
        A1 = Pcov + x_s[-1].T.dot(x_s[-2]) / N
        for t in range(T - 1, 1, -1):
            Pcov = (V_f[t - 1] + J[t - 1].dot(Pcov - A.dot(V_f[t - 1]))).dot(J[t - 2].T)
            A1 += Pcov + x_s[t - 1].T.dot(x_s[t - 2]) / N

        return x_s, V_s, Vtsum, YX, A1, A2, A3, likelihood

    @staticmethod
    def _m_step(YY, x_s, V_s, Vtsum, YX, A1, A2, A3):
        """ Maximization step in Expectation-Maximization algorithm

        Args:
            YX (np.ndarray): (d_obs, d_obs) sum of variance of Y over time
            x_s (np.ndarray): (T, N, d_latent) or (T, d_latent), smoothed latent state
            V_s (np.ndarray): (T, N, d_latent, d_latent) or (T, d_latent, d_latent), smoothed variance of latent state
            Vtsum (np.ndarray): (N, d_latent, d_latent) or (d_latent, d_latent), sum of V_s across time
            YX (np.ndarray): (d_obs, d_latent) sum of covariance between Ys and Xs across time
            A1, A2, A3: other quantities necessary in M step

        Returns:
            y (np.ndarray): (T, N, d_obs) or (T, d_obs), observed data
            A (np.ndarray): State dynamics matrix (d_latent, d_latent)
            C (np.ndarray): Observation matrix (d_obs, d_latent)
            Q (np.ndarray): Covariance matrix of state noise (d_latent, d_latent)
            R (np.ndarray): Covariance "matrix" of observation noise; only diagonal elements (d_obs,)
            pi_0 (np.ndarray): Initial state estimate (d_latent, )
            V_0 (np.ndarray): Initial state covariance estimate (d_latent, )
        """
        T = x_s.shape[0]
        N = x_s.shape[1]
        pi_0 = x_s[0, 0]
        V_0 = V_s[0]
        C = YX.dot(LA.pinv(Vtsum)) / (N)
        R = YY - np.diag(C.dot(YX.T)) / (T * N)
        A = A1.dot(LA.pinv(A2))
        Q = (1 / (T - 1)) * np.diag(np.diag((A3 - A.dot(A1.T))))

        return A, C, Q, R, pi_0, V_0

    def kfold_cross_val(self, y, d_latent, trial_type, num_iterations=100, k=3):
        """K-fold cross validation

        Args:
            y ():
            d_latent ():
            trial_type ():
            num_iterations ():
            k ():
        """
        model_explained_variances = []
        psth_explained_variances = []
        kf = model_selection.KFold(k)
        cur_k = 1
        for training_trials, test_trials in kf.split(y[0]):  # for shape
            print("Fold {} of {}".format(cur_k, k))
            y_train = y[:, training_trials]
            y_test = y[:, test_trials]
            y_hat = np.zeros_like(y_test)

            # train dynamics model
            A, C, Q, R, pi_0, V_0 = self._initialize(y_train, d_latent)
            T = y_train.shape[0]
            N = y_train.shape[1]

            YY = np.sum(y * y, axis=(0, 1)).T / (T * N)
            x_s, V_s, Vtsum, YX, A1, A2, A3, lik = self._e_step(y, A, C, Q, R, pi_0, V_0)

            for _ in range(num_iterations):
                A, C, Q, R, pi_0, V_0 = self._m_step(YY, x_s, V_s, Vtsum, YX, A1, A2, A3)
                x_s, V_s, Vtsum, YX, A1, A2, A3, lik = self._e_step(y, A, C, Q, R, pi_0, V_0)

            for holdout_ind in range(y_test.shape[2]):
                y_test_holdout = np.delete(y_test, holdout_ind, 2)
                C_holdout = np.delete(C, holdout_ind, 0)
                R_holdout = np.delete(R, holdout_ind, 0)
                returns = self._e_step(y_test_holdout, A, C_holdout, Q, R_holdout, pi_0, V_0)
                x_hat = returns[0]
                y_hat[:, :, holdout_ind] = x_hat.dot(C[holdout_ind].T)

            unique_trial_types = np.unique(trial_type[test_trials])
            psth_explained_variance = 0
            overall_neuron_mean = np.mean(y_test,
                                          axis=(1, 2))  # my guess is that this should be test, but need to confirm

            normalization = np.sum(np.power(y_test.transpose(2, 1, 0) - overall_neuron_mean, 2).transpose(2, 1, 0),
                                   axis=(0, 1))
            model_error = np.sum(np.power(y_test - y_hat, 2), axis=(0, 1))
            model_explained_variance = 1 - np.mean(model_error / normalization)

            # compare to PSTH, per trial contingency
            for t in unique_trial_types:
                trial_types_inds = trial_type[test_trials] == t
                y_test_of_trial_type = y_test[:, trial_types_inds]
                mean_y_test = np.mean(y_test_of_trial_type, axis=1)
                normalization = np.sum(
                    np.power(y_test_of_trial_type.transpose(2, 1, 0) - overall_neuron_mean, 2).transpose(2, 1, 0),
                    axis=(0, 1))
                model_error = np.sum(
                    np.power(y_test_of_trial_type.transpose(1, 0, 2) - mean_y_test, 2).transpose(1, 0, 2), axis=(0, 1))
                psth_explained_variance += 1 - np.mean(model_error / normalization)
            psth_explained_variance = psth_explained_variance / unique_trial_types.shape[0]

            model_explained_variances.append(model_explained_variance)
            psth_explained_variances.append(psth_explained_variance)
            cur_k += 1
        return model_explained_variances, psth_explained_variances
    @staticmethod
    def get_likelihood(y, A, C, Q, R, pi_0, V_0):
        if Q.ndim == 2:
            Q = np.diag(Q)
        if R.ndim == 2:
            R = np.diag(R)
        ret = gLDS._e_step(y, A, C, np.diag(Q), R, pi_0, V_0)
        likelihood = ret[-1]
        return likelihood

def sort_by_column_norm(A, rowargs=[], colargs=[]):
    descending_column_order = np.argsort(LA.norm(A, axis=0))[::-1]
    sorted_args = []
    for arg in rowargs:
        sorted_args.append(arg[descending_column_order])
    for arg in colargs:
        sorted_args.append(arg[:,descending_column_order])
    if len(sorted_args) > 0:
        return A[:, descending_column_order], sorted_args
    else:
        return A[:, descending_column_order]

