from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.optimize import fmin_slsqp
from toolz import partial
from sklearn.model_selection import KFold, TimeSeriesSplit, RepeatedKFold
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from bayes_opt import BayesianOptimization


class Optimize(object):
    ####
    # Synthetic Difference in Differences (SDID)
    ####
    def est_zeta(self, Y_pre_c) -> float:
        """
        # SDID
        Parameter to adjust the L2 penalty term
        """
        return (self.n_treat * self.n_post_term) ** (1 / 4) * np.std(
            Y_pre_c.diff().dropna().values
        )
    
    def sdid_sse_loss(self, W, X, y, zeta, T_pre) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        _X["intersept"] = 1
        return np.sum((y - _X.dot(W)) ** 2) + zeta ** 2 * T_pre * np.sum(W[:-1] ** 2)

    def est_omega(self, Y_pre_c, Y_pre_t, zeta=0):
        """
        # SDID
        estimating omega
        """
        Y_pre_t = Y_pre_t.copy()
        n_features = Y_pre_c.shape[1]
        T_pre = Y_pre_c.shape[0]  # T_pre

        _w = np.repeat(1 / n_features, n_features)
        _w0 = 1

        start_w = np.append(_w, _w0)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        # Required to have non negative values
        max_bnd = abs(Y_pre_t.mean()) * 2
        # (0, 1) for w_1, ..., w_n, (max_bnd * -1, max_bnd)
        w_bnds = tuple(
            (0, 1) if i < n_features else (max_bnd * -1, max_bnd) for i in range(n_features + 1)
        )

        caled_w = fmin_slsqp(
            partial(self.sdid_sse_loss, X=Y_pre_c, y=Y_pre_t, zeta=zeta, T_pre=T_pre),
            start_w,
            f_eqcons=lambda x: np.sum(x[:n_features]) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w

    def est_lambda(self, Y_pre_c, Y_post_c, zeta=0):
        """
        # SDID
        estimating lambda
        """
        Y_pre_c_T = Y_pre_c.T
        Y_post_c_T = Y_post_c.T
        
        T_pre = Y_pre_c.shape[0]

        n_pre_term = Y_pre_c_T.shape[1]

        _lambda = np.repeat(1 / n_pre_term, n_pre_term)
        _lambda0 = 1

        start_lambda = np.append(_lambda, _lambda0)

        if type(Y_post_c_T) == pd.core.frame.DataFrame:
            Y_post_c_T = Y_post_c_T.mean(axis=1)

        max_bnd = abs(Y_post_c_T.mean()) * 2
        lambda_bnds = tuple(
            (0, 1) if i < n_pre_term else (max_bnd * -1, max_bnd)
            for i in range(n_pre_term + 1)
        )

        caled_lambda = fmin_slsqp(
            partial(self.sdid_sse_loss, X=Y_pre_c_T, y=Y_post_c_T, zeta=zeta, T_pre=T_pre),
            start_lambda,
            f_eqcons=lambda x: np.sum(x[:n_pre_term]) - 1,
            bounds=lambda_bnds,
            disp=False,
        )

        return caled_lambda[:n_pre_term]
    

    ####
    # Synthetic Control Method (SC)
    ####
    def sc_mse_loss(self, W, X, y, zeta=0, T_pre=0) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        return np.mean((y - _X.dot(W)) ** 2) + 0.5 * zeta ** 2 * np.sum(W ** 2)

    def est_omega_sc(self, Y_pre_c, Y_pre_t, zeta=0):
        """
        # SC
        estimating omega for synthetic control method (not for synthetic diff.-in-diff.)
        """
        Y_pre_t = Y_pre_t.copy()

        n_features = Y_pre_c.shape[1]
        T_pre = Y_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)

        if type(Y_pre_t) == pd.core.frame.DataFrame:
            Y_pre_t = Y_pre_t.mean(axis=1)

        # Required to have non negative values
        w_bnds = tuple((0, 1) for i in range(n_features))

        caled_w = fmin_slsqp(
            partial(self.sc_mse_loss, X=Y_pre_c, y=Y_pre_t, zeta=zeta, T_pre=T_pre),
            _w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w


    ####
    # Growth Synthetic Control Method (GSC)
    ####
    def gsc_rmse_loss(self, W, X, y, zeta=0, T_pre=0) -> float:
        if type(y) == pd.core.frame.DataFrame:
            y = y.mean(axis=1)
        _X = X.copy()
        y_pred = _X.dot(W)
        loss = np.sqrt(np.mean((y - y_pred) ** 2)) + 0.01 * np.sum(W ** 2)
        return loss

    def est_omega_gsc(self, R_pre_c, R_pre_t, zeta=0):
        """
        # SC
        estimating omega for growth synthetic control method
        """
        R_pre_t = R_pre_t.copy()

        n_features = R_pre_c.shape[1]
        T_pre = R_pre_c.shape[0]

        _w = np.repeat(1 / n_features, n_features)

        if type(R_pre_t) == pd.core.frame.DataFrame:
            R_pre_t = R_pre_t.mean(axis=1)

        # Required to have non negative values
        max_bnd = 5
        # (0, 1) for w_1, ..., w_n, (max_bnd * -1, max_bnd)
        w_bnds = tuple(
            (0, 1) for i in range(n_features)
        )

        caled_w = fmin_slsqp(
            partial(self.gsc_rmse_loss, X=R_pre_c, y=R_pre_t, zeta=zeta, T_pre=T_pre),
            _w,
            f_eqcons=lambda x: np.sum(x) - 1,
            bounds=w_bnds,
            disp=False,
        )

        return caled_w