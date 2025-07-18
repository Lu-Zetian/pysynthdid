import pandas as pd
import numpy as np
import statsmodels.api as sm

from synthdid.optimizer import Optimize
from synthdid.plot import Plot
from synthdid.variance import Variance
from synthdid.summary import Summary


class SynthDID(Optimize, Plot, Variance, Summary):
    """
    Synthetic Difference in Differences
    df              : pandas.DataFrame
    pre_term        : term before treatment
    post_term       : term after treatmetn
    treatment_unit  : treatment columns names list

    [example]
    df = fetch_CaliforniaSmoking()
    sdid = SynthDID(df, [1970, 1979], [1980, 1988], ["California"])
    sdid.fit()
    """

    def __init__(
        self, df: pd.DataFrame, pre_term, post_term, treatment_unit: list, **kwargs
    ):
        # first val
        self.df = df
        self.df_growth = df.pct_change().fillna(0) + 1
        self.pre_term = pre_term
        self.post_term = post_term
        # able to change
        self.treatment = treatment_unit
        self.control = [col for col in df.columns if col not in self.treatment]
        self._divide_data()

        # params
        self.hat_zeta = None
        self.base_zeta = None
        self.hat_omega_sc = None
        self.hat_omega_gsc = None
        self.hat_omega = None
        self.hat_lambda = None
        self.hat_omega_ElasticNet = None
        self.hat_omega_Lasso = None
        self.hat_omega_Ridge = None
        self.sdid_se = None
        self.sc_se = None
        self.did_se = None

    def _divide_data(self):

        self.Y_pre_c = self.df.loc[self.pre_term[0]:self.pre_term[1], self.control]
        self.Y_pre_t = self.df.loc[self.pre_term[0]:self.pre_term[1], self.treatment]

        self.Y_post_c = self.df.loc[self.post_term[0]:self.post_term[1], self.control]
        self.Y_post_t = self.df.loc[self.post_term[0]:self.post_term[1], self.treatment]
        
        self.R_pre_c = self.df_growth.loc[self.pre_term[0]:self.pre_term[1], self.control]
        self.R_pre_t = self.df_growth.loc[self.pre_term[0]:self.pre_term[1], self.treatment]

        self.R_post_c = self.df_growth.loc[self.post_term[0]:self.post_term[1], self.control]
        self.R_post_t = self.df_growth.loc[self.post_term[0]:self.post_term[1], self.treatment]

        self.n_treat = len(self.treatment)
        self.n_post_term = len(self.Y_post_t)

    def fit(
        self,
        force_zeta=None,
    ):

        self.zeta = self.est_zeta(self.Y_pre_c)

        if force_zeta != None:
            self.zeta = force_zeta

        self.hat_omega = self.est_omega(self.Y_pre_c, self.Y_pre_t, self.zeta)
        
        self.hat_lambda = self.est_lambda(self.Y_pre_c, self.Y_post_c)
        
        self.hat_omega_sc = self.est_omega_sc(self.Y_pre_c, self.Y_pre_t, self.zeta)
        
        self.hat_omega_gsc = self.est_omega_gsc(self.R_pre_c, self.R_pre_t, self.zeta)
        

    def did_potential_outcome(self):
        """
        return potential outcome
        """
        Y_pre_c = self.Y_pre_c.copy()
        Y_pre_t = self.Y_pre_t.copy()
        Y_post_c = self.Y_post_c.copy()
        Y_post_t = self.Y_post_t.copy()

        if type(Y_pre_t) != pd.DataFrame:
            Y_pre_t = pd.DataFrame(Y_pre_t)

        if type(Y_post_t) != pd.DataFrame:
            Y_post_t = pd.DataFrame(Y_post_t)

        Y_pre_t["did"] = Y_pre_t.mean(axis=1)

        base_trend = Y_post_c.mean(axis=1)

        Y_post_t["did"] = base_trend + (
            Y_pre_t.mean(axis=1).mean() - Y_pre_c.mean(axis=1).mean()
        )

        return pd.concat([Y_pre_t["did"], Y_post_t["did"]], axis=0)

    def sc_potential_outcome(self):
        sc = pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_sc)
        
        return sc
    
    def gsc_potential_outcome(self):
        gsc = pd.concat([self.Y_pre_c, self.Y_post_c]).dot(self.hat_omega_gsc)
        
        y_f = np.squeeze(np.array(self.Y_pre_t))
        y_sc = np.array(gsc.loc[: self.pre_term[1]])
        ols = sm.OLS(y_f, y_sc)  # No intercept
        gsc_ratio = ols.fit().params[0]
        gsc = gsc * gsc_ratio
        
        return gsc

    def sparceReg_potential_outcome(self, model="ElasticNet"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1

        if model == "ElasticNet":
            s_omega = self.hat_omega_ElasticNet
        elif model == "Lasso":
            s_omega = self.hat_omega_Lasso
        elif model == "Ridge":
            s_omega = self.hat_omega_Ridge
        else:
            print(f"model={model} is not supported")
            return None
        return pd.concat([Y_pre_c_intercept, Y_post_c_intercept]).dot(s_omega)

    def sdid_trajectory(self):
        hat_omega = self.hat_omega[:-1]
        Y_c = pd.concat([self.Y_pre_c, self.Y_post_c])
        n_features = self.Y_pre_c.shape[1]
        start_w = np.repeat(1 / n_features, n_features)

        _intercept = (start_w - hat_omega) @ self.Y_pre_c.T @ self.hat_lambda

        return Y_c.dot(hat_omega) + _intercept

    def sdid_potential_outcome(self):
        Y_pre_c = self.Y_pre_c.copy()
        Y_post_c = self.Y_post_c.copy()
        hat_omega = self.hat_omega[:-1]
        
        pre_outcome = Y_pre_c.dot(hat_omega)

        base_sc = Y_post_c @ hat_omega
        pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
        pre_control_base = pre_outcome @ self.hat_lambda
        
        post_outcome = base_sc + pre_treat_base - pre_control_base

        return pd.concat([pre_outcome, post_outcome], axis=0)

    def sparce_sdid_potential_outcome(self, model="ElasticNet"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1

        if model == "ElasticNet":
            s_omega = self.hat_omega_ElasticNet
        elif model == "Lasso":
            s_omega = self.hat_omega_Lasso
        elif model == "Ridge":
            s_omega = self.hat_omega_Ridge
        else:
            print(f"model={model} is not supported")
            return None

        base_sc = Y_post_c_intercept @ s_omega
        pre_treat_base = (self.Y_pre_t.T @ self.hat_lambda).values[0]
        pre_control_base = Y_pre_c_intercept @ s_omega @ self.hat_lambda

        post_outcome = base_sc + pre_treat_base - pre_control_base

        return pd.concat([Y_pre_c_intercept.dot(s_omega), post_outcome], axis=0)

    def target_y(self):
        return self.df.loc[self.pre_term[0]:self.post_term[1], self.treatment].mean(
            axis=1
        )

    def estimated_params(self, model="sdid"):
        Y_pre_c_intercept = self.Y_pre_c.copy()
        Y_post_c_intercept = self.Y_post_c.copy()
        Y_pre_c_intercept["intercept"] = 1
        Y_post_c_intercept["intercept"] = 1
        if model == "sdid":
            return (
                pd.DataFrame(
                    {
                        "features": Y_pre_c_intercept.columns,
                        "sdid_weight": np.round(self.hat_omega, 3),
                    }
                ),
                pd.DataFrame(
                    {
                        "time": Y_pre_c_intercept.index,
                        "sdid_weight": np.round(self.hat_lambda, 3),
                    }
                ),
            )
        elif model == "sc":
            return pd.DataFrame(
                {
                    "features": self.Y_pre_c.columns,
                    "sc_weight": np.round(self.hat_omega_sc, 3),
                }
            )
        elif model == "gsc":
            return pd.DataFrame(
                {
                    "features": self.Y_pre_c.columns,
                    "gsc_weight": np.round(self.hat_omega_gsc, 3),
                }
            )
        elif model == "ElasticNet":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "ElasticNet_weight": np.round(self.hat_omega_ElasticNet, 3),
                }
            )
        elif model == "Lasso":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "Lasso_weight": np.round(self.hat_omega_Lasso, 3),
                }
            )
        elif model == "Ridge":
            return pd.DataFrame(
                {
                    "features": Y_pre_c_intercept.columns,
                    "Ridge_weight": np.round(self.hat_omega_Ridge, 3),
                }
            )
        else:
            return None

    def hat_tau(self, model="sdid"):
        """
        return ATT
        """
        result = pd.DataFrame({"actual_y": self.target_y()})
        post_actural_treat = result.loc[self.post_term[0] :, "actual_y"].mean()

        if model == "sdid":
            result["sdid"] = self.sdid_trajectory()

            pre_sdid = result["sdid"].head(len(self.hat_lambda)) @ self.hat_lambda
            post_sdid = result.loc[self.post_term[0] :, "sdid"].mean()

            pre_treat = (self.Y_pre_t.T @ self.hat_lambda).values[0]
            counterfuctual_post_treat = pre_treat + (post_sdid - pre_sdid)

        elif model == "sc":
            result["sc"] = self.sc_potential_outcome()
            post_sc = result.loc[self.post_term[0] :, "sc"].mean()
            counterfuctual_post_treat = post_sc
            
        elif model == "gsc":
            result["gsc"] = self.gsc_potential_outcome()
            post_gsc = result.loc[self.post_term[0] :, "gsc"].mean()
            counterfuctual_post_treat = post_gsc

        elif model == "did":
            Y_pre_t = self.Y_pre_t.copy()
            Y_post_t = self.Y_post_t.copy()
            if type(Y_pre_t) != pd.DataFrame:
                Y_pre_t = pd.DataFrame(Y_pre_t)

            if type(Y_post_t) != pd.DataFrame:
                Y_post_t = pd.DataFrame(Y_post_t)

            # actural treat
            post_actural_treat = (
                Y_post_t.mean(axis=1).mean() - Y_pre_t.mean(axis=1).mean()
            )
            counterfuctual_post_treat = (
                self.Y_post_c.mean(axis=1).mean() - self.Y_pre_c.mean(axis=1).mean()
            )

        return post_actural_treat - counterfuctual_post_treat

    def cal_se(self, algo="placebo", replications=200):

        sdid_var, sc_var, gsc_var, did_var = self.estimate_variance(
            algo=algo, replications=replications
        )

        self.sdid_se = np.sqrt(sdid_var)
        self.sc_se = np.sqrt(sc_var)
        self.gsc_se = np.sqrt(gsc_var)
        self.did_se = np.sqrt(did_var)


if __name__ == "__main__":
    from sample_data import fetch_CaliforniaSmoking

    df = fetch_CaliforniaSmoking()

    PRE_TEREM = [1970, 1979]
    POST_TEREM = [1980, 1988]
    TREATMENT = ["California"]

    sdid = SynthDID(df, PRE_TEREM, POST_TEREM, TREATMENT)
    sdid.fit()
    sdid.plot(model="sdid")
