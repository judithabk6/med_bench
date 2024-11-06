import numpy as np


def ipw_risk(y, t, hat_y, hat_e, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = t / clipped_hat_e + (1 - t) / (1 - clipped_hat_e)
    return np.sum(((y - hat_y) ** 2) * ipw_weights) / len(y)


def r_risk(y, t, hat_m, hat_e, hat_tau):
    return np.mean(((y - hat_m) - (t - hat_e) * hat_tau) ** 2)


def u_risk(y, t, hat_m, hat_e, hat_tau):
    return np.mean(((y - hat_m) / (t - hat_e) - hat_tau) ** 2)


def w_risk(y, t, hat_e, hat_tau, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    pseudo_outcome = (y * (t - clipped_hat_e)) / (clipped_hat_e * (1 - clipped_hat_e))
    return np.mean((pseudo_outcome - hat_tau) ** 2)


def ipw_r_risk(y, t, hat_mu_0, hat_mu_1, hat_e, hat_m, trimming=None):
    if trimming is not None:
        clipped_hat_e = np.clip(hat_e, trimming, 1 - trimming)
    else:
        clipped_hat_e = hat_e
    ipw_weights = t / clipped_hat_e + (1 - t) / (1 - clipped_hat_e)
    hat_tau = hat_mu_1 - hat_mu_0

    return np.sum((((y - hat_m) - (t - hat_e) * (hat_tau)) ** 2) * ipw_weights) / len(y)


def ipw_r_risk_oracle(y, t, hat_mu_0, hat_mu_1, e, mu_1, mu_0):
    m = mu_0 * (1 - e) + mu_1 * e
    return ipw_r_risk(y=y, t=t, hat_mu_0=hat_mu_0, hat_mu_1=hat_mu_1, hat_e=e, hat_m=m)
