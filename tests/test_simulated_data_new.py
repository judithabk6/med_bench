"""test_simulated_data_new::simulate_data

Rappel :
p_t = P(T=1|X)
th_p_t_mx = P(T=1|X,M)
"""

import pytest

from judith_abecassis.src.get_simulated_data_new import simulate_data

from pprint import pprint
from numpy.random import default_rng
import numpy as np
from judith_abecassis.src.get_simulated_data_new import simulate_data
from judith_abecassis.src.test_simulation_settings_new import get_estimation


data = simulate_data(
    n=1,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="continuous",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=10000,
)
print(data)

# effects = np.array(data[4:9])
# print(effects)


# base data
(
    array([[0.30471708]]),
    array([[1]]),
    array([[0]]),
    array([[1.97499944]]),
    1.3124754041240658,
    1.2,
    1.2,
    0.11247540412406576,
    0.11247540412406576,
    array([0.57559524]),
    array([0.41595136]),
)


# "continuous"
(
    array([[0.30471708]]),
    array([[1]]),
    array([[1.480531]]),
    array([[2.71526494]]),
    2.180531003718505,
    1.2,
    1.2,
    0.9805310037185047,
    0.9805310037185047,
    array([0.57559524]),
    array([0.7649375]),
)


# TESTS
# total = indirect + direct (direct 0 et indirect 1)
# mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_y=True : 5!=6!=gamma_t ;  7!=8
# Tester qu'avec même seed, génère mêmes données qu'avant?
# vérifier la binarité, la continuité, la dimension de x,t,m,y

# Tester ce qui devrait être modifié et ce qui ne le devrait pas
# Faire un test d'égalité sur chaque sortie => laquelle on a modifié

# Egalité de chaque paramètre pour petits exemples tant qu'à faire (robuste mais moche?)
# (sinon juste pour nous, ça garantit qu'on a rien changé à la fonction)


# QUI FAIT VARIER QUOI
# n : taille des données
# n : 4 7 8 ; mis_spec_y=False ; 5=6=gamma_t
# n : 4 5 6 7 8 ; mis_spec_y=True ; 5!=6!=gamma_t
# rg : all (modifie les données ET les effets)
# seed : 4 5 6 7 8 10 (modifie les effets, pas les données)
# mis_spec_m : 4 (5) 7 8 (y=True)
# mis_spec_y : 4 5 6 7 8 (??)
# mis_spec_m=False, mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_m=True, mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_m=False, mis_spec_y=True : 5!=6!=gamma_t ;  7!=8
# dim_x : x[0] idem (sinon rg)
# dim_x : 4 5 8 9 10
# "continuous" : 3 4 7 8 10
# "continuous" : 3 4 (5) (6) 7 8 10 (y=True)
# sigma_y : 3
# sigma_m : _ (binary)
# sigma_m : 2 3 4 7 8 10 (continuous)
# beta_t_factor : 3 4 7 8 10
# beta_m_factor : 3 4 7 8


# CORRECTIONS OU IMPLEMENTATIONS
# Problèmes nuls : Donner des arguments invalides, on s'en préoccupe? Si oui comment? (n, dim_x, dim_m <1)
# n=1, ne pas avoir une liste de liste
# Forcer l'argument "continuous"? Poser un warning si c'est une autre lettre que "c"?
# Faut-il retirer le main?
# Pouvoir fixer gamma_t = 1.2?
# Donner les bornes dans lesquelles sont générées les données ou les effets?
# Tout générer entre 0 et 1, et puis l'utilisateur se débrouille?
# Normaliser les effets? Poser l'ATE=1?
# Pourquoi beta_t_factor et pas omega_t? ou carrément beta_t?
# les omega n'existent pas dans la fonction
# Si l'utilisateur a le papier devant lui, on peut énumérer en entrée tous les paramètres alpha, beta, gamma?
# Pas de sigma_x?
# Quelle formule pour les probas?


# dim_m>1 ; n>1 ; "binary" ; l39
data = simulate_data(
    n=7,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=3,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
)

# dim_m>1 ; n=1 ; l58
data = simulate_data(
    n=1,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=2,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
)

# n=0 : Warnings
data = simulate_data(
    n=0,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
)
print(data)

# n<0 : l19 ; ValueError: negative dimensions are not allowed
data = simulate_data(
    n=-1,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
)

# sigma_m grand ; "continuous" ; P(T=1|X,M) = NaN
data = simulate_data(
    n=1,
    rg=default_rng(42),
    mis_spec_m=True,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="fds",
    sigma_y=0.5,
    sigma_m=5351,
    beta_t_factor=1,
    beta_m_factor=1,
)
print(data)

# sigma_m=0 ; "continuous" ; P(T=1|X,M) = NaN
data = simulate_data(
    n=1,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="continuous",
    sigma_y=0.5,
    sigma_m=0,
    beta_t_factor=1,
    beta_m_factor=1,
)

# Les coefficients qu'on ne voit pas c'est dommage (ici gamma_m)
data = simulate_data(
    n=1,
    rg=default_rng(42),
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=1,
    type_m="continuous",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=10000,
    beta_m_factor=1,
)
print(data)




# Brouillon de la fonction :


def simulate_data(
    n,
    rg,
    mis_spec_m=False,
    mis_spec_y=False,
    dim_x=1,
    dim_m=1,
    seed=None,
    type_m="binary",
    sigma_y=0.5,
    sigma_m=0.5,
    beta_t_factor=1,
    beta_m_factor=1,
):
    rg_coef = default_rng(seed)
    x = rg.standard_normal(n * dim_x).reshape((n, dim_x))
    alphas = np.ones(dim_x) / dim_x
    p_t = expit(alphas.dot(x.T))
    t = rg.binomial(1, p_t, n).reshape(-1, 1)
    t0 = np.zeros((n, 1))
    t1 = np.ones((n, 1))

    # generate M
    beta_x = rg_coef.standard_normal((dim_x, dim_m)) * 1 / (dim_m * dim_x)
    beta_t = np.ones((1, dim_m)) * beta_t_factor
    if mis_spec_m:
        beta_xt = rg_coef.standard_normal((dim_x, dim_m)) * 1 / (dim_m * dim_x)
    else:
        beta_xt = np.zeros((dim_x, dim_m))

    if type_m == "binary":
        p_m0 = expit(x.dot(beta_x) + beta_t * t0 + x.dot(beta_xt) * t0)
        p_m1 = expit(x.dot(beta_x) + beta_t * t1 + x.dot(beta_xt) * t1)
        pre_m = rg.random(n)
        m0 = ((pre_m < p_m0.ravel()) * 1).reshape(-1, 1)
        m1 = ((pre_m < p_m1.ravel()) * 1).reshape(-1, 1)
        m_2d = np.hstack((m0, m1))
        m = m_2d[np.arange(n), t[:, 0]].reshape(-1, 1)
    else:
        random_noise = sigma_m * rg.standard_normal((n, dim_m))
        m0 = x.dot(beta_x) + t0.dot(beta_t) + t0 * (x.dot(beta_xt)) + random_noise
        m1 = x.dot(beta_x) + t1.dot(beta_t) + t1 * (x.dot(beta_xt)) + random_noise
        m = x.dot(beta_x) + t.dot(beta_t) + t * (x.dot(beta_xt)) + random_noise

    # generate Y
    gamma_m = np.ones((dim_m, 1)) * 0.5 / dim_m * beta_m_factor
    gamma_x = np.ones((dim_x, 1)) / dim_x**2
    gamma_t = 1.2
    if mis_spec_y:
        gamma_t_m = np.ones((dim_m, 1)) * 0.5 / dim_m
    else:
        gamma_t_m = np.zeros((dim_m, 1))

    y = (
        x.dot(gamma_x)
        + gamma_t * t
        + m.dot(gamma_m)
        + m.dot(gamma_t_m) * t
        + sigma_y * rg.standard_normal((n, 1))
    )

    if type_m == "binary":
        theta_1 = gamma_t + gamma_t_m * np.mean(p_m1)
        theta_0 = gamma_t + gamma_t_m * np.mean(p_m0)
        delta_1 = np.mean((p_m1 - p_m0) * (gamma_m.flatten() + gamma_t_m.dot(t1.T)))
        delta_0 = np.mean((p_m1 - p_m0) * (gamma_m.flatten() + gamma_t_m.dot(t0.T)))
    else:
        theta_1 = gamma_t + gamma_t_m.T.dot(
            np.mean(m1, axis=0)
        )  # to do mean(m1) pour avoir un vecteur de taille dim_m
        theta_0 = gamma_t + gamma_t_m.T.dot(np.mean(m0, axis=0))
        delta_1 = (
            gamma_t * t1
            + m1.dot(gamma_m)
            + m1.dot(gamma_t_m) * t1
            - gamma_t * t1
            + m0.dot(gamma_m)
            + m0.dot(gamma_t_m) * t1
        ).mean()
        delta_0 = (
            gamma_t * t0
            + m1.dot(gamma_m)
            + m1.dot(gamma_t_m) * t0
            - gamma_t * t0
            + m0.dot(gamma_m)
            + m0.dot(gamma_t_m) * t0
        ).mean()

    if type_m == "binary":
        pre_pm = np.hstack((p_m0.reshape(-1, 1), p_m1.reshape(-1, 1)))
        pre_pm[m.ravel() == 0, :] = 1 - pre_pm[m.ravel() == 0, :]
        pm = pre_pm[:, 1].reshape(-1, 1)
    else:
        p_m0 = np.prod(
            stats.norm.pdf(
                (m - x.dot(beta_x)) - t0.dot(beta_t) - t0 * (x.dot(beta_xt)) / sigma_m
            ),
            axis=1,
        )
        p_m1 = np.prod(
            stats.norm.pdf(
                (m - x.dot(beta_x)) - t1.dot(beta_t) - t1 * (x.dot(beta_xt)) / sigma_m
            ),
            axis=1,
        )
        pre_pm = np.hstack((p_m0.reshape(-1, 1), p_m1.reshape(-1, 1)))
        pm = pre_pm[:, 1].reshape(-1, 1)

    px = np.prod(stats.norm.pdf(x), axis=1)

    pre_pt = np.hstack(((1 - p_t).reshape(-1, 1), p_t.reshape(-1, 1)))
    double_px = np.hstack((px.reshape(-1, 1), px.reshape(-1, 1)))
    denom = np.sum(pre_pm * pre_pt * double_px, axis=1)
    num = pm.ravel() * p_t.ravel() * px.ravel()
    th_p_t_mx = num.ravel() / denom

    return (
        x,
        t,
        m,
        y,
        theta_1.flatten()[0] + delta_0.flatten()[0],
        theta_1.flatten()[0],
        theta_0.flatten()[0],
        delta_1.flatten()[0],
        delta_0.flatten()[0],
        p_t,
        th_p_t_mx,
    )
