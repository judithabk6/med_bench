"""test_simulated_data_new::simulate_data

Rappel :
p_t = P(T=1|X)
th_p_t_mx = P(T=1|X,M)
"""

from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from med_bench.src.get_simulated_data import simulate_data


@pytest.fixture(autouse=True)
def data():
    n = 3
    rg = default_rng(42)
    mis_spec_m = False
    mis_spec_y = False
    dim_x = 1
    dim_m = 1
    seed = 1
    type_m = "binary"
    sigma_y = 0.5
    sigma_m = 0.5
    beta_t_factor = 10
    beta_m_factor = 1

    return simulate_data(
        n,
        rg,
        mis_spec_m,
        mis_spec_y,
        dim_x,
        dim_m,
        seed,
        type_m,
        sigma_y,
        sigma_m,
        beta_t_factor,
        beta_m_factor,
    )


@pytest.fixture(autouse=True)
def effects():
    return np.array(data[4:9])


def test_total_is_direct_plus_indirect():
    assert effects[0] == effects[1] + effects[4]  # total = theta_1 + delta_0
    assert effects[0] == effects[2] + effects[3]  # total = theta_0 + delta_1


def test_effects_are_equals_if_y_well_specified(mis_spec_y):
    if mis_spec_y:
        assert effects[1] != effects[2]
        assert effects[3] != effects[4]
    else:
        assert effects[1] == effects[2]
        assert effects[3] == effects[4]


def test_m_is_binary(type_m):
    if type_m == "binary":
        for m in data[2]:
            assert m in {0, 1}
    else:
        for m in data[2]:
            assert not m in {0, 1}


def test_dimension_x(n, dim_x):
    assert (data[0]).shape == (n, dim_x)


def test_dimension_t(n):
    assert (data[1]).shape == (n, 1)


def test_dimension_m(n, dim_m):
    assert (data[2]).shape == (n, dim_m)


def test_dimension_y(n):
    assert (data[3]).shape == (n, 1)


# dim_m>1 ; n>1 ; "binary" ; l39
@pytest.mark.xfail
def test_m_multidimensional_binary_works():
    with pytest.raises(Exception):
        simulate_data(
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
@pytest.mark.xfail
def test_m_multidimensional_binary_works1():
    with pytest.raises(Exception):
        simulate_data(
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
@pytest.mark.xfail
def test_n_null_should_fail():
    with pytest.raises(Exception):
        simulate_data(
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


# n<0 : l19 ; ValueError: negative dimensions are not allowed
@pytest.mark.xfail
def test_n_negative_should_fail():
    with pytest.raises(Exception):
        simulate_data(
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
@pytest.mark.xfail
def test_huge_sigma_m_makes_nan():
    with pytest.raises(Exception):
        simulate_data(
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


# sigma_m=0 ; "continuous" ; P(T=1|X,M) = NaN
@pytest.mark.xfail
def test_null_sigma_m_makes_nan():
    with pytest.raises(Exception):
        simulate_data(
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


# TESTS IMPLEMENTES
# total = indirect + direct (direct 0 et indirect 1)
# mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_y=True : 5!=6!=gamma_t ;  7!=8
# vérifier la binarité, la continuité, la dimension de x,t,m,y

# RENVOI D'ERREUR
# dim_m>1 ; n>1 ; "binary" ; l39
# dim_m>1 ; n=1 ; l58
# n=0 : Warnings
# n<0 : l19 ; ValueError: negative dimensions are not allowed
# sigma_m grand ; "continuous" ; P(T=1|X,M) = NaN
# sigma_m=0 ; "continuous" ; P(T=1|X,M) = NaN


# TESTS RESTANTS
# Tester qu'avec même seed, génère mêmes données qu'avant?
# histoire d'être un vecteur ou une matrix? (huber_no_reg plutôt?)
# Tester les sorties qui devraient être modifiées et celles qui ne le devraient pas


# CORRECTIONS OU IMPLEMENTATIONS
# Problèmes nuls : Donner des arguments invalides, on s'en préoccupe? Si oui comment? (n, dim_x, dim_m <1)
# n=1, ne pas avoir une liste de liste
# Forcer l'argument "continuous"? Poser un warning si c'est une autre lettre que "c"?
# Faut-il retirer le main?
# Pourquoi beta_t_factor et pas omega_t? ou carrément beta_t?
# les omega n'existent pas dans la fonction

# FEATURES DE SECOND ORDRE
# Pouvoir fixer gamma_t = 1.2?
# Pas de sigma_x?
# Si l'utilisateur a le papier devant lui, on peut énumérer en entrée tous les paramètres alpha, beta, gamma?

# FEATURES DE TROISIEME ORDRE
# Donner les bornes dans lesquelles sont générées les données ou les effets?
# Tout générer entre 0 et 1, et puis l'utilisateur se débrouille?
# Normaliser les effets? Poser l'ATE=1?
