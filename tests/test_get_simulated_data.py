"""
test_simulated_data_new::simulate_data

We test :
- The dimensions of the outputs
- Whether they should be binary or not
- Whether the effects are coherent
- Whether forbidden inputs return an error

We pinpoint aberrant behavior reguarding some input combinaisons

Reminder :
p_t = P(T=1|X)
th_p_t_mx = P(T=1|X,M)
"""

from pprint import pprint
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


def test_dimension_x(n, dim_x):
    assert (data[0]).shape == (n, dim_x)


def test_dimension_t(n):
    assert (data[1]).shape == (n, 1)


def test_dimension_m(n, dim_m):
    assert (data[2]).shape == (n, dim_m)


def test_dimension_y(n):
    assert (data[3]).shape == (n, 1)


def test_m_is_binary(type_m):
    if type_m == "binary":
        for m in data[2]:
            assert m in {0, 1}
    else:
        for m in data[2]:
            assert not m in {0, 1}


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


# dim_x=0 : No Warning
@pytest.mark.xfail
def test_dim_x_null_should_fail():
    with pytest.raises(Exception):
        simulate_data(
            n=10,
            rg=default_rng(42),
            mis_spec_m=False,
            mis_spec_y=False,
            dim_x=0,
            dim_m=1,
            seed=1,
            type_m="binary",
            sigma_y=0.5,
            sigma_m=0.5,
            beta_t_factor=1,
            beta_m_factor=1,
        )


# dim_m=0 ; l134 : ValueError
@pytest.mark.xfail
def test_dim_m_null_should_fail():
    with pytest.raises(Exception):
        simulate_data(
            n=10,
            rg=default_rng(42),
            mis_spec_m=False,
            mis_spec_y=False,
            dim_x=1,
            dim_m=0,
            seed=1,
            type_m="binary",
            sigma_y=0.5,
            sigma_m=0.5,
            beta_t_factor=1,
            beta_m_factor=1,
        )


# dim_x<0 : l115 ; ValueError: negative dimensions are not allowed
@pytest.mark.xfail
def test_dim_x_negative_should_fail():
    with pytest.raises(Exception):
        simulate_data(
            n=10,
            rg=default_rng(42),
            mis_spec_m=False,
            mis_spec_y=False,
            dim_x=-1,
            dim_m=1,
            seed=1,
            type_m="binary",
            sigma_y=0.5,
            sigma_m=0.5,
            beta_t_factor=1,
            beta_m_factor=1,
        )


# dim_m<0 : l123 ; ValueError: negative dimensions are not allowed
@pytest.mark.xfail
def test_dim_m_negative_should_fail():
    with pytest.raises(Exception):
        simulate_data(
            n=10,
            rg=default_rng(42),
            mis_spec_m=False,
            mis_spec_y=False,
            dim_x=1,
            dim_m=-1,
            seed=1,
            type_m="binary",
            sigma_y=0.5,
            sigma_m=0.5,
            beta_t_factor=1,
            beta_m_factor=1,
        )


# dim_m>1 ; n>1 ; "binary" ; l39
@pytest.mark.xfail
def test_m_multidimensional_binary_works():
    try:
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
    except ValueError as err:
        pprint(err)
        assert False
    else:
        pass


# dim_m>1 ; n=1 ; l58
@pytest.mark.xfail
def test_m_multidimensional_binary_works1():
    try:
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
    except ValueError as err:
        pprint(err)
        assert False
    else:
        pass


# sigma_m grand ; "continuous" ; P(T=1|X,M) = NaN
@pytest.mark.xfail
def test_huge_sigma_m_makes_nan():
    with pytest.raises(Warning):
        data_temp = simulate_data(
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
    assert data_temp[10] != np.nan


# sigma_m=0 ; "continuous" ; P(T=1|X,M) = NaN
@pytest.mark.xfail
def test_null_sigma_m_makes_nan():
    with pytest.raises(Exception):
        data_temp = simulate_data(
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
    assert data_temp[10] != np.nan


# TESTS IMPLEMENTES
# total = indirect + direct (direct 0 et indirect 1)
# mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_y=True : 5!=6!=gamma_t ;  7!=8
# vérifier la binarité, la continuité, la dimension de x,t,m,y

# RENVOI D'ERREUR
# n<0 ou dim<0 ; ValueError: negative dimensions are not allowed

# A CORRIGER
# dim_x=0 : No Warning
# dim_m=0 ; l134 : ValueError [Mais pas pour les bonnes raisons]
# dim_m>1 ; n>1 ; "binary" ; l39 [cas non implémenté]
# dim_m>1 ; n=1 ; l58 [cas non implémenté]
# n=0 : Warnings [on pourrait l'interdire à l'utilisateur]
# sigma_m grand ; "continuous" ; P(T=1|X,M) = NaN
# sigma_m=0 ; "continuous" ; P(T=1|X,M) = NaN

# CORRECTIONS ADDITIONNELLES
# Donner des arguments invalides, on s'en préoccupe? Si oui comment? (n, dim_x, dim_m <1)
# n=1, ne pas avoir une liste de liste?
# Forcer l'argument "continuous"? Poser un warning si c'est une autre lettre que "c"?
# Retirer le main?
# Pourquoi beta_t_factor et pas omega_t? ou carrément beta_t?
# Les omega existent pas l'article mais pas dans la fonction


# TESTS RESTANTS
# Tester qu'avec deux seeds différentes, les données sont différentes? (rigoureux? utile?)
# Histoire d'être un vecteur ou une matrix? (huber_no_reg plutôt?)
# Tester les sorties qui devraient être modifiées et celles qui ne le devraient pas
# au vu du modèle posé dans l'article? (trop long? utile?)


# FEATURES DE SECOND ORDRE
# Pouvoir fixer gamma_t = 1.2?
# Pas de sigma_x?
# Si l'utilisateur a le papier devant lui, on peut énumérer en entrée
# tous les paramètres alpha, beta, gamma?

# FEATURES DE TROISIEME ORDRE
# Donner les bornes dans lesquelles sont générées les données ou les effets?
# Tout générer entre 0 et 1, et puis l'utilisateur se débrouille?
# Normaliser les effets? Poser l'ATE=1?
