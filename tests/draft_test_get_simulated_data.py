from pprint import pprint
import pytest
import itertools
import numpy as np
from numpy.random import default_rng
from ..src.get_simulated_data import simulate_data


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


data=[]

try:
    data = simulate_data(
        n=3,
        rg=default_rng(42),
        mis_spec_m=False,
        mis_spec_y=False,
        dim_x=1,
        dim_m=1,
        seed=1,
        type_m="binary",
        sigma_y=0.5,
        sigma_m=0.5,
        beta_t_factor=10,
        beta_m_factor=1)
except ValueError as err:
    print(err)
else:
    print(data)



@pytest.mark.parametrize("n", [1, 2, 100])
@pytest.mark.parametrize("rg", [default_rng(1), default_rng(10)])
@pytest.mark.parametrize("mis_spec_m", [False, True])
@pytest.mark.parametrize("mis_spec_y", [False, True])
@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("dim_m", [1, 2, 3])
@pytest.mark.parametrize("seed", [None, 2, 20])
@pytest.mark.parametrize("type_m", ["binary", "continuous"])
@pytest.mark.parametrize("sigma_y", [0.5, 0.05, 5])
@pytest.mark.parametrize("sigma_m", [0.5, 0.05, 5])
@pytest.mark.parametrize("beta_t_factor", [1, 0.1, 10])
@pytest.mark.parametrize("beta_m_factor", [1, 0.1, 10])




@pytest.fixture(autouse=True)
    def effects(
        self,
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
    ):
        setup(
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
        return np.array(data[4:9])


# TESTS
# total = indirect + direct (direct 0 et indirect 1)
# mis_spec_y=False : 5=6=gamma_t ;  7=8
# mis_spec_y=True : 5!=6!=gamma_t ;  7!=8
# Tester qu'avec même seed, génère mêmes données qu'avant?
# vérifier la binarité, la continuité, la dimension de x,t,m,y
# histoire d'être un vecteur ou une matrix???

# Tester ce qui devrait être modifié et ce qui ne le devrait pas
# Faire un test d'égalité sur chaque sortie => laquelle on a modifié

# Egalité de chaque paramètre pour petits exemples tant qu'à faire (robuste mais moche?)
# (sinon juste pour nous, ça garantit qu'on a rien changé à la fonction)


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