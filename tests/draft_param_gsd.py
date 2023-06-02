from pprint import pprint
import itertools
import pytest
import numpy as np
from numpy.random import default_rng
from .get_simulated_data import simulate_data


# use directly itertools, param_list useless?
parameter_list = []
for parameter in itertools.product(
    [1, 2, 100],
    [default_rng(1), default_rng(10)],
    [False, True],
    [False, True],
    [1, 2, 3],
    [1, 2, 3],
    [None, 2, 20],
    ["binary", "continuous"],
    [0.5, 0.05, 5],
    [0.5, 0.05, 5],
    [1, 0.1, 10],
    [1, 0.1, 10],
):
    parameter_list.append(parameter)


@pytest.mark.parametrize(
    [
        "n",
        "rg",
        "mis_spec_m",
        "mis_spec_y",
        "dim_x",
        "dim_m",
        "seed",
        "type_m",
        "sigma_y",
        "sigma_m",
        "beta_t_factor",
        "beta_m_factor",
    ],
    parameter_list,
)
class ParametrizedTest:
    def test_total_is_direct_plus_indirect(
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
        data = simulate_data(
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
        effects = np.array(data[4:9])

        assert effects[0] == effects[1] + effects[4]  # total = theta_1 + delta_0
        assert effects[0] == effects[2] + effects[3]  # total = theta_0 + delta_1

    def test_effects_are_equals_if_y_well_specified(self):
        pass


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
