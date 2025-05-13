"""
Exemple 2
=========

Ceci est un exemple simple montrant comment tracer une fonction sinus.

.. image:: ../images/linear_reg_ex.png
   :alt: Exemple de régression linéaire
   :width: 400px
   :align: center

Description
-----------

Nous allons générer des données à l'aide de NumPy et les tracer avec Matplotlib.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# Générer des données
x = np.linspace(0, 10, 100)
y = np.sin(x)

# %%

# Tracer les données
plt.plot(x, y)
plt.title("Exemple de tracé")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
