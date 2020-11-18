import numpy as np
import os


def compute_tau_interaction(X, fz, Delta_q):
    """où $X$ est la distance entre le coude humain et l'attache,
    $fz$ est la force renvoyée sur l'axe z du capteur (repère capteur)
    et $\Delta{q}$ est le décalage angulaire moyen que nous identifions
    sur la plage de mouvement.

    Args:
        X ([type]): [description]
        fz ([type]): [description]
        Delta_q ([type]): [description]
    """
    return X*fz*np.cos(Delta_q)

