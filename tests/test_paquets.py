"""Tests de la logique des paquets (sans pyannote : rapides)."""
import numpy as np
from pyannote.core import Annotation, Segment

from whospoke.paquets import Relieur, bornes_paquets, placer


def test_bornes_sans_fusion():
    # 130 s à 10 Hz -> paquets de 60 s : [0-60], [60-120], [120-130]
    assert bornes_paquets(1300, 10, paquet=60, fusion=0) == [(0, 600), (600, 1200), (1200, 1300)]


def test_bornes_fusion_du_dernier_paquet_court():
    # le dernier bout (10 s) dure moins de 15 s -> collé au précédent
    assert bornes_paquets(1300, 10, paquet=60, fusion=15) == [(0, 600), (600, 1300)]


def test_bornes_fichier_plus_court_qu_un_paquet():
    assert bornes_paquets(300, 10, paquet=60, fusion=10) == [(0, 300)]


def paquet(*labels):
    """Annotation locale avec un tour de parole d'1 s par locuteur."""
    locale = Annotation()
    for i, label in enumerate(labels):
        locale[Segment(i, i + 1)] = label
    return locale


def test_relieur_reconnait_la_meme_voix():
    relieur = Relieur(seuil=0.5)
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    assert relieur.relier(paquet("S0", "S1"), [a, b]) == {"S0": "SPK00", "S1": "SPK01"}
    # paquet suivant : les noms locaux sont inversés, mais les voix sont les mêmes
    assert relieur.relier(paquet("S0", "S1"), [b * 0.9, a]) == {"S0": "SPK01", "S1": "SPK00"}


def test_relieur_nouvelle_voix_sous_le_seuil():
    relieur = Relieur(seuil=0.5)
    relieur.relier(paquet("S0"), [np.array([1.0, 0.0])])
    assert relieur.relier(paquet("S0"), [np.array([0.0, 1.0])]) == {"S0": "SPK01"}


def test_relieur_deux_voix_d_un_paquet_jamais_fusionnees():
    relieur = Relieur(seuil=0.5)
    relieur.relier(paquet("S0"), [np.array([1.0, 0.0])])
    # deux voix proches de SPK00 dans le même paquet : une seule peut être SPK00
    sortie = relieur.relier(paquet("S0", "S1"), [np.array([1.0, 0.1]), np.array([1.0, 0.2])])
    assert sorted(sortie.values()) == ["SPK00", "SPK01"]


def test_relieur_empreinte_nulle_donne_un_nouveau_locuteur():
    relieur = Relieur(seuil=0.5)
    relieur.relier(paquet("S0"), [np.array([1.0, 0.0])])
    assert relieur.relier(paquet("S0"), [np.zeros(2)]) == {"S0": "SPK01"}


def test_placer_decale_les_tours():
    resultat = Annotation()
    placer(resultat, paquet("S0"), {"S0": "SPK00"}, decalage=60.0)
    assert list(resultat.itertracks(yield_label=True))[0][0] == Segment(60, 61)
