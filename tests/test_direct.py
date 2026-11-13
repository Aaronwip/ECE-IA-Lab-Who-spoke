"""Tests du direct avec un FAUX pyannote (instantané) : on vérifie le découpage au fil de l'eau,
les décalages, le rééchantillonnage et la remise à zéro, pas la qualité de la diarisation."""
from types import SimpleNamespace

import numpy as np
from pyannote.core import Annotation, Segment

from whospoke.direct import SR, Direct, en_float_mono


class FauxPipeline:
    """Un seul locuteur, qui parle pendant tout le paquet, toujours avec la même voix."""

    def __init__(self):
        self.durees = []  # durée (s) de chaque paquet reçu

    def __call__(self, entree):
        duree = entree["waveform"].shape[1] / entree["sample_rate"]
        self.durees.append(duree)
        locale = Annotation()
        locale[Segment(0, duree)] = "SPEAKER_00"
        return SimpleNamespace(speaker_diarization=locale, speaker_embeddings=np.array([[1.0, 0.0]]))


def micro(direct, secondes, sr, morceau=0.5):
    """Simule le navigateur : un morceau d'entiers 16 bits toutes les 0,5 s."""
    n = int(morceau * sr)
    for _ in range(int(secondes / morceau)):
        direct.ajouter(sr, (np.random.randn(n) * 1000).astype(np.int16))


def test_paquets_au_fil_de_l_eau_et_reste_final():
    faux = FauxPipeline()
    direct = Direct(faux, paquet=60, minimum=10)
    micro(direct, 130, sr=48000)   # 130 s à 48 kHz -> 2 paquets pleins + un reste de 10 s
    direct.terminer()
    assert faux.durees == [60, 60, 10]
    etat = direct.etat()
    assert len(etat["audio"]) == 130 * SR              # tout est rééchantillonné en 16 kHz
    debuts = sorted(s.start for s in etat["resultat"].itersegments())
    assert debuts == [0, 60, 120]                      # chaque paquet à sa vraie place
    assert etat["resultat"].labels() == ["SPK00"]      # même voix -> un seul locuteur
    assert etat["analyse"] == 130


def test_reste_trop_court_ignore():
    faux = FauxPipeline()
    direct = Direct(faux, paquet=60, minimum=10)
    micro(direct, 65, sr=SR)       # reste de 5 s < 10 s : pas analysé
    direct.terminer()
    assert faux.durees == [60]
    assert direct.etat()["recu"] == 65


def test_reinitialiser_repart_de_zero():
    direct = Direct(FauxPipeline(), paquet=60)
    micro(direct, 60, sr=SR)
    direct.terminer()
    direct.reinitialiser()
    etat = direct.etat()
    assert etat["recu"] == 0 and len(etat["resultat"]) == 0 and len(etat["audio"]) == 0


def test_en_float_mono():
    stereo = np.array([[32767, -32767], [0, 0]], dtype=np.int16)
    assert np.allclose(en_float_mono(stereo), [0.0, 0.0])
    assert en_float_mono(np.array([32767], dtype=np.int16))[0] == 1.0
