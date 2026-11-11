"""Diarisation par paquets : le code commun au script d'évaluation (scripts/run_paquets.py)
et à la démo (app/demo.py). Les deux utilisent exactement les mêmes fonctions,
donc la démo se comporte comme le système mesuré dans results/runs.csv.

Principe :
  1. on découpe l'audio en paquets de N secondes (bornes_paquets) ;
  2. pyannote traite chaque paquet séparément (diariser_paquet) ;
     dans un paquet, les noms SPEAKER_00, SPEAKER_01... sont LOCAUX ;
  3. le Relieur compare l'empreinte de voix de chaque locuteur local aux locuteurs
     déjà connus (similarité cosinus) : au-dessus du seuil -> même personne, sinon -> nouveau ;
  4. placer() recopie les tours de parole du paquet à leur vraie position dans la conversation.
"""
import os

import numpy as np
import torch
from pyannote.core import Segment

MODELE = "pyannote/speaker-diarization-community-1"


def charger_pipeline(modele=MODELE, segmentation=None):
    """Charge le pipeline pyannote. `segmentation` : checkpoint .ckpt fine-tuné (optionnel)."""
    from dotenv import load_dotenv
    from pyannote.audio import Model, Pipeline

    load_dotenv()
    pipeline = Pipeline.from_pretrained(modele, token=os.getenv("HF_TOKEN"))
    if segmentation:
        # Même architecture, nouveaux poids : on remplace seulement les poids du réseau de segmentation
        poids = Model.from_pretrained(segmentation).state_dict()
        pipeline._segmentation.model.load_state_dict(poids)
        print(f"Segmentation remplacée par {segmentation}")
    return pipeline


def bornes_paquets(nb_echantillons, sr, paquet=60.0, fusion=10.0):
    """Liste des paquets sous forme (début, fin) en échantillons.

    Un dernier paquet plus court que `fusion` secondes (ex. 2 s) donne une empreinte
    inutilisable : on le colle au paquet précédent (run D). fusion=0 : jamais (run B).
    """
    taille = int(paquet * sr)
    bornes = [(d, min(d + taille, nb_echantillons)) for d in range(0, nb_echantillons, taille)]
    if len(bornes) > 1 and bornes[-1][1] - bornes[-1][0] < fusion * sr:
        dernier = bornes.pop()
        bornes[-1] = (bornes[-1][0], dernier[1])
    return bornes


def diariser_paquet(pipeline, morceau, sr):
    """pyannote sur un seul paquet (tableau numpy mono float32).
    Renvoie (annotation locale, empreintes) ; empreintes[k] = voix du locuteur locale.labels()[k]."""
    waveform = torch.from_numpy(np.ascontiguousarray(morceau, dtype=np.float32)).unsqueeze(0)
    output = pipeline({"waveform": waveform, "sample_rate": sr})
    return output.speaker_diarization, output.speaker_embeddings


def cosinus(a, b):
    """Similarité cosinus : 1 = même direction (voix très proches), 0 = sans rapport."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class Relieur:
    """Mémoire des locuteurs d'UNE conversation, d'un paquet à l'autre.

    Chaque locuteur global garde la somme de ses empreintes et leur nombre :
    sa voix de référence est la moyenne (somme / n), qui s'affine à chaque paquet.
    """

    def __init__(self, seuil=0.5):
        self.seuil = seuil
        self.globaux = {}  # nom -> [somme des empreintes, nombre de paquets]

    def relier(self, locale, empreintes):
        """Renvoie le dictionnaire nom local -> nom global pour un nouveau paquet."""
        correspondance = {}
        deja_pris = set()  # deux voix d'un même paquet ne peuvent pas être la même personne
        for k, local in enumerate(locale.labels()):
            e = empreintes[k]
            meilleur, meilleure_sim = None, -1.0
            if np.linalg.norm(e) > 0:  # pyannote met une empreinte nulle s'il n'a pas pu la calculer
                for nom, (somme, n) in self.globaux.items():
                    if nom in deja_pris or n == 0:  # n == 0 : aucune empreinte connue, rien à comparer
                        continue
                    sim = cosinus(e, somme / n)
                    if sim > meilleure_sim:
                        meilleur, meilleure_sim = nom, sim
            if meilleur is None or meilleure_sim < self.seuil:
                meilleur = f"SPK{len(self.globaux):02d}"  # nouveau locuteur
                self.globaux[meilleur] = [np.zeros_like(e), 0]
            if np.linalg.norm(e) > 0:
                self.globaux[meilleur][0] += e
                self.globaux[meilleur][1] += 1
            correspondance[local] = meilleur
            deja_pris.add(meilleur)
        return correspondance


def placer(resultat, locale, correspondance, decalage):
    """Recopie les tours de parole d'un paquet dans `resultat`, décalés de `decalage` secondes.
    La piste (2e clé) rend chaque tour unique, même si deux voix ont les mêmes bornes."""
    for tour, piste, local in locale.itertracks(yield_label=True):
        segment = Segment(tour.start + decalage, tour.end + decalage)
        resultat[segment, f"{decalage:.0f}_{piste}_{local}"] = correspondance[local]
