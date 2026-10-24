"""Lecture des fichiers RTTM (format standard « qui parle quand »)."""
from pathlib import Path

from pyannote.core import Annotation, Segment


def lire_rttm(chemin):
    """Lit un fichier RTTM et renvoie une Annotation pyannote.

    Une ligne RTTM ressemble à :
        SPEAKER abjxc 1 0.400 6.640 <NA> <NA> spk00 <NA> <NA>
    champ 1 = identifiant du fichier, 3 = début (s), 4 = durée (s), 7 = locuteur.
    """
    chemin = Path(chemin)
    annotation = Annotation(uri=chemin.stem)
    with open(chemin, encoding="utf-8") as f:
        for numero, ligne in enumerate(f):
            champs = ligne.split()
            if not champs or champs[0] != "SPEAKER":
                continue  # ignore les lignes vides ou d'un autre type
            debut = float(champs[3])
            duree = float(champs[4])
            locuteur = champs[7]
            # Le numéro de ligne sert d'identifiant de piste : sans lui, deux locuteurs
            # qui ont exactement le même segment (chevauchement) s'écraseraient.
            annotation[Segment(debut, debut + duree), numero] = locuteur
    return annotation
