"""Diarisation « en direct » simulée : on découpe chaque audio en paquets de N secondes,
on lance pyannote sur chaque paquet séparément, puis on relie les locuteurs d'un paquet à l'autre.

Usage :
    uv run python scripts/run_paquets.py --limite 10 --paquet 60 --seuil 0.5

Le problème à résoudre : dans chaque paquet, pyannote nomme les locuteurs SPEAKER_00, SPEAKER_01...
Ces noms sont LOCAUX : le SPEAKER_00 du paquet 2 n'est pas forcément le SPEAKER_00 du paquet 1.

La solution : pyannote renvoie aussi une empreinte de voix (embedding, un vecteur) par locuteur.
On garde une liste de locuteurs « globaux », chacun avec la moyenne de ses empreintes.
Pour chaque locuteur d'un nouveau paquet, on cherche le locuteur global le plus ressemblant
(similarité cosinus) :
  - si la ressemblance dépasse le seuil -> c'est la même personne ;
  - sinon -> c'est un nouveau locuteur.

Critère « temps réel » : chaque paquet doit être traité en moins de N secondes,
sinon le retard s'accumule pendant la soutenance.
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment

parser = argparse.ArgumentParser(description="Diarisation pyannote par paquets")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--sortie", default=None, help="dossier des RTTM (défaut : outputs/paquets_<paquet>s_<seuil>)")
parser.add_argument("--limite", type=int, default=10, help="nombre de fichiers (tri alphabétique)")
parser.add_argument("--paquet", type=float, default=60.0, help="durée d'un paquet en secondes")
parser.add_argument("--seuil", type=float, default=0.5, help="similarité cosinus minimale pour dire « même personne »")
parser.add_argument("--fusion", type=float, default=0.0,
                    help="dernier paquet collé au précédent s'il dure moins de N s (0 = jamais, comportement du run B)")
parser.add_argument("--modele", default="pyannote/speaker-diarization-community-1")
args = parser.parse_args()

nom = f"paquets_{args.paquet:.0f}s_{args.seuil}" + (f"_fusion{args.fusion:.0f}" if args.fusion else "")
sortie = Path(args.sortie or f"outputs/{nom}")
os.makedirs(sortie, exist_ok=True)
cache = Path(f"outputs/cache_paquets_{args.paquet:.0f}s")  # résultats pyannote bruts, par fichier
os.makedirs(cache, exist_ok=True)
fichiers = sorted(Path(args.audio).glob("*.wav"))[: args.limite]

load_dotenv()
pipeline = Pipeline.from_pretrained(args.modele, token=os.getenv("HF_TOKEN"))  # chargé même si tout est en cache : plus simple


def cosinus(a, b):
    """Similarité cosinus : 1 = même direction (voix très proches), 0 = sans rapport."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


temps_total = 0.0
pire_paquet = 0.0  # le paquet le plus lent, rapporté à sa durée (doit rester < 1)

for i, chemin in enumerate(fichiers, start=1):
    data, sr = sf.read(chemin, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    taille = int(args.paquet * sr)  # nombre d'échantillons par paquet

    # Bornes des paquets, en échantillons : (début, fin)
    bornes = [(d, min(d + taille, len(data))) for d in range(0, len(data), taille)]
    # Un dernier paquet trop court (ex. 2 s) donne une empreinte inutilisable :
    # on le colle au paquet précédent
    if len(bornes) > 1 and bornes[-1][1] - bornes[-1][0] < args.fusion * sr:
        dernier = bornes.pop()
        bornes[-1] = (bornes[-1][0], dernier[1])

    # Étape 1 (lente) : pyannote sur chaque paquet. Le résultat est mis en cache, paquet par paquet,
    # car il ne dépend pas du seuil : on peut ensuite essayer plusieurs seuils sans relancer,
    # et un paquet déjà calculé avec les mêmes bornes n'est jamais recalculé.
    paquets = []  # liste de (décalage en s, annotation locale, empreintes, durée de calcul en s)
    for debut_ech, fin_ech in bornes:
        fichier_cache = cache / f"{chemin.stem}_{debut_ech}_{fin_ech}.pkl"
        if fichier_cache.exists():
            locale, empreintes, duree = pickle.loads(fichier_cache.read_bytes())
        else:
            waveform = torch.from_numpy(data[debut_ech:fin_ech]).unsqueeze(0)
            t0 = time.perf_counter()
            output = pipeline({"waveform": waveform, "sample_rate": sr})
            duree = time.perf_counter() - t0
            locale, empreintes = output.speaker_diarization, output.speaker_embeddings
            fichier_cache.write_bytes(pickle.dumps((locale, empreintes, duree)))
        paquets.append((debut_ech / sr, locale, empreintes, duree))

    # Étape 2 (rapide) : relier les locuteurs d'un paquet à l'autre
    # Locuteurs globaux de ce fichier : nom -> [somme des empreintes, nombre de paquets]
    globaux = {}
    resultat = Annotation(uri=chemin.stem)

    for (decalage, locale, empreintes, duree), (debut_ech, fin_ech) in zip(paquets, bornes):
        # empreintes[k] = empreinte du locuteur locale.labels()[k]
        temps_total += duree
        pire_paquet = max(pire_paquet, duree / ((fin_ech - debut_ech) / sr))

        # Relier chaque locuteur local à un locuteur global
        correspondance = {}
        deja_pris = set()  # deux voix d'un même paquet ne peuvent pas être la même personne
        for k, local in enumerate(locale.labels()):
            e = empreintes[k]
            meilleur, meilleure_sim = None, -1.0
            if np.linalg.norm(e) > 0:  # pyannote met une empreinte nulle s'il n'a pas pu la calculer
                for nom, (somme, n) in globaux.items():
                    if nom in deja_pris or n == 0:  # n == 0 : aucune empreinte connue, rien à comparer
                        continue
                    sim = cosinus(e, somme / n)
                    if sim > meilleure_sim:
                        meilleur, meilleure_sim = nom, sim
            if meilleur is None or meilleure_sim < args.seuil:
                meilleur = f"SPK{len(globaux):02d}"  # nouveau locuteur
                globaux[meilleur] = [np.zeros_like(e), 0]
            if np.linalg.norm(e) > 0:
                globaux[meilleur][0] += e
                globaux[meilleur][1] += 1
            correspondance[local] = meilleur
            deja_pris.add(meilleur)

        # Recopier les tours de parole du paquet, décalés à leur vraie position
        # La piste (2e clé) rend chaque tour unique, même si deux voix ont les mêmes bornes
        for tour, piste, local in locale.itertracks(yield_label=True):
            segment = Segment(tour.start + decalage, tour.end + decalage)
            resultat[segment, f"{decalage:.0f}_{piste}_{local}"] = correspondance[local]

    with open(sortie / f"{chemin.stem}.rttm", "w") as f:
        resultat.write_rttm(f)
    print(f"[{i}/{len(fichiers)}] {chemin.stem} : {len(data) / sr:.0f} s, "
          f"{len(globaux)} locuteurs trouvés")

(sortie / "temps.txt").write_text(f"{temps_total:.1f}")
print(f"Temps total : {temps_total:.1f} s -> RTTM dans {sortie}")
print(f"Paquet le plus lent : {pire_paquet:.2f} x sa durée (temps réel tenu si < 1)")
