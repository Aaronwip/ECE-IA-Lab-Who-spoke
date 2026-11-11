"""Diarisation « en direct » simulée : on découpe chaque audio en paquets de N secondes,
on lance pyannote sur chaque paquet séparément, puis on relie les locuteurs d'un paquet à l'autre.

Usage :
    uv run python scripts/run_paquets.py --limite 10 --paquet 60 --seuil 0.5 --fusion 10

Le problème à résoudre : dans chaque paquet, pyannote nomme les locuteurs SPEAKER_00, SPEAKER_01...
Ces noms sont LOCAUX : le SPEAKER_00 du paquet 2 n'est pas forcément le SPEAKER_00 du paquet 1.

La solution : pyannote renvoie aussi une empreinte de voix (embedding, un vecteur) par locuteur.
On garde une liste de locuteurs « globaux », chacun avec la moyenne de ses empreintes.
Pour chaque locuteur d'un nouveau paquet, on cherche le locuteur global le plus ressemblant
(similarité cosinus) :
  - si la ressemblance dépasse le seuil -> c'est la même personne ;
  - sinon -> c'est un nouveau locuteur.
Le code de ces étapes est dans src/whospoke/paquets.py, partagé avec la démo.

Critère « temps réel » : chaque paquet doit être traité en moins de N secondes,
sinon le retard s'accumule pendant la soutenance.
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import soundfile as sf
from pyannote.core import Annotation

from whospoke.paquets import MODELE, Relieur, bornes_paquets, charger_pipeline, diariser_paquet, placer

parser = argparse.ArgumentParser(description="Diarisation pyannote par paquets")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--sortie", default=None, help="dossier des RTTM (défaut : outputs/paquets_<paquet>s_<seuil>)")
parser.add_argument("--limite", type=int, default=10, help="nombre de fichiers (tri alphabétique)")
parser.add_argument("--paquet", type=float, default=60.0, help="durée d'un paquet en secondes")
parser.add_argument("--seuil", type=float, default=0.5, help="similarité cosinus minimale pour dire « même personne »")
parser.add_argument("--fusion", type=float, default=0.0,
                    help="dernier paquet collé au précédent s'il dure moins de N s (0 = jamais, comportement du run B)")
parser.add_argument("--modele", default=MODELE)
parser.add_argument("--segmentation", default=None,
                    help="checkpoint .ckpt d'une segmentation fine-tunée (ex. models/finetune_sdm_2026-11-11/segmentation.ckpt)")
args = parser.parse_args()

# Étiquette du modèle de segmentation : le cache et les sorties d'un modèle ne doivent jamais servir à un autre
etiquette = f"_seg-{Path(args.segmentation).parent.name}" if args.segmentation else ""
nom = f"paquets_{args.paquet:.0f}s_{args.seuil}" + (f"_fusion{args.fusion:.0f}" if args.fusion else "") + etiquette
sortie = Path(args.sortie or f"outputs/{nom}")
os.makedirs(sortie, exist_ok=True)
cache = Path(f"outputs/cache_paquets_{args.paquet:.0f}s{etiquette}")  # résultats pyannote bruts, par paquet
os.makedirs(cache, exist_ok=True)
fichiers = sorted(Path(args.audio).glob("*.wav"))[: args.limite]

pipeline = charger_pipeline(args.modele, args.segmentation)  # chargé même si tout est en cache : plus simple

temps_total = 0.0
pire_paquet = 0.0  # le paquet le plus lent, rapporté à sa durée (doit rester < 1)

for i, chemin in enumerate(fichiers, start=1):
    data, sr = sf.read(chemin, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    bornes = bornes_paquets(len(data), sr, args.paquet, args.fusion)

    # Étape 1 (lente) : pyannote sur chaque paquet. Le résultat est mis en cache, paquet par paquet,
    # car il ne dépend pas du seuil : on peut ensuite essayer plusieurs seuils sans relancer,
    # et un paquet déjà calculé avec les mêmes bornes n'est jamais recalculé.
    paquets = []  # liste de (annotation locale, empreintes, durée de calcul en s)
    for debut_ech, fin_ech in bornes:
        fichier_cache = cache / f"{chemin.stem}_{debut_ech}_{fin_ech}.pkl"
        if fichier_cache.exists():
            locale, empreintes, duree = pickle.loads(fichier_cache.read_bytes())
        else:
            t0 = time.perf_counter()
            locale, empreintes = diariser_paquet(pipeline, data[debut_ech:fin_ech], sr)
            duree = time.perf_counter() - t0
            fichier_cache.write_bytes(pickle.dumps((locale, empreintes, duree)))
        paquets.append((locale, empreintes, duree))

    # Étape 2 (rapide) : relier les locuteurs d'un paquet à l'autre
    relieur = Relieur(args.seuil)
    resultat = Annotation(uri=chemin.stem)
    for (locale, empreintes, duree), (debut_ech, fin_ech) in zip(paquets, bornes):
        temps_total += duree
        pire_paquet = max(pire_paquet, duree / ((fin_ech - debut_ech) / sr))
        correspondance = relieur.relier(locale, empreintes)
        placer(resultat, locale, correspondance, debut_ech / sr)

    with open(sortie / f"{chemin.stem}.rttm", "w") as f:
        resultat.write_rttm(f)
    print(f"[{i}/{len(fichiers)}] {chemin.stem} : {len(data) / sr:.0f} s, "
          f"{len(relieur.globaux)} locuteurs trouvés")

(sortie / "temps.txt").write_text(f"{temps_total:.1f}")
print(f"Temps total : {temps_total:.1f} s -> RTTM dans {sortie}")
print(f"Paquet le plus lent : {pire_paquet:.2f} x sa durée (temps réel tenu si < 1)")
