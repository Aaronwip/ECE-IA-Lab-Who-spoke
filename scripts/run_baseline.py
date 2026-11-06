"""Lance la baseline sur les premiers fichiers de data/raw/audio et écrit un RTTM prédit par fichier.

Sans option : toute la parole est attribuée à un seul locuteur.
Avec --oracle : MFCC + clustering, avec le nombre de locuteurs lu dans la référence.
C'est un diagnostic, pas un vrai système : sur un audio inconnu, ce nombre n'est pas disponible.

Usage :
    uv run python scripts/run_baseline.py --limite 10
    uv run python scripts/run_baseline.py --limite 10 --oracle --sortie outputs/baseline_mfcc_oracle
"""
import argparse
import os
import time
from pathlib import Path

from whospoke.baseline import SR, charger_audio, diariser
from whospoke.io_rttm import lire_rttm

parser = argparse.ArgumentParser(description="Diarisation baseline d'un dossier d'audios")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--sortie", default="outputs/baseline_vad", help="dossier des RTTM prédits")
parser.add_argument("--limite", type=int, default=3, help="nombre de fichiers à traiter")
parser.add_argument("--oracle", action="store_true", help="lit le nombre de locuteurs dans la référence")
parser.add_argument("--references", default="data/raw/dev", help="dossier des RTTM de référence")
args = parser.parse_args()

# Tri alphabétique : on prend toujours les mêmes fichiers, donc les runs restent comparables
fichiers = sorted(Path(args.audio).glob("*.wav"))[: args.limite]

sortie = Path(args.sortie)
os.makedirs(sortie, exist_ok=True)
temps_total = 0.0

# Le tout premier appel charge les modules de librosa (plusieurs secondes) : on le fait hors chrono
diariser(charger_audio(fichiers[0]), fichiers[0].stem, nb_locuteurs=2)

for i, chemin in enumerate(fichiers, start=1):
    signal = charger_audio(chemin)
    nb_locuteurs = 1
    if args.oracle:
        nb_locuteurs = len(lire_rttm(Path(args.references) / f"{chemin.stem}.rttm").labels())

    # On ne chronomètre que la diarisation, pas le chargement du fichier
    debut = time.perf_counter()
    diarization = diariser(signal, chemin.stem, nb_locuteurs)
    duree = time.perf_counter() - debut
    temps_total += duree

    with open(sortie / f"{chemin.stem}.rttm", "w") as f:
        diarization.write_rttm(f)
    print(f"[{i}/{len(fichiers)}] {chemin.stem} : {len(signal) / SR:.0f} s d'audio, "
          f"{nb_locuteurs} locuteur(s), traité en {duree:.2f} s")

# Lu ensuite par evaluate.py pour remplir la colonne temps_s de results/runs.csv
(sortie / "temps.txt").write_text(f"{temps_total:.1f}")
print(f"Temps total : {temps_total:.1f} s -> RTTM dans {sortie}")
