"""Lance la baseline (détection de parole par énergie) sur les premiers fichiers de data/raw/audio.

Écrit un RTTM prédit par fichier. Pour l'instant toute la parole est attribuée à un seul locuteur.

Usage :
    uv run python scripts/run_baseline.py --limite 25
"""
import argparse
import os
import time
from pathlib import Path

from whospoke.baseline import SR, charger_audio, detecter_parole, trames_vers_annotation

parser = argparse.ArgumentParser(description="Diarisation baseline d'un dossier d'audios")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--sortie", default="outputs/baseline_vad", help="dossier des RTTM prédits")
parser.add_argument("--limite", type=int, default=3, help="nombre de fichiers à traiter")
args = parser.parse_args()

# Tri alphabétique : on prend toujours les mêmes fichiers, donc les runs restent comparables
fichiers = sorted(Path(args.audio).glob("*.wav"))[: args.limite]

sortie = Path(args.sortie)
os.makedirs(sortie, exist_ok=True)
temps_total = 0.0

for i, chemin in enumerate(fichiers, start=1):
    signal = charger_audio(chemin)

    # On ne chronomètre que la diarisation, pas le chargement du fichier
    debut = time.perf_counter()
    parole = detecter_parole(signal)
    etiquettes = ["spk0" if p else None for p in parole]  # un seul locuteur pour l'instant
    diarization = trames_vers_annotation(etiquettes, chemin.stem)
    duree = time.perf_counter() - debut
    temps_total += duree

    with open(sortie / f"{chemin.stem}.rttm", "w") as f:
        diarization.write_rttm(f)
    print(f"[{i}/{len(fichiers)}] {chemin.stem} : {len(signal) / SR:.0f} s d'audio, traité en {duree:.2f} s")

# Lu ensuite par evaluate.py pour remplir la colonne temps_s de results/runs.csv
(sortie / "temps.txt").write_text(f"{temps_total:.1f}")
print(f"Temps total : {temps_total:.1f} s -> RTTM dans {sortie}")
