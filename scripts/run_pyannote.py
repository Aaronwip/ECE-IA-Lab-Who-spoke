"""Lance pyannote sur les premiers fichiers de data/raw/audio et écrit un RTTM prédit par fichier.

Usage :
    uv run python scripts/run_pyannote.py --limite 3
"""
import argparse
import os
import time
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

parser = argparse.ArgumentParser(description="Diarisation pyannote d'un dossier d'audios")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--sortie", default="outputs/pyannote_dev", help="dossier des RTTM prédits")
parser.add_argument("--limite", type=int, default=3, help="nombre de fichiers à traiter")
args = parser.parse_args()

# Tri alphabétique : on prend toujours les mêmes fichiers, donc les runs restent comparables
fichiers = sorted(Path(args.audio).glob("*.wav"))[: args.limite]

load_dotenv()  # charge HF_TOKEN depuis le fichier .env
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)

sortie = Path(args.sortie)
os.makedirs(sortie, exist_ok=True)
temps_total = 0.0

for i, chemin in enumerate(fichiers, start=1):
    # Charge l'audio avec soundfile (évite torchcodec sous Windows)
    data, sr = sf.read(chemin, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)  # stéréo -> mono
    waveform = torch.from_numpy(data).unsqueeze(0)  # forme (1, nb_échantillons)

    # On ne chronomètre que la diarisation, pas le chargement du fichier
    debut = time.perf_counter()
    output = pipeline({"waveform": waveform, "sample_rate": sr})
    duree = time.perf_counter() - debut
    temps_total += duree

    diarization = output.speaker_diarization
    diarization.uri = chemin.stem  # même identifiant que dans le RTTM de référence
    with open(sortie / f"{chemin.stem}.rttm", "w") as f:
        diarization.write_rttm(f)
    print(f"[{i}/{len(fichiers)}] {chemin.stem} : {len(data) / sr:.0f} s d'audio, traité en {duree:.1f} s")

# Lu ensuite par evaluate.py pour remplir la colonne temps_s de results/runs.csv
(sortie / "temps.txt").write_text(f"{temps_total:.1f}")
print(f"Temps total : {temps_total:.1f} s -> RTTM dans {sortie}")
