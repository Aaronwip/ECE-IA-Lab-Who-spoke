"""Mesure le temps passé par pyannote dans chacune de ses étapes, sur un seul fichier.

Usage :
    uv run python scripts/profil_pyannote.py data/raw/audio/ahnss.wav

Principe : le pipeline appelle une fonction « hook » à la fin de chaque étape
(segmentation, comptage des locuteurs, embeddings, reconstruction).
Notre hook note l'heure à chaque appel. La durée d'une étape = heure de sa fin
moins heure de fin de l'étape précédente.
"""
import argparse
import os
import time

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

parser = argparse.ArgumentParser(description="Temps par étape de pyannote")
parser.add_argument("fichier", help="chemin du .wav (16 kHz mono)")
parser.add_argument("--modele", default="pyannote/speaker-diarization-community-1")
args = parser.parse_args()

load_dotenv()
pipeline = Pipeline.from_pretrained(args.modele, token=os.getenv("HF_TOKEN"))

# Réglages qui influencent la vitesse : on les affiche pour pouvoir les citer
print(f"Threads CPU utilisés par torch : {torch.get_num_threads()}")
print(f"Taille des lots - segmentation : {pipeline.segmentation_batch_size}, "
      f"embeddings : {pipeline.embedding_batch_size}")

data, sr = sf.read(args.fichier, dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)
waveform = torch.from_numpy(data).unsqueeze(0)
duree_audio = len(data) / sr

# fin_etape["embeddings"] = heure du DERNIER appel du hook pour cette étape.
# Le hook est aussi appelé pendant l'étape (barre de progression) : on écrase
# la valeur à chaque fois, donc il ne reste que l'heure de fin.
fin_etape = {}


def hook(nom_etape, artefact, file=None, total=None, completed=None):
    fin_etape[nom_etape] = time.perf_counter()


debut = time.perf_counter()
pipeline({"waveform": waveform, "sample_rate": sr}, hook=hook)
fin = time.perf_counter()

# Les étapes arrivent dans l'ordre d'exécution (un dict garde l'ordre d'insertion)
noms = {
    "segmentation": "segmentation (qui parle quand, par fenêtre)",
    "speaker_counting": "comptage des locuteurs simultanés",
    "embeddings": "embeddings (empreinte de chaque voix)",
    "discrete_diarization": "clustering + reconstruction",
}
total = fin - debut
precedent = debut
print(f"\nFichier : {args.fichier} ({duree_audio:.0f} s d'audio)\n")
for etape, heure in fin_etape.items():
    duree = heure - precedent
    print(f"{noms.get(etape, etape):45s} {duree:7.1f} s  ({100 * duree / total:4.1f} %)")
    precedent = heure
print(f"{'fin (mise en forme du résultat)':45s} {fin - precedent:7.1f} s")
print(f"\nTotal : {total:.1f} s, soit {total / duree_audio:.2f} x la durée de l'audio")
print(f"Pour 20 min de soutenance, il faudrait environ {total / duree_audio * 20:.1f} min")
