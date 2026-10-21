import argparse
import os
from pathlib import Path

import soundfile as sf
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

# Lit le chemin de l'audio passé en argument
parser = argparse.ArgumentParser(description="Diarisation d'un fichier audio avec pyannote")
parser.add_argument("audio", help="chemin du fichier audio (wav mono 16 kHz conseillé)")
args = parser.parse_args()
audio_path = Path(args.audio)

load_dotenv()  # charge HF_TOKEN depuis le fichier .env

# Télécharge le modèle depuis Hugging Face (lent la 1re fois, ensuite en cache)
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.getenv("HF_TOKEN"),
)

# Charge l'audio en mémoire avec soundfile (évite torchcodec sous Windows)
data, sr = sf.read(audio_path, dtype="float32")
if data.ndim > 1:
    data = data.mean(axis=1)  # stéréo -> mono
waveform = torch.from_numpy(data).unsqueeze(0)  # forme (1, nb_échantillons)

# Lance la diarisation
output = pipeline({"waveform": waveform, "sample_rate": sr})
diarization = output.speaker_diarization

# Affiche qui parle quand
for turn, speaker in diarization:
    print(f"{speaker} parle de {turn.start:.1f}s à {turn.end:.1f}s")

# Écrit le résultat au format RTTM dans outputs/<nom>.rttm
os.makedirs("outputs", exist_ok=True)
rttm_path = Path("outputs") / f"{audio_path.stem}.rttm"
diarization.uri = audio_path.stem  # identifiant du fichier écrit dans chaque ligne RTTM
with open(rttm_path, "w") as f:
    diarization.write_rttm(f)
print(f"RTTM écrit dans {rttm_path}")
