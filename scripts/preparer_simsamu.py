"""Prépare Simsamu (français) en local : wav 16 kHz + RTTM de référence, découpé par séance.

Usage :
    uv run python scripts/preparer_simsamu.py

Résultat :
    data/raw/simsamu/validation/{audio,rttm}/   séance dj_2023 (7 appels) : pour décider
    data/raw/simsamu/test/{audio,rttm}/         séance reu (9 appels)     : mesure finale UNIQUEMENT

Le découpage est le même que dans le notebook Kaggle (train = dj_2022 + dj_mai, jamais
utilisé ici). Le rééchantillonnage 8 kHz -> 16 kHz aussi (resample_poly), pour que le modèle
voie en local exactement le même audio qu'à l'entraînement.
pyarrow : lit le format parquet, celui des jeux de données Hugging Face.
"""
import io
from pathlib import Path

import pyarrow.parquet as pq
import soundfile as sf
from huggingface_hub import hf_hub_download
from scipy.signal import resample_poly

RACINE = Path("data/raw/simsamu")
PARQUET = RACINE / "train-00000-of-00001.parquet"   # Simsamu n'a qu'un fichier, nommé « train » par HF
PARTITIONS = {"validation": ("dj_2023_",), "test": ("reu_",)}

if not PARQUET.exists():
    hf_hub_download("diarizers-community/simsamu", "data/train-00000-of-00001.parquet",
                    repo_type="dataset", local_dir=RACINE)
    (RACINE / "data" / PARQUET.name).rename(PARQUET)

compte = {p: 0 for p in PARTITIONS}
for lot in pq.ParquetFile(PARQUET).iter_batches(batch_size=1):
    ligne = lot.to_pylist()[0]
    uri = Path(ligne["audio"]["path"]).stem   # ex. reu_covid
    partition = next((p for p, prefixes in PARTITIONS.items() if uri.startswith(prefixes)), None)
    if partition is None:
        continue   # séances d'entraînement : inutiles en local
    for dossier in ["audio", "rttm"]:
        (RACINE / partition / dossier).mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(io.BytesIO(ligne["audio"]["bytes"]), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = resample_poly(audio, 16000, sr).astype("float32")
    sf.write(RACINE / partition / "audio" / f"{uri}.wav", audio, 16000, subtype="PCM_16")
    with open(RACINE / partition / "rttm" / f"{uri}.rttm", "w") as f:
        for debut, fin, locuteur in zip(ligne["timestamps_start"], ligne["timestamps_end"], ligne["speakers"]):
            f.write(f"SPEAKER {uri} 1 {debut:.3f} {fin - debut:.3f} <NA> <NA> {locuteur} <NA> <NA>\n")
    compte[partition] += 1

print(compte)
assert compte == {"validation": 7, "test": 9}, "découpage inattendu"
