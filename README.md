# Who Spoke? — diarisation de locuteurs

Projet individuel Intelligence Lab (ECE Paris, 16/10 → 16/11/2026, mentor Mathis Escriva).
**Question : qui parle, et quand ?** sur des conversations jamais vues, avec une démo qui peut tourner en direct.

## Approche
- **Baselines faites main** (comparaison) : détection de parole par énergie, puis MFCC + k-means.
- **Système final** : `pyannote/speaker-diarization-community-1`, open source et auto-hébergeable,
  traité **par paquets de 60 s** pour suivre une conversation en direct (`src/whospoke/paquets.py`).
- **Fine-tuning** de la segmentation sur Kaggle (`kaggle/finetune_segmentation.ipynb`) :
  AMI (anglais, sans gain) puis Simsamu (appels au SAMU simulés, en français).

## Résultats (VoxConverse dev, 10 fichiers, DER en %, plus bas = mieux)
| Approche | DER strict | DER standard (±0,25 s) | JER |
|---|---|---|---|
| Baseline énergie | 49,48 | 46,92 | 81,2 |
| Baseline MFCC + k-means | 41,74 | 38,45 | 56,8 |
| pyannote, fichier entier | 5,51 | 2,53 | 15,3 |
| pyannote par paquets de 60 s (système de la démo) | 6,02 | 2,99 | 16,8 |

Sur Simsamu (validation `dj_2023`), le même système fait 13,22 % de DER standard. Le fine-tuning divise
par deux le DER de validation de la segmentation (0,198 → 0,098). La mesure sur le test `reu` reste à faire.
Tous les runs : [`results/runs.csv`](results/runs.csv). Progression : [`docs/journal.md`](docs/journal.md).
Données : [`docs/fiche_corpus.md`](docs/fiche_corpus.md).

## Installation (Windows, Python 3.11, [uv](https://docs.astral.sh/uv/), ffmpeg)
```powershell
uv sync
# Jeton Hugging Face (accès accepté au modèle pyannote/speaker-diarization-community-1)
"HF_TOKEN=hf_xxx" | Out-File -Encoding utf8 .env
```

## Reproduire
```powershell
uv run python scripts/telecharger_voxconverse.py          # VoxConverse dev -> data/raw
uv run python scripts/preparer_simsamu.py                 # Simsamu, découpé par séance
uv run python scripts/run_paquets.py --fusion 10          # système par paquets -> outputs/
uv run python -m whospoke.evaluate outputs/paquets_60s_0.5_fusion10 --approche pyannote-paquets --jeu voxconverse-dev-10
uv run pytest                                             # tests
uv run python app/demo.py                                 # démo Gradio : http://127.0.0.1:7860
```

## Structure
- `src/whospoke/` : code partagé (lecture RTTM, baseline, paquets, évaluation DER/JER).
- `scripts/` : téléchargement des données, lancement des runs, figures.
- `app/demo.py` : démo (fichier ou micro, timeline, écoute des segments, export RTTM/JSON).
- `kaggle/` : notebook de fine-tuning. `results/` : runs, figures, résumés des entraînements.
- `data/`, `outputs/`, `models/` : non versionnés (audio, prédictions, poids).
