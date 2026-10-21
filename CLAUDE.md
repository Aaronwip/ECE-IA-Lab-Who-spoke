# Who Spoke? — diarisation de locuteurs

Projet individuel Intelligence Lab (ECE Paris), mentor Mathis Escriva, du 16/10 au 16/11/2026.
Objectif : répondre à « qui parle, et quand ? » sur des conversations jamais vues.
Évaluation : démo sur des enregistrements inconnus préparés par l'organisation.

## Livrables
- Repo GitHub reproductible : code, dépendances, instructions, scripts d'évaluation.
- `docs/fiche_corpus.md` : source, accès, licence, volume, composition, limites, risques de fuite.
- Tableau comparatif des approches + ablation (un changement à la fois, mêmes fichiers de validation).
- Démo : charger un audio, timeline des locuteurs, écoute des segments, export RTTM/JSON.
- Métriques : DER avec ses 3 composantes (parole ratée, fausse alerte, confusion), JER, temps d'inférence, cas d'échec.
  Toujours donner un DER strict (collar 0, chevauchements inclus) et un DER standard (collar 0,25 s).

## Environnement
- Windows, PowerShell, VS Code. Python 3.11 géré par uv : `uv add <paquet>`, `uv run python ...`.
- ffmpeg installé. Conversion : `ffmpeg -i entree.mp4 -ac 1 -ar 16000 sortie.wav` (mono, 16 kHz).
- Audio chargé avec soundfile puis passé à pyannote sous la forme `{"waveform", "sample_rate"}` (torchcodec pose problème sous Windows).
- `HF_TOKEN` dans `.env`, modèle `pyannote/speaker-diarization-community-1`.

## Structure
- `data/` : samples (tests rapides), raw (corpus téléchargés), own (enregistrements consentis). Ignoré par Git.
- `outputs/` : RTTM prédits. Ignoré par Git.
- `src/whospoke/` (prévu) : io_rttm.py, baseline.py, embeddings.py, evaluate.py
- `scripts/` : téléchargement des données, lancement des runs.
- `results/runs.csv` : une ligne par run (date, approche, paramètres, jeu, DER, JER, temps). Versionné.
- `docs/` : fiche_corpus.md, journal.md. `app/` : démo Gradio.

## Données
- VoxConverse : régler sur `dev` (20-30 fichiers pour itérer), mesurer sur `test`.
- AMI : découpage officiel par réunion.
- Simsamu (FR) et enregistrements personnels consentis : test uniquement, jamais pour régler.

## Commandes
- Test pyannote : `uv run python test_pyannote.py data/samples/test1.wav` → `outputs/test1.rttm`

## Règles
- Ne jamais versionner `data/`, `outputs/`, `.env`, ni aucun audio.
- Ne jamais régler de paramètre sur le jeu de test.
- Une tâche = un commit. Chaque run d'évaluation ajoute une ligne à `results/runs.csv`.
- Pas de nouvelle dépendance sans expliquer pourquoi.
- L'utilisateur débute en IA et veut apprendre : expliquer chaque choix en français, garder un code simple
  qu'il peut défendre ligne par ligne en soutenance, le guider plutôt que tout faire à sa place.
