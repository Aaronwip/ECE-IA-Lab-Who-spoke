# Rapport de session autonome — 31/10/2026

Branche : `auto/harness-data` (8 commits, rien poussé, `main` intact).
Tes fichiers non suivis (`AGENTS.md`, `scripts/run_baseline.py`, `src/whospoke/baseline.py`) n'ont **pas été touchés ni commités**.
Toutes les décisions prises seul sont dans [`decisions.md`](decisions.md).

## En bref

| Étape | État |
|---|---|
| 1. `eval/score.py` + tests | ✅ Fini. 11 tests passent. Les scores sont identiques à ton `whospoke.evaluate` sur les 25 fichiers dev. |
| 2. Scripts de données | ✅ Écrits et testés. ⚠️ Téléchargement AMI **interrompu** : 3 réunions train sur 30. |
| 3. Vérification sans fuite | ✅ Script + 4 tests. ⚠️ **Pas encore exécutable sur le vrai AMI** : listes dev/test manquantes, le script répond `INCOMPLET`. |
| 4. `docs/corpus.md` | ✅ Fiches + tableau généré par script, sur les données présentes. |
| 5. pyannote 3.1 | ❌ Accès refusé (403, modèle « gated »). Repli sur `community-1`, lui aussi interrompu après 1 fichier sur 3. |

**Ce qui s'est passé :** vers 12 h 45, Claude Code a tué mes deux tâches de fond (téléchargement AMI et
run pyannote) parce que **la machine manquait de mémoire**. La consigne de l'outil est de ne pas les
relancer automatiquement. Je ne l'ai donc pas fait et j'ai fini le reste sur les données partielles.

## Ce qui marche

- `uv run pytest` → 11 tests passent :
  - `test_score.py` : RTTM identiques → DER = 0 ; décalage de 1 s → 15 % strict et 11,84 % avec collar (calculés à la main dans les commentaires) ; locuteur manquant → raté = 33,3 %, JER = 50 % ; mode dossier (MOYENNE ≠ GLOBAL) ; CSV ; référence vide ignorée.
  - `test_verifier_splits.py` : partitions propres, réunion partagée, locuteur partagé, fichier mal rangé.
- `eval/score.py`, en mode fichier et en mode dossier :
  ```powershell
  uv run python eval/score.py --ref data/raw/dev --hyp outputs/pyannote_dev --csv results/scores_dev25.csv
  ```
  Sur tes 25 RTTM du 25/10 : DER strict 6,87 %, standard 3,58 %, JER 19,09 %. C'est **exactement** la ligne de `results/runs.csv`, donc les deux implémentations se recoupent.
- `scripts/telecharger_voxconverse.py` : 216 RTTM réécrits depuis le commit figé, audio déjà présent (pas de nouveau téléchargement du zip).
- `scripts/telecharger_ami.py` : listes + RTTM de train (136 réunions) et 3 audios (`EN2001a/b/d`), vérifiés complets et en 16 kHz mono.
- `scripts/stats_corpus.py` : a rempli le tableau de `docs/corpus.md`.
- Aucun audio, ni `data/`, ni `.env` dans Git (vérifié avec `git ls-files`). `.gitignore` complété (`*.zip`, `*.part`, `*.ogg`, `*.opus`).

## Ce qui a échoué / reste à faire

1. **Téléchargement AMI incomplet.** À relancer quand la machine a de la mémoire libre (ferme le navigateur, etc.) :
   ```powershell
   uv run python scripts/telecharger_ami.py --par-split 10
   ```
   Le script reprend là où il s'est arrêté (les .wav déjà présents sont sautés). Il est lent (~0,3 Mo/s depuis Édimbourg), compte ~1 à 2 h.
   ⚠️ **Plafond disque :** les 10 premières réunions *train* sont toutes des `EN20xx` longues (1 à 1,5 h, 110-170 Mo chacune). Avec les 2,2 Go de VoxConverse, l'ensemble pourrait **dépasser 5 Go**. Le script s'arrête alors proprement avec un message « Plafond disque ». Si ça arrive, utilise `--par-split 5`.
2. **Vérification des fuites AMI pas encore faite** sur dev/test. Après le téléchargement :
   ```powershell
   uv run python scripts/verifier_splits.py   # 0 = OK, 1 = fuite, 2 = incomplet
   uv run python scripts/stats_corpus.py
   ```
   Si le script trouve des locuteurs en commun, **c'est une vraie information sur la partition officielle**, pas un bug à masquer. Dis-le dans la fiche corpus.
3. **pyannote 3.1 : accès refusé.** Va sur https://huggingface.co/pyannote/speaker-diarization-3.1 **et** https://huggingface.co/pyannote/segmentation-3.0, connecte-toi et accepte les conditions. Ensuite :
   ```powershell
   uv run python scripts/run_pyannote.py --modele pyannote/speaker-diarization-3.1 --limite 3 --sortie results/pyannote/3.1
   uv run python eval/score.py --ref data/raw/dev --hyp results/pyannote/3.1 --csv results/pyannote/3.1/scores.csv
   uv run python -m whospoke.evaluate results/pyannote/3.1 --approche pyannote-3.1 --jeu voxconverse-dev   # ligne dans runs.csv
   ```
   Je n'ai pas pu tester que 3.1 se charge avec pyannote.audio 4.x. J'ai ajouté `getattr(output, "speaker_diarization", output)` pour gérer les deux formats de sortie, mais c'est à vérifier.
4. Run `community-1` partiel : seul `abjxc` a été produit. Il est identique à ton RTTM du 24/10 (DER strict 0,60 %), donc il n'apporte rien de nouveau. Je n'ai **pas** ajouté de ligne dans `runs.csv`.

## À vérifier en priorité

1. **`data/raw/dev/dhorc.rttm` était vide** dans ta copie locale, alors que la version officielle (commit `24bf60b`) contient 23 segments. Le script l'a réécrit. Conséquences :
   - `docs/fiche_corpus.md` affirme « dhorc.rttm est vide » et « de 0 à 20 locuteurs » : **c'est faux**, corrige (1 à 20).
   - `dhorc` n'est pas dans tes 25 premiers fichiers : `runs.csv` n'est pas affecté.
   - Vérifie s'il y a d'autres RTTM corrompus ailleurs (`data/raw/` pour les prochains corpus).
2. **Le collar.** `--collar 0.25` veut dire ±0,25 s autour de chaque frontière (convention NIST). Pour pyannote, c'est `collar=0.5`, parce que pyannote attend la largeur totale. C'est le même choix que dans ton `evaluate.py`. `test_decalage_connu` le prouve par un calcul à la main : relis-le, c'est une bonne question de soutenance.
3. **Deux fiches corpus** coexistent maintenant : `fiche_corpus.md` (ta v1, détaillée sur VoxConverse) et `corpus.md` (demandée dans cette session). Choisis-en une et fusionne. Je n'ai pas modifié la tienne.
4. **Rangement des données** : VoxConverse est resté dans `data/raw/audio` + `data/raw/dev` pour ne pas casser tes scripts ; AMI est dans `data/raw/ami/<split>/`. À harmoniser si tu veux (`data/raw/voxconverse/dev/`).
5. **Scripts dans `scripts/`, pas dans `data/`**, parce que `data/` est ignoré par Git (voir `decisions.md`).
6. Biais du sous-ensemble AMI : « 10 premières par ordre alphabétique » donne un train composé uniquement de réunions réelles `EN`, sans réunion à scénario. Pour la démo, un tirage aléatoire avec graine serait plus représentatif.

## Fichiers ajoutés ou modifiés

- `eval/score.py`, `tests/test_score.py`, `tests/test_verifier_splits.py`
- `src/whospoke/donnees.py` (téléchargement, plafond disque, conversion 16 kHz mono, découpe RTTM)
- `scripts/telecharger_ami.py`, `scripts/telecharger_voxconverse.py`, `scripts/verifier_splits.py`, `scripts/stats_corpus.py`
- `scripts/run_pyannote.py` : option `--modele` (défaut inchangé)
- `results/pyannote/community-1/` (1 RTTM + scores.csv)
- `docs/corpus.md`, `docs/decisions.md`, `docs/SESSION_REPORT.md`
- `pyproject.toml` / `uv.lock` : `pytest` en dépendance dev + config pytest ; `.gitignore`

Petit détail : dans un terminal Git Bash, les accents s'affichent mal (`�`) parce que la sortie passe par un tube. Les fichiers eux-mêmes sont bien en UTF-8.
