# Journal des décisions

Choix pris pendant la session autonome du 31/10/2026 (branche `auto/harness-data`), quand la consigne était ambiguë.
Format : date — choix — alternative écartée — raison.

| Date | Choix | Alternative | Raison |
|---|---|---|---|
| 2026-10-31 | `--collar 0.25` = marge de **chaque côté** de la frontière ; converti en `collar=0.5` pour pyannote.metrics | Passer 0.25 tel quel à pyannote | pyannote attend la largeur totale (vérifié dans `pyannote/metrics/utils.py` : `t ± 0.5*collar`). La convention NIST/md-eval « collar 0,25 s » veut dire ±0,25 s. Le test `test_decalage_connu` le vérifie. |
| 2026-10-31 | `score.py` donne **toujours** le DER strict (collar 0) en plus du DER standard | Uniquement collar 0,25 s comme demandé | CLAUDE.md exige les deux. |
| 2026-10-31 | Mode dossier : on score chaque RTTM de `--hyp` apparié à la référence de même nom ; références sans hypothèse ignorées (signalé sur stderr) | Compter une hypothèse absente comme 100 % ratée | Permet de scorer un sous-ensemble (3 fichiers) sans copier les références. Même logique que `whospoke/evaluate.py`. |
| 2026-10-31 | CSV : une ligne par fichier + `MOYENNE` (moyenne simple) + `GLOBAL` (erreurs cumulées / parole cumulée) | Seulement la moyenne demandée | Le GLOBAL est le chiffre des articles et de `results/runs.csv` ; la moyenne seule surpondère les fichiers courts. |
| 2026-10-31 | Pas de fichier UEM : pyannote prend l'étendue réf ∪ hyp | Lire les UEM d'AMI | Les UEM AMI couvrent tout l'enregistrement : différence limitée aux silences en bord de fichier. Code plus simple. |
| 2026-10-31 | Référence vide (ex. `dhorc`) : fichier ignoré avec un message | Planter / DER infini | Le DER n'est pas défini (division par zéro). |
| 2026-10-31 | `score.py` réutilise `whospoke.io_rttm.lire_rttm` | Réécrire un lecteur RTTM | Une seule façon de lire un RTTM dans le projet. |
| 2026-10-31 | Ajout de `pytest` en dépendance **dev** (`uv add --dev pytest`) | unittest de la bibliothèque standard | Demandé explicitement (« pytest passe ») ; groupe dev = pas installé pour la démo. |
