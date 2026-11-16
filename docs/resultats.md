# Résultats : tableau comparatif et ablation

Toutes les lignes viennent de [`results/runs.csv`](../results/runs.csv) (une ligne par run, calculée par
`whospoke.evaluate`). Le raisonnement et les paris faits avant chaque run sont dans [`journal.md`](journal.md).

**Lecture.** DER = part de la parole de référence mal attribuée (plus bas = mieux), décomposée en
**raté** (parole non détectée), **FA** (fausse alerte : parole détectée là où personne ne parle) et
**conf.** (confusion : la bonne parole attribuée au mauvais locuteur).
**Strict** : aucune tolérance, chevauchements inclus. **Standard** : ±0,25 s de tolérance autour de chaque
changement de locuteur. JER : erreur moyenne par locuteur (pénalise les locuteurs qui parlent peu).

## 1. Tableau comparatif — VoxConverse dev-10 (anglais, 10 fichiers, 39 min)

Toutes les approches sont mesurées sur les **mêmes 10 fichiers** (les 10 premiers de `dev` par ordre alphabétique).

| Approche | DER strict (raté / FA / conf.) | DER standard (raté / FA / conf.) | JER | Temps |
|---|---|---|---|---|
| Baseline 1 : détection d'énergie, 1 seul locuteur | 49,48 (9,42 / 2,61 / 37,46) | 46,92 (7,42 / 1,50 / 38,01) | 81,2 | 4 s |
| Baseline 2 : MFCC + k-means, nombre de locuteurs donné (oracle) | 41,74 (9,42 / 2,61 / 29,71) | 38,45 (7,42 / 1,50 / 29,54) | 56,8 | 2 s |
| pyannote community-1, fichier entier (run A) | 5,51 (2,24 / 2,05 / 1,21) | **2,53** (1,19 / 0,64 / 0,69) | 15,3 | — |
| pyannote par paquets de 60 s (run B) | 6,27 (2,33 / 2,02 / 1,92) | 3,25 (1,28 / 0,63 / 1,34) | 16,8 | 1 400 s |
| **+ fusion du dernier paquet < 10 s (run D, système de la démo)** | 6,02 (2,33 / 2,02 / 1,67) | **2,99** (1,26 / 0,63 / 1,09) | 16,8 | 1 369 s |
| D + segmentation fine-tunée AMI (run E) | 6,21 (1,79 / 2,70 / 1,73) | 2,99 (0,82 / 1,03 / 1,14) | 17,9 | 1 438 s |
| D + segmentation fine-tunée Simsamu (run H) | 8,89 (4,32 / 2,45 / 2,12) | 4,69 (2,45 / 0,79 / 1,45) | 21,6 | 1 443 s |

La baseline 2 triche en faveur d'elle-même (on lui donne le vrai nombre de locuteurs) et reste 13 fois moins
bonne que pyannote : l'essentiel de son erreur est de la **confusion**, les MFCC ne distinguent pas bien les voix.

## 2. Tableau comparatif — Simsamu (français, appels au SAMU simulés, téléphone 8 kHz)

| Jeu | Segmentation | DER strict (raté / FA / conf.) | DER standard (raté / FA / conf.) | JER |
|---|---|---|---|---|
| Validation `dj_2023` (7 appels, 25 min) | d'origine (run F) | 20,30 (14,13 / 4,43 / 1,74) | 13,22 (9,06 / 2,98 / 1,17) | 23,7 |
| | fine-tunée Simsamu (run G) | 11,62 (6,10 / 2,00 / 3,51) | **6,38** (3,32 / 0,43 / 2,63) | 15,8 |
| **Test `reu`** (9 appels, 32 min, jamais vus) | d'origine | 16,99 (7,49 / 7,28 / 2,22) | 12,10 (5,21 / 5,34 / 1,55) | 18,0 |
| | **fine-tunée Simsamu** | 11,88 (1,91 / 5,34 / 4,63) | **5,82** (0,78 / 1,70 / 3,34) | 15,1 |

Les quatre lignes utilisent le système complet de la démo (paquets de 60 s, fusion < 10 s, seuil 0,5).

### Enregistrement personnel : choix du modèle de la démo en direct

`reunion_mathis` : réunion de 7 min en français, 2 personnes, micro de PC (conditions proches de la soutenance).
Référence : pré-annotation du système corrigée à l'oreille dans Audacity (`scripts/rttm_audacity.py`).

| Segmentation | DER strict (raté / FA / conf.) | DER standard (raté / FA / conf.) | JER |
|---|---|---|---|
| **d'origine** | **11,26** (2,55 / 1,74 / 6,96) | **8,28** (0,52 / 1,56 / 6,19) | 17,2 |
| fine-tunée Simsamu | 16,18 (4,35 / 6,08 / 5,74) | 9,38 (1,07 / 3,28 / 5,03) | 20,5 |

→ **La démo en direct garde la segmentation d'origine.** Le modèle fine-tuné double les fausses alertes hors du
téléphone 8 kHz. La pré-annotation venait du modèle fine-tuné : le biais jouait en sa faveur, il perd quand même.
Limites : un seul fichier, ajouts manuels précis à ±0,5 s environ.

## 3. Analyse d'ablation (un seul changement à la fois)

| # | Changement | Jeu | DER standard avant → après | Ce qu'on en conclut |
|---|---|---|---|---|
| 1 | Fichier entier → **paquets de 60 s** (A → B) | Vox dev-10 | 2,53 → 3,25 | Le direct coûte 0,7 point, presque tout en confusion (+0,65) : relier les locuteurs d'un paquet à l'autre est le point faible. |
| 2 | **Seuil de liaison** des locuteurs 0,3 / 0,4 / 0,5 / 0,6 / 0,7 | Vox dev-10 | 3,38 / 3,25 / 3,25 / 3,25 / 3,25 | Paramètre non critique entre 0,4 et 0,7 : on garde 0,5. |
| 3 | **Fusion** du dernier paquet s'il dure < 10 s (B → D) | Vox dev-10 | 3,25 → 2,99 | Un paquet trop court donne une empreinte vocale peu fiable ; le coller au précédent récupère 0,26 point. |
| 4 | Segmentation **fine-tunée AMI** (D → E) | Vox dev-10 | 2,99 → 2,99 | Aucun gain : moins de parole ratée (−0,44) mais plus de fausses alertes (+0,40). AMI faisait déjà partie de l'entraînement de pyannote. |
| 5 | Segmentation **fine-tunée Simsamu** (F → G) | Simsamu val | 13,22 → 6,38 | DER divisé par 2 en français au téléphone : parole ratée 9,06 → 3,32, fausses alertes 2,98 → 0,43. |
| 5 bis | idem, sur le **test** `reu` | Simsamu test | 12,10 → 5,82 | Le gain se confirme sur des données jamais vues : la validation était représentative. |
| 5 ter | idem, sur l'**anglais** (D → H) | Vox dev-10 | 2,99 → 4,69 | **Oubli catastrophique** : en se spécialisant, le modèle perd en anglais (surtout de la parole ratée). |
| 5 quater | idem, sur une **réunion au micro de PC** (français) | `reunion_mathis` | 8,28 → 9,38 | Hors du téléphone, le fine-tuné perd (fausses alertes 1,56 → 3,28) : le gain est propre au domaine. |
| 6 | **Taux d'apprentissage** du fine-tuning Simsamu | Simsamu val (DER par fenêtre de 10 s) | voir ci-dessous | 1e-5 : sous-apprentissage ; 1e-4 : bon réglage. |

**Taux d'apprentissage (ligne 6).** Même code, mêmes données, seul le taux change. DER de validation
calculé par fenêtres de 10 s pendant l'entraînement (avant fine-tuning : 0,1975).

| Taux | Meilleur DER val | Meilleure époque | Remarque |
|---|---|---|---|
| 1e-5 | 0,148 | 29 (dernière) | encore en baisse après 30 époques : **sous-apprentissage** |
| 1e-4 (run du 12/11) | 0,0976 | 16 | arrêt anticipé à 22 époques |
| 1e-4 (relance du 16/11) | 0,1009 | 27 | même réglage : **le hasard seul fait varier le score d'environ 0,003** |
| 1e-3 | *en cours* | | |

Le modèle à 1e-4 atteint un plateau vers l'époque 15 ; la petite remontée vue le 12/11 est de la taille du bruit
mesuré par la relance : **pas de surapprentissage net**. Courbes : `results/finetune/*/par_epoque.csv`,
`results/finetune/2026-11-12_nuit_simsamu/logs/version_0/metrics.csv`.

## 4. Temps d'inférence (CPU de PC portable, sans GPU)

| Jeu | Durée audio | Temps total | Paquet le plus lent / sa durée |
|---|---|---|---|
| Vox dev-10, run D | 39 min | 1 369 s (≈ 0,6 × la durée) | 0,63 |
| Simsamu `reu`, origine / fine-tunée | 32 min | 956 s / 830 s | 0,57 / 0,57 |

Chaque paquet de 60 s est traité en moins de 40 s : **le direct est tenu** sur un portable.
97 % du temps de pyannote est passé dans le calcul des empreintes vocales (`scripts/profil_pyannote.py`).

## 5. Cas d'échec

| Fichier | Système | Ce qui se passe |
|---|---|---|
| `aufkn`, `azisu` (Vox) | pyannote | locuteurs qui parlent très peu : non détectés ou fusionnés avec un autre (JER jusqu'à 68 %) |
| `afjiv` (Vox) | pyannote | musique détectée comme de la parole (fausses alertes) |
| `reu_brulures` (Simsamu) | fine-tunée | 3 locuteurs trouvés pour 2 : une voix coupée en deux (confusion 16,9 % en strict) |
| `reu_grand_mere` (Simsamu) | les deux | confusion 9 à 10 % ; l'original invente un 3e locuteur |
| `dj_2023_covid` (Simsamu) | fine-tunée | confusion 11,9 % en strict |

Point commun : le fine-tuning améliore **quand** on parle (segmentation), pas **qui** parle (empreintes vocales
et liaison entre paquets, non entraînées). C'est la suite logique du projet.

## 6. Limites

- VoxConverse : 10 fichiers de `dev` seulement ; le jeu `test` (plusieurs dizaines d'heures) n'a pas été mesuré, trop long sur CPU.
- Simsamu : téléphone 8 kHz, scénarios joués ; le gain ne garantit pas un gain sur un micro de PC.
- Le DER « par fenêtre » de l'entraînement est optimiste (la validation sert aussi à choisir l'époque) ;
  les chiffres de référence sont ceux des tableaux 1 et 2, calculés sur les fichiers entiers.
- Risques de fuite (données vues par pyannote pendant son entraînement) : voir [`fiche_corpus.md`](fiche_corpus.md).
