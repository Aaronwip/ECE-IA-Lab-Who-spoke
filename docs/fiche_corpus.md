# Fiche corpus

Version 2 (12/11/2026) : fusion de `fiche_corpus.md` (v1 du 25/10) et `corpus.md`, ajout de Simsamu,
du fine-tuning, des enregistrements personnels et d'une vue d'ensemble des risques de fuite.
Pour chaque corpus : source, accès, licence, volume, composition, limites, risques de fuite.

## Rôle des corpus dans le projet

| Corpus | Langue | Usage | Statut |
|---|---|---|---|
| VoxConverse dev | Anglais | Réglage et itération : les 10 premiers fichiers (« dev-10 ») pour l'ablation | Téléchargé (216 fichiers) |
| VoxConverse test | Anglais | Mesure finale uniquement, jamais pour régler | À télécharger |
| AMI (sdm) | Anglais | Fine-tuning de la segmentation (train, avec la validation officielle) | Utilisé sur Kaggle le 11/11 |
| Simsamu train (`dj_2022` + `dj_mai`) | Français | Fine-tuning de la segmentation | Utilisé sur Kaggle le 12/11 |
| Simsamu validation (`dj_2023`) | Français | Choix de l'époque, puis comparaison des modèles | Préparé en local |
| Simsamu test (`reu`) | Français | Mesure finale uniquement | Préparé en local, jamais regardé |
| Enregistrements personnels | Français | Réglage ou test selon l'enregistrement (voir plus bas) | 2 enregistrements |

## Statistiques (calculées)

Les chiffres du tableau sont calculés par `uv run python scripts/stats_corpus.py`, qui réécrit seulement
la zone entre les marqueurs. Tout le reste de la fiche est écrit à la main.

- **Annoté complet** : toutes les réunions de la partition officielle, d'après les RTTM.
- **Audio local** : seulement les réunions dont l'audio a été téléchargé (plafond disque de 5 Go).
- *Heures de parole* : temps où au moins une personne parle (chevauchements comptés une seule fois).
- *Chevauchement* : part du temps de parole où au moins deux personnes parlent en même temps.
- *Locuteurs distincts* : identités globales chez AMI (ex. `FEE005`). Chez VoxConverse, les labels sont propres
  à chaque fichier. Chez Simsamu, ce sont des rôles (`medecin`, `patient`) : on ne peut pas compter les personnes.

<!-- STATS:debut -->
| Jeu | Enregistrements annotés | Avec audio local | Heures d'audio local | Heures de parole annotée | Chevauchement (% de la parole) | Locuteurs par fichier (min / médiane / max) | Locuteurs distincts |
|---|---|---|---|---|---|---|---|
| AMI train (annoté complet) | 136 | 3 | 3.4 | 66.0 | 13.4 | 3 / 4 / 5 | 152 |
| AMI train (audio local) | 3 | 3 | 3.4 | 2.8 | 8.8 | 4 / 5 / 5 | 5 |
| VoxConverse dev | 216 | 216 | 20.3 | 18.9 | 3.7 | 1 / 4 / 20 | n/a (labels locaux) |
| Simsamu validation (dj_2023) | 7 | 7 | 0.4 | 0.3 | 0.0 | 2 / 2 / 2 | n/a (rôles) |
| Simsamu test (reu) | 9 | 9 | 0.5 | 0.4 | 0.0 | 2 / 2 / 2 | n/a (rôles) |
<!-- STATS:fin -->

### Reproduire

```powershell
uv run python scripts/telecharger_voxconverse.py   # RTTM dev (commit figé) + audio dev
uv run python scripts/telecharger_ami.py --par-split 10
uv run python scripts/verifier_splits.py           # échoue (code 1) s'il y a une fuite entre partitions AMI
uv run python scripts/preparer_simsamu.py          # validation dj_2023 + test reu, en 16 kHz
uv run python scripts/stats_corpus.py              # met à jour le tableau ci-dessus
```

---

## Risques de fuite : vue d'ensemble

Une fuite se produit quand le modèle a déjà vu, pendant son entraînement ou nos réglages, les données sur lesquelles on le mesure.
Le score paraît alors meilleur qu'il ne le sera sur les enregistrements inconnus de la soutenance.

**Ce que pyannote a vu.** La fiche de `pyannote/segmentation-3.0` indique qu'il a été entraîné sur
*« the training sets of AISHELL, AliMeeting, AMI, AVA-AVD, DIHARD, Ego4D, MSDWild, REPERE, and VoxConverse »*.
La fiche de `community-1`, notre pipeline, ne liste pas ses données. Elle publie ses scores sur ces mêmes corpus :
on suppose donc des données d'entraînement proches.

| Corpus | Vu par pyannote ? | Vu par notre fine-tuning ? | Conséquence |
|---|---|---|---|
| VoxConverse dev | **Probablement** : pas de partie *train* officielle, donc le *train* cité est sans doute dev | Non | Scores dev optimistes : 6,87 % (DER strict, nos 25 fichiers) contre 11,2 % publiés sur test. **La comparaison juste se fait sur test.** |
| VoxConverse test | Non (en principe) | Non | Mesure finale |
| AMI | **Oui**, la partition train | Oui (train) | Fine-tuning sans effet (0,1952 → 0,1946) : le modèle connaissait déjà ces données |
| Simsamu | **Non**, absent de la liste | Train et validation | Fine-tuning efficace (validation 0,1975 → 0,0976). Le test `reu` reste vierge |
| Enregistrements personnels | Non | Non | Les plus proches de la soutenance ; peu nombreux |

**Fuites que nous évitons nous-mêmes :**
- On règle sur VoxConverse dev et sur Simsamu `dj_2023`. On mesure sur VoxConverse test et sur Simsamu `reu`, **une seule fois**, avec la configuration finale.
- La validation Simsamu `dj_2023` a servi à choisir l'époque du fine-tuning : ses scores sont donc **légèrement optimistes** pour le modèle fine-tuné. Seul `reu` donne une mesure honnête.
- Un enregistrement personnel dont on a regardé les erreurs pour régler quelque chose devient un jeu de **réglage**. Il ne peut plus servir de test.

---

## VoxConverse

| | |
|---|---|
| **Source** | Chung, Huh, Nagrani, Afouras, Zisserman, *Spot the conversation: speaker diarisation in the wild*, Interspeech 2020 (arXiv:2007.01216). Oxford VGG. https://www.robots.ox.ac.uk/~vgg/data/voxconverse/ |
| **Annotations** | [joonson/voxconverse](https://github.com/joonson/voxconverse), dossier `dev/`, commit `24bf60b` (v0.3). |
| **Accès** | Audio dev : `voxconverse_dev_wav.zip`, lien donné dans le README du dépôt GitHub. Aucun compte. Rangement local : `data/raw/audio/*.wav` et `data/raw/dev/*.rttm`. |
| **Licence** | CC BY 4.0, pour la recherche. Les droits des vidéos restent à leurs propriétaires (YouTube). On cite, on ne redistribue pas l'audio. |
| **Format** | WAV mono 16 kHz, aucune conversion. |
| **Volume (dev)** | 216 fichiers, 20,3 h. De 22 s à 18 min 17 s par fichier, médiane 4 min 16 s. Test : 232 fichiers. |
| **Composition** | Extraits YouTube : débats politiques, plateaux télé, journaux, interviews. De 1 à 20 locuteurs par fichier, médiane 4. La parole couvre 92,8 % de la durée. |

**Limites**
- Presque uniquement en anglais.
- Son « broadcast » : bonne qualité, peu de bruit. Plus facile qu'un PC portable posé sur une table.
- Annotations imparfaites : la v0.3 a dû corriger des RTTM de test.
- Le sous-ensemble de réglage (dev-10, puis dev-25 : premiers fichiers par ordre alphabétique) n'est pas tiré au hasard, donc pas forcément représentatif.
- Correction de la v1 : `dhorc.rttm` n'est **pas** vide. La copie locale du 25/10 l'était, mais la version officielle contient 23 segments. `telecharger_voxconverse.py` réécrit désormais tous les RTTM depuis le commit figé.

**Risques de fuite** : voir la vue d'ensemble (pyannote a probablement vu dev). Les labels étant locaux à chaque fichier,
on ne peut pas vérifier si une même personnalité apparaît dans dev et dans test.

---

## AMI Meeting Corpus

| | |
|---|---|
| **Source** | Carletta et al., *The AMI Meeting Corpus: a pre-announcement*, MLMI 2005. Édimbourg, IDIAP, TNO. https://groups.inf.ed.ac.uk/ami/corpus/ |
| **Deux copies utilisées** | 1. **Évaluation locale (Mix-Headset)** : partitions et RTTM de [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup), commit `2509d89` (partition *Full-corpus-ASR*, références `only_words`). Audio du miroir public d'Édimbourg. 2. **Fine-tuning (Kaggle)** : [`diarizers-community/ami`](https://huggingface.co/datasets/diarizers-community/ami), configuration `sdm` (un seul micro distant), format parquet. |
| **Licence** | CC BY 4.0 (corpus, annotations et copie Hugging Face). Dépôt BUT : Apache 2.0. On ne redistribue pas l'audio. |
| **Langue** | Anglais, beaucoup de locuteurs non natifs. |
| **Composition** | Réunions de 3 à 5 personnes (le plus souvent 4). Réunions *à scénario* (ES, IS, TS : une équipe qui conçoit une télécommande) et réunions *réelles* (EN, IB, IN…). |
| **Volume** | ≈ 100 h. Partition BUT : 136 réunions train, 18 dev, 16 test. Fine-tuning `sdm` : 134 réunions train (79,7 h) et 18 de validation (9,7 h). |
| **Micros** | *Mix-Headset* / *ihm* = micros-casques additionnés : son proche et propre. *sdm* = un seul micro de table, la condition la plus proche d'un PC posé sur la table. |

**Limites**
- Sous-ensemble local non tiré au hasard : les 10 premières réunions train sont toutes de type `EN2001`–`EN2009` (réunions réelles). Le téléchargement a été interrompu le 31/10 : seules 3 réunions train ont leur audio en local.
- Références `only_words` : rires et toux ne sont pas annotés comme parole, et comptent en fausse alerte s'ils sont détectés.
- Les mêmes personnes reviennent dans une série `a, b, c, d` d'une même partition. C'est normal à l'intérieur d'une partition.

**Risques de fuite**
- La partition *Full-corpus-ASR* ne doit partager aucun locuteur entre train, dev et test. `scripts/verifier_splits.py` le vérifie, mais **sans conclusion pour l'instant** : les listes dev et test manquent en local (code 2, `INCOMPLET`).
- **pyannote a été entraîné sur AMI train.** C'est l'explication retenue de l'échec du fine-tuning AMI (run E : DER standard inchangé à 2,99 sur VoxConverse dev-10, parole ratée échangée contre des fausses alertes).

---

## Simsamu

| | |
|---|---|
| **Source** | Enregistrements d'internes en médecine d'urgence qui simulent des appels au SAMU (régulation médicale). Publié par medkit : [`medkit/simsamu`](https://huggingface.co/datasets/medkit/simsamu). Copie utilisée, préparée pour la diarisation : [`diarizers-community/simsamu`](https://huggingface.co/datasets/diarizers-community/simsamu). |
| **Accès** | Un fichier parquet de 172 Mo sur Hugging Face, sans compte. `scripts/preparer_simsamu.py` le télécharge et écrit `data/raw/simsamu/{validation,test}/{audio,rttm}`. |
| **Licence** | MIT. |
| **Langue** | **Français.** |
| **Composition** | 61 appels simulés, 3 h 15 au total, 3 min 11 en moyenne. Les internes jouent tour à tour l'appelant et le médecin régulateur. La situation (accident, douleur thoracique, brûlures…) et l'attitude de l'appelant (timide, en colère, paniqué, incompréhensible…) sont tirées au sort. **2 locuteurs** par appel (`medecin`, `patient`), sauf un appel à 3. |
| **Son** | **8 kHz**, qualité téléphone. La voix du médecin est captée directement par un micro. Celle de l'appelant passe par le réseau téléphonique et le haut-parleur d'un téléphone avant d'être captée : elle est plus dégradée. Nous rééchantillonnons en 16 kHz (`resample_poly`), à l'identique sur Kaggle et en local. |

**Découpage (décision du 12/11).** Simsamu n'a pas de découpage officiel : les 61 appels sont dans une seule partition « train ».
Nous découpons **par séance**, d'après le préfixe du nom de fichier. Les mêmes internes jouent plusieurs scènes pendant une séance :
découper au hasard mettrait les mêmes voix dans l'entraînement et dans le test.

| Ensemble | Séances | Appels | Durée | Usage |
|---|---|---|---|---|
| train | `dj_2022` + `dj_mai` | 45 | 2,3 h | Fine-tuning |
| validation | `dj_2023` | 7 | 25 min | Choix de l'époque, comparaison des modèles |
| test | `reu` | 9 | 32 min | Mesure finale uniquement |

**Limites**
- Le 8 kHz téléphonique diffère beaucoup d'un micro de PC portable. Un gain sur Simsamu ne garantit pas un gain en soutenance.
- Toujours le même scénario : un appel au SAMU à 2 personnes, avec des tours de parole très réguliers. Pas de réunion à 4 ou plus.
- **Aucun chevauchement annoté** (0,0 % d'après le tableau) : les deux voix ne se recouvrent jamais dans les références. Le modèle fine-tuné apprend donc à ne jamais prédire deux voix en même temps, ce qui peut lui coûter de la parole ratée sur VoxConverse ou en réunion.
- Petit volume (2,3 h d'entraînement) : le modèle surapprend au-delà d'une quinzaine d'époques (courbe du 12/11).
- Les labels sont des rôles : impossible de compter les personnes distinctes.

**Risques de fuite**
- Les séances séparent les groupes d'internes, sans le garantir : rien ne prouve qu'un même interne n'a pas participé à deux séances. `reu` (probablement La Réunion) est la séance la plus susceptible d'impliquer d'autres personnes : c'est pour ça qu'elle sert de test.
- Absent de l'entraînement de pyannote (voir la vue d'ensemble).

---

## Enregistrements personnels

| | |
|---|---|
| **Protocole** | Conversations réelles en français, enregistrées avec un téléphone (WhatsApp) ou le micro d'un PC portable, puis converties en WAV mono 16 kHz par la démo (ffmpeg). Rangés dans `data/own/`, jamais versionnés. |
| **Consentement** | Accord oral de chaque personne enregistrée pour un usage limité au projet. **À confirmer par écrit** (un message suffit) avant la soutenance, surtout pour les enregistrements où apparaissent d'autres personnes. |
| **Annotation** | Une référence RTTM est corrigée à la main en écoutant, à partir de la sortie de la démo. Sans référence corrigée, un enregistrement ne sert qu'à observer, pas à mesurer. |

| Enregistrement | Durée | Personnes | Référence | Usage |
|---|---|---|---|---|
| `WhatsApp Audio 2026-11-11 at 23.16.58` | 2 min 21 | à préciser (3 trouvées par la démo) | à annoter | test |
| `REU-MATHIS-AARON-` (réunion avec le mentor) | 7 min | 2 | à annoter | **réglage** : ses erreurs ont déjà été analysées (4 faux locuteurs de 0,1 à 1,4 s). Servira aussi à enregistrer la voix du mentor pour la démo |

**Limites** : très peu d'enregistrements et peu de personnes. Ils servent à vérifier la tendance en conditions réelles,
pas à produire un score précis. Il en faudrait 2 ou 3 de plus pour le test, avec des personnes ou des séances différentes.

**Risques de fuite** : un enregistrement de réglage ne doit jamais servir de test. Les deux ensembles doivent venir de séances différentes.
