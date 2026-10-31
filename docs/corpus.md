# Corpus

Fiche par corpus utilisé pour l'évaluation. Les **chiffres** du tableau ci-dessous sont calculés par
`uv run python scripts/stats_corpus.py`, qui réécrit uniquement la zone entre les marqueurs.
Les fiches en dessous sont écrites à la main. La fiche détaillée de VoxConverse (première version)
se trouve dans [`fiche_corpus.md`](fiche_corpus.md).

## Comment reproduire

```powershell
uv run python scripts/telecharger_voxconverse.py   # RTTM dev + audio dev (zip sauté si déjà là)
uv run python scripts/telecharger_ami.py --par-split 10
uv run python scripts/verifier_splits.py           # échoue (code 1) s'il y a une fuite
uv run python scripts/stats_corpus.py              # met à jour le tableau ci-dessous
```

## Statistiques (calculées)

- **Annoté complet** : toutes les réunions de la partition officielle, d'après les RTTM.
- **Audio local** : uniquement les réunions dont l'audio a été téléchargé (plafond disque de 5 Go).
- *Heures de parole* : temps où au moins une personne parle (les chevauchements ne sont comptés qu'une fois).
- *Chevauchement* : part de ce temps de parole où au moins deux personnes parlent en même temps.
- *Locuteurs distincts* : identités globales chez AMI (ex. `FEE005`). Chez VoxConverse, les labels
  (`spk00`...) sont propres à chaque fichier : on ne peut pas compter les personnes distinctes.

<!-- STATS:debut -->
(à remplir par scripts/stats_corpus.py)
<!-- STATS:fin -->

---

## AMI Meeting Corpus (micro Mix-Headset)

| | |
|---|---|
| **Source** | Carletta et al., *The AMI Meeting Corpus: a pre-announcement*, MLMI 2005. Université d'Édimbourg, IDIAP, TNO. https://groups.inf.ed.ac.uk/ami/corpus/ |
| **Partitions / annotations** | [BUTSpeechFIT/AMI-diarization-setup](https://github.com/BUTSpeechFIT/AMI-diarization-setup), commit `2509d89`. Partition officielle *Full-corpus-ASR* d'AMI, références `only_words` (dérivées des annotations manuelles v1.6.2). |
| **Accès** | Audio : miroir public d'Édimbourg, `amicorpus/<réunion>/audio/<réunion>.Mix-Headset.wav`, sans compte. |
| **Licence** | Corpus et annotations : CC BY 4.0. Dépôt BUT (listes + RTTM) : Apache 2.0. On cite, on ne redistribue pas l'audio. |
| **Langue** | Anglais, avec beaucoup de locuteurs non natifs. |
| **Contenu** | Réunions de 3 à 5 personnes (le plus souvent 4). Réunions *à scénario* (ES, IS, TS : équipe qui conçoit une télécommande) et réunions *réelles* (EN, IB, IN...). |
| **Micro** | *Mix-Headset* = somme des micros-casques individuels : son proche et propre. Plus facile qu'un micro de table unique. |
| **Volume complet** | ≈ 100 h au total. Partition BUT : 136 réunions train, 18 dev, 16 test (voir le tableau pour les heures). |
| **Sous-ensemble local** | Les **10 premières réunions par ordre alphabétique** de chaque partition (30 réunions). Les RTTM des 170 réunions sont gardés (quelques Mo), ce qui permet de vérifier les fuites sur la partition entière. |

**Limites**
- Le sous-ensemble n'est pas tiré au hasard. En train, les 10 premières réunions sont toutes de type `EN2001`–`EN2009` (réunions réelles, souvent longues). Elles ne représentent pas les réunions à scénario. En dev et en test, les 10 premières couvrent 2 à 3 groupes de 4 réunions (ex. `ES2011a-d`, `IB4001-...`).
- Dans une même partition, les **mêmes personnes** reviennent sur plusieurs réunions (une série `a, b, c, d` = un même groupe). C'est voulu, et ça ne pose pas de problème à l'intérieur d'une partition.
- Références `only_words` : les rires et les toux ne sont pas annotés comme de la parole. Un système qui les détecte sera compté en *fausse alerte*.
- Mix-Headset est une condition « micro proche » : les résultats ne disent rien d'un téléphone posé au milieu de la table (condition de la démo ?).

**Risques de fuite**
- La partition *Full-corpus-ASR* est conçue pour que train, dev et test n'aient **aucun locuteur en commun**. `scripts/verifier_splits.py` le vérifie sur les listes officielles et sur les RTTM.
- **pyannote a été entraîné sur AMI** (train de cette même partition) : sur AMI dev/test, pyannote est évalué dans des conditions qu'il connaît. Son score y est donc optimiste par rapport à des enregistrements inconnus.

---

## VoxConverse (dev)

| | |
|---|---|
| **Source** | Chung et al., *Spot the conversation: speaker diarisation in the wild*, Interspeech 2020. Oxford VGG. https://www.robots.ox.ac.uk/~vgg/data/voxconverse/ |
| **Annotations** | [joonson/voxconverse](https://github.com/joonson/voxconverse), dossier `dev/`, commit `24bf60b` (v0.3). |
| **Accès** | Audio : `voxconverse_dev_wav.zip` (lien du README), sans compte. Déjà en 16 kHz mono. |
| **Licence** | CC BY 4.0 ; les droits des vidéos restent à leurs propriétaires (YouTube). |
| **Langue** | Anglais (presque uniquement). |
| **Contenu** | Extraits YouTube : débats, plateaux télé, journaux, interviews. De 1 à plus de 20 locuteurs par fichier. |
| **Volume** | 216 fichiers, ≈ 20 h d'audio (voir le tableau). |
| **Usage dans le projet** | Réglage et itération. Pas de partie *train* officielle ; *test* (232 fichiers) uniquement pour la mesure finale, pas encore téléchargé. |

**Limites**
- Son de type « broadcast », bonne qualité, différent d'une réunion enregistrée au téléphone.
- Les labels de locuteurs sont **locaux à chaque fichier** : impossible de vérifier si une même personnalité apparaît dans dev et dans test.
- ⚠️ `dhorc.rttm` était **vide** dans la copie locale téléchargée le 25/10. La version officielle (commit `24bf60b`) contient 23 segments. `telecharger_voxconverse.py` réécrit désormais tous les RTTM depuis le commit figé. `fiche_corpus.md` (« dhorc est vide ») doit donc être corrigée.

**Risques de fuite**
- pyannote cite VoxConverse dans ses données d'entraînement (voir `fiche_corpus.md`) : son score sur dev est probablement optimiste.
- Règle : on règle sur dev, on mesure sur test, jamais l'inverse.
