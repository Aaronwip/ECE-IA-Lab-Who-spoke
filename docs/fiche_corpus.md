# Fiche corpus

Version 1 (25/10/2026) : VoxConverse dev. AMI, Simsamu et enregistrements personnels à compléter.

## Rôle des corpus dans le projet

| Corpus | Langue | Usage | Statut |
|---|---|---|---|
| VoxConverse dev | Anglais | Réglage et itération (sous-ensemble fixe de 25 fichiers) | Téléchargé |
| VoxConverse test | Anglais | Mesure finale uniquement, jamais pour régler | À télécharger |
| AMI | Anglais | Second corpus, découpage officiel par réunion | À télécharger |
| Simsamu | Français | Test uniquement | À télécharger |
| Enregistrements personnels | Français | Test uniquement, avec consentement écrit | À enregistrer |

---

## VoxConverse

### Source
- Chung, Huh, Nagrani, Afouras, Zisserman, *Spot the conversation: speaker diarisation in the wild*, Interspeech 2020 (arXiv:2007.01216). Université d'Oxford (VGG).
- Page officielle : https://www.robots.ox.ac.uk/~vgg/data/voxconverse/
- Annotations (RTTM) : https://github.com/joonson/voxconverse

### Accès
- Audio dev : `voxconverse_dev_wav.zip`, lien donné dans le README du dépôt GitHub (pas sur la page officielle).
- RTTM : dossier `dev/` du dépôt GitHub, version v0.3 (branche principale).
- Rangement local : `data/raw/audio/*.wav` et `data/raw/dev/*.rttm`, un RTTM par audio, même nom de fichier.
- Aucun compte ni inscription nécessaire.

### Licence
- Creative Commons Attribution 4.0 (CC BY 4.0), pour la recherche.
- Les droits d'auteur des vidéos restent à leurs propriétaires d'origine (YouTube).
- Conséquence : on peut utiliser et citer le corpus, mais on ne redistribue pas l'audio (il reste hors du dépôt Git).

### Volume (dev, mesuré sur nos fichiers)
- 216 fichiers, 20,3 h d'audio.
- Durée par fichier : de 22 s à 18 min 17 s, médiane 4 min 16 s.
- Format : WAV mono 16 kHz, aucune conversion nécessaire.

### Composition (dev)
- Extraits de vidéos YouTube : débats politiques, plateaux télé, journaux, interviews.
- Locuteurs par fichier : de 0 à 20, médiane 4. 22 fichiers n'ont qu'un locuteur, 16 en ont 10 ou plus.
- La parole couvre 92,8 % de la durée audio.
- Chevauchement : 3,7 % de la durée de parole a au moins deux locuteurs simultanés.

### Limites
- Langue presque uniquement anglaise : le corpus ne dit rien des performances en français (d'où Simsamu et les enregistrements personnels).
- Conditions « broadcast » : bonne qualité sonore, peu de bruit de fond. C'est plus facile qu'un enregistrement au téléphone ou dans une salle.
- Les annotations ne sont pas parfaites : la v0.3 a dû corriger des erreurs dans des RTTM de test.
- `dhorc.rttm` est vide (aucun locuteur annoté). Son DER par fichier n'est pas défini (division par zéro) : à exclure ou à traiter à part.
- Le sous-ensemble de réglage (25 premiers fichiers par ordre alphabétique) n'est pas tiré au hasard : il n'est pas forcément représentatif des 216 fichiers.

### Risques de fuite
- **pyannote a vu VoxConverse à l'entraînement.** La fiche de `pyannote/segmentation-3.0` cite « the training sets of [...] VoxConverse ». VoxConverse n'a pas de partie *train* officielle, donc la partie utilisée est très probablement dev. Le DER de pyannote sur dev est donc sans doute optimiste : 6,87 % mesurés sur nos 25 fichiers, contre 11,2 % publiés par pyannote sur VoxConverse v0.3 (collar 0, chevauchements inclus). La comparaison juste se fera sur test.
- **Réglage sur dev, mesure sur test.** Aucun paramètre n'est choisi en regardant test.
- **Mêmes personnes dans dev et test** : des personnalités publiques peuvent apparaître dans les deux. Non vérifié ; c'est un risque possible pour une méthode à base d'empreintes vocales.

---

## AMI

*À compléter : source, accès, licence, volume, composition, limites, risques de fuite.*

## Simsamu

*À compléter.*

## Enregistrements personnels

*À compléter : protocole d'enregistrement, consentement, nombre de personnes, durée, annotation manuelle.*
