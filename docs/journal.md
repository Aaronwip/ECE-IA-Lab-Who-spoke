1. Composantes : par rapport à vad-energie (ratée 9,42 %, fausse alerte 2,61 %), lesquelles doivent rester identiques ? Pourquoi ? Si ta réponse est juste, ce sera aussi un test que le code n'a rien cassé.
-> Je ne sais pas 
Rep 1 : On peut dire que les composantes restent stables. 
2. Confusion : elle était de 37,46 %, et pyannote fait 1,21 %. Tu la vois où : moins de 10 %, entre 10 et 25 %, entre 25 et 37 %, ou au-dessus de 37 % ?
-> entre 25 et 37
Rep 1 : le pari etait bon mais tout dépendra de la suite pour savoir quoi cibler comme tuning (a mon avis)
3. Musique : dans afjiv, on impose 5 groupes pour 5 vrais locuteurs, mais tu as entendu de la musique que le détecteur laisse passer. Que peut-il arriver à cette musique, et à l'un des locuteurs ?
-> Elle peut etre interprétée comme un autre locuteur alors que c'est de la musique. Le systeme ne se trompera pas et interpretera le bon nombre de locuteur
Rep 1 : A confirmer par la suite 

Test du 10/11 : 
1. Quelle étape prend le plus de temps : segmentation, embeddings ou clustering ? Indice : relis le nombre de fois que chaque réseau tourne. --> la segmentation 
2. Quelle part du total prend cette étape : moins de 50 %, entre 50 et 80 %, ou plus de 80 % ? --> moins de 50%
3. Vitesse sur ce fichier : moins de 0,5 fois la durée de l'audio, entre 0,5 et 1 fois, ou plus d'une fois ? Indice : la moyenne sur tes 25 fichiers était d'environ 0,7 fois. --> entre 0,5 et 1 fois

Résultat test 1 : 
                     Pari                      │     Résultat      │ Verdict │
├────────────────────────────────────────────────┼───────────────────┼─────────┤
│ Q1 : l'étape la plus lente est la segmentation │ Embeddings, 451 s │ ❌      │
├────────────────────────────────────────────────┼───────────────────┼─────────┤
│ Q2 : cette étape prend moins de 50 %           │ 97 %              │ ❌      │
├────────────────────────────────────────────────┼───────────────────┼─────────┤
│ Q3 : entre 0,5 et 1 fois la durée de l'audio   │ 0,70 fois         │ ✅      │

Test 2 du 10/11 : 
1. DER standard : pyannote sur le fichier entier fait environ 3,6 %. Avec les paquets, tu le vois à moins de 5 %, entre 5 et 15 %, ou au-dessus de 15 % ? --> moins de 5 %
2. Quelle composante va le plus augmenter : parole ratée, fausse alerte ou confusion ? Indice : qu'est-ce qui peut mal se passer dans la mise en relation des locuteurs ? --> confusion
3. Seuil trop haut (0,7) : tu prévois trop de locuteurs ou pas assez ? Et quelle composante du DER en souffrira ? --> Pas assez

Run A avec pyannote :
Résultat marquant : Le MFCC avec k oracle fait 41,7 % en strict, pyannote 5,5 %, soit environ 7,5 fois moins d'erreur. Et pyannote doit en plus trouver seul le nombre de locuteurs.

Analyse : Les tests doivent desormais poursuivre sur pyannote. Le DER passe de 5,51 % à 2,53 % dès qu'on ignore 0,25 s de chaque côté de chaque changement de locuteur

│ Fichier │     Référence      │    pyannote    │ DER strict │  JER   │
├─────────┼────────────────────┼────────────────┼────────────┼────────┤
│ aufkn   │ 172 s / 15 s / 1 s │ 172 s / 9 s    │ 7,3 %      │ 49,6 % │
├─────────┼────────────────────┼────────────────┼────────────┼────────┤
│ azisu   │ 91 / 67 / 60 / 5 s │ 85 / 68 / 65 s │ 9,5 %      │ 32,5 % │
├─────────┼────────────────────┼────────────────┼────────────┼────────┤
│ aisvi   │ 8 locuteurs        │ 8 locuteurs ✅ │ 3,8 %      │ 13,0 % │

Run A (fichier entier) contre run B (paquets de 60 s)

┌──────────────────┬────────┬──────┬─────────────┬───────────┬────────┬──────────────────────────────────────────┐
│                  │  DER   │ Raté │   Fausse    │ Confusion │  JER   │                 Vitesse                  │
│                  │  std   │      │   alerte    │           │        │                                          │
├──────────────────┼────────┼──────┼─────────────┼───────────┼────────┼──────────────────────────────────────────┤
│ A (fichier       │ 2,53 % │ 1,19 │ 0,64        │ 0,69      │ 15,29  │ 0,70 fois la durée de l'audio,           │
│ entier)          │        │      │             │           │ %      │ impossible en direct                     │
├──────────────────┼────────┼──────┼─────────────┼───────────┼────────┼──────────────────────────────────────────┤
│ B (paquets de 60 │ 3,25 % │ 1,28 │ 0,63        │ 1,34      │ 16,84  │ 0,60 fois, direct possible               │
│  s)              │        │      │             │           │ %      │                                          │
└──────────────────┴────────┴──────┴─────────────┴───────────┴────────┴──────────────────────────────────────────┘

RUN C :  parmi 0,3, 0,4, 0,5, 0,6 et 0,7, lequel donnera la confusion la plus basse ? Inférieur à 0,5, 0,5 lui-même, ou supérieur ? Donne une raison en une ligne. -> 0,5 la plus basse.inferieur a 0.5 car plus precis semble correct au vu des derniers tests

Resultats du run C :
┌───────────┬─────────┬───────────┬────────────────────────────────────────────┐
│   Seuil   │ DER std │ Confusion │ Locuteurs dans aisvi (8 dans la référence) │
├───────────┼─────────┼───────────┼────────────────────────────────────────────┤
│ 0,3       │ 3,38 %  │ 1,47      │ 7 (deux voix fusionnées à tort)            │
├───────────┼─────────┼───────────┼────────────────────────────────────────────┤
│ 0,4 à 0,7 │ 3,25 %  │ 1,34      │ 8                                          │
└───────────┴─────────┴───────────┴────────────────────────────────────────────┘

Découverte du principe de : seuil bas donne deux locuteurs fusionnées
Choix pour le seuil de 0,5 

Run D : 
1. atgpi (1 seul vrai locuteur, 2 trouvés au run B) : combien de locuteurs trouvés maintenant ? --> 1
2. La confusion globale (1,34 au run B) : elle baisse beaucoup (vers 0,69, le niveau du fichier entier), un peu, ou pas du tout ? Indice : combien de secondes de parole sont en jeu dans ces petits bouts de fin, sur 39 min d'audio ? --> baisse beaucoup car les denrieres secondes d'un fichier restaient le porbleme majeur

Resultats et analyse : le run D nous odnne des résultats interessants car nous decidons de garder un seuil de 0,5, faire en sorte de fusionner les dernieres trames faisant <10s et de prendre des paquets de 60s. Nosu partons donc avec les réglages suivants pour le démo.

KAggle test 1 : 
Métrique suivie : DiarizationErrorRate (min) | avant : 0.2002
"mode": "test",
    "config": "sdm",
    "lr": 0.0001,
    "reunions_train": 8,
    "heures_train": 4.9,
    "reunions_val": 6,
    "heures_val": 3.5,
    "epoques": 1,
    "duree_min": 1.3,
    "metrique": "DiarizationErrorRate",
    "avant": 0.2002,
    "meilleur": 0.2052,
    "fichier": "meilleur-epoch=00.ckpt"

    Résultats du test de nuit 11/11 : 
    12 epochs, -0,0006 mais une courbe finalement plate oscillant d'epoch en epoch. L'explication la plus probable est celle évoquée hier : AMI faisait déjà partie des données d'entraînement du modèle de segmentation de pyannote. Le réentraîner sur AMI, c'est lui remontrer ce qu'il a déjà appris. Il était déjà à son meilleur niveau sur ce type de données.
    (0,1952 → 0,1946, dans le bruit)

    Run E : 

┌─────────────────────┬─────────────────┬───────────────────────┬──────────────┐
│                     │ Run D (origine) │ Run E (fine-tuné AMI) │    Écart     │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ DER standard        │ 2,99            │ 2,99                  │ 0            │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ – parole ratée      │ 1,26            │ 0,82                  │ −0,44        │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ – fausse alerte     │ 0,63            │ 1,03                  │ +0,40        │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ – confusion         │ 1,09            │ 1,14                  │ +0,05        │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ DER strict          │ 6,02            │ 6,21                  │ +0,19        │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ JER                 │ 16,75           │ 17,88                 │ +1,1         │
├─────────────────────┼─────────────────┼───────────────────────┼──────────────┤
│ Paquet le plus lent │ 0,63            │ 0,65                  │ même vitesse │
└─────────────────────┴─────────────────┴───────────────────────┴──────────────┘

Test de la nuit sur simsamu 12/11 : 
DER de validation en baisse de plus de 0,03. Meilleur perf dans la tranche entre 5e et la 15e 

Run F  (community-1 sur dj_2023) : 
Résultat (12/11) : DER strict 20,30 (raté 14,13 / FA 4,43 / conf. 1,74), DER standard 13,22 (raté 9,06 / FA 2,98 / conf. 1,17), JER 23,73.
→ Sur du français au téléphone, pyannote fait 4 fois plus d'erreurs que sur VoxConverse (2,99). L'erreur dominante est la parole ratée : c'est le travail de la segmentation.

Run G  (modèle Simsamu sur dj_2023) :
Paris : simulation sur simsamu pour validation. Peu d'amelioration mais DER en baisse 
Résultat (14/11) :

┌─────────────────────┬──────────────────┬────────────────────────┬────────┐
│ Simsamu dj_2023     │ Run F (origine)  │ Run G (fine-tuné SAMU) │ Écart  │
├─────────────────────┼──────────────────┼────────────────────────┼────────┤
│ DER standard        │ 13,22            │ 6,38                   │ −6,84  │
│ – parole ratée      │ 9,06             │ 3,32                   │ −5,74  │
│ – fausse alerte     │ 2,98             │ 0,43                   │ −2,55  │
│ – confusion         │ 1,17             │ 2,63                   │ +1,46  │
│ DER strict          │ 20,30            │ 11,62                  │ −8,68  │
│ JER                 │ 23,73            │ 15,82                  │ −7,91  │
│ Paquet le plus lent │ —                │ 0,62                   │        │
└─────────────────────┴──────────────────┴────────────────────────┴────────┘

→ Pari raté dans le bon sens : le DER est divisé par deux, pas « peu d'amélioration ».
  Le gain vient de la parole ratée et des fausses alertes (la segmentation), la confusion augmente un peu
  (surtout dj_2023_covid : 11,9 % de confusion en strict).
→ Limite : dj_2023 a servi à choisir l'époque du fine-tuning, ce score est donc un peu optimiste.
  La mesure honnête sera sur le test `reu`.

Run H (modèle Simsamu sur VoxConverse dev-10) :
DER strict < 11,62 % 
DER Standard < 6,38%
Meilleures performance overall 
Résultat (14/11) :

┌─────────────────────┬─────────────────┬────────────────────────┬────────┐
│ VoxConverse dev-10  │ Run D (origine) │ Run H (fine-tuné SAMU) │ Écart  │
├─────────────────────┼─────────────────┼────────────────────────┼────────┤
│ DER standard        │ 2,99            │ 4,69                   │ +1,70  │
│ – parole ratée      │ 1,26            │ 2,45                   │ +1,19  │
│ – fausse alerte     │ 0,63            │ 0,79                   │ +0,16  │
│ – confusion         │ 1,09            │ 1,45                   │ +0,36  │
│ DER strict          │ 6,02            │ 8,89                   │ +2,87  │
│ JER                 │ 16,75           │ 21,61                  │ +4,86  │
│ Paquet le plus lent │ 0,63            │ 0,65                   │ même   │
└─────────────────────┴─────────────────┴────────────────────────┴────────┘

→ Pari à moitié juste : les chiffres sont bien sous ceux de G, mais le modèle est MOINS bon que l'original
  sur l'anglais (+1,70 point). Ce n'est pas « meilleur overall ».
→ C'est de l'oubli (catastrophic forgetting) : en s'adaptant au français au téléphone, le modèle perd
  une partie de ce qu'il savait faire ailleurs. Fichiers les plus touchés : azisu (raté 12,9 %), ahnss, aufkn.
→ Décision à prendre : quel modèle pour la démo ? Elle se fera sur des enregistrements inconnus,
  ni téléphone 8 kHz, ni débat télé anglais. Il faut trancher sur une validation proche de la démo
  (enregistrements personnels en français au micro), jamais sur le test.
