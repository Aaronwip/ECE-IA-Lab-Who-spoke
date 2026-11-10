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