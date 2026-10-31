"""Vérifie qu'il n'y a aucune fuite entre partitions : ni enregistrement, ni locuteur partagé.

Pourquoi : si un même enregistrement (ou une même voix) est à la fois en train et en test,
le score de test est trop optimiste, car le système a « déjà vu » la réponse.

Vérifications :
1. AMI, listes officielles : aucune réunion et aucun locuteur dans deux partitions.
2. AMI, RTTM : les locuteurs annotés de chaque partition sont disjoints, et chaque RTTM
   ne parle que de sa propre réunion (découpe par enregistrement, jamais par segment).
3. AMI, fichiers sur disque : chaque .wav/.rttm rangé dans <split>/ appartient bien à la liste
   officielle de ce split.
4. Entre corpus : aucun identifiant d'enregistrement commun (AMI, VoxConverse dev, et
   VoxConverse test s'il est téléchargé).

Limite : les locuteurs VoxConverse s'appellent spk00, spk01... dans CHAQUE fichier ; ce ne sont
pas des identités globales, donc on ne peut pas vérifier les locuteurs entre fichiers VoxConverse.

Usage :
    uv run python scripts/verifier_splits.py      # code de sortie 1 si une fuite est trouvée
"""
import argparse
import sys
from itertools import combinations
from pathlib import Path

SPLITS = ["train", "dev", "test"]


def lire_liste(chemin):
    return set(Path(chemin).read_text(encoding="utf-8").split())


def lire_rttm_brut(chemin):
    """Renvoie (identifiants d'enregistrement, locuteurs) cités dans un RTTM."""
    enregistrements, locuteurs = set(), set()
    for ligne in Path(chemin).read_text(encoding="utf-8").splitlines():
        champs = ligne.split()
        if champs and champs[0] == "SPEAKER":
            enregistrements.add(champs[1])
            locuteurs.add(champs[7])
    return enregistrements, locuteurs


def verifier_ami(dossier):
    """Renvoie (liste d'erreurs, {split: réunions}) pour AMI."""
    erreurs = []
    dossier = Path(dossier)
    reunions = {s: lire_liste(dossier / "lists" / f"{s}.meetings.txt") for s in SPLITS}
    locuteurs_listes = {s: lire_liste(dossier / "lists" / f"{s}.speakers.txt") for s in SPLITS}

    # Locuteurs réellement présents dans les RTTM de chaque partition
    locuteurs_rttm = {s: set() for s in SPLITS}
    for split in SPLITS:
        for chemin in sorted((dossier / split / "rttm").glob("*.rttm")):
            enregistrements, locuteurs = lire_rttm_brut(chemin)
            locuteurs_rttm[split] |= locuteurs
            if enregistrements != {chemin.stem}:
                erreurs.append(f"AMI {split}/{chemin.name} contient d'autres enregistrements : {enregistrements}")
            if chemin.stem not in reunions[split]:
                erreurs.append(f"AMI {split}/rttm/{chemin.name} n'est pas dans la liste officielle {split}")
        for chemin in sorted((dossier / split / "wav").glob("*.wav")):
            if chemin.stem not in reunions[split]:
                erreurs.append(f"AMI {split}/wav/{chemin.name} n'est pas dans la liste officielle {split}")
        # Les locuteurs annotés doivent être ceux déclarés pour ce split
        hors_liste = locuteurs_rttm[split] - locuteurs_listes[split]
        if hors_liste:
            erreurs.append(f"AMI {split} : locuteurs des RTTM absents de {split}.speakers.txt : {sorted(hors_liste)}")

    for a, b in combinations(SPLITS, 2):
        for nom, ensembles in [("réunions", reunions), ("locuteurs (listes)", locuteurs_listes),
                               ("locuteurs (RTTM)", locuteurs_rttm)]:
            communs = ensembles[a] & ensembles[b]
            if communs:
                erreurs.append(f"AMI {a}/{b} : {len(communs)} {nom} en commun : {sorted(communs)}")
    return erreurs, reunions


def ids_voxconverse(dossier_rttm):
    return {f.stem for f in Path(dossier_rttm).glob("*.rttm")}


def main():
    parser = argparse.ArgumentParser(description="Vérifie l'absence de fuite entre partitions")
    parser.add_argument("--ami", default="data/raw/ami")
    parser.add_argument("--voxconverse-dev", default="data/raw/dev")
    parser.add_argument("--voxconverse-test", default="data/raw/test")
    args = parser.parse_args()

    erreurs, reunions = verifier_ami(args.ami)
    for split in SPLITS:
        print(f"AMI {split:<5} : {len(reunions[split])} réunions dans la liste officielle")

    # Entre corpus : on rassemble tous les identifiants et on cherche les doublons
    ensembles = {f"AMI {s}": reunions[s] for s in SPLITS}
    ensembles["VoxConverse dev"] = ids_voxconverse(args.voxconverse_dev)
    if Path(args.voxconverse_test).exists():
        ensembles["VoxConverse test"] = ids_voxconverse(args.voxconverse_test)
    print(f"VoxConverse dev : {len(ensembles['VoxConverse dev'])} enregistrements")
    for a, b in combinations(ensembles, 2):
        communs = ensembles[a] & ensembles[b]
        if communs:
            erreurs.append(f"{a} / {b} : {len(communs)} enregistrements en commun : {sorted(communs)}")

    if erreurs:
        print("\nFUITE DÉTECTÉE :")
        for e in erreurs:
            print(" -", e)
        sys.exit(1)
    print("\nOK : aucun enregistrement ni locuteur connu partagé entre partitions.")


if __name__ == "__main__":
    main()
