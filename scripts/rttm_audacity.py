"""Convertit un RTTM en étiquettes Audacity, et inversement : sert à annoter nos propres enregistrements.

Méthode : on part de la sortie du système (pré-annotation), on ouvre l'audio et les étiquettes dans
Audacity (Fichier > Importer > Étiquettes), on corrige à l'oreille en voyant la forme d'onde (bornes,
nom du locuteur, segments manquants ou en trop), on exporte (Fichier > Exporter > Exporter les étiquettes),
puis on reconvertit en RTTM de référence.

Usage :
    uv run python scripts/rttm_audacity.py vers-audacity outputs/own_simsamu/reunion_mathis.rttm data/own/annot/reunion_mathis.txt
    uv run python scripts/rttm_audacity.py vers-rttm data/own/annot/reunion_mathis.txt data/own/rttm/reunion_mathis.rttm
"""
import argparse
from pathlib import Path


def rttm_vers_audacity(rttm, etiquettes):
    """Une ligne RTTM (début + durée) devient une étiquette Audacity : début<TAB>fin<TAB>locuteur."""
    lignes = []
    for ligne in Path(rttm).read_text(encoding="utf-8").splitlines():
        champs = ligne.split()
        if not champs or champs[0] != "SPEAKER":
            continue
        debut, duree, locuteur = float(champs[3]), float(champs[4]), champs[7]
        lignes.append(f"{debut:.3f}\t{debut + duree:.3f}\t{locuteur}")
    Path(etiquettes).parent.mkdir(parents=True, exist_ok=True)
    Path(etiquettes).write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return len(lignes)


def audacity_vers_rttm(etiquettes, rttm):
    """Inverse : chaque étiquette (début, fin, locuteur) redevient une ligne RTTM. Le nom du fichier sert d'identifiant."""
    uri = Path(rttm).stem
    lignes = []
    for ligne in Path(etiquettes).read_text(encoding="utf-8-sig").splitlines():
        champs = ligne.split("\t")
        # Audacity peut ajouter des lignes « \ » (étiquettes de fréquence) : on garde seulement les vraies étiquettes
        if len(champs) < 3 or champs[0].startswith("\\"):
            continue
        debut, fin, locuteur = float(champs[0]), float(champs[1]), champs[2].strip().replace(" ", "_")
        if fin <= debut or not locuteur:
            raise SystemExit(f"Étiquette invalide (fin avant début ou sans nom) : {ligne!r}")
        lignes.append(f"SPEAKER {uri} 1 {debut:.3f} {fin - debut:.3f} <NA> <NA> {locuteur} <NA> <NA>")
    Path(rttm).parent.mkdir(parents=True, exist_ok=True)
    Path(rttm).write_text("\n".join(lignes) + "\n", encoding="utf-8")
    return len(lignes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RTTM <-> étiquettes Audacity")
    parser.add_argument("sens", choices=["vers-audacity", "vers-rttm"])
    parser.add_argument("entree")
    parser.add_argument("sortie")
    args = parser.parse_args()
    if args.sens == "vers-audacity":
        n = rttm_vers_audacity(args.entree, args.sortie)
    else:
        n = audacity_vers_rttm(args.entree, args.sortie)
    print(f"{n} segments écrits dans {args.sortie}")
