"""Compare des RTTM prédits aux RTTM de référence et ajoute une ligne à results/runs.csv.

Usage :
    uv run python -m whospoke.evaluate outputs/pyannote_dev --approche pyannote-community-1 --jeu voxconverse-dev
"""
import argparse
import csv
import os
import warnings
from datetime import date
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from whospoke.io_rttm import lire_rttm

# VoxConverse ne fournit pas d'UEM (zone à évaluer) : pyannote prend alors toute l'étendue
# de la référence et de la prédiction, ce qui nous convient. On masque l'avertissement répété.
warnings.filterwarnings("ignore", message=".*approximated.*")

COLONNES = [
    "date", "approche", "parametres", "jeu", "nb_fichiers",
    "der_strict", "miss_strict", "fa_strict", "conf_strict",
    "der_standard", "miss_standard", "fa_standard", "conf_standard",
    "jer", "temps_s",
]


def pourcent(x):
    return round(100 * x, 2)


def composantes(metrique):
    """DER global et ses 3 composantes (en %), cumulés sur tous les fichiers."""
    total = metrique["total"]  # durée totale de parole dans la référence
    return {
        "der": pourcent(abs(metrique)),
        "miss": pourcent(metrique["missed detection"] / total),
        "fa": pourcent(metrique["false alarm"] / total),
        "conf": pourcent(metrique["confusion"] / total),
    }


def evaluer(dossier_pred, dossier_ref):
    """Évalue chaque RTTM de dossier_pred contre le RTTM du même nom dans dossier_ref."""
    # Dans pyannote, collar = durée TOTALE ignorée autour de chaque frontière (moitié avant,
    # moitié après). Le DER « standard » à 0,25 s de chaque côté s'écrit donc collar=0.5.
    der_strict = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    der_standard = DiarizationErrorRate(collar=0.5, skip_overlap=False)
    jer = JaccardErrorRate(collar=0.0, skip_overlap=False)

    fichiers = sorted(Path(dossier_pred).glob("*.rttm"))
    if not fichiers:
        raise SystemExit(f"Aucun RTTM dans {dossier_pred}")

    print(f"{'fichier':<10} {'DER strict':>10} {'DER std':>8} {'JER':>7}")
    for chemin_pred in fichiers:
        reference = lire_rttm(Path(dossier_ref) / chemin_pred.name)
        prediction = lire_rttm(chemin_pred)
        # Chaque appel calcule le score du fichier ET l'ajoute au cumul global
        d_strict = der_strict(reference, prediction)
        d_standard = der_standard(reference, prediction)
        j = jer(reference, prediction)
        print(f"{chemin_pred.stem:<10} {100 * d_strict:>9.1f}% {100 * d_standard:>7.1f}% {100 * j:>6.1f}%")

    # Global = cumul des durées d'erreur / cumul de la parole (pas la moyenne des DER par fichier)
    strict = composantes(der_strict)
    standard = composantes(der_standard)
    return {
        "nb_fichiers": len(fichiers),
        "der_strict": strict["der"], "miss_strict": strict["miss"],
        "fa_strict": strict["fa"], "conf_strict": strict["conf"],
        "der_standard": standard["der"], "miss_standard": standard["miss"],
        "fa_standard": standard["fa"], "conf_standard": standard["conf"],
        "jer": pourcent(abs(jer)),
    }


def ajouter_ligne(chemin_csv, ligne):
    """Ajoute une ligne au CSV des runs ; crée le dossier et l'en-tête s'ils n'existent pas."""
    chemin_csv = Path(chemin_csv)
    nouveau = not chemin_csv.exists()
    os.makedirs(chemin_csv.parent, exist_ok=True)
    with open(chemin_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES)
        if nouveau:
            writer.writeheader()
        writer.writerow(ligne)


def main():
    parser = argparse.ArgumentParser(description="Calcule DER et JER d'un dossier de RTTM prédits")
    parser.add_argument("predictions", help="dossier des RTTM prédits (ex. outputs/pyannote_dev)")
    parser.add_argument("--references", default="data/raw/dev", help="dossier des RTTM de référence")
    parser.add_argument("--approche", required=True, help="nom de l'approche (ex. pyannote-community-1)")
    parser.add_argument("--parametres", default="defaut", help="réglages utilisés, en texte libre")
    parser.add_argument("--jeu", required=True, help="jeu évalué (ex. voxconverse-dev)")
    parser.add_argument("--runs", default="results/runs.csv", help="fichier CSV où ajouter le run")
    args = parser.parse_args()

    scores = evaluer(args.predictions, args.references)

    # Temps d'inférence total, écrit par le script qui a produit les prédictions (s'il existe)
    fichier_temps = Path(args.predictions) / "temps.txt"
    temps = fichier_temps.read_text().strip() if fichier_temps.exists() else ""

    ligne = {
        "date": date.today().isoformat(),
        "approche": args.approche,
        "parametres": args.parametres,
        "jeu": args.jeu,
        **scores,
        "temps_s": temps,
    }
    ajouter_ligne(args.runs, ligne)

    print(f"\n{scores['nb_fichiers']} fichiers")
    print(f"DER strict   : {scores['der_strict']}% (raté {scores['miss_strict']}, "
          f"fausse alerte {scores['fa_strict']}, confusion {scores['conf_strict']})")
    print(f"DER standard : {scores['der_standard']}% (raté {scores['miss_standard']}, "
          f"fausse alerte {scores['fa_standard']}, confusion {scores['conf_standard']})")
    print(f"JER          : {scores['jer']}%")
    print(f"Ligne ajoutée à {args.runs}")


if __name__ == "__main__":
    main()
