"""Évaluation : DER (strict et standard) et JER d'un dossier de RTTM prédits.

Chaque RTTM prédit est comparé au RTTM de référence du même nom. Le script affiche le score
de chaque fichier, puis ajoute une ligne (score GLOBAL) à results/runs.csv.

Usage :
    uv run python -m whospoke.evaluate outputs/pyannote_dev --references data/raw/dev \
        --approche pyannote-community-1 --jeu voxconverse-dev
    # + les scores fichier par fichier dans un CSV (utilisé par scripts/figure_erreurs.py)
    ... --csv results/scores_mfcc_oracle.csv

Deux DER sont toujours calculés (cf. CLAUDE.md) :
- strict   : collar 0, chevauchements inclus ;
- standard : collar ±0,25 s autour de chaque frontière, chevauchements inclus.
"""
import argparse
import csv
import os
import sys
import warnings
from datetime import date
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from whospoke.io_rttm import lire_rttm

# Sans fichier UEM (zone à évaluer), pyannote prend l'étendue de la référence et de
# l'hypothèse réunies. C'est ce qu'on veut ; on masque l'avertissement répété à chaque fichier.
warnings.filterwarnings("ignore", message=".*approximated.*")

# Colonnes du CSV « par fichier » (option --csv)
COLONNES_FICHIER = [
    "fichier",
    "der_strict", "miss_strict", "fa_strict", "conf_strict",
    "der_standard", "miss_standard", "fa_standard", "conf_standard",
    "jer", "parole_ref_s",
]
# Colonnes de results/runs.csv : une ligne par run
COLONNES_RUNS = [
    "date", "approche", "parametres", "jeu", "nb_fichiers",
    "der_strict", "miss_strict", "fa_strict", "conf_strict",
    "der_standard", "miss_standard", "fa_standard", "conf_standard",
    "jer", "temps_s",
]


# ---------------- 1. Les métriques ----------------

def creer_metriques(collar=0.25):
    """Crée les 3 métriques. `collar` est la marge de CHAQUE côté d'une frontière (en s).

    Attention : pyannote.metrics attend la largeur TOTALE du collar (moitié avant la frontière,
    moitié après). Un collar de ±0,25 s (convention NIST/md-eval) s'écrit donc collar=0.5.
    """
    return {
        "strict": DiarizationErrorRate(collar=0.0, skip_overlap=False),
        "standard": DiarizationErrorRate(collar=2 * collar, skip_overlap=False),
        "jer": JaccardErrorRate(collar=0.0, skip_overlap=False),
    }


def detail(composants):
    """Transforme les durées renvoyées par pyannote (en s) en pourcentages de la parole de référence."""
    total = composants["total"]  # durée totale de parole dans la référence
    return {
        "der": 100 * (composants["missed detection"] + composants["false alarm"]
                      + composants["confusion"]) / total,
        "miss": 100 * composants["missed detection"] / total,
        "fa": 100 * composants["false alarm"] / total,
        "conf": 100 * composants["confusion"] / total,
    }


def ligne_scores(nom, strict, standard, jer, parole):
    """Range les scores d'un fichier (ou du total) sous les noms de colonnes du CSV."""
    return {
        "fichier": nom,
        "der_strict": strict["der"], "miss_strict": strict["miss"],
        "fa_strict": strict["fa"], "conf_strict": strict["conf"],
        "der_standard": standard["der"], "miss_standard": standard["miss"],
        "fa_standard": standard["fa"], "conf_standard": standard["conf"],
        "jer": jer,
        "parole_ref_s": parole,
    }


# ---------------- 2. Scorer fichier par fichier, puis le total ----------------

def scorer_paire(reference, hypothese, metriques):
    """Score une paire (référence, hypothèse) et l'ajoute au cumul des métriques.

    Renvoie une ligne de résultats (pourcentages) pour ce fichier.
    """
    # detailed=True renvoie les durées de chaque type d'erreur ; l'appel met aussi à jour le cumul
    strict = detail(metriques["strict"](reference, hypothese, detailed=True))
    standard = detail(metriques["standard"](reference, hypothese, detailed=True))
    jer = metriques["jer"](reference, hypothese)
    parole = reference.get_timeline().support().duration()
    return ligne_scores(reference.uri, strict, standard, 100 * jer, parole)


def ligne_globale(metriques, lignes):
    """Score global : erreurs cumulées sur tous les fichiers / parole cumulée.

    C'est le chiffre habituel des articles (et celui de results/runs.csv) : un long fichier pèse plus qu'un court.
    """
    strict = detail(metriques["strict"].accumulated_)
    standard = detail(metriques["standard"].accumulated_)
    # JER global de pyannote = moyenne sur tous les locuteurs de tous les fichiers
    jer = 100 * abs(metriques["jer"])
    return ligne_scores("GLOBAL", strict, standard, jer, sum(l["parole_ref_s"] for l in lignes))


def ligne_moyenne(lignes):
    """Moyenne simple des scores par fichier : chaque fichier pèse autant, quelle que soit sa durée."""
    moyenne = {"fichier": "MOYENNE"}
    for colonne in COLONNES_FICHIER[1:]:
        moyenne[colonne] = sum(l[colonne] for l in lignes) / len(lignes)
    return moyenne


def apparier(ref, hyp):
    """Renvoie la liste des paires (chemin_ref, chemin_hyp) à scorer.

    Fichier + fichier : une seule paire. Dossier + dossier : chaque RTTM de `hyp` est apparié
    au RTTM de même nom dans `ref`. Les références sans hypothèse ne sont pas scorées (on le signale).
    """
    ref, hyp = Path(ref), Path(hyp)
    if ref.is_file() and hyp.is_file():
        return [(ref, hyp)]
    if not (ref.is_dir() and hyp.is_dir()):
        raise SystemExit("références et prédictions doivent être deux fichiers ou deux dossiers")

    paires = []
    for chemin_hyp in sorted(hyp.glob("*.rttm")):
        chemin_ref = ref / chemin_hyp.name
        if not chemin_ref.exists():
            raise SystemExit(f"Pas de référence pour {chemin_hyp.name} dans {ref}")
        paires.append((chemin_ref, chemin_hyp))
    if not paires:
        raise SystemExit(f"Aucun RTTM dans {hyp}")

    sans_hyp = len(list(ref.glob("*.rttm"))) - len(paires)
    if sans_hyp:
        print(f"Info : {sans_hyp} référence(s) sans prédiction, non scorée(s).", file=sys.stderr)
    return paires


def scorer(paires, collar=0.25):
    """Score toutes les paires. Renvoie (lignes par fichier, ligne moyenne, ligne globale)."""
    metriques = creer_metriques(collar)
    lignes = []
    for chemin_ref, chemin_hyp in paires:
        reference = lire_rttm(chemin_ref)
        hypothese = lire_rttm(chemin_hyp)
        if not reference:
            # Ex. dhorc.rttm (VoxConverse) : aucune parole annotée, le DER divise par zéro.
            print(f"Ignoré : {chemin_ref.name} ne contient aucune parole de référence.", file=sys.stderr)
            continue
        hypothese.uri = reference.uri  # au cas où les noms internes diffèrent
        lignes.append(scorer_paire(reference, hypothese, metriques))
    if not lignes:
        raise SystemExit("Aucun fichier scorable.")
    return lignes, ligne_moyenne(lignes), ligne_globale(metriques, lignes)


# ---------------- 3. Affichage et sauvegarde ----------------

def afficher(lignes):
    print(f"{'fichier':<12} {'DER strict':>10} {'(raté/FA/conf)':>20} {'DER std':>8} {'JER':>7}")
    for l in lignes:
        composantes = f"{l['miss_strict']:.1f}/{l['fa_strict']:.1f}/{l['conf_strict']:.1f}"
        print(f"{l['fichier']:<12} {l['der_strict']:>9.2f}% {composantes:>20} "
              f"{l['der_standard']:>7.2f}% {l['jer']:>6.2f}%")


def ecrire_csv(chemin, lignes):
    """Écrit les scores par fichier (+ MOYENNE + GLOBAL) dans un CSV, arrondis au millième."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_FICHIER)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in ligne.items()})


def ajouter_ligne(chemin_csv, ligne):
    """Ajoute une ligne à results/runs.csv ; crée le dossier et l'en-tête s'ils n'existent pas."""
    chemin_csv = Path(chemin_csv)
    nouveau = not chemin_csv.exists()
    os.makedirs(chemin_csv.parent, exist_ok=True)
    with open(chemin_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES_RUNS)
        if nouveau:
            writer.writeheader()
        writer.writerow(ligne)


def ligne_run(globale, nb_fichiers, approche, parametres, jeu, temps):
    """Ligne de results/runs.csv : le score GLOBAL arrondi à 2 décimales, plus le contexte du run."""
    scores = {k: round(v, 2) for k, v in globale.items() if k in COLONNES_RUNS}
    return {"date": date.today().isoformat(), "approche": approche, "parametres": parametres,
            "jeu": jeu, "nb_fichiers": nb_fichiers, **scores, "temps_s": temps}


# ---------------- 4. Ligne de commande ----------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Calcule DER (strict et standard) et JER d'un dossier de RTTM prédits")
    parser.add_argument("predictions", help="dossier (ou fichier) des RTTM prédits, ex. outputs/pyannote_dev")
    parser.add_argument("--references", default="data/raw/dev", help="dossier (ou fichier) des RTTM de référence")
    parser.add_argument("--approche", required=True, help="nom de l'approche (ex. pyannote-community-1)")
    parser.add_argument("--parametres", default="defaut", help="réglages utilisés, en texte libre")
    parser.add_argument("--jeu", required=True, help="jeu évalué (ex. voxconverse-dev)")
    parser.add_argument("--runs", default="results/runs.csv", help="fichier CSV où ajouter le run")
    parser.add_argument("--csv", help="écrit aussi les scores par fichier + MOYENNE + GLOBAL dans ce CSV")
    args = parser.parse_args(argv)

    lignes, moyenne, globale = scorer(apparier(args.references, args.predictions))
    afficher(lignes + [moyenne, globale])
    if args.csv:
        ecrire_csv(args.csv, lignes + [moyenne, globale])
        print(f"Scores par fichier écrits dans {args.csv}")

    # Temps d'inférence total, écrit par le script qui a produit les prédictions (s'il existe)
    fichier_temps = Path(args.predictions) / "temps.txt"
    temps = fichier_temps.read_text().strip() if fichier_temps.exists() else ""

    ligne = ligne_run(globale, len(lignes), args.approche, args.parametres, args.jeu, temps)
    ajouter_ligne(args.runs, ligne)
    print(f"\n{ligne['nb_fichiers']} fichiers")
    print(f"DER strict   : {ligne['der_strict']}% (raté {ligne['miss_strict']}, "
          f"fausse alerte {ligne['fa_strict']}, confusion {ligne['conf_strict']})")
    print(f"DER standard : {ligne['der_standard']}% (raté {ligne['miss_standard']}, "
          f"fausse alerte {ligne['fa_standard']}, confusion {ligne['conf_standard']})")
    print(f"JER          : {ligne['jer']}%")
    print(f"Ligne ajoutée à {args.runs}")


if __name__ == "__main__":
    main()
