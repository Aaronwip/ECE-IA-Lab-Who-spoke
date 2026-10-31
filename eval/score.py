"""Score un RTTM prédit (hypothèse) contre un RTTM de référence : DER et JER.

Usage :
    # un seul fichier
    uv run python eval/score.py --ref data/voxconverse/dev/rttm/abjxc.rttm --hyp results/pyannote/abjxc.rttm
    # mode dossier : un RTTM prédit par fichier, apparié au RTTM de référence de même nom
    uv run python eval/score.py --ref data/voxconverse/dev/rttm --hyp results/pyannote --csv results/pyannote/scores.csv

Deux DER sont toujours calculés (cf. CLAUDE.md) :
- strict   : collar 0, chevauchements inclus ;
- standard : collar ±0,25 s autour de chaque frontière, chevauchements inclus.
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate

from whospoke.io_rttm import lire_rttm

# Sans fichier UEM (zone à évaluer), pyannote prend l'étendue de la référence et de
# l'hypothèse réunies. C'est ce qu'on veut ; on masque l'avertissement répété à chaque fichier.
warnings.filterwarnings("ignore", message=".*approximated.*")

COLONNES = [
    "fichier",
    "der_strict", "miss_strict", "fa_strict", "conf_strict",
    "der_standard", "miss_standard", "fa_standard", "conf_standard",
    "jer", "parole_ref_s",
]


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
    total = composants["total"]
    return {
        "der": 100 * (composants["missed detection"] + composants["false alarm"]
                      + composants["confusion"]) / total,
        "miss": 100 * composants["missed detection"] / total,
        "fa": 100 * composants["false alarm"] / total,
        "conf": 100 * composants["confusion"] / total,
    }


def scorer_paire(reference, hypothese, metriques):
    """Score une paire (référence, hypothèse) et l'ajoute au cumul des métriques.

    Renvoie une ligne de résultats (pourcentages) pour ce fichier.
    """
    # detailed=True renvoie les durées de chaque type d'erreur ; l'appel met aussi à jour le cumul
    strict = detail(metriques["strict"](reference, hypothese, detailed=True))
    standard = detail(metriques["standard"](reference, hypothese, detailed=True))
    jer = metriques["jer"](reference, hypothese)
    return {
        "fichier": reference.uri,
        "der_strict": strict["der"], "miss_strict": strict["miss"],
        "fa_strict": strict["fa"], "conf_strict": strict["conf"],
        "der_standard": standard["der"], "miss_standard": standard["miss"],
        "fa_standard": standard["fa"], "conf_standard": standard["conf"],
        "jer": 100 * jer,
        "parole_ref_s": reference.get_timeline().support().duration(),
    }


def ligne_globale(metriques, lignes):
    """Score global : erreurs cumulées sur tous les fichiers / parole cumulée.

    C'est le chiffre habituel des articles : un long fichier pèse plus qu'un court.
    """
    strict = detail(metriques["strict"].accumulated_)
    standard = detail(metriques["standard"].accumulated_)
    return {
        "fichier": "GLOBAL",
        "der_strict": strict["der"], "miss_strict": strict["miss"],
        "fa_strict": strict["fa"], "conf_strict": strict["conf"],
        "der_standard": standard["der"], "miss_standard": standard["miss"],
        "fa_standard": standard["fa"], "conf_standard": standard["conf"],
        # JER global de pyannote = moyenne sur tous les locuteurs de tous les fichiers
        "jer": 100 * abs(metriques["jer"]),
        "parole_ref_s": sum(l["parole_ref_s"] for l in lignes),
    }


def ligne_moyenne(lignes):
    """Moyenne simple des scores par fichier : chaque fichier pèse autant, quelle que soit sa durée."""
    moyenne = {"fichier": "MOYENNE"}
    for colonne in COLONNES[1:]:
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
        raise SystemExit("--ref et --hyp doivent être deux fichiers ou deux dossiers")

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
        print(f"Info : {sans_hyp} référence(s) sans hypothèse, non scorée(s).", file=sys.stderr)
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


def ecrire_csv(chemin, lignes):
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    with open(chemin, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLONNES)
        writer.writeheader()
        for ligne in lignes:
            writer.writerow({k: round(v, 3) if isinstance(v, float) else v for k, v in ligne.items()})


def afficher(lignes):
    print(f"{'fichier':<12} {'DER strict':>10} {'(raté/FA/conf)':>20} {'DER std':>8} {'JER':>7}")
    for l in lignes:
        composantes = f"{l['miss_strict']:.1f}/{l['fa_strict']:.1f}/{l['conf_strict']:.1f}"
        print(f"{l['fichier']:<12} {l['der_strict']:>9.2f}% {composantes:>20} "
              f"{l['der_standard']:>7.2f}% {l['jer']:>6.2f}%")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Calcule DER (strict et standard) et JER")
    parser.add_argument("--ref", required=True, help="RTTM de référence (fichier ou dossier)")
    parser.add_argument("--hyp", required=True, help="RTTM prédit (fichier ou dossier)")
    parser.add_argument("--collar", type=float, default=0.25,
                        help="marge ignorée de chaque côté des frontières pour le DER standard (s)")
    parser.add_argument("--csv", help="écrit les scores par fichier + MOYENNE + GLOBAL dans ce CSV")
    args = parser.parse_args(argv)

    lignes, moyenne, globale = scorer(apparier(args.ref, args.hyp), args.collar)
    afficher(lignes + [moyenne, globale])
    if args.csv:
        ecrire_csv(args.csv, lignes + [moyenne, globale])
        print(f"Scores écrits dans {args.csv}")


if __name__ == "__main__":
    main()
