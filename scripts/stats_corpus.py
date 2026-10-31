"""Calcule les statistiques des corpus et les écrit dans docs/corpus.md.

Les chiffres sont insérés entre les marqueurs <!-- STATS:debut --> et <!-- STATS:fin --> :
relancer le script met à jour les tableaux sans toucher au texte écrit à la main autour.

Deux niveaux pour AMI :
- « annoté » : toutes les réunions de la partition officielle (RTTM complets) ;
- « audio local » : seulement les réunions dont l'audio a été téléchargé (sous-ensemble).

Usage :
    uv run python scripts/stats_corpus.py
"""
import argparse
import statistics
from pathlib import Path

import soundfile as sf

from whospoke.io_rttm import lire_rttm

DEBUT, FIN = "<!-- STATS:debut -->", "<!-- STATS:fin -->"


def stats_jeu(dossier_rttm, dossier_wav, seulement_avec_audio=False):
    """Statistiques d'un ensemble de RTTM (et des .wav de même nom, s'ils existent)."""
    rttms = sorted(Path(dossier_rttm).glob("*.rttm"))
    if seulement_avec_audio:
        rttms = [r for r in rttms if (Path(dossier_wav) / f"{r.stem}.wav").exists()]

    duree_audio = parole = chevauchement = 0.0
    avec_audio = 0
    locuteurs_par_fichier, locuteurs = [], set()
    for chemin in rttms:
        annotation = lire_rttm(chemin)
        wav = Path(dossier_wav) / f"{chemin.stem}.wav"
        if wav.exists():
            duree_audio += sf.info(wav).duration
            avec_audio += 1
        parole += annotation.get_timeline().support().duration()  # temps où au moins 1 personne parle
        chevauchement += annotation.get_overlap().duration()        # temps où au moins 2 parlent
        locuteurs_par_fichier.append(len(annotation.labels()))
        locuteurs |= {(chemin.stem, l) for l in annotation.labels()}  # (fichier, label)
    return {
        "fichiers": len(rttms),
        "avec_audio": avec_audio,
        "heures_audio": duree_audio / 3600,
        "heures_parole": parole / 3600,
        "chevauchement_pct": 100 * chevauchement / parole if parole else 0,
        "loc_min": min(locuteurs_par_fichier, default=0),
        "loc_med": statistics.median(locuteurs_par_fichier) if locuteurs_par_fichier else 0,
        "loc_max": max(locuteurs_par_fichier, default=0),
        # Labels globaux (AMI : FEE005...) -> on compte les identités distinctes
        "labels_distincts": len({l for _, l in locuteurs}),
    }


def ligne(nom, s, identites_globales):
    distincts = str(s["labels_distincts"]) if identites_globales else "n/a (labels locaux)"
    audio = f"{s['heures_audio']:.1f}" if s["avec_audio"] else "—"
    return (f"| {nom} | {s['fichiers']} | {s['avec_audio']} | {audio} | {s['heures_parole']:.1f} | "
            f"{s['chevauchement_pct']:.1f} | {s['loc_min']} / {s['loc_med']:g} / {s['loc_max']} | {distincts} |")


def tableau(args):
    entete = [
        "| Jeu | Enregistrements annotés | Avec audio local | Heures d'audio local | Heures de parole annotée "
        "| Chevauchement (% de la parole) | Locuteurs par fichier (min / médiane / max) | Locuteurs distincts |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lignes = []
    for split in ["train", "dev", "test"]:
        rttm = Path(args.ami) / split / "rttm"
        wav = Path(args.ami) / split / "wav"
        if rttm.exists():
            lignes.append(ligne(f"AMI {split} (annoté complet)", stats_jeu(rttm, wav), True))
            lignes.append(ligne(f"AMI {split} (audio local)", stats_jeu(rttm, wav, True), True))
    if Path(args.voxconverse_rttm).exists():
        s = stats_jeu(args.voxconverse_rttm, args.voxconverse_wav)
        lignes.append(ligne("VoxConverse dev", s, False))
    return "\n".join(entete + lignes)


def main():
    parser = argparse.ArgumentParser(description="Statistiques des corpus -> docs/corpus.md")
    parser.add_argument("--ami", default="data/raw/ami")
    parser.add_argument("--voxconverse-rttm", default="data/raw/dev")
    parser.add_argument("--voxconverse-wav", default="data/raw/audio")
    parser.add_argument("--doc", default="docs/corpus.md")
    args = parser.parse_args()

    texte_stats = tableau(args)
    print(texte_stats)

    doc = Path(args.doc)
    contenu = doc.read_text(encoding="utf-8")
    if DEBUT not in contenu or FIN not in contenu:
        raise SystemExit(f"Marqueurs {DEBUT} / {FIN} absents de {doc}")
    avant, reste = contenu.split(DEBUT, 1)
    _, apres = reste.split(FIN, 1)
    doc.write_text(f"{avant}{DEBUT}\n{texte_stats}\n{FIN}{apres}", encoding="utf-8")
    print(f"\nTableau mis à jour dans {doc}")


if __name__ == "__main__":
    main()
