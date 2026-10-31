"""Télécharge AMI (micro Mix-Headset) avec les annotations et partitions officielles BUT.

- Partitions et RTTM : https://github.com/BUTSpeechFIT/AMI-diarization-setup
  (partition « Full-corpus-ASR » d'AMI, références `only_words`).
- Audio : miroir officiel de l'université d'Édimbourg, fichier <réunion>.Mix-Headset.wav.

Ce qui est téléchargé :
- les listes de réunions et de locuteurs des 3 partitions (train, dev, test) ;
- les RTTM de TOUTES les réunions (quelques Ko, utiles pour vérifier l'absence de fuite) ;
- l'audio des `--par-split` premières réunions de chaque partition (ordre alphabétique),
  pour respecter le plafond disque de 5 Go.

Rangement :
    data/raw/ami/lists/{train,dev,test}.{meetings,speakers}.txt
    data/raw/ami/{train,dev,test}/rttm/<réunion>.rttm
    data/raw/ami/{train,dev,test}/wav/<réunion>.wav     (16 kHz mono)

Usage :
    uv run python scripts/telecharger_ami.py --par-split 10
"""
import argparse
from pathlib import Path

from whospoke.donnees import (assurer_16k_mono, decouper_rttm, lire_texte, telecharger,
                              taille_distante, verifier_plafond)

# Commit figé du dépôt BUT : les partitions ne changeront pas sous nos pieds (reproductibilité).
BUT = "https://raw.githubusercontent.com/BUTSpeechFIT/AMI-diarization-setup/{commit}"
AUDIO = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{reunion}/audio/{reunion}.Mix-Headset.wav"
SPLITS = ["train", "dev", "test"]


def main():
    parser = argparse.ArgumentParser(description="Téléchargement d'AMI Mix-Headset + partitions BUT")
    parser.add_argument("--data", default="data", help="racine des données (pour le plafond disque)")
    parser.add_argument("--sortie", default="data/raw/ami", help="dossier de destination")
    parser.add_argument("--par-split", type=int, default=10, help="nb de réunions audio par partition")
    parser.add_argument("--commit", default="2509d8933721023fab4def2618aabd5c28eb82e9",
                        help="commit du dépôt BUT (figé par défaut)")
    args = parser.parse_args()

    base = BUT.format(commit=args.commit)
    sortie = Path(args.sortie)

    for split in SPLITS:
        # 1. Listes officielles : on les garde telles quelles, ce sont elles qui définissent la partition
        for liste in ["meetings", "speakers"]:
            texte = lire_texte(f"{base}/lists/{split}.{liste}.txt")
            chemin = sortie / "lists" / f"{split}.{liste}.txt"
            chemin.parent.mkdir(parents=True, exist_ok=True)
            chemin.write_text(texte, encoding="utf-8", newline="")
        reunions = sorted(
            (sortie / "lists" / f"{split}.meetings.txt").read_text(encoding="utf-8").split()
        )

        # 2. RTTM : un seul fichier par partition chez BUT, qu'on découpe en un fichier par réunion
        par_reunion = decouper_rttm(lire_texte(f"{base}/only_words/rttms/{split}.rttm"))
        dossier_rttm = sortie / split / "rttm"
        dossier_rttm.mkdir(parents=True, exist_ok=True)
        for reunion, lignes in par_reunion.items():
            (dossier_rttm / f"{reunion}.rttm").write_text("\n".join(lignes) + "\n", encoding="utf-8",
                                                          newline="")
        print(f"[{split}] {len(reunions)} réunions dans la liste, {len(par_reunion)} RTTM écrits")

        # 3. Audio : les N premières réunions par ordre alphabétique (choix reproductible)
        dossier_wav = sortie / split / "wav"
        obtenues = 0
        for reunion in reunions:
            if obtenues >= args.par_split:
                break
            destination = dossier_wav / f"{reunion}.wav"
            if destination.exists():
                obtenues += 1
                continue
            url = AUDIO.format(reunion=reunion)
            try:
                verifier_plafond(args.data, taille_distante(url))
                telecharger(url, destination)
            except RuntimeError:
                raise  # plafond disque dépassé : on s'arrête net
            except Exception as erreur:  # fichier absent du miroir, coupure réseau...
                print(f"  {reunion} : échec du téléchargement ({erreur}), réunion suivante")
                continue
            converti = assurer_16k_mono(destination)
            obtenues += 1
            print(f"  {reunion} : {destination.stat().st_size / 1e6:.0f} Mo"
                  + (" (converti en 16 kHz mono)" if converti else ""))
        print(f"[{split}] {obtenues} audios présents dans {dossier_wav}")


if __name__ == "__main__":
    main()
