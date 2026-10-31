"""Télécharge VoxConverse dev : RTTM du dépôt officiel + audio (archive zip d'Oxford).

- RTTM : https://github.com/joonson/voxconverse (dossier dev/, commit figé).
- Audio : voxconverse_dev_wav.zip (lien du README du dépôt), déjà en 16 kHz mono.

Rangement (celui déjà utilisé par scripts/run_pyannote.py et whospoke.evaluate) :
    data/raw/dev/<id>.rttm
    data/raw/audio/<id>.wav

Si tous les .wav attendus sont déjà là, l'archive (~2 Go) n'est pas retéléchargée.

Usage :
    uv run python scripts/telecharger_voxconverse.py
"""
import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

from whospoke.donnees import (assurer_16k_mono, lire_texte, telecharger, taille_distante,
                              verifier_plafond)

COMMIT = "24bf60be297701cd7e4ef18550c6d390c1b87365"
RTTM = "https://raw.githubusercontent.com/joonson/voxconverse/{commit}/dev/{nom}"
ARBRE = "https://api.github.com/repos/joonson/voxconverse/git/trees/{commit}?recursive=1"
ZIP = "https://www.robots.ox.ac.uk/~vgg/data/voxconverse/data/voxconverse_dev_wav.zip"


def lister_rttm_dev(commit):
    """Liste des RTTM du dossier dev/ du dépôt, via l'API GitHub."""
    with urllib.request.urlopen(ARBRE.format(commit=commit), timeout=60) as reponse:
        arbre = json.load(reponse)["tree"]
    return sorted(Path(e["path"]).name for e in arbre
                  if e["path"].startswith("dev/") and e["path"].endswith(".rttm"))


def main():
    parser = argparse.ArgumentParser(description="Téléchargement de VoxConverse dev")
    parser.add_argument("--data", default="data", help="racine des données (pour le plafond disque)")
    parser.add_argument("--rttm", default="data/raw/dev", help="dossier des RTTM de référence")
    parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
    parser.add_argument("--commit", default=COMMIT, help="commit du dépôt joonson/voxconverse")
    args = parser.parse_args()

    # 1. RTTM : un fichier par enregistrement
    dossier_rttm = Path(args.rttm)
    dossier_rttm.mkdir(parents=True, exist_ok=True)
    noms = lister_rttm_dev(args.commit)
    for nom in noms:
        # Toujours réécrits (quelques Ko) : on est sûr d'avoir la version du commit figé.
        # newline="" : on écrit les octets tels quels, sans conversion Windows des fins de ligne.
        texte = lire_texte(RTTM.format(commit=args.commit, nom=nom))
        (dossier_rttm / nom).write_text(texte, encoding="utf-8", newline="")
    print(f"{len(noms)} RTTM dev dans {dossier_rttm}")

    # 2. Audio : on ne télécharge l'archive que s'il manque des fichiers
    dossier_audio = Path(args.audio)
    attendus = {Path(n).stem for n in noms}
    presents = {f.stem for f in dossier_audio.glob("*.wav")}
    manquants = attendus - presents
    if manquants:
        print(f"{len(manquants)} audios manquants : téléchargement de l'archive")
        archive = Path(args.data) / "raw" / "voxconverse_dev_wav.zip"
        # L'archive puis son contenu décompressé occupent environ 2 fois sa taille
        verifier_plafond(args.data, 2 * taille_distante(ZIP))
        telecharger(ZIP, archive)
        dossier_audio.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            for membre in z.namelist():
                nom = Path(membre).name
                if nom.endswith(".wav") and Path(nom).stem in manquants:
                    (dossier_audio / nom).write_bytes(z.read(membre))
        archive.unlink()  # on libère la place une fois décompressé

    # 3. Format : tout doit être en 16 kHz mono
    convertis = sum(assurer_16k_mono(f) for f in sorted(dossier_audio.glob("*.wav")))
    restants = attendus - {f.stem for f in dossier_audio.glob("*.wav")}
    print(f"{len(attendus) - len(restants)}/{len(attendus)} audios dans {dossier_audio}, "
          f"{convertis} converti(s) en 16 kHz mono")
    if restants:
        print(f"Toujours manquants : {sorted(restants)}")


if __name__ == "__main__":
    main()
