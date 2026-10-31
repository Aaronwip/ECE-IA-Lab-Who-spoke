"""Outils communs aux scripts de téléchargement : téléchargement, plafond disque, format audio.

Uniquement la bibliothèque standard + soundfile + ffmpeg (déjà installés) : pas de nouvelle dépendance.
"""
import shutil
import subprocess
import urllib.request
from pathlib import Path

import soundfile as sf

# Plafond fixé pour le projet : tout le dossier data/ doit tenir dans 5 Go.
PLAFOND_OCTETS = 5 * 1024**3


def taille_dossier(dossier):
    """Taille totale (en octets) de tous les fichiers sous `dossier`."""
    dossier = Path(dossier)
    if not dossier.exists():
        return 0
    return sum(f.stat().st_size for f in dossier.rglob("*") if f.is_file())


def taille_distante(url):
    """Taille annoncée par le serveur (en-tête Content-Length), ou 0 si inconnue."""
    requete = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(requete, timeout=60) as reponse:
        return int(reponse.headers.get("Content-Length", 0))


def verifier_plafond(dossier_data, octets_a_ajouter):
    """Lève une erreur si ajouter `octets_a_ajouter` ferait dépasser le plafond de data/."""
    actuel = taille_dossier(dossier_data)
    if actuel + octets_a_ajouter > PLAFOND_OCTETS:
        raise RuntimeError(
            f"Plafond disque : data/ fait {actuel / 1e9:.2f} Go, "
            f"+{octets_a_ajouter / 1e9:.2f} Go dépasserait {PLAFOND_OCTETS / 1e9:.2f} Go."
        )


def telecharger(url, destination):
    """Télécharge `url` vers `destination` (via un fichier .part, pour ne jamais laisser
    un fichier à moitié écrit sous son vrai nom si le téléchargement est interrompu)."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partiel = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as reponse, open(partiel, "wb") as f:
        shutil.copyfileobj(reponse, f, length=1024 * 1024)
    partiel.replace(destination)


def lire_texte(url):
    """Télécharge un petit fichier texte (liste, RTTM) et renvoie son contenu."""
    with urllib.request.urlopen(url, timeout=60) as reponse:
        return reponse.read().decode("utf-8")


def assurer_16k_mono(chemin):
    """Vérifie qu'un .wav est en 16 kHz mono ; sinon le convertit sur place avec ffmpeg.

    Renvoie True si une conversion a été faite.
    """
    chemin = Path(chemin)
    info = sf.info(chemin)
    if info.samplerate == 16000 and info.channels == 1:
        return False
    temporaire = chemin.with_name(chemin.stem + ".tmp.wav")
    # Même commande que dans CLAUDE.md : -ac 1 = mono, -ar 16000 = 16 kHz
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(chemin), "-ac", "1", "-ar", "16000", str(temporaire)],
        check=True,
    )
    temporaire.replace(chemin)
    return True


def decouper_rttm(texte):
    """Découpe un RTTM qui contient plusieurs enregistrements en {identifiant: lignes}.

    Le champ 2 d'une ligne RTTM (index 1) est l'identifiant de l'enregistrement.
    """
    par_fichier = {}
    for ligne in texte.splitlines():
        champs = ligne.split()
        if champs and champs[0] == "SPEAKER":
            par_fichier.setdefault(champs[1], []).append(ligne)
    return par_fichier
