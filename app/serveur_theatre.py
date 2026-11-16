"""Serveur du Théâtre des voix : sert la page ET analyse un audio qui n'a pas de RTTM.

Lancement (depuis la racine du repo) :
    uv run python app/serveur_theatre.py
    puis ouvrir http://127.0.0.1:8000/app/theatre.html

Option : --segmentation chemin/vers/poids.ckpt   (segmentation fine-tunée, sinon celle d'origine)
         --port 8000

Pourquoi ce fichier ? La page theatre.html ne fait qu'AFFICHER une diarisation : le modèle pyannote
tourne en Python, pas dans le navigateur. Ce serveur fait le lien :
  - GET  /etat      -> le nom du modèle chargé (la page l'affiche en petit) ;
  - POST /diariser  -> la page envoie l'audio, le serveur renvoie le texte du RTTM ;
  - tout le reste   -> les fichiers du repo (page, three.js, exemples), comme `python -m http.server`.

L'analyse est exactement celle de la démo Gradio et du run D (results/runs.csv) : mêmes fonctions
de src/whospoke/paquets.py, mêmes réglages. Aucune nouvelle dépendance : http.server fait partie de Python.
"""
import argparse
import io
import json
import subprocess
import tempfile
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import librosa
from pyannote.core import Annotation

from whospoke.paquets import MODELE, Relieur, bornes_paquets, charger_pipeline, diariser_paquet, placer

# Réglages retenus après l'ablation (run D), les mêmes que dans app/demo.py
SR = 16000      # pyannote travaille en 16 kHz mono
PAQUET = 60.0   # durée d'un paquet (s)
SEUIL = 0.5     # similarité cosinus minimale pour dire « même personne »
FUSION = 10.0   # dernier paquet de moins de 10 s collé au précédent

RACINE = Path(__file__).resolve().parent.parent   # racine du repo : c'est elle qui est servie

pipeline = None               # chargé une seule fois, au démarrage
description = ""              # texte affiché par la page : quel modèle tourne
verrou = threading.Lock()     # une seule analyse à la fois (le modèle n'est pas fait pour deux en parallèle)


def charger_audio(chemin):
    """N'importe quel format -> tableau numpy mono 16 kHz (même fonction que dans app/demo.py).
    librosa lit wav/flac/mp3 ; pour le reste (m4a, vidéo...), on convertit d'abord avec ffmpeg."""
    try:
        audio, _ = librosa.load(chemin, sr=SR, mono=True)
    except Exception:
        wav = Path(chemin).with_name("converti.wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(chemin), "-ac", "1", "-ar", str(SR), str(wav)],
                       check=True, capture_output=True)
        audio, _ = librosa.load(wav, sr=SR, mono=True)
    return audio


def diariser(chemin):
    """Audio -> texte RTTM. Paquet par paquet, comme dans la démo et dans scripts/run_paquets.py."""
    audio = charger_audio(chemin)
    relieur = Relieur(SEUIL)
    resultat = Annotation(uri="audio")
    for debut, fin in bornes_paquets(len(audio), SR, PAQUET, FUSION):
        locale, empreintes = diariser_paquet(pipeline, audio[debut:fin], SR)
        placer(resultat, locale, relieur.relier(locale, empreintes), debut / SR)
    texte = io.StringIO()
    resultat.write_rttm(texte)
    return texte.getvalue()


class Requete(SimpleHTTPRequestHandler):
    """Une requête du navigateur. Par défaut, SimpleHTTPRequestHandler renvoie les fichiers du dossier."""

    def repondre(self, code, texte, type_mime):
        donnees = texte.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", type_mime + "; charset=utf-8")
        self.send_header("Content-Length", str(len(donnees)))
        self.end_headers()
        self.wfile.write(donnees)

    def do_GET(self):
        if urlparse(self.path).path == "/etat":
            self.repondre(200, json.dumps({"modele": description}), "application/json")
        else:
            super().do_GET()          # un fichier du repo

    def do_POST(self):
        adresse = urlparse(self.path)
        if adresse.path != "/diariser":
            self.send_error(404)
            return
        # Le corps de la requête est le fichier audio brut ; ?ext=.mp3 donne son extension.
        extension = parse_qs(adresse.query).get("ext", [".wav"])[0]
        donnees = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        with tempfile.TemporaryDirectory() as dossier:
            chemin = Path(dossier) / ("audio" + Path("x" + extension).suffix)
            chemin.write_bytes(donnees)
            t0 = time.perf_counter()
            try:
                with verrou:
                    rttm = diariser(chemin)
            except Exception as erreur:
                self.repondre(500, str(erreur), "text/plain")
                return
        print(f"Analyse : {len(donnees) / 1e6:.1f} Mo traités en {time.perf_counter() - t0:.0f} s")
        self.repondre(200, rttm, "text/plain")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serveur du Théâtre des voix")
    parser.add_argument("--segmentation", help="checkpoint .ckpt de segmentation fine-tunée (optionnel)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("Chargement du modèle pyannote...")
    pipeline = charger_pipeline(segmentation=args.segmentation)
    segmentation = f"segmentation fine-tunée ({Path(args.segmentation).name})" if args.segmentation else "segmentation d'origine"
    description = f"{MODELE} · {segmentation} · paquets de {PAQUET:.0f} s"

    # 127.0.0.1 : le serveur n'est visible que depuis cet ordinateur.
    serveur = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Requete, directory=str(RACINE)))
    print(f"Modèle : {description}")
    print(f"Ouvrir http://127.0.0.1:{args.port}/app/theatre.html   (Ctrl+C pour arrêter)")
    serveur.serve_forever()
