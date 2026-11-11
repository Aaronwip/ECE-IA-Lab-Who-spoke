"""Démo Who Spoke? : qui parle, et quand ?

Lancement :
    uv run python app/demo.py      puis ouvrir http://127.0.0.1:7860

On charge un audio. Il est traité paquet par paquet (60 s), exactement comme dans le
run D (scripts/run_paquets.py, mêmes fonctions de src/whospoke/paquets.py) :
la timeline se remplit après chaque paquet, comme elle le ferait en direct.
On peut cliquer sur un segment du tableau pour l'écouter, puis exporter en RTTM et JSON.

Gradio : bibliothèque Python qui fabrique une page web (chargement de fichier, micro,
lecteur audio, téléchargement) sans écrire de HTML ni de JavaScript.
"""
import json
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import librosa
import matplotlib
import numpy as np
from matplotlib.figure import Figure
from pyannote.core import Annotation

from whospoke.paquets import Relieur, bornes_paquets, charger_pipeline, diariser_paquet, placer

# Réglages retenus après l'ablation (run D, results/runs.csv)
SR = 16000      # pyannote travaille en 16 kHz mono
PAQUET = 60.0   # durée d'un paquet (s) : contrainte du direct
SEUIL = 0.5     # similarité cosinus minimale pour dire « même personne »
FUSION = 10.0   # dernier paquet de moins de 10 s collé au précédent

print("Chargement du modèle pyannote...")
pipeline = charger_pipeline()  # segmentation d'origine : le fine-tuning AMI n'apportait rien (run E)


def charger_audio(chemin):
    """N'importe quel format -> tableau numpy mono 16 kHz.
    librosa lit wav/flac/mp3 ; pour le reste (m4a, vidéo...), on convertit d'abord avec ffmpeg."""
    try:
        audio, _ = librosa.load(chemin, sr=SR, mono=True)
    except Exception:
        wav = Path(tempfile.mkdtemp()) / "converti.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(chemin), "-ac", "1", "-ar", str(SR), str(wav)],
                       check=True, capture_output=True)
        audio, _ = librosa.load(wav, sr=SR, mono=True)
    return audio


def dessiner(resultat, duree, traite):
    """Timeline : une ligne par locuteur, une barre par tour de parole.
    La zone grise est la partie de l'audio pas encore traitée."""
    locuteurs = sorted(resultat.labels())
    fig = Figure(figsize=(11, 1.2 + 0.45 * max(len(locuteurs), 1)))
    ax = fig.subplots()
    couleurs = matplotlib.colormaps["tab10"]
    for i, locuteur in enumerate(locuteurs):
        tours = [(s.start, s.duration) for s in resultat.label_timeline(locuteur)]
        ax.broken_barh(tours, (i - 0.4, 0.8), color=couleurs(i % 10))
    if traite < duree:
        ax.axvspan(traite, duree, color="0.92", zorder=0)
    ax.set_yticks(range(len(locuteurs)), locuteurs)
    ax.set_ylim(max(len(locuteurs), 1) - 0.5, -0.5)  # premier locuteur en haut
    ax.set_xlim(0, duree)
    ax.set_xlabel("temps (s)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def tableau(resultat):
    """Tours de parole, triés dans le temps. support() recolle les morceaux d'un même
    locuteur coupés à la frontière de deux paquets."""
    lignes = [[locuteur, round(s.start, 2), round(s.end, 2), round(s.duration, 2)]
              for s, _, locuteur in resultat.support().itertracks(yield_label=True)]
    return sorted(lignes, key=lambda ligne: ligne[1])


def exporter(resultat, duree):
    """Écrit le RTTM (format standard, comparable aux références) et un JSON lisible."""
    dossier = Path(tempfile.mkdtemp())
    rttm = dossier / f"{resultat.uri}.rttm"
    with open(rttm, "w") as f:
        resultat.write_rttm(f)
    contenu = {
        "fichier": resultat.uri,
        "duree_s": round(duree, 2),
        "reglages": {"paquet_s": PAQUET, "seuil": SEUIL, "fusion_s": FUSION},
        "locuteurs": sorted(resultat.labels()),
        "segments": [{"locuteur": l, "debut": d, "fin": f} for l, d, f, _ in tableau(resultat)],
    }
    js = dossier / f"{resultat.uri}.json"
    js.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")
    return [str(rttm), str(js)]


def analyser(chemin):
    """Générateur : chaque `yield` met la page à jour, une fois par paquet traité."""
    if chemin is None:
        raise gr.Error("Charge d'abord un fichier audio (ou enregistre-toi au micro).")
    audio = charger_audio(chemin)
    duree = len(audio) / SR
    bornes = bornes_paquets(len(audio), SR, PAQUET, FUSION)
    relieur = Relieur(SEUIL)
    resultat = Annotation(uri=Path(chemin).stem)
    pire = 0.0  # paquet le plus lent, rapporté à sa durée (direct tenu si < 1)

    yield dessiner(resultat, duree, 0), [], f"Paquet 1/{len(bornes)} en cours...", audio, None
    for n, (debut, fin) in enumerate(bornes, start=1):
        t0 = time.perf_counter()
        locale, empreintes = diariser_paquet(pipeline, audio[debut:fin], SR)
        calcul = time.perf_counter() - t0
        rapport = calcul / ((fin - debut) / SR)
        pire = max(pire, rapport)
        placer(resultat, locale, relieur.relier(locale, empreintes), debut / SR)

        statut = (f"**Paquet {n}/{len(bornes)}** : {(fin - debut) / SR:.0f} s d'audio traités en {calcul:.1f} s "
                  f"({rapport:.2f} x sa durée) · {len(resultat.labels())} locuteur(s) pour l'instant")
        if n < len(bornes):
            yield dessiner(resultat, duree, fin / SR), tableau(resultat), statut + " · paquet suivant en cours...", audio, None

    verdict = "direct tenu" if pire < 1 else "trop lent pour le direct"
    statut = (f"**Terminé** : {duree:.0f} s d'audio, {len(resultat.labels())} locuteur(s). "
              f"Paquet le plus lent : {pire:.2f} x sa durée ({verdict}). "
              f"Clique sur une ligne du tableau pour écouter le segment.")
    yield dessiner(resultat, duree, duree), tableau(resultat), statut, audio, exporter(resultat, duree)


def ecouter(audio, evt: gr.SelectData):
    """Clic sur une ligne du tableau -> extrait audio de ce tour de parole."""
    if audio is None or evt.row_value is None:
        return None
    _, debut, fin, _ = evt.row_value
    extrait = audio[int(float(debut) * SR): int(float(fin) * SR)]
    return SR, (extrait * 32767).astype(np.int16)  # le lecteur attend des entiers 16 bits


with gr.Blocks(title="Who Spoke?") as demo:
    gr.Markdown("# Who Spoke? — qui parle, et quand ?\n"
                f"pyannote community-1, traité par paquets de {PAQUET:.0f} s "
                f"(seuil {SEUIL}, fusion du dernier paquet < {FUSION:.0f} s).")
    with gr.Row():
        entree = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
        with gr.Column():
            bouton = gr.Button("Analyser", variant="primary")
            fichiers = gr.File(label="Export RTTM / JSON", file_count="multiple")
    statut = gr.Markdown()
    timeline = gr.Plot(label="Timeline des locuteurs")
    with gr.Row():
        segments = gr.Dataframe(headers=["locuteur", "début (s)", "fin (s)", "durée (s)"],
                                interactive=False, label="Tours de parole (clic = écouter)", max_height=350)
        extrait = gr.Audio(label="Segment sélectionné", interactive=False, autoplay=True)
    audio_charge = gr.State()  # l'audio 16 kHz, gardé pour l'écoute des segments

    bouton.click(analyser, inputs=entree, outputs=[timeline, segments, statut, audio_charge, fichiers])
    segments.select(ecouter, inputs=audio_charge, outputs=extrait)

if __name__ == "__main__":
    demo.launch()
