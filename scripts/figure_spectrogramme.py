"""Graphique D : spectrogramme d'un fichier, avec la référence et la prédiction alignées dessous.

Une même couleur désigne le même vrai locuteur dans les deux chronologies.

Usage :
    uv run python scripts/figure_spectrogramme.py afjiv outputs/baseline_mfcc_oracle --sortie results/figures/spectrogramme_afjiv.png
"""
import argparse
import os
import warnings
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from pyannote.metrics.diarization import DiarizationErrorRate

from whospoke.baseline import PAS, SR, TRAME, charger_audio
from whospoke.io_rttm import lire_rttm

warnings.filterwarnings("ignore", message=".*approximated.*")

FOND = "#fcfcfb"
TEXTE = "#52514e"
GRILLE = "#e1e0d9"
GRIS = "#898781"  # locuteurs sans couleur attribuée
# Teintes choisies pour rester distinctes en cas de daltonisme ; au-delà de 8 locuteurs, on passe au gris
COULEURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# Une seule teinte pour une intensité : clair = faible, foncé = fort
ECHELLE = LinearSegmentedColormap.from_list("energie", [FOND, "#86b6ef", "#2a78d6", "#0d366b"])


def tracer_chronologie(ax, annotation, lignes):
    """Dessine une ligne par locuteur. `lignes` = liste de (label, texte affiché, couleur)."""
    for rang, (label, _, couleur) in enumerate(lignes):
        segments = [(s.start, s.duration) for s, _, l in annotation.itertracks(yield_label=True) if l == label]
        ax.broken_barh(segments, (rang - 0.35, 0.7), color=couleur, linewidth=0)
    ax.set_yticks(range(len(lignes)), [texte for _, texte, _ in lignes])
    ax.set_ylim(len(lignes) - 0.5, -0.5)  # premier locuteur en haut


parser = argparse.ArgumentParser(description="Spectrogramme + référence + prédiction d'un fichier")
parser.add_argument("fichier", help="nom du fichier sans extension (ex. afjiv)")
parser.add_argument("predictions", help="dossier des RTTM prédits")
parser.add_argument("--audio", default="data/raw/audio", help="dossier des .wav")
parser.add_argument("--references", default="data/raw/dev", help="dossier des RTTM de référence")
parser.add_argument("--sortie", required=True, help="image à écrire (.png)")
args = parser.parse_args()

signal = charger_audio(Path(args.audio) / f"{args.fichier}.wav")
reference = lire_rttm(Path(args.references) / f"{args.fichier}.rttm")
prediction = lire_rttm(Path(args.predictions) / f"{args.fichier}.rttm")

# Mêmes réglages que calculer_mfcc : c'est ce spectrogramme que les MFCC résument ensuite
mel = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=TRAME, hop_length=PAS, n_mels=40)
mel_db = librosa.power_to_db(mel, ref=np.max, top_db=80)

# Les groupes prédits n'ont pas de nom commun avec la référence (spk2 n'est pas spk02) :
# on associe chaque groupe à un vrai locuteur, avec le même calcul que celui du DER
correspondance = DiarizationErrorRate().optimal_mapping(reference, prediction)

locuteurs = sorted(reference.labels())
couleur_de = {l: COULEURS[i] if i < len(COULEURS) else GRIS for i, l in enumerate(locuteurs)}
lignes_ref = [(l, l, couleur_de[l]) for l in locuteurs]
# On range les groupes prédits dans l'ordre de leur vrai locuteur, pour faciliter la comparaison
groupes = sorted(prediction.labels(), key=lambda g: locuteurs.index(correspondance[g]) if g in correspondance else len(locuteurs))
lignes_pred = [(g, f"{g} → {correspondance.get(g, 'aucun')}", couleur_de.get(correspondance.get(g), GRIS)) for g in groupes]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 7), facecolor=FOND, layout="constrained",
                         height_ratios=[3, max(len(lignes_ref), 2), max(len(lignes_pred), 2)])
duree = len(signal) / SR

image = axes[0].imshow(mel_db, origin="lower", aspect="auto", cmap=ECHELLE, extent=[0, duree, 0, mel_db.shape[0]])
axes[0].set_ylabel("Bande mel (grave → aigu)", color=TEXTE)
axes[0].set_title(f"{args.fichier} : ce qu'on entend, ce qui est annoté, ce qui est prédit", loc="left", color="#0b0b0b", fontsize=12)
legende = axes[0].inset_axes([1.01, 0, 0.012, 1])  # hors du graphique, pour garder les 3 axes du temps alignés
barre = fig.colorbar(image, cax=legende)
barre.set_label("dB", color=TEXTE)
barre.ax.tick_params(colors=TEXTE, length=0)
barre.outline.set_visible(False)

tracer_chronologie(axes[1], reference, lignes_ref)
axes[1].set_ylabel("Référence", color=TEXTE)
tracer_chronologie(axes[2], prediction, lignes_pred)
axes[2].set_ylabel("Prédiction", color=TEXTE)
axes[2].set_xlabel("Temps (s)", color=TEXTE)
axes[2].set_xlim(0, duree)

for ax in axes:
    ax.set_facecolor(FOND)
    ax.tick_params(colors=TEXTE, length=0)
    ax.grid(axis="x", color=GRILLE, linewidth=0.8)
    ax.set_axisbelow(True)
    for cote in ("top", "right", "left", "bottom"):
        ax.spines[cote].set_visible(False)
axes[0].grid(False)

os.makedirs(Path(args.sortie).parent, exist_ok=True)
fig.savefig(args.sortie, dpi=150, facecolor=FOND)
print(f"Graphique écrit dans {args.sortie}")
