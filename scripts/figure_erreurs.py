"""Graphique A : erreurs par fichier (parole ratée, fausse alerte, confusion) en barres empilées.

Lit le CSV par fichier écrit par eval/score.py. Scores en DER strict (collar 0).

Usage :
    uv run python eval/score.py --ref data/raw/dev --hyp outputs/baseline_mfcc_oracle --csv results/scores_mfcc_oracle.csv
    uv run python scripts/figure_erreurs.py results/scores_mfcc_oracle.csv --titre mfcc-oracle --sortie results/figures/erreurs_mfcc_oracle.png
"""
import argparse
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt

FOND = "#fcfcfb"
TEXTE = "#52514e"
GRILLE = "#e1e0d9"
# Une couleur par type d'erreur ; teintes choisies pour rester distinctes en cas de daltonisme
COMPOSANTES = [
    ("miss_strict", "Parole ratée", "#2a78d6"),
    ("fa_strict", "Fausse alerte", "#eb6834"),
    ("conf_strict", "Confusion", "#1baf7a"),
]

parser = argparse.ArgumentParser(description="Barres empilées des erreurs par fichier")
parser.add_argument("scores", help="CSV par fichier écrit par eval/score.py")
parser.add_argument("--titre", required=True, help="nom de l'approche, affiché dans le titre")
parser.add_argument("--sortie", required=True, help="image à écrire (.png)")
args = parser.parse_args()

with open(args.scores, encoding="utf-8") as f:
    lignes = list(csv.DictReader(f))

# Le CSV finit par deux lignes de synthèse : on garde GLOBAL pour le titre et on trace le reste
glob = next(l for l in lignes if l["fichier"] == "GLOBAL")
fichiers = [l for l in lignes if l["fichier"] not in ("GLOBAL", "MOYENNE")]
# barh trace de bas en haut : en triant du plus petit au plus grand, le pire fichier est en haut
fichiers.sort(key=lambda l: float(l["der_strict"]))

fig, ax = plt.subplots(figsize=(9, 0.42 * len(fichiers) + 1.8), facecolor=FOND, layout="constrained")
ax.set_facecolor(FOND)
noms = [l["fichier"] for l in fichiers]
gauche = [0.0] * len(fichiers)  # où commence le prochain morceau de chaque barre
for colonne, nom, couleur in COMPOSANTES:
    valeurs = [float(l[colonne]) for l in fichiers]
    # Le contour de la couleur du fond crée un petit espace entre les morceaux
    ax.barh(noms, valeurs, left=gauche, height=0.65, color=couleur, label=nom, edgecolor=FOND, linewidth=1.5)
    gauche = [g + v for g, v in zip(gauche, valeurs)]

# Le total au bout de chaque barre : c'est le DER strict du fichier
for i, l in enumerate(fichiers):
    ax.text(gauche[i] + 0.6, i, f"{float(l['der_strict']):.1f} %", va="center", color=TEXTE, fontsize=9)

ax.set_title(f"{args.titre} : erreurs par fichier (DER strict global {float(glob['der_strict']):.1f} %)",
             loc="left", color="#0b0b0b", fontsize=12)
ax.set_xlabel("% du temps de parole de la référence", color=TEXTE)
ax.set_xlim(0, max(gauche) * 1.12)
ax.tick_params(colors=TEXTE, length=0)
ax.grid(axis="x", color=GRILLE, linewidth=0.8)
ax.set_axisbelow(True)
for cote in ("top", "right", "left", "bottom"):
    ax.spines[cote].set_visible(False)
ax.legend(loc="lower right", frameon=False, labelcolor=TEXTE)

os.makedirs(Path(args.sortie).parent, exist_ok=True)
fig.savefig(args.sortie, dpi=150, facecolor=FOND)
print(f"Graphique écrit dans {args.sortie}")
