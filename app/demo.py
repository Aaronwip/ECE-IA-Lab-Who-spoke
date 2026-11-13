"""Démo Who Spoke? : qui parle, et quand ?

Lancement :
    uv run python app/demo.py      puis ouvrir http://127.0.0.1:7860

Onglet « Fichier » : on charge un audio. Il est traité paquet par paquet (60 s), exactement
comme dans le run D (scripts/run_paquets.py, mêmes fonctions de src/whospoke/paquets.py) :
la timeline se remplit après chaque paquet.
Onglet « Direct (micro) » : on parle, un paquet part toutes les 60 s PENDANT l'enregistrement
(src/whospoke/direct.py), et la timeline se remplit au fil de l'eau.
Dans les deux cas : clic sur un segment du tableau pour l'écouter, export RTTM et JSON.

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

from whospoke.direct import Direct
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


# ---------------- Direct au micro (logique dans src/whospoke/direct.py) ----------------
# Démo locale pour une seule personne : un seul objet Direct, partagé.
direct = Direct(pipeline, PAQUET, SEUIL, minimum=FUSION)
affichage = {"actif": False, "cle": None}  # évite de redessiner si rien n'a changé


def minsec(s):
    return f"{int(s // 60)} min {int(s % 60):02d} s"


def statut_direct(e):
    morceaux = [f"**Reçu** : {minsec(e['recu'])} · **analysé** : {minsec(e['analyse'])}"]
    if e["en_cours"] is not None:
        morceaux.append(f"paquet à partir de {minsec(e['en_cours'])} en calcul")
    if e["attente"]:
        morceaux.append(f"{e['attente']} paquet(s) en attente")
    if e["calculs"]:
        duree, calcul = e["calculs"][-1]
        morceaux.append(f"dernier paquet : {calcul:.0f} s de calcul pour {duree:.0f} s ({calcul / duree:.2f} x)")
    morceaux.append(f"{len(e['resultat'].labels())} locuteur(s)")
    if e["erreur"]:
        morceaux.append("**Erreur dans le calcul, voir le terminal**")
    return " · ".join(morceaux)


def demarrer():
    direct.reinitialiser()
    affichage.update(actif=True, cle=None)
    return (None, [], f"Enregistrement en cours. Premier résultat après {PAQUET:.0f} s de parole "
            "+ environ 40 s de calcul.", None, None)


def recevoir(morceau):
    """Toutes les 0,5 s : le navigateur envoie le dernier morceau du micro."""
    if morceau is not None:
        sr, donnees = morceau
        direct.ajouter(sr, donnees)


def rafraichir():
    """Toutes les 3 s : redessine la timeline si quelque chose a changé."""
    e = direct.etat()
    cle = (int(e["recu"]), len(e["calculs"]), e["en_cours"])
    if not affichage["actif"] or e["recu"] == 0 or cle == affichage["cle"]:
        return gr.skip(), gr.skip()
    affichage["cle"] = cle
    return dessiner(e["resultat"], max(e["recu"], 1), e["analyse"]), statut_direct(e)


def finir():
    """Arrêt du micro : on analyse le dernier morceau, puis tableau, écoute et export comme pour un fichier."""
    affichage["actif"] = False
    yield gr.skip(), gr.skip(), "Arrêt : analyse des derniers paquets en cours...", gr.skip(), gr.skip()
    direct.terminer()
    e = direct.etat()
    resultat, duree = e["resultat"], len(e["audio"]) / SR
    if not len(resultat):
        yield gr.skip(), [], f"Rien d'analysé : il faut au moins {FUSION:.0f} s d'enregistrement.", None, None
        return
    resultat.uri = "direct_" + time.strftime("%Y%m%d_%H%M%S")
    pire = max(calcul / d for d, calcul in e["calculs"])
    statut = (f"**Terminé** : {minsec(e['recu'])} enregistrées, {len(resultat.labels())} locuteur(s). "
              f"Paquet le plus lent : {pire:.2f} x sa durée ({'direct tenu' if pire < 1 else 'trop lent'}). "
              "Clique sur une ligne du tableau pour écouter le segment.")
    yield dessiner(resultat, duree, duree), tableau(resultat), statut, e["audio"], exporter(resultat, duree)


def zone_resultats():
    """Statut, timeline, tableau des tours, lecteur et export : identiques dans les deux onglets."""
    statut = gr.Markdown()
    timeline = gr.Plot(label="Timeline des locuteurs (zone grise : pas encore analysée)")
    with gr.Row():
        segments = gr.Dataframe(headers=["locuteur", "début (s)", "fin (s)", "durée (s)"],
                                interactive=False, label="Tours de parole (clic = écouter)", max_height=350)
        with gr.Column():
            extrait = gr.Audio(label="Segment sélectionné", interactive=False, autoplay=True)
            fichiers = gr.File(label="Export RTTM / JSON", file_count="multiple")
    audio_charge = gr.State()  # l'audio 16 kHz, gardé pour l'écoute des segments
    segments.select(ecouter, inputs=audio_charge, outputs=extrait)
    return [timeline, segments, statut, audio_charge, fichiers]


with gr.Blocks(title="Who Spoke?") as demo:
    gr.Markdown("# Who Spoke? — qui parle, et quand ?\n"
                f"pyannote community-1, traité par paquets de {PAQUET:.0f} s "
                f"(seuil {SEUIL}, fusion du dernier paquet < {FUSION:.0f} s).")
    with gr.Tab("Fichier"):
        with gr.Row():
            entree = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio")
            bouton = gr.Button("Analyser", variant="primary", scale=0)
        sorties_fichier = zone_resultats()
        bouton.click(analyser, inputs=entree, outputs=sorties_fichier)

    with gr.Tab("Direct (micro)"):
        gr.Markdown(f"Clique sur le micro et parle : un paquet part toutes les {PAQUET:.0f} s pendant "
                    "l'enregistrement, et la timeline se remplit au fil de l'eau. Arrête l'enregistrement pour "
                    "analyser la fin, écouter les segments et exporter.")
        micro = gr.Audio(sources=["microphone"], streaming=True, type="numpy", label="Micro")
        sorties_direct = zone_resultats()
        timeline_direct, _, statut_direct_md, _, _ = sorties_direct
        micro.start_recording(demarrer, outputs=sorties_direct)
        micro.stream(recevoir, inputs=micro, outputs=None, stream_every=0.5)
        micro.stop_recording(finir, outputs=sorties_direct)
        gr.Timer(3.0).tick(rafraichir, outputs=[timeline_direct, statut_direct_md])

if __name__ == "__main__":
    demo.launch()
