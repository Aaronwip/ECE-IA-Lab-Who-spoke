"""Baseline maison : détection de parole par énergie + MFCC + clustering."""
import librosa
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment

SR = 16000  # fréquence d'échantillonnage attendue (16 000 valeurs par seconde)
TRAME = 400  # 25 ms à 16 kHz : longueur de la fenêtre d'analyse
PAS = 160  # 10 ms : décalage entre deux trames, soit 100 trames par seconde


def charger_audio(chemin):
    """Charge un wav mono 16 kHz sous forme de tableau numpy."""
    signal, sr = sf.read(chemin, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)  # stéréo -> mono
    assert sr == SR, f"{chemin} : {sr} Hz au lieu de {SR}"
    return signal


def energie_db(signal):
    """Énergie (RMS) de chaque trame, en décibels."""
    rms = librosa.feature.rms(y=signal, frame_length=TRAME, hop_length=PAS)[0]
    return 20 * np.log10(rms + 1e-10)  # + 1e-10 évite log(0) sur un silence parfait


def detecter_parole(signal, seuil_db=-40):
    """Renvoie un tableau de booléens : True si la trame contient de la parole."""
    energie = energie_db(signal)
    return energie > energie.max() + seuil_db


def trames_vers_annotation(etiquettes, uri):
    """Regroupe les trames consécutives de même étiquette en segments (None = silence)."""
    annotation = Annotation(uri=uri)
    debut = 0  # indice de la première trame du segment en cours
    for i in range(1, len(etiquettes) + 1):
        # Un segment se termine à la fin du fichier ou quand l'étiquette change
        if i == len(etiquettes) or etiquettes[i] != etiquettes[debut]:
            if etiquettes[debut] is not None:  # on n'écrit pas les silences
                annotation[Segment(debut * PAS / SR, i * PAS / SR)] = etiquettes[debut]
            debut = i
    return annotation
