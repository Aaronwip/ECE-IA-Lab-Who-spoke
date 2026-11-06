"""Baseline maison : détection de parole par énergie + MFCC + clustering."""
import librosa
import numpy as np
import soundfile as sf
from pyannote.core import Annotation, Segment
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

SR = 16000  # fréquence d'échantillonnage attendue (16 000 valeurs par seconde)
TRAME = 400  # 25 ms à 16 kHz : longueur de la fenêtre d'analyse
PAS = 160  # 10 ms : décalage entre deux trames, soit 100 trames par seconde
FENETRE = 100  # 1 s : on attribue un seul locuteur par tranche de 100 trames


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


def calculer_mfcc(signal):
    """Renvoie un tableau (nb_trames, 12) qui décrit le timbre de chaque trame."""
    mfcc = librosa.feature.mfcc(y=signal, sr=SR, n_mfcc=13, n_fft=TRAME, hop_length=PAS, n_mels=40)
    # librosa renvoie (coefficients, trames) : on inverse pour avoir une ligne par trame.
    # On retire le coefficient 0 : il mesure surtout le volume, pas la voix.
    return mfcc.T[:, 1:]


def regrouper_trames(mfcc, parole, nb_locuteurs):
    """Attribue un locuteur à chaque trame de parole (None = silence)."""
    # 1. On découpe en fenêtres de 1 s et on garde celles qui contiennent de la parole
    debuts = [d for d in range(0, len(parole), FENETRE) if parole[d:d + FENETRE].any()]
    # 2. Chaque fenêtre devient un seul vecteur : la moyenne des MFCC de ses trames de parole
    vecteurs = [mfcc[d:d + FENETRE][parole[d:d + FENETRE]].mean(axis=0) for d in debuts]
    # 3. Même échelle pour tous les coefficients, sinon les plus grands dominent les distances
    vecteurs = StandardScaler().fit_transform(vecteurs)
    # 4. Clustering : on fusionne les fenêtres les plus proches jusqu'à obtenir nb_locuteurs groupes
    groupes = AgglomerativeClustering(n_clusters=nb_locuteurs, linkage="ward").fit_predict(vecteurs)
    # 5. Chaque trame de parole reçoit le groupe de sa fenêtre
    etiquettes = [None] * len(parole)
    for d, groupe in zip(debuts, groupes):
        for i in range(d, min(d + FENETRE, len(parole))):
            if parole[i]:
                etiquettes[i] = f"spk{groupe}"
    return etiquettes


def diariser(signal, uri, nb_locuteurs=1):
    """Chaîne complète : parole -> MFCC -> clustering -> Annotation."""
    parole = detecter_parole(signal)
    if nb_locuteurs == 1:  # un seul locuteur : rien à regrouper
        etiquettes = ["spk0" if p else None for p in parole]
    else:
        etiquettes = regrouper_trames(calculer_mfcc(signal), parole, nb_locuteurs)
    return trames_vers_annotation(etiquettes, uri)


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
