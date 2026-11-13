"""Diarisation EN DIRECT : l'audio arrive du micro par petits morceaux (toutes les 0,5 s).

Le problème : pyannote met environ 40 s à traiter un paquet de 60 s. Pendant ce temps,
le micro continue d'envoyer du son, qui ne doit pas être perdu.

La solution, avec deux « fils d'exécution » (threads) qui tournent en même temps :
  - le fil de RÉCEPTION (ajouter) range chaque morceau dans un tampon ; dès que le tampon
    contient 60 s, il découpe un paquet et le met dans une FILE D'ATTENTE. C'est instantané ;
  - le fil de CALCUL (_travailler) prend les paquets de la file un par un, lance pyannote,
    puis relie les locuteurs avec le même Relieur que le run D.
Tant que chaque paquet est traité en moins de 60 s (0,63 x mesuré), la file ne grossit pas :
le retard sur la parole reste d'environ un paquet + son temps de calcul.

Un verrou (Lock) empêche les deux fils de modifier les mêmes données au même moment.
"""
import queue
import threading
import time
import traceback

import numpy as np
from pyannote.core import Annotation
from scipy.signal import resample_poly

from whospoke.paquets import Relieur, diariser_paquet, placer

SR = 16000  # pyannote travaille en 16 kHz ; le micro du navigateur envoie souvent du 48 kHz


def en_float_mono(morceau):
    """Morceau du micro (souvent des entiers 16 bits, parfois stéréo) -> float32 mono entre -1 et 1."""
    morceau = np.asarray(morceau)
    if np.issubdtype(morceau.dtype, np.integer):
        morceau = morceau.astype(np.float32) / np.iinfo(morceau.dtype).max
    if morceau.ndim > 1:
        morceau = morceau.mean(axis=1)
    return morceau.astype(np.float32)


class Direct:
    def __init__(self, pipeline, paquet=60.0, seuil=0.5, minimum=10.0):
        self.pipeline = pipeline
        self.paquet = paquet      # durée d'un paquet (s)
        self.seuil = seuil        # seuil du Relieur
        self.minimum = minimum    # à l'arrêt, un reste plus court n'est pas analysé (empreinte inutilisable)
        self.verrou = threading.Lock()
        self.file = queue.Queue()
        self.session = 0          # change à chaque nouvel enregistrement : un vieux paquet encore
                                  # en calcul ne doit pas atterrir dans le nouveau résultat
        self.reinitialiser()
        threading.Thread(target=self._travailler, daemon=True).start()

    def reinitialiser(self):
        """Nouvel enregistrement : on oublie tout (locuteurs compris)."""
        with self.verrou:
            self.session += 1
            self.sr_micro = None
            self.tampon = []          # morceaux reçus depuis le dernier paquet (fréquence du micro)
            self.taille_tampon = 0    # en échantillons
            self.audio = np.zeros(0, dtype=np.float32)  # tout l'audio parti en paquets, en 16 kHz
            self.recu = 0.0           # secondes reçues du micro
            self.analyse = 0.0        # secondes déjà analysées (fin du dernier paquet traité)
            self.en_cours = None      # début (s) du paquet en cours de calcul, sinon None
            self.calculs = []         # (durée du paquet, temps de calcul) pour chaque paquet
            self.relieur = Relieur(self.seuil)
            self.resultat = Annotation(uri="direct")
            self.erreur = None

    def ajouter(self, sr, morceau):
        """Appelé à chaque morceau du micro (fil de réception). Doit rester rapide."""
        morceau = en_float_mono(morceau)
        with self.verrou:
            self.sr_micro = sr
            self.tampon.append(morceau)
            self.taille_tampon += len(morceau)
            self.recu += len(morceau) / sr
            if self.taille_tampon >= self.paquet * sr:
                self._envoyer(int(self.paquet * sr))

    def terminer(self):
        """Fin de l'enregistrement : on envoie le reste s'il est assez long, puis on attend
        que tous les paquets soient traités."""
        with self.verrou:
            if self.sr_micro and self.taille_tampon >= self.minimum * self.sr_micro:
                self._envoyer(self.taille_tampon)
        self.file.join()  # attend la fin du calcul (hors verrou, sinon le fil de calcul serait bloqué)

    def _envoyer(self, n):
        """Découpe les n premiers échantillons du tampon en un paquet et le met dans la file.
        (Appelé avec le verrou déjà pris.)"""
        tout = np.concatenate(self.tampon)
        paquet, reste = tout[:n], tout[n:]
        self.tampon = [reste] if len(reste) else []
        self.taille_tampon = len(reste)
        if self.sr_micro != SR:
            paquet = resample_poly(paquet, SR, self.sr_micro).astype(np.float32)
        debut = len(self.audio) / SR
        self.audio = np.concatenate([self.audio, paquet])
        self.file.put((self.session, debut, paquet))

    def _travailler(self):
        """Fil de calcul : tourne en permanence, traite les paquets dans l'ordre d'arrivée."""
        while True:
            session, debut, paquet = self.file.get()
            try:
                if session != self.session:
                    continue  # paquet d'un enregistrement abandonné
                self.en_cours = debut
                t0 = time.perf_counter()
                locale, empreintes = diariser_paquet(self.pipeline, paquet, SR)
                calcul = time.perf_counter() - t0
                with self.verrou:
                    if session == self.session:
                        placer(self.resultat, locale, self.relieur.relier(locale, empreintes), debut)
                        self.analyse = debut + len(paquet) / SR
                        self.calculs.append((len(paquet) / SR, calcul))
            except Exception:
                self.erreur = traceback.format_exc()
                print(self.erreur)
            finally:
                self.en_cours = None
                self.file.task_done()

    def etat(self):
        """Copie cohérente de l'état, pour l'affichage (sans bloquer le calcul longtemps)."""
        with self.verrou:
            return {
                "resultat": self.resultat.copy(), "recu": self.recu, "analyse": self.analyse,
                "en_cours": self.en_cours, "calculs": list(self.calculs),
                "attente": self.file.qsize(), "erreur": self.erreur, "audio": self.audio,
            }
