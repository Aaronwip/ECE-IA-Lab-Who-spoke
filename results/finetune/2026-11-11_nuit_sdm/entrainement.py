"""Fine-tuning de la segmentation de pyannote community-1 sur AMI (lancé par le notebook)."""
import argparse, io, json, os, shutil, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["test", "nuit"], required=True)
parser.add_argument("--config", default="sdm")
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max-heures", type=float, default=10)
args = parser.parse_args()

NB_FICHIERS_TRAIN = 1 if args.mode == "test" else 19   # train = 19 fichiers parquet (~7 réunions chacun)
NB_FICHIERS_VAL = 1 if args.mode == "test" else 3
MAX_EPOQUES = 1 if args.mode == "test" else 30
HF_TOKEN = os.environ["HF_TOKEN"]   # transmis par le notebook
RACINE = Path("/tmp/ami")           # données de travail (non sauvegardées)
SORTIE = Path("/kaggle/working")    # ce qui est sauvegardé (modèle, courbe, résumé)
print("Réglages :", vars(args), "| époques max", MAX_EPOQUES)
print("numpy", np.__version__, "| torch", torch.__version__, "| GPU", torch.cuda.is_available())
print("Espace libre sur /tmp : %.0f Go" % (shutil.disk_usage("/tmp").free / 1e9))

# ---------------- 1. Données : AMI depuis Hugging Face, réécrit au format pyannote ----------------
# wav/<id>.wav (audio 16 kHz mono), rttm/<id>.rttm (qui parle quand), uem/<id>.uem (zone annotée),
# lists/<partition>.txt (identifiants). Le test AMI n'est jamais téléchargé.
for dossier in ["wav", "rttm", "uem", "lists"]:
    (RACINE / dossier).mkdir(parents=True, exist_ok=True)


def convertir(partition_hf, partition, nb_fichiers, total_fichiers):
    uris, heures = [], 0.0
    for n in range(nb_fichiers):
        nom = f"{args.config}/{partition_hf}-{n:05d}-of-{total_fichiers:05d}.parquet"
        chemin = hf_hub_download("diarizers-community/ami", nom, repo_type="dataset",
                                 token=HF_TOKEN, local_dir="/tmp/hf")
        # Lecture réunion par réunion pour ne pas tout charger en mémoire
        for lot in pq.ParquetFile(chemin).iter_batches(batch_size=1):
            ligne = lot.to_pylist()[0]
            audio, sr = sf.read(io.BytesIO(ligne["audio"]["bytes"]), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            assert sr == 16000, f"fréquence inattendue : {sr}"
            chemin_origine = ligne["audio"].get("path") or f"{partition}{n}_{len(uris)}"
            uri = Path(chemin_origine).name.split(".")[0]   # ex. EN2001a
            duree = len(audio) / sr
            sf.write(RACINE / "wav" / f"{uri}.wav", audio, sr, subtype="PCM_16")
            with open(RACINE / "rttm" / f"{uri}.rttm", "w") as f:
                for debut, fin, locuteur in zip(ligne["timestamps_start"], ligne["timestamps_end"], ligne["speakers"]):
                    f.write(f"SPEAKER {uri} 1 {debut:.3f} {fin - debut:.3f} <NA> <NA> {locuteur} <NA> <NA>\n")
            (RACINE / "uem" / f"{uri}.uem").write_text(f"{uri} 1 0.000 {duree:.3f}\n")
            uris.append(uri)
            heures += duree / 3600
        os.remove(chemin)  # libère le disque
        print(f"{nom} converti", flush=True)
    (RACINE / "lists" / f"{partition}.txt").write_text("\n".join(uris) + "\n")
    print(f"{partition} : {len(uris)} réunions, {heures:.1f} h d'audio", flush=True)
    return len(uris), heures


debut = time.time()
nb_train, h_train = convertir("train", "train", NB_FICHIERS_TRAIN, 19)
nb_val, h_val = convertir("validation", "development", NB_FICHIERS_VAL, 3)
print(f"Préparation des données : {(time.time() - debut) / 60:.1f} min")

# Contrôle : une réunion, ses locuteurs, et des temps cohérents avec la durée de l'audio
uri = (RACINE / "lists" / "train.txt").read_text().split()[0]
lignes = (RACINE / "rttm" / f"{uri}.rttm").read_text().splitlines()
duree = sf.info(RACINE / "wav" / f"{uri}.wav").duration
fin_max = max(float(l.split()[3]) + float(l.split()[4]) for l in lignes)
print(uri, f"{duree:.0f} s,", len(lignes), "tours de parole,",
      len({l.split()[7] for l in lignes}), "locuteurs, dernier tour finit à", f"{fin_max:.0f} s")
assert fin_max <= duree + 1, "les temps du RTTM dépassent la durée de l'audio"

# ---------------- 2. Protocole pyannote : où trouver audio, références et listes ----------------
from pyannote.database import FileFinder, registry

yml = f"""
Databases:
  AMI: {RACINE}/wav/{{uri}}.wav

Protocols:
  AMI:
    SpeakerDiarization:
      {args.config}:
        train:
          uri: {RACINE}/lists/train.txt
          annotation: {RACINE}/rttm/{{uri}}.rttm
          annotated: {RACINE}/uem/{{uri}}.uem
        development:
          uri: {RACINE}/lists/development.txt
          annotation: {RACINE}/rttm/{{uri}}.rttm
          annotated: {RACINE}/uem/{{uri}}.uem
"""
(RACINE / "database.yml").write_text(yml)
registry.load_database(str(RACINE / "database.yml"))
protocole = registry.get_protocol(f"AMI.SpeakerDiarization.{args.config}", preprocessors={"audio": FileFinder()})

# ---------------- 3. Modèle et tâche ----------------
# On part des poids de segmentation de community-1 (c'est ça, le fine-tuning), avec exactement
# les mêmes réglages : fenêtres de 10 s, 3 locuteurs max par fenêtre, 2 en même temps.
from pyannote.audio import Model
from pyannote.audio.tasks import SpeakerDiarization

modele = Model.from_pretrained("pyannote/speaker-diarization-community-1", subfolder="segmentation", token=HF_TOKEN)
specs = modele.specifications
tache = SpeakerDiarization(protocole, duration=specs.duration, max_speakers_per_chunk=len(specs.classes),
                           max_speakers_per_frame=specs.powerset_max_classes, batch_size=32, num_workers=4)
modele.task = tache


def configure_optimizers():
    # taux 10x plus petit que le défaut (1e-3) pour ne pas « casser » ce que le modèle sait déjà
    return torch.optim.Adam(modele.parameters(), lr=args.lr)


modele.configure_optimizers = configure_optimizers

# ---------------- 4. Score AVANT entraînement (DER par fenêtre de 10 s, validation AMI) ----------------
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

# inference_mode=False : sinon les compteurs du score sont figés et l'entraînement qui suit plante
avant = Trainer(accelerator="gpu", devices=1, logger=False, enable_checkpointing=False,
                enable_progress_bar=False, inference_mode=False).validate(modele, verbose=False)[0]
nom_metrique, sens = tache.val_monitor
print(f"AVANT : {nom_metrique} = {avant[nom_metrique]:.4f}", flush=True)

# ---------------- 5. Entraînement ----------------
debut = time.time()


class Afficher(Callback):
    """Une ligne par époque dans les logs (les barres de progression sont illisibles dans un log)."""

    def on_validation_end(self, trainer, pl_module):   # même moment que ModelCheckpoint
        valeur = trainer.callback_metrics.get(nom_metrique)
        if trainer.sanity_checking or valeur is None:   # sanity check = 2 lots de validation avant de commencer
            return
        print(f"époque {trainer.current_epoch} : {nom_metrique} = {float(valeur):.4f} "
              f"({(time.time() - debut) / 60:.0f} min écoulées)", flush=True)


sauvegarde = ModelCheckpoint(dirpath=str(SORTIE / "checkpoints"), monitor=nom_metrique, mode=sens,
                             save_top_k=1, save_last=True, filename="meilleur-{epoch:02d}")
arret = EarlyStopping(monitor=nom_metrique, mode=sens, patience=5)   # 5 époques sans progrès -> stop
journal = CSVLogger(str(SORTIE), name="logs")
entraineur = Trainer(accelerator="gpu", devices=1, max_epochs=MAX_EPOQUES, max_time={"hours": args.max_heures},
                     callbacks=[sauvegarde, arret, Afficher()], logger=journal, enable_progress_bar=False)
entraineur.fit(modele)
duree_entrainement = time.time() - debut
epoques = entraineur.current_epoch
print(f"Entraînement : {duree_entrainement / 60:.1f} min, {epoques} époque(s), "
      f"soit {duree_entrainement / 60 / max(epoques, 1):.1f} min par époque")

# ---------------- 6. Courbe, meilleur modèle, résumé ----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

metriques = pd.read_csv(Path(journal.log_dir) / "metrics.csv")
val = metriques.dropna(subset=[nom_metrique])[["epoch", nom_metrique]]
print(val.to_string(index=False))
plt.figure(figsize=(6, 3.5))
plt.axhline(avant[nom_metrique], color="grey", linestyle="--", label="avant fine-tuning")
plt.plot(val["epoch"], val[nom_metrique], marker="o", label="après chaque époque")
plt.xlabel("époque"); plt.ylabel(nom_metrique + " (validation AMI, par fenêtre)")
plt.title(f"Fine-tuning segmentation sur AMI {args.config}"); plt.legend(); plt.tight_layout()
plt.savefig(SORTIE / "courbe_validation.png", dpi=150)

meilleur = Path(sauvegarde.best_model_path)
shutil.copy(meilleur, SORTIE / "segmentation_finetunee.ckpt")
resume = {
    "mode": args.mode, "config": args.config, "lr": args.lr,
    "reunions_train": nb_train, "heures_train": round(h_train, 1),
    "reunions_val": nb_val, "heures_val": round(h_val, 1),
    "epoques": epoques, "duree_min": round(duree_entrainement / 60, 1),
    "metrique": nom_metrique, "avant": round(avant[nom_metrique], 4),
    "meilleur": round(float(sauvegarde.best_model_score), 4), "fichier": meilleur.name,
}
(SORTIE / "resume.json").write_text(json.dumps(resume, indent=2, ensure_ascii=False))
print(json.dumps(resume, indent=2, ensure_ascii=False))
