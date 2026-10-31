"""Tests du vérificateur de fuite (scripts/verifier_splits.py) sur un faux AMI miniature."""
import verifier_splits


def faux_ami(racine, partitions):
    """partitions = {split: {réunion: [locuteurs]}}. Crée listes + RTTM comme le vrai script."""
    (racine / "lists").mkdir(parents=True)
    for split in verifier_splits.SPLITS:
        reunions = partitions.get(split, {})
        locuteurs = sorted({l for locs in reunions.values() for l in locs})
        (racine / "lists" / f"{split}.meetings.txt").write_text("\n".join(reunions))
        (racine / "lists" / f"{split}.speakers.txt").write_text("\n".join(locuteurs))
        (racine / split / "rttm").mkdir(parents=True)
        for reunion, locs in reunions.items():
            lignes = [f"SPEAKER {reunion} 1 {i}.0 1.0 <NA> <NA> {l} <NA> <NA>" for i, l in enumerate(locs)]
            (racine / split / "rttm" / f"{reunion}.rttm").write_text("\n".join(lignes))
    return racine


def test_partitions_propres(tmp_path):
    ami = faux_ami(tmp_path, {
        "train": {"R1": ["A", "B"]},
        "dev": {"R2": ["C"]},
        "test": {"R3": ["D", "E"]},
    })
    erreurs, _ = verifier_splits.verifier_ami(ami)
    assert erreurs == []


def test_reunion_partagee_detectee(tmp_path):
    ami = faux_ami(tmp_path, {"train": {"R1": ["A"]}, "test": {"R1": ["A"]}})
    erreurs, _ = verifier_splits.verifier_ami(ami)
    assert any("réunions en commun" in e for e in erreurs)


def test_locuteur_partage_detecte(tmp_path):
    # Réunions différentes mais même personne (B) en train et en test : c'est une fuite.
    ami = faux_ami(tmp_path, {"train": {"R1": ["A", "B"]}, "test": {"R2": ["B", "C"]}})
    erreurs, _ = verifier_splits.verifier_ami(ami)
    assert any("locuteurs (RTTM) en commun" in e for e in erreurs)


def test_fichier_mal_range_detecte(tmp_path):
    ami = faux_ami(tmp_path, {"train": {"R1": ["A"]}, "test": {"R2": ["B"]}})
    # Un audio de test posé par erreur dans train/wav
    (ami / "train" / "wav").mkdir()
    (ami / "train" / "wav" / "R2.wav").write_bytes(b"")
    erreurs, _ = verifier_splits.verifier_ami(ami)
    assert any("train/wav/R2.wav" in e for e in erreurs)
