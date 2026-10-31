"""Tests du harnais d'évaluation (eval/score.py) sur de petits RTTM construits à la main.

Les valeurs attendues sont calculées à la main dans les commentaires : si un test casse,
c'est soit le code, soit notre compréhension de la métrique qui est fausse.
"""
import pytest

import score


def ecrire_rttm(chemin, segments):
    """segments = liste de (début, fin, locuteur). Écrit un RTTM au format standard."""
    with open(chemin, "w", encoding="utf-8") as f:
        for debut, fin, locuteur in segments:
            f.write(f"SPEAKER {chemin.stem} 1 {debut:.3f} {fin - debut:.3f} <NA> <NA> {locuteur} <NA> <NA>\n")
    return chemin


def scorer_un(tmp_path, ref, hyp, collar=0.25):
    """Écrit ref et hyp dans deux dossiers, les score, renvoie la ligne du fichier."""
    (tmp_path / "ref").mkdir()
    (tmp_path / "hyp").mkdir()
    chemin_ref = ecrire_rttm(tmp_path / "ref" / "f.rttm", ref)
    chemin_hyp = ecrire_rttm(tmp_path / "hyp" / "f.rttm", hyp)
    lignes, _, _ = score.scorer([(chemin_ref, chemin_hyp)], collar)
    return lignes[0]


def test_rttm_identiques_donnent_der_nul(tmp_path):
    segments = [(0, 10, "A"), (8, 15, "B"), (20, 30, "A")]  # avec un chevauchement 8-10
    ligne = scorer_un(tmp_path, segments, segments)
    assert ligne["der_strict"] == pytest.approx(0)
    assert ligne["der_standard"] == pytest.approx(0)
    assert ligne["jer"] == pytest.approx(0)


def test_noms_de_locuteurs_differents_ne_comptent_pas(tmp_path):
    # Le DER cherche la meilleure correspondance A->spk1, B->spk0 : les noms n'ont pas d'importance.
    ref = [(0, 10, "A"), (10, 20, "B")]
    hyp = [(0, 10, "spk1"), (10, 20, "spk0")]
    assert scorer_un(tmp_path, ref, hyp)["der_strict"] == pytest.approx(0)


def test_decalage_connu(tmp_path):
    # Référence : A de 0 à 10, B de 10 à 20 (20 s de parole).
    # Hypothèse : la même chose décalée de +1 s.
    ref = [(0, 10, "A"), (10, 20, "B")]
    hyp = [(1, 11, "A"), (11, 21, "B")]
    ligne = scorer_un(tmp_path, ref, hyp)

    # Strict : 0-1 raté (1 s), 10-11 confusion (ref B, hyp A : 1 s), 20-21 fausse alerte (1 s)
    # DER = 3 / 20 = 15 %
    assert ligne["miss_strict"] == pytest.approx(5)
    assert ligne["conf_strict"] == pytest.approx(5)
    assert ligne["fa_strict"] == pytest.approx(5)
    assert ligne["der_strict"] == pytest.approx(15)

    # Standard (±0,25 s autour des frontières de référence 0, 10 et 20) :
    # parole évaluée = 20 - 0,25 - 0,5 - 0,25 = 19 s
    # chaque erreur passe de 1 s à 0,75 s -> DER = 3 * 0,75 / 19 = 11,84 %
    assert ligne["der_standard"] == pytest.approx(100 * 2.25 / 19)
    assert ligne["miss_standard"] == pytest.approx(100 * 0.75 / 19)


def test_locuteur_manquant(tmp_path):
    # Le système ne détecte pas du tout B (5 s sur 15 s de parole).
    ref = [(0, 10, "A"), (10, 15, "B")]
    hyp = [(0, 10, "A")]
    ligne = scorer_un(tmp_path, ref, hyp)
    assert ligne["miss_strict"] == pytest.approx(100 * 5 / 15)
    assert ligne["fa_strict"] == pytest.approx(0)
    assert ligne["conf_strict"] == pytest.approx(0)
    assert ligne["der_strict"] == pytest.approx(100 * 5 / 15)
    # JER = moyenne par locuteur de référence : A parfait (0), B jamais trouvé (1) -> 50 %
    assert ligne["jer"] == pytest.approx(50)


def test_mode_dossier_moyenne_et_global(tmp_path):
    # Deux fichiers : un parfait de 10 s, un raté à moitié de 30 s.
    (tmp_path / "ref").mkdir()
    (tmp_path / "hyp").mkdir()
    ecrire_rttm(tmp_path / "ref" / "a.rttm", [(0, 10, "A")])
    ecrire_rttm(tmp_path / "hyp" / "a.rttm", [(0, 10, "A")])
    ecrire_rttm(tmp_path / "ref" / "b.rttm", [(0, 30, "A")])
    ecrire_rttm(tmp_path / "hyp" / "b.rttm", [(0, 15, "A")])

    paires = score.apparier(tmp_path / "ref", tmp_path / "hyp")
    lignes, moyenne, globale = score.scorer(paires)
    assert [l["fichier"] for l in lignes] == ["a", "b"]
    # Moyenne simple : (0 % + 50 %) / 2 = 25 %
    assert moyenne["der_strict"] == pytest.approx(25)
    # Global : 15 s ratées / 40 s de parole = 37,5 % (le long fichier pèse plus)
    assert globale["der_strict"] == pytest.approx(37.5)


def test_cli_ecrit_le_csv(tmp_path):
    (tmp_path / "ref").mkdir()
    (tmp_path / "hyp").mkdir()
    ecrire_rttm(tmp_path / "ref" / "a.rttm", [(0, 10, "A")])
    ecrire_rttm(tmp_path / "hyp" / "a.rttm", [(0, 10, "A")])
    sortie = tmp_path / "scores.csv"
    score.main(["--ref", str(tmp_path / "ref"), "--hyp", str(tmp_path / "hyp"), "--csv", str(sortie)])
    contenu = sortie.read_text(encoding="utf-8").splitlines()
    assert contenu[0].startswith("fichier,der_strict")
    assert [l.split(",")[0] for l in contenu[1:]] == ["a", "MOYENNE", "GLOBAL"]


def test_reference_vide_est_ignoree(tmp_path):
    # Un fichier sans parole de référence (comme dhorc dans VoxConverse) ne doit pas faire planter.
    (tmp_path / "ref").mkdir()
    (tmp_path / "hyp").mkdir()
    ecrire_rttm(tmp_path / "ref" / "vide.rttm", [])
    ecrire_rttm(tmp_path / "hyp" / "vide.rttm", [(0, 5, "A")])
    ecrire_rttm(tmp_path / "ref" / "ok.rttm", [(0, 10, "A")])
    ecrire_rttm(tmp_path / "hyp" / "ok.rttm", [(0, 10, "A")])
    lignes, _, _ = score.scorer(score.apparier(tmp_path / "ref", tmp_path / "hyp"))
    assert [l["fichier"] for l in lignes] == ["ok"]
