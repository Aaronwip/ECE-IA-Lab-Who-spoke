"""Aller-retour RTTM -> étiquettes Audacity -> RTTM : les segments doivent être identiques."""
from rttm_audacity import audacity_vers_rttm, rttm_vers_audacity


def test_aller_retour(tmp_path):
    rttm = tmp_path / "f.rttm"
    rttm.write_text("SPEAKER f 1 0.500 2.000 <NA> <NA> A <NA> <NA>\n"
                    "SPEAKER f 1 2.000 1.250 <NA> <NA> B <NA> <NA>\n", encoding="utf-8")
    rttm_vers_audacity(rttm, tmp_path / "f.txt")
    assert (tmp_path / "f.txt").read_text(encoding="utf-8").splitlines() == ["0.500\t2.500\tA", "2.000\t3.250\tB"]
    audacity_vers_rttm(tmp_path / "f.txt", tmp_path / "retour" / "f.rttm")
    assert (tmp_path / "retour" / "f.rttm").read_text(encoding="utf-8") == rttm.read_text(encoding="utf-8")


def test_espaces_dans_le_nom_et_lignes_de_frequence(tmp_path):
    # Un nom tapé à la main dans Audacity peut contenir des espaces (interdits dans un RTTM)
    (tmp_path / "g.txt").write_text("1.0\t2.0\tAaron W\n\\\t100\t200\n", encoding="utf-8")
    audacity_vers_rttm(tmp_path / "g.txt", tmp_path / "g.rttm")
    assert (tmp_path / "g.rttm").read_text(encoding="utf-8").split()[7] == "Aaron_W"


def test_espaces_au_lieu_des_tabulations(tmp_path):
    # Fichier réenregistré par VS Code : tabulations devenues espaces, fins de ligne Windows
    (tmp_path / "h.txt").write_bytes(b"0.030   31.550  Mathis\r\n4.000   4.500   Aaron\r\n")
    assert audacity_vers_rttm(tmp_path / "h.txt", tmp_path / "h.rttm") == 2
    assert (tmp_path / "h.rttm").read_text(encoding="utf-8").split()[3:5] == ["0.030", "31.520"]


def test_fichier_sans_etiquette(tmp_path):
    # Avant : 0 segment écrit en silence. Maintenant : arrêt avec un message clair.
    import pytest
    (tmp_path / "vide.txt").write_text("\n\\\t100\t200\n", encoding="utf-8")  # ligne vide + ligne de fréquence
    with pytest.raises(SystemExit):
        audacity_vers_rttm(tmp_path / "vide.txt", tmp_path / "vide.rttm")
