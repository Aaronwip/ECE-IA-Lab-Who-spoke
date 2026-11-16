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
