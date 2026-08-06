from urban_monitor.compat import write_text


def test_compat_write_text_accepts_newline(tmp_path):
    path = tmp_path / "utf8.txt"
    write_text(path, "a\nb\n", newline="")
    assert path.read_text(encoding="utf-8") == "a\nb\n"
