from pathlib import Path

from urban_monitor.cli import build_parser


def test_readme_is_short_chinese_plug_and_train_guide():
    text = Path("README.md").read_text(encoding="utf-8")
    assert len(text.splitlines()) <= 250
    assert "城市更新遥感监测平台" in text
    assert "python run.py all --aoi beijing_demo" in text
    assert "data/beijing_demo/" in text
    assert "training/checkpoints/best.pth" in text


def test_documented_commands_exist_in_parser():
    parser = build_parser()
    parser.parse_args(["doctor"])
    parser.parse_args(["prepare", "--aoi", "beijing_demo"])
    parser.parse_args(["train", "--aoi", "beijing_demo", "--resume"])
    parser.parse_args(["predict", "--aoi", "beijing_demo"])
    parser.parse_args(["all", "--aoi", "beijing_demo", "--force"])
    parser.parse_args(["demo"])
