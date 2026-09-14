from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_data_has_only_three_top_level_folders():
    entries = sorted(path.name for path in (ROOT / "data").iterdir() if path.is_dir())
    assert entries == ["claims", "kaggle", "policies"]


def test_backend_only_no_frontend_folder():
    assert not (ROOT / "frontend").exists()


def test_no_deprecated_term_in_text_project_files():
    deprecated = "syn" + "thetic"
    extensions = {".py", ".md", ".json", ".yml", ".yaml", ".txt"}
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue
        if deprecated in path.read_text(encoding="utf-8", errors="ignore").lower():
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []
