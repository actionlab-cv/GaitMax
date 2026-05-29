from pathlib import Path


def src_root(st: Path = Path(__file__).resolve()) -> Path:
    for p in st.parents:
        if (p / ".git").exists() or (p / "setup.py").exists() or (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Could not find project root")
