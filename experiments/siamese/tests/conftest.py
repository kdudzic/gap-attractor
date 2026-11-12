import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / "tests"), str(REPO / "experiments" / "context_file"), str(REPO / "src")]
