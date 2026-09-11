"""Run from any working directory after installing the package."""
from pathlib import Path
import sys

from traj_course.cli import main

if __name__ == "__main__":
    main(["--root", str(Path(__file__).resolve().parents[1]), "run", *sys.argv[1:]])
