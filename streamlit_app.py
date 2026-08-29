"""Streamlit Community Cloud entrypoint for the hosted demo branch."""

from pathlib import Path
import runpy

APP = Path(__file__).resolve().parent / "demo" / "app.py"
runpy.run_path(str(APP), run_name="__main__")
