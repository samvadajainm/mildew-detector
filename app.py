"""Render entry point — thin wrapper that runs serve_example.py from the
draft3-yoloSeg subfolder. Render's start command: gunicorn app:app"""
import sys
import os

# Add the draft3 folder to Python's module search path so plantseg imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "draft3-yoloSeg"))

# Override the WEIGHTS path since plantseg.py looks for model/ relative to itself
import plantseg
plantseg.WEIGHTS = os.path.join(os.path.dirname(__file__),
                                "draft3-yoloSeg", "model", "plantseg.pt")

from serve_example import app  # noqa: E402, F401
