"""
Every other script imports DATA_DIR and DOCS_DIR from here instead of
hardcoding a path. Paths are resolved relative to THIS FILE's own location
on disk, not a fixed absolute path -- so the project works no matter what
OS you're on or where you unzip the folder.
"""
import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
DOCS_DIR = os.path.join(ROOT_DIR, "docs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)
