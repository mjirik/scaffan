# import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import subprocess

try:

    __version__ = subprocess.check_output("git describe".split(" "), cwd="..").strip()
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    __version__ = "0.34.1"
"""
Used for scaffold analysis
"""
