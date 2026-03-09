"""PyInstaller entry point for srclight.

This thin wrapper avoids the 'relative import with no known parent package'
error that occurs when PyInstaller tries to run cli.py directly (which uses
``from . import __version__``).
"""

from srclight.cli import main

if __name__ == "__main__":
    main()
