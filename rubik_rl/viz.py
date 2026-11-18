"""Open cube states in viz/cube_viewer.html."""
from __future__ import annotations

import webbrowser
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .cube import cube as C

VIEWER = Path(__file__).resolve().parent.parent / "viz" / "cube_viewer.html"


def viewer_url(state: np.ndarray | str | None = None, moves=None) -> str:
    """file:// URL that opens the viewer at `state` (array or facelet string) and then
    animates `moves` (move indices, names, or a "R U R'" string) on top of it."""
    parts = []
    if state is not None:
        s = state if isinstance(state, str) else C.to_string(state)
        parts.append("state=" + s)
    if moves is not None:
        if isinstance(moves, str):
            moves = C.parse_moves(moves)
        names = [m if isinstance(m, str) else C.MOVE_NAMES[int(m)] for m in moves]
        parts.append("moves=" + quote("_".join(names), safe="_"))
    url = VIEWER.as_uri()
    return url + ("#" + "&".join(parts) if parts else "")


def show(state=None, moves=None) -> str:
    """Open the viewer in the default browser. Returns the URL."""
    url = viewer_url(state, moves)
    webbrowser.open(url)
    return url
