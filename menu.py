"""Nuke menu: adds bam > bamDeVe to the Nodes toolbar and the main menu bar.

Nuke runs this automatically for every folder on its plugin path, e.g. with
``nuke.pluginAddPath("/path/to/nuke_tools")`` in ``~/.nuke/init.py`` or the
folder on ``NUKE_PATH``. Nothing heavy is imported at startup, so a missing
numpy only shows up when the tool is used.
"""

import os
import sys

import nuke

_NK = "bamDeVe.nk"


def _tool_dir():
    """This folder: from __file__, else the plugin path entry holding the .nk."""
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        for d in nuke.pluginPath() + sys.path:
            if os.path.isfile(os.path.join(d, _NK)):
                return d
    return ""


_DIR = _tool_dir()


def create_bamdeve():
    """Paste the bamDeVe group (connected to the selected node, as nodePaste does)."""
    path = os.path.join(_DIR, _NK)
    if not os.path.isfile(path):
        nuke.message("bamDeVe: %s not found next to menu.py" % _NK)
        return None
    if _DIR not in sys.path:                      # the buttons import vignette_solve
        sys.path.insert(0, _DIR)
    nuke.nodePaste(path)
    return nuke.selectedNode()


for _bar in ("Nodes", "Nuke"):
    nuke.menu(_bar).addMenu("bam").addCommand("bamDeVe", create_bamdeve)
