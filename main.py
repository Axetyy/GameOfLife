"""
Pygame viewer for the HashLife engine.

    python3 viewer.py patterns/TetrisOTCAMP.mc.gz        # the Quest-for-Tetris computer
    python3 viewer.py my_pattern.rle                      # any RLE / MacroCell file (.gz ok)
    python3 viewer.py                                      # built-in Gosper glider gun

Keys / mouse
    mouse wheel      zoom (around the cursor)         left-drag   pan
    SPACE            run / pause                       N or RIGHT  one step (2^j generations)
    UP / DOWN        step size 2^j  (j+1 / j-1)        R           reset to generation 0
    F                fit whole pattern                 H           help on/off
    S                save screenshot (png)             ESC / Q     quit
    click pattern buttons / drag bottom slider to load patterns and change step size

Rendering is fully vectorised with numpy: the quadtree is mirrored into flat arrays and only the visible
nodes are expanded, level by level.  Zoomed in you see individual cells; zoomed far out each pixel shows
the population density of the square of cells it covers, so a 16-million-cell-wide computer stays crisp.
"""

import argparse
import time
import pygame
from logic.hashlife import load_pattern, load_rle
from simulation.sim import DEFAULT_RATE_IDX, RATES,Sim
from patterns.patterns import BUILTIN_PATTERNS
from ui import Viewer

def main():
    ap = argparse.ArgumentParser(description="Fast HashLife Game of Life viewer")
    ap.add_argument("pattern", nargs="?", help=".mc / .rle (optionally .gz); default: Gosper glider gun")
    ap.add_argument("--j", type=int, default=0, help="initial step exponent (2^j generations per step)")
    ap.add_argument("--size", default="1280x800", help="window size, e.g. 1600x900")
    ap.add_argument("--max-nodes", type=int, default=100_000_000, help="garbage-collect above this many nodes")
    ap.add_argument("--rate", type=float, default=RATES[DEFAULT_RATE_IDX],
                    help="playback speed in updates/second while running (0 = unlimited)")
    ap.add_argument("--run", action="store_true", help="start running immediately")
    a = ap.parse_args()

    universe = universe()
    t = time.time()
    root = load_pattern(universe, a.pattern) if a.pattern else load_rle(universe, BUILTIN_PATTERNS["Gosper gun"])
    print(f"loaded in {time.time() - t:.2f}s: universe 2^{universe.LVL[root]} wide, {len(universe):,} nodes, "
          f"population {universe.POP[root]:,}")

    pygame.init()
    w, h = (int(v) for v in a.size.lower().split("x"))
    sim = Sim(universe, root, a.max_nodes)
    sim.j = a.j
    target = a.rate or 1e9
    sim.rate_idx = min(range(len(RATES)), key=lambda i: abs((RATES[i] or 1e9) - target))
    sim.running = a.run
    sim.start()
    v = Viewer(universe, sim, (w, h), selected_pattern=None if a.pattern else "Gosper gun")
    if not a.pattern:
        v.scale = 12.0
    try:
        v.run()
    finally:
        sim.stop()
        pygame.quit()


if __name__ == "__main__":
    main()