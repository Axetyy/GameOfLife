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
import math
import sys
import threading
import time
import traceback

import numpy as np
import pygame

from hashlife import Universe, Aborted, load_pattern, load_rle, GOSPER_GUN


PANEL_H = 92

# Updates per second while running (0 = as fast as the CPU allows).
RATES = [1, 2, 5, 10, 20, 30, 60, 120, 0]
DEFAULT_RATE_IDX = 3   # 10 updates / second

# Built-in Life patterns (all use the standard B3/S23 rule).
BUILTIN_PATTERNS = {
    "Gosper gun": """x = 36, y = 9, rule = B3/S23
24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4b
obo$10bo5bo7bo$11bo3bo$12b2o!""",
    "Glider": """x = 3, y = 3, rule = B3/S23
bo$2bo$3o!""",
    "R-pentomino": """x = 3, y = 3, rule = B3/S23
b2o$2o$bo!""",
    "Acorn": """x = 7, y = 3, rule = B3/S23
b2o$4bo$2bo3b3o!""",
    "Pulsar": """x = 13, y = 13, rule = B3/S23
2b3o3b3o$13b$bo2bo3bo2bo$bo2bo3bo2bo$bo2bo3bo2bo$2b3o3b3o$13b$2b3o3b3o$bo2bo3bo2bo$bo2bo3bo2bo$bo2bo3bo2bo$13b$2b3o3b3o!""",
    "LW spaceship": """x = 5, y = 4, rule = B3/S23
bo2bo$o4b$o3bo$4o!""",
}


class Sim(threading.Thread):
    def __init__(self, universe, root, max_nodes):
        super().__init__(daemon=True)
        self.u, self.root, self.init_root = universe, root, root
        self.gen = 0
        self.lock = threading.RLock()
        self.running = False
        self.once = False
        self.j = 0
        self.quit = False
        self.busy = False
        self.reset_req = False
        self.last_dt = 0.0
        self.max_nodes = max_nodes
        self.gcs = 0
        self.rate_idx = DEFAULT_RATE_IDX

    def request_reset(self):
        self.reset_req = True
        self.u.abort = True

    def stop(self):
        self.quit = True
        self.u.abort = True

    def run(self):
        u = self.u
        while not self.quit:
            if self.reset_req:
                with self.lock:
                    self.root, self.gen = self.init_root, 0
                    self.reset_req = False
                    u.abort = False
                continue
            if not (self.running or self.once):
                time.sleep(0.01)
                continue
            self.once = False
            j = self.j
            self.busy = True
            t0 = time.time()
            try:
                new = u.advance(self.root, j)
            except Aborted:
                if not self.quit and not self.reset_req:
                    u.abort = False
                continue
            except Exception:
                # Never leave the UI waiting on a dead/stuck worker.
                traceback.print_exc()
                self.running = False
                u.abort = False
                continue
            finally:
                self.busy = False
            with self.lock:
                self.root = new
                self.gen += 1 << j
                self.last_dt = time.time() - t0
            if len(u) > self.max_nodes:
                print(f"Warning: universe has {len(u)} nodes, exceeding the limit of {self.max_nodes}.")
                print("         Consider increasing the limit with --max-nodes.")
                with self.lock:
                    self.root, self.init_root = u.compact([self.root, self.init_root])
                    self.gcs += 1
                    self.max_nodes = max(self.max_nodes, 2 * len(u))
            # Throttle: while running, wait so we don't exceed the chosen updates/second.
            while (not self.quit and not self.reset_req and self.running and not self.once):
                rate = RATES[self.rate_idx]
                if not rate or time.time() >= t0 + 1.0 / rate:
                    break
                time.sleep(0.005)


class Mirror:
    def __init__(self, uni):
        self.u = uni
        self.epoch = -1
        self.m = 0
        self.cap = 0
        self.NWa = self.NEa = self.SWa = self.SEa = self.POPa = self.BITSa = None

    def _alloc(self, cap):
        def grow(old, dtype):
            new = np.zeros(cap, dtype)
            if old is not None:
                new[:self.m] = old[:self.m]
            return new
        old = (self.NWa, self.NEa, self.SWa, self.SEa, self.POPa, self.BITSa)
        dt = (np.int64, np.int64, np.int64, np.int64, np.float64, np.uint64)
        self.NWa, self.NEa, self.SWa, self.SEa, self.POPa, self.BITSa = [grow(o, d) for o, d in zip(old, dt)]
        self.cap = cap

    def sync(self):
        u = self.u
        if self.epoch != u.epoch:
            self.epoch, self.m = u.epoch, 0
            self.NWa = self.NEa = self.SWa = self.SEa = self.POPa = self.BITSa = None
            self._alloc(1 << 16)
        n = min(len(u.NW), len(u.NE), len(u.SW), len(u.SE), len(u.POP), len(u.BITS), len(u.LVL))
        if n > self.m:
            if n > self.cap:
                self._alloc(max(n, self.cap * 2))
            m = self.m
            self.NWa[m:n] = u.NW[m:n]
            self.NEa[m:n] = u.NE[m:n]
            self.SWa[m:n] = u.SW[m:n]
            self.SEa[m:n] = u.SE[m:n]
            self.POPa[m:n] = np.array(u.POP[m:n], dtype=np.float64)
            self.BITSa[m:n] = np.array(u.BITS[m:n], dtype=np.uint64)
            self.m = n

    def frontier(self, root, stop, vx0, vy0, vx1, vy1):
        """All non-empty level-`stop` nodes intersecting the viewport: (ids, x0, y0) arrays."""
        k = self.u.LVL[root]
        half = 1 << (k - 1)
        ids = np.array([root], np.int64)
        xs = np.array([-half], np.int64)
        ys = xs.copy()
        lvl = k
        while lvl > stop and ids.size:
            h = 1 << (lvl - 1)
            ids = np.concatenate((self.NWa[ids], self.NEa[ids], self.SWa[ids], self.SEa[ids]))
            xs = np.concatenate((xs, xs + h, xs, xs + h))
            ys = np.concatenate((ys, ys, ys + h, ys + h))
            lvl -= 1
            sz = 1 << lvl
            keep = (self.POPa[ids] > 0) & (xs + sz > vx0) & (xs < vx1) & (ys + sz > vy0) & (ys < vy1)
            ids, xs, ys = ids[keep], xs[keep], ys[keep]
        return ids, xs, ys


def make_palette():
    pal = []
    bg, mid, top = np.array((9, 11, 18)), np.array((30, 110, 190)), np.array((190, 255, 215))
    for i in range(256):
        t = i / 255.0
        c = bg + (mid - bg) * min(1, t * 2) if t < 0.5 else mid + (top - mid) * (t - 0.5) * 2
        pal.append(tuple(int(v) for v in c))
    return pal


class Viewer:
    def __init__(self, universe, sim, size, selected_pattern=None):
        self.u, self.sim = universe, sim
        self.mir = Mirror(universe)
        self.pal = make_palette()
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Game of Life")
        self.font = pygame.font.Font(None, 22)
        self.cx = self.cy = 0.0
        self.scale = 1.0
        self.show_help = True
        self.last_draw_ms = 0.0
        self.selected_pattern = selected_pattern
        self.pattern_buttons = {}
        self.speed_drag = None      # None, 'step' or 'rate'
        self.speed_slider_rect = pygame.Rect(150, 0, 300, 14)
        self.rate_slider_rect = pygame.Rect(150, 0, 300, 14)
        self.speed_max_j = 20
        self.fit()

    def fit(self):
        W, H = self.screen.get_size()
        H = max(1, H - PANEL_H)          # don't fit underneath the control panel
        with self.sim.lock:
            self.mir.sync()
            root = self.sim.root
            k = self.u.LVL[root]
            ids, xs, ys = self.mir.frontier(root, max(4, k - 12), -1e30, -1e30, 1e30, 1e30)
            if ids.size == 0:
                return
            sz = 1 << max(4, k - 12)
            x0, x1, y0, y1 = xs.min(), xs.max() + sz, ys.min(), ys.max() + sz

        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2 - (PANEL_H / 2) / 1.0 * 0  # recentred below
        self.scale = min(64.0, max(1e-10, 0.92 * min(W / max(1, x1 - x0), H / max(1, y1 - y0))))
        # shift so the pattern is centred in the area above the panel
        self.cy += (PANEL_H / 2) / self.scale

    def zoom_at(self, mx, my, factor):
        W, H = self.screen.get_size()
        wx = self.cx + (mx - W / 2) / self.scale
        wy = self.cy + (my - H / 2) / self.scale
        self.scale = min(64.0, max(1e-10, self.scale * factor))
        self.cx = wx - (mx - W / 2) / self.scale
        self.cy = wy - (my - H / 2) / self.scale

    def render_cells(self, root, vx0, vy0, vx1, vy1):
        ox, oy = int(math.floor(vx0)) - 8, int(math.floor(vy0)) - 8
        bw, bh = int(math.ceil(vx1)) - ox + 9, int(math.ceil(vy1)) - oy + 9
        buf = np.zeros((bh, bw), np.uint8)
        ids, xs, ys = self.mir.frontier(root, 3, vx0, vy0, vx1, vy1)
        if ids.size:
            bits = np.ascontiguousarray(self.mir.BITSa[ids])
            ub = np.unpackbits(bits.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little").reshape(-1, 8, 8)
            i, yy, xx = np.nonzero(ub)
            buf[ys[i] - oy + yy, xs[i] - ox + xx] = 255
        return buf, (bw, bh), (ox, oy)

    def render_density(self, root, vx0, vy0, vx1, vy1, W, H):
        scale = self.scale
        k = self.u.LVL[root]
        # FIX: never pick a block level above the root, or the root gets drawn as if it were 2^L wide.
        L = min(k, max(4, int(math.floor(math.log2(1.0 / scale)))))
        ids, xs, ys = self.mir.frontier(root, L, vx0, vy0, vx1, vy1)
        img = np.zeros((H, W), np.float64)
        if ids.size == 0:
            return img.astype(np.uint8), L
        pop = self.mir.POPa[ids]
        px = np.floor((xs - vx0) * scale).astype(np.int64)
        py = np.floor((ys - vy0) * scale).astype(np.int64)
        size_px = (1 << L) * scale
        if size_px <= 1.0:
            ok = (px >= 0) & (px < W) & (py >= 0) & (py < H)
            flat = py[ok] * W + px[ok]
            acc = np.bincount(flat, weights=pop[ok], minlength=W * H).reshape(H, W)
            dens = acc * scale * scale
        else:
            w = int(math.ceil(size_px))
            d = pop / float(4 ** L)
            dens = img
            for dy in range(w):
                for dx in range(w):
                    qx, qy = px + dx, py + dy
                    ok = (qx >= 0) & (qx < W) & (qy >= 0) & (qy < H)
                    dens[qy[ok], qx[ok]] = d[ok]
        b = 1.0 - np.exp(-dens * 55.0)
        out = np.where(dens > 0, np.maximum(b, 0.22), 0.0)
        return (out * 255).astype(np.uint8), L

    def load_builtin_pattern(self, name):
        """Safely replace the current pattern after stopping any in-flight HashLife step."""
        if name not in BUILTIN_PATTERNS:
            return
        self.sim.running = False
        self.sim.once = False
        self.sim.request_reset()
        # Let the worker acknowledge the abort/reset before mutating its Universe.
        # FIX: bounded wait, and bail out if the worker thread is dead, so the UI can't hang forever.
        deadline = time.time() + 10.0
        while (self.sim.busy or self.sim.reset_req) and self.sim.is_alive() and time.time() < deadline:
            pygame.time.wait(1)
        root = load_rle(self.u, BUILTIN_PATTERNS[name])
        with self.sim.lock:
            self.sim.root = root
            self.sim.init_root = root
            self.sim.gen = 0
            self.sim.reset_req = False
            self.u.abort = False
        self.selected_pattern = name
        self.fit()

    def set_speed_from_x(self, x):
        rect = self.speed_slider_rect
        fraction = (x - rect.left) / max(1, rect.width)
        self.sim.j = max(0, min(self.speed_max_j, round(fraction * self.speed_max_j)))

    def set_rate_from_x(self, x):
        rect = self.rate_slider_rect
        fraction = (x - rect.left) / max(1, rect.width)
        n = len(RATES) - 1
        self.sim.rate_idx = max(0, min(n, round(fraction * n)))

    @staticmethod
    def rate_label(idx):
        r = RATES[idx]
        return "unlimited" if r == 0 else f"{r} updates/s"

    def _draw_slider(self, rect, fraction):
        pygame.draw.line(self.screen, (80, 100, 125), (rect.left, rect.centery),
                         (rect.right, rect.centery), 4)
        knob_x = int(rect.left + min(1.0, max(0.0, fraction)) * rect.width)
        pygame.draw.circle(self.screen, (100, 195, 245), (knob_x, rect.centery), 8)
        pygame.draw.circle(self.screen, (235, 245, 255), (knob_x, rect.centery), 3)

    def draw_controls(self):
        """Draw clickable pattern presets, a step-size slider and a playback-speed slider."""
        W, H = self.screen.get_size()
        top = H - PANEL_H
        panel = pygame.Rect(0, top, W, PANEL_H)
        pygame.draw.rect(self.screen, (13, 17, 27), panel)
        pygame.draw.line(self.screen, (55, 70, 90), (0, top), (W, top), 1)

        label = self.font.render("PATTERNS", True, (150, 175, 205))
        self.screen.blit(label, (8, top + 10))
        x = 105
        self.pattern_buttons = {}
        for name in BUILTIN_PATTERNS:
            text = self.font.render(name, True, (245, 248, 255))
            bw = text.get_width() + 18
            rect = pygame.Rect(x, top + 5, bw, 27)
            fill = (38, 105, 155) if name == self.selected_pattern else (35, 43, 58)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, (75, 95, 120), rect, 1, border_radius=5)
            self.screen.blit(text, (x + 9, top + 10))
            self.pattern_buttons[name] = rect
            x += bw + 7

        slider_left = self.font.size("STEP SIZE  2^20  (1,048,576 gen/update)")[0] + 24
        slider_right = max(slider_left + 100, min(W - 24, slider_left + 340))

        step_text = self.font.render(
            f"STEP SIZE  2^{self.sim.j}  ({1 << self.sim.j:,} gen/update)", True, (225, 235, 245))
        self.screen.blit(step_text, (8, top + 40))
        self.speed_slider_rect = pygame.Rect(slider_left, top + 38, slider_right - slider_left, 14)
        self._draw_slider(self.speed_slider_rect, self.sim.j / self.speed_max_j)

        idx = self.sim.rate_idx
        rate_text = self.font.render(f"SPEED  {self.rate_label(idx)}", True, (225, 235, 245))
        self.screen.blit(rate_text, (8, top + 66))
        self.rate_slider_rect = pygame.Rect(slider_left, top + 64, slider_right - slider_left, 14)
        self._draw_slider(self.rate_slider_rect, idx / (len(RATES) - 1))

    def draw(self):
        t0 = time.time()
        W, H = self.screen.get_size()
        scale = self.scale
        vx0, vx1 = self.cx - W / (2 * scale), self.cx + W / (2 * scale)
        vy0, vy1 = self.cy - H / (2 * scale), self.cy + H / (2 * scale)
        with self.sim.lock:
            self.mir.sync()
            root, gen = self.sim.root, self.sim.gen
            pop, lvl, nodes = self.u.POP[root], self.u.LVL[root], len(self.u)
            if scale >= 0.5:
                buf, (bw, bh), (ox, oy) = self.render_cells(root, vx0, vy0, vx1, vy1)
                s8 = pygame.Surface((bw, bh), 0, 8)
                s8.set_palette(self.pal)
                pygame.surfarray.blit_array(s8, buf.T)
                scaled = pygame.transform.scale(s8, (max(1, round(bw * scale)), max(1, round(bh * scale))))
                self.screen.fill(self.pal[0])
                self.screen.blit(scaled, (-(vx0 - ox) * scale, -(vy0 - oy) * scale))
            else:
                img, _ = self.render_density(root, vx0, vy0, vx1, vy1, W, H)
                s8 = pygame.Surface((W, H), 0, 8)
                s8.set_palette(self.pal)
                pygame.surfarray.blit_array(s8, img.T)
                self.screen.blit(s8, (0, 0))
        self.last_draw_ms = (time.time() - t0) * 1000
        self.hud(gen, pop, lvl, nodes)
        self.draw_controls()

    def hud(self, gen, pop, lvl, nodes):
        sim = self.sim
        status = "COMPUTING..." if sim.busy else ("RUNNING" if sim.running else "paused")
        lines = [
            f"generation  {gen:,}",
            f"population  {pop:,}",
            f"step        2^{sim.j} = {1 << sim.j:,} generations   [{status}]   speed {self.rate_label(sim.rate_idx)}",
            f"universe    {2 ** lvl:,} x {2 ** lvl:,} cells   zoom {self.scale:.3g} px/cell",
            f"hashlife    {nodes:,} nodes   last step {sim.last_dt:.2f}s   draw {self.last_draw_ms:.0f} ms"
            + (f"   gc x{sim.gcs}" if sim.gcs else ""),
        ]
        if self.show_help:
            lines += ["", "SPACE run/pause   N step   UP/DOWN step size   [ ] playback speed   wheel zoom   drag pan",
                      "F fit   R reset   S screenshot   H hide help   ESC quit",
                      "Click a pattern below to load it; sliders change step size and playback speed."]
        y = 6
        for ln in lines:
            surf = self.font.render(ln, True, (225, 235, 245), (9, 11, 18))
            self.screen.blit(surf, (8, y))
            y += surf.get_height() + 1

    def run(self):
        clock = pygame.time.Clock()
        drag = None
        while True:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    return
                if e.type == pygame.KEYDOWN:
                    k = e.key
                    if k in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    elif k == pygame.K_SPACE:
                        self.sim.running = not self.sim.running
                    elif k in (pygame.K_n, pygame.K_RIGHT):
                        self.sim.once = True
                    elif k in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                        self.sim.j = min(60, self.sim.j + 1)
                    elif k in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
                        self.sim.j = max(0, self.sim.j - 1)
                    elif k == pygame.K_LEFTBRACKET:
                        self.sim.rate_idx = max(0, self.sim.rate_idx - 1)
                    elif k == pygame.K_RIGHTBRACKET:
                        self.sim.rate_idx = min(len(RATES) - 1, self.sim.rate_idx + 1)
                    elif k == pygame.K_r:
                        self.sim.running = False
                        self.sim.request_reset()
                    elif k == pygame.K_f:
                        self.fit()
                    elif k == pygame.K_h:
                        self.show_help = not self.show_help
                    elif k == pygame.K_s:
                        name = time.strftime("life_%Y%m%d_%H%M%S.png")
                        pygame.image.save(self.screen, name)
                        print("saved", name)
                elif e.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    self.zoom_at(mx, my, 1.25 ** e.y)
                elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    in_panel = e.pos[1] >= self.screen.get_height() - PANEL_H
                    clicked_pattern = next(
                        (name for name, rect in self.pattern_buttons.items() if rect.collidepoint(e.pos)),
                        None)
                    if clicked_pattern is not None:
                        self.load_builtin_pattern(clicked_pattern)
                        drag = None
                    elif self.speed_slider_rect.inflate(0, 18).collidepoint(e.pos):
                        self.speed_drag = 'step'
                        self.set_speed_from_x(e.pos[0])
                    elif self.rate_slider_rect.inflate(0, 18).collidepoint(e.pos):
                        self.speed_drag = 'rate'
                        self.set_rate_from_x(e.pos[0])
                    elif not in_panel:          # FIX: clicks on the panel no longer start a pan
                        drag = e.pos
                elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                    drag = None
                    self.speed_drag = None
                elif e.type == pygame.MOUSEMOTION:
                    if self.speed_drag == 'step':
                        self.set_speed_from_x(e.pos[0])
                    elif self.speed_drag == 'rate':
                        self.set_rate_from_x(e.pos[0])
                    elif drag is not None:
                        self.cx -= (e.pos[0] - drag[0]) / self.scale
                        self.cy -= (e.pos[1] - drag[1]) / self.scale
                        drag = e.pos
            self.draw()
            pygame.display.flip()
            clock.tick(30)


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

    universe = Universe()
    t = time.time()
    root = load_pattern(universe, a.pattern) if a.pattern else load_rle(universe, GOSPER_GUN)
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
    # FIX: only highlight "Gosper gun" when that's actually what was loaded.
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