import numpy as np
from logic.hashlife import *
from patterns.patterns import BUILTIN_PATTERNS
from simulation.sim import Sim, RATES, DEFAULT_RATE_IDX
from logic.mirror import Mirror
import pygame
import math
import time
PANEL_H = 92


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