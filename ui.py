import numpy as np
from logic.hashlife import *
from patterns.patterns import BUILTIN_PATTERNS
from simulation.sim import Sim, RATES, DEFAULT_RATE_IDX
from logic.mirror import Mirror
import pygame
import math
import time

THEMES = [
    dict(name="Ocean", bg=(9, 11, 18), mid=(30, 110, 190), top=(190, 255, 215),
         age=[(255, 255, 255), (120, 255, 220), (40, 150, 230), (60, 60, 170)]),
    dict(name="Ember", bg=(14, 8, 8), mid=(190, 60, 20), top=(255, 225, 140),
         age=[(255, 250, 210), (255, 200, 60), (235, 90, 25), (140, 25, 40)]),
    dict(name="Neon", bg=(10, 6, 20), mid=(130, 40, 200), top=(90, 255, 235),
         age=[(255, 255, 255), (255, 80, 200), (150, 70, 255), (50, 130, 255)]),
]

AGE_CAP = 1 << 40
AGE_LOG_MAX = 24.0

PANEL_BASE = 92 
PANEL_H = 92


@property
def panel_h(self):
    W = self.screen.get_width()
    return PANEL_BASE + self.layout_patterns(W)[1]

def make_palette(theme):
    """Static palette: index 0 = background, 255 = brightest (a live cell)."""
    pal = []
    bg, mid, top = (np.array(theme[k], dtype=float) for k in ("bg", "mid", "top"))
    for i in range(256):
        t = i / 255.0
        c = bg + (mid - bg) * min(1, t * 2) if t < 0.5 else mid + (top - mid) * (t - 0.5) * 2
        pal.append(tuple(int(v) for v in c))
    return pal


def make_age_palette(theme):
    """Age palette: index 0 = background, 1 = newborn ... 255 = oldest."""
    stops = [np.array(s, dtype=float) for s in theme["age"]]
    pal = [tuple(theme["bg"])]
    n = 255
    for i in range(n):
        t = i / (n - 1) * (len(stops) - 1)
        k = min(int(t), len(stops) - 2)
        f = t - k
        c = stops[k] + (stops[k + 1] - stops[k]) * f
        pal.append(tuple(int(v) for v in c))
    return pal


class Viewer:
    def __init__(self, universe, sim, size, selected_pattern=None):
        self.u, self.sim = universe, sim
        self.mir = Mirror(universe)
        self.theme_idx = 0
        self.color_mode = 'age'
        self.age_prev = None
        self._build_palettes()
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Game of Life")
        self.font = pygame.font.Font(None, 22)
        self.cx = self.cy = 0.0
        self.scale = 1.0
        self.show_help = True
        self.last_draw_ms = 0.0
        self.selected_pattern = selected_pattern
        self.pattern_buttons = {}
        self.color_btn = pygame.Rect(0, 0, 0, 0)
        self.theme_btn = pygame.Rect(0, 0, 0, 0)
        self.speed_drag = None
        self.speed_slider_rect = pygame.Rect(150, 0, 300, 14)
        self.rate_slider_rect = pygame.Rect(150, 0, 300, 14)
        self.speed_max_j = 20
        self.fit()

        
    def _build_palettes(self):
        theme = THEMES[self.theme_idx]
        self.static_pal = make_palette(theme)
        self.age_pal = make_age_palette(theme)
    def layout_patterns(self, W):
        """Wrap pattern buttons; returns (rects, extra_height)."""
        x, y = 105, 5
        rects = {}
        for name in BUILTIN_PATTERNS:
            tw = self.font.size(name)[0] + 18
            if x + tw > W - 8 and x > 8:
                x = 8
                y += 32
            rects[name] = pygame.Rect(x, y, tw, 27)
            x += tw + 7
        return rects, y - 5

    def toggle_color_mode(self):
        self.color_mode = 'static' if self.color_mode == 'age' else 'age'
        self.age_prev = None

    def cycle_theme(self):
        self.theme_idx = (self.theme_idx + 1) % len(THEMES)
        self._build_palettes()

    def age_colorize(self, alive_buf, ox, oy, gen):
        alive = alive_buf > 0
        ages = np.full(alive_buf.shape, -1, np.int64)
        ages[alive] = 0
        prev = self.age_prev
        if prev is not None and gen >= prev[2]:
            pa, (pox, poy), pgen = prev
            dg = min(gen - pgen, AGE_CAP)
            bh, bw = alive_buf.shape
            ph, pw = pa.shape
            x0, x1 = max(ox, pox), min(ox + bw, pox + pw)
            y0, y1 = max(oy, poy), min(oy + bh, poy + ph)
            if x1 > x0 and y1 > y0:
                old = pa[y0 - poy:y1 - poy, x0 - pox:x1 - pox]
                cur = ages[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
                both = (old >= 0) & (cur >= 0)
                cur[both] = np.minimum(old[both] + dg, AGE_CAP)
        self.age_prev = (ages, (ox, oy), gen)

        idx = np.zeros(alive_buf.shape, np.uint8)
        a = ages[alive].astype(np.float64)
        idx[alive] = (1 + np.minimum(254.0, np.log2(1.0 + a) * (254.0 / AGE_LOG_MAX))).astype(np.uint8)
        return idx
    def fit(self):
        W, H = self.screen.get_size()
        H = max(1, H - PANEL_H)
        with self.sim.lock:
            self.mir.sync()
            root = self.sim.root
            k = self.u.LVL[root]
            ids, xs, ys = self.mir.frontier(root, max(4, k - 12), -1e30, -1e30, 1e30, 1e30)
            if ids.size == 0:
                return
            sz = 1 << max(4, k - 12)
            x0, x1, y0, y1 = xs.min(), xs.max() + sz, ys.min(), ys.max() + sz

        self.cx, self.cy = (x0 + x1) / 2, (y0 + y1) / 2 - (PANEL_H / 2) / 1.0 * 0  
        self.scale = min(64.0, max(1e-10, 0.92 * min(W / max(1, x1 - x0), H / max(1, y1 - y0))))
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
        self.age_prev = None
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

    def _draw_button(self, rect, text, active=False):
        fill = (38, 105, 155) if active else (35, 43, 58)
        pygame.draw.rect(self.screen, fill, rect, border_radius=5)
        pygame.draw.rect(self.screen, (75, 95, 120), rect, 1, border_radius=5)
        surf = self.font.render(text, True, (245, 248, 255))
        self.screen.blit(surf, (rect.x + (rect.width - surf.get_width()) // 2,
                                rect.y + (rect.height - surf.get_height()) // 2))

    def draw_controls(self):
        """Draw clickable pattern presets, sliders and colour-mode buttons."""
        W, H = self.screen.get_size()
        rects, extra = self.layout_patterns(W)
        ph = PANEL_BASE + extra
        top = H - ph
        ctl = top + extra                      # controls start below the last row of pattern buttons

        panel = pygame.Rect(0, top, W, ph)
        pygame.draw.rect(self.screen, (13, 17, 27), panel)
        pygame.draw.line(self.screen, (55, 70, 90), (0, top), (W, top), 1)

        label = self.font.render("PATTERNS", True, (150, 175, 205))
        self.screen.blit(label, (8, top + 10))

        # layout_patterns returns rects with y relative to the panel top, so offset them
        self.pattern_buttons = {}
        for name, r in rects.items():
            rect = r.move(0, top)
            self.pattern_buttons[name] = rect
            text = self.font.render(name, True, (245, 248, 255))
            fill = (38, 105, 155) if name == self.selected_pattern else (35, 43, 58)
            pygame.draw.rect(self.screen, fill, rect, border_radius=5)
            pygame.draw.rect(self.screen, (75, 95, 120), rect, 1, border_radius=5)
            self.screen.blit(text, (rect.x + 9, rect.y + 5))

        slider_left = self.font.size("STEP SIZE  2^20  (1,048,576 gen/update)")[0] + 24
        slider_right = max(slider_left + 100, min(W - 24, slider_left + 340))

        step_text = self.font.render(
            f"STEP SIZE  2^{self.sim.j}  ({1 << self.sim.j:,} gen/update)", True, (225, 235, 245))
        self.screen.blit(step_text, (8, ctl + 40))
        self.speed_slider_rect = pygame.Rect(slider_left, ctl + 38, slider_right - slider_left, 14)
        self._draw_slider(self.speed_slider_rect, self.sim.j / self.speed_max_j)

        idx = self.sim.rate_idx
        rate_text = self.font.render(f"SPEED  {self.rate_label(idx)}", True, (225, 235, 245))
        self.screen.blit(rate_text, (8, ctl + 66))
        self.rate_slider_rect = pygame.Rect(slider_left, ctl + 64, slider_right - slider_left, 14)
        self._draw_slider(self.rate_slider_rect, idx / (len(RATES) - 1))

        bx = slider_right + 24
        self.color_btn = pygame.Rect(bx, ctl + 34, 210, 24)
        self.theme_btn = pygame.Rect(bx, ctl + 62, 210, 24)
        self._draw_button(self.color_btn,
                        f"COLOR: {'AGE' if self.color_mode == 'age' else 'STATIC'}  (C)",
                        active=self.color_mode == 'age')
        self._draw_button(self.theme_btn, f"PALETTE: {THEMES[self.theme_idx]['name']}  (V)")
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
                if self.color_mode == 'age':
                    buf = self.age_colorize(buf, ox, oy, gen)
                    pal = self.age_pal
                else:
                    self.age_prev = None
                    pal = self.static_pal
                s8 = pygame.Surface((bw, bh), 0, 8)
                s8.set_palette(pal)
                pygame.surfarray.blit_array(s8, buf.T)
                scaled = pygame.transform.scale(s8, (max(1, round(bw * scale)), max(1, round(bh * scale))))
                self.screen.fill(pal[0])
                self.screen.blit(scaled, (-(vx0 - ox) * scale, -(vy0 - oy) * scale))
            else:
                self.age_prev = None
                img, _ = self.render_density(root, vx0, vy0, vx1, vy1, W, H)
                s8 = pygame.Surface((W, H), 0, 8)
                s8.set_palette(self.static_pal)
                pygame.surfarray.blit_array(s8, img.T)
                self.screen.blit(s8, (0, 0))
        self.last_draw_ms = (time.time() - t0) * 1000
        self.hud(gen, pop, lvl, nodes)
        self.draw_controls()

    def hud(self, gen, pop, lvl, nodes):
        sim = self.sim
        status = "COMPUTING..." if sim.busy else ("RUNNING" if sim.running else "paused")
        mode = "cell age" if self.color_mode == 'age' else "static"
        if self.color_mode == 'age' and self.scale < 0.5:
            mode += " (zoom in to see ages)"
        lines = [
            f"generation  {gen:,}",
            f"population  {pop:,}",
            f"step        2^{sim.j} = {1 << sim.j:,} generations   [{status}]   speed {self.rate_label(sim.rate_idx)}",
            f"universe    {2 ** lvl:,} x {2 ** lvl:,} cells   zoom {self.scale:.3g} px/cell",
            f"hashlife    {nodes:,} nodes   last step {sim.last_dt:.2f}s   draw {self.last_draw_ms:.0f} ms"
            + (f"   gc x{sim.gcs}" if sim.gcs else ""),
            f"colors      {mode}   palette {THEMES[self.theme_idx]['name']}",
        ]
        if self.show_help:
            lines += ["", "SPACE run/pause   N step   UP/DOWN step size   [ ] playback speed   wheel zoom   drag pan",
                      "C age/static colors   V next palette   F fit   R reset   S screenshot   H hide help   ESC quit",
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
                        self.age_prev = None
                        self.sim.request_reset()
                    elif k == pygame.K_f:
                        self.fit()
                    elif k == pygame.K_h:
                        self.show_help = not self.show_help
                    elif k == pygame.K_c:
                        self.toggle_color_mode()
                    elif k == pygame.K_v:
                        self.cycle_theme()
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
                    elif self.color_btn.collidepoint(e.pos):
                        self.toggle_color_mode()
                        drag = None
                    elif self.theme_btn.collidepoint(e.pos):
                        self.cycle_theme()
                        drag = None
                    elif self.speed_slider_rect.inflate(0, 18).collidepoint(e.pos):
                        self.speed_drag = 'step'
                        self.set_speed_from_x(e.pos[0])
                    elif self.rate_slider_rect.inflate(0, 18).collidepoint(e.pos):
                        self.speed_drag = 'rate'
                        self.set_rate_from_x(e.pos[0])
                    elif not in_panel:        
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