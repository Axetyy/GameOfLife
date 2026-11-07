
import threading
import time
import traceback

from logic.hashlife import Aborted

DEFAULT_RATE_IDX = 3 
RATES = [1, 2, 5, 10, 20, 30, 60, 120, 0]

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
