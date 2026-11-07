import numpy as np

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

