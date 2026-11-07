"""
Hashlife from scratch.
"""
import gzip
import re

import numpy as np

M256 = (1 << 256) - 1

class Aborted(Exception):
    pass

def _life256(b):
    """
    Compute the next generation of a 256-bit life board.
    (bit= y*16 + x)
    """
    n1 = (b << 1) & M256
    n2 = b>>1
    n3 = (b << 16) & M256
    n4 = b>>16
    n5 = (b << 17) & M256
    n6 = b >> 17
    n7 = (b << 15) & M256
    n8 = b >> 15
    x = n1 ^ n2
    s1 = x ^ n3
    c1 = (n1 & n2) | ( x & n3)
    x = n4 ^ n5
    s2 = x ^ n6
    c2 = (n4 & n5) | ( x & n6)
    s3 = n7 ^ n8
    c3 = n7 & n8
    x = s1 ^ s2
    ones = x ^ s3
    c4 = (s1 & s2) | ( x & s3)
    x = c1 ^ c2
    t1 = x ^ c3
    u1 = (c1 & c2) | ( x & c3)
    twos = t1 ^ c4
    fours = u1 | (t1 & c4)
    return twos & ~fours & (ones | b) & M256


class Universe:
    """
    “If you wish to make an apple pie from scratch, you must first invent the universe”
        - Carl Sagan, Cosmos
    Hashlife universe
    """
    
    def __init__(self):
        self.epoch = 0
        self.abort = False
        self.reset_tables()
        
    def reset_tables(self):
        self.LVL,self.NW,self.NE,self.SW,self.SE = [],[],[],[],[]
        self.POP,self.BITS,self.RES = [],[],[]
        self.table = {}
        self.leaf_table = {}
        self.empty = {}
        self.slow = {}
        self.slow_j = None
        self.center4 = {}
        self.spread_cache = {}
        self.epoch+=1   
        self.empty[3] = self.leaf(0)
    def __len__(self):
        return len(self.LVL)
    
    def leaf(self,bits):
        n = self.leaf_table.get (bits)
        if n is None:
            n = len(self.LVL)
            self.NW.append(-1)
            self.NE.append(-1)
            self.SW.append(-1)
            self.SE.append(-1)
            self.POP.append(bits.bit_count())
            self.BITS.append(bits)
            self.RES.append(-1)
            self.LVL.append(3)
            self.leaf_table[bits] = n
        return n
    
    def mk(self,nw,ne,sw,se):
        key = (nw,ne,sw,se)
        n = self.table.get(key)
        
        if n is None:
            P = self.POP
            n = len(self.LVL)
            self.NW.append(nw)
            self.NE.append(ne)
            self.SW.append(sw)
            self.SE.append(se)
            self.POP.append(P[nw]+P[ne]+P[sw]+P[se])
            self.BITS.append(0)
            self.RES.append(-1)
            self.LVL.append(self.LVL[nw]+1)
            self.table[key] = n
        return n
    
    def empty_node(self,k):
        e = self.empty.get(k)
        if e is None:
            c = self.empty_node(k-1)
            e = self.mk(c,c,c,c)
            self.empty[k] = e
        return e
    
    def expand(self,n):
        k = self.LVL[n]
        e = self.empty_node(k-1)
        mk = self.mk
        return mk(mk(e, e, e, self.NW[n]), mk(e, e, self.NE[n], e),
                  mk(e, self.SW[n], e, e), mk(self.SE[n], e, e, e))
        
    def padded(self,n):
        """
        True when all live cells sit in the central half of the root square.
        """
        NW,NE,SW,SE,P = self.NW,self.NE,self.SW,self.SE,self.POP
        return P[n] == (P[SE[NW[n]]] + P[SW[NE[n]]] + P[NE[SW[n]]] + P[NW[SE[n]]])
    
    def spread(self,bits):
        v = self.spread_cache.get(bits)
        if v is None:
            v = 0
            for r in range(8):
                v |= ((bits >> (8 * r)) & 255) << (16 * r)
            self.spread_cache[bits] = v
        return v

    def base(self,n,g):
        sp = self.spread
        B = self.BITS
        board = (sp(B[self.NW[n]]) | (sp(B[self.NE[n]]) << 8) |
                 (sp(B[self.SW[n]]) << 128) | (sp(B[self.SE[n]]) << 136))
        for _ in range(g):
            board = _life256(board)
        out = 0
        for r in range(8):
            out |= ((board >> ((r + 4) * 16 + 4)) & 255) << (8 * r)
        return self.leaf(out)
    
    def center(self,n):
        
        if self.LVL[n] == 4:
            c = self.center4.get(n)
            if c is None:
                c = self.center4[n] = self.base(n,0)
            return c
        return self.mk(self.SE[self.NW[n]], self.SW[self.NE[n]], self.NE[self.SW[n]], self.NW[self.SE[n]])    
    def result(self,n,j):
        k = self.LVL[n]
        full = (j == k - 2)
        if full:
            r = self.RES[n]
            if r >=0:
                return r
        else:
            r = self.slow.get(n)
            if r is not None:
                return r
        if self.POP[n] == 0:
            r = self.empty_node(k - 1)
        elif k == 4:
            r = self.base(n,1<<j)
        else:
            if self.abort:
                raise Aborted
            NW, NE, SW, SE, mk, res = self.NW, self.NE, self.SW, self.SE, self.mk, self.result
            a,b,c,d = NW[n], NE[n], SW[n], SE[n]
            n01 = mk(NE[a], NW[b], SE[a], SW[b])
            n10 = mk(SW[a], SE[a], NW[c], NE[c])
            n11 = mk(SE[a], SW[b], NE[c], NW[d])
            n12 = mk(SW[b], SE[b], NW[d], NE[d])
            n21 = mk(NE[c], NW[d], SE[c], SW[d])
            if full:
                j1 = j - 1
                r00 = res(a, j1); r01 = res(n01, j1); r02 = res(b, j1)
                r10 = res(n10, j1); r11 = res(n11, j1); r12 = res(n12, j1)
                r20 = res(c, j1); r21 = res(n21, j1); r22 = res(d, j1)
                r = mk(res(mk(r00, r01, r10, r11), j1), res(mk(r01, r02, r11,r12), j1),res(mk(r10, r11, r20, r21), j1), res(mk(r11, r12, r21,r22), j1))
            else:
                cen = self.center
                r00 = cen(a); r01 = cen(n01); r02 = cen(b)
                r10 = cen(n10); r11 = cen(n11); r12 = cen(n12)
                r20 = cen(c); r21 = cen(n21); r22 = cen(d)
                r = mk(res(mk(r00, r01, r10, r11), j), res(mk(r01, r02, r11, r12), j),
                       res(mk(r10, r11, r20, r21), j), res(mk(r11, r12, r21, r22), j))
        if full:
            self.RES[n] = r
        else:
            self.slow[n] = r
        return r
    
    def advance(self,root,j):
        """Return a new root, 2^j generations later."""
        if self.slow_j != j:
            self.slow = {}
            self.slow_j = j
        need = max(5,j+2)
        while self.LVL[root] < need or not self.padded(root):
            root = self.expand(root)
        return self.result(self.expand(root), j)
    
    def compact(self,roots):
        """Return a list of compacted roots."""
        L, NW, NE, SW, SE, B = self.LVL, self.NW, self.NE, self.SW, self.SE, self.BITS
        memo = {}
        self.reset_tables()
 
        def cp(n):
            r = memo.get(n)
            if r is None:
                r = self.leaf(B[n]) if L[n] == 3 else self.mk(cp(NW[n]), cp(NE[n]), cp(SW[n]), cp(SE[n]))
                memo[n] = r
            return r
        return [cp(r) for r in roots]
    
    def live_cells(self,root,limit=10_000_000):
        """Set of (x, y) live cells (use only for small patterns / tests)."""
        out = set()
        k = self.LVL[root]
 
        def walk(n, x, y):
            if self.POP[n] == 0:
                return
            if self.LVL[n] == 3:
                bits = self.BITS[n]
                for i in range(64):
                    if (bits >> i) & 1:
                        out.add((x + (i & 7), y + (i >> 3)))
                return
            h = 1 << (self.LVL[n] - 1)
            walk(self.NW[n], x, y)
            walk(self.NE[n], x + h, y)
            walk(self.SW[n], x, y + h)
            walk(self.SE[n], x + h, y + h)
        walk(root, -(1 << (k - 1)), -(1 << (k - 1)))
        return out
    
    
def _read_text(path):
    opener = gzip.open if str(path).lower().endswith(".gz") else open
    with opener(path, "rb") as f:
        return f.read().decode("latin1")

def load_mc(uni, text):
    ids = [0]
    for line in text.splitlines():
        line = line.strip()
        if not line or line[0] == "[":
            continue
        if line[0] == "#":
            if line.startswith("#R") and line.split()[-1].upper() not in ("B3/S23", "23/3"):
                print("warning: pattern rule is", line, "- this engine only runs B3/S23")
            continue
        if line[0] in ".*$":
            bits = x = y = 0
            for ch in line:
                if ch == ".":
                    x += 1
                elif ch == "*":
                    bits |= 1 << (y * 8 + x)
                    x += 1
                elif ch == "$":
                    y += 1
                    x = 0
            ids.append(uni.leaf(bits))
        else:
            p = line.split()
            lvl = int(p[0])
            kids = [uni.empty_node(lvl - 1) if int(q) == 0 else ids[int(q)] for q in p[1:5]]
            ids.append(uni.mk(*kids))
    return ids[-1]

def from_array(uni, arr):
    h, w = arr.shape
    S = 16
    while S < max(h, w) * 2:
        S *= 2
    full = np.zeros((S, S), bool)
    oy, ox = (S - h) // 2, (S - w) // 2
    full[oy:oy + h, ox:ox + w] = arr
    B = S // 8
    blocks = full.reshape(B, 8, B, 8).transpose(0, 2, 1, 3).reshape(B, B, 64)
    weights = np.uint64(1) << np.arange(64, dtype=np.uint64)
    vals = (blocks.astype(np.uint64) * weights).sum(-1)
    ids = np.zeros((B, B), np.int64)
    for by, bx in zip(*np.nonzero(vals)):
        ids[by, bx] = uni.leaf(int(vals[by, bx]))
    level = 3
    while ids.shape[0] > 1:
        nb = ids.shape[0] // 2
        a, b, c, d = ids[0::2, 0::2], ids[0::2, 1::2], ids[1::2, 0::2], ids[1::2, 1::2]
        e = uni.empty_node(level)
        new = np.full((nb, nb), uni.empty_node(level + 1), np.int64)
        for y, x in zip(*np.nonzero((a != e) | (b != e) | (c != e) | (d != e))):
            new[y, x] = uni.mk(int(a[y, x]), int(b[y, x]), int(c[y, x]), int(d[y, x]))
        ids, level = new, level + 1
    return int(ids[0, 0])

 
def load_rle(uni, text):
    rows, body = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] == "#":
            continue
        if s[0] in "xX" and "=" in s:
            m = re.search(r"rule\s*=\s*(\S+)", s, re.I)
            if m and m.group(1).upper() not in ("B3/S23", "23/3"):
                print("warning: RLE rule is", m.group(1), "- this engine only runs B3/S23")
            continue
        body.append(s)
    data = "".join(body).split("!")[0]
    x = y = 0
    cells = []
    for num, tag in re.findall(r"(\d*)([a-zA-Z.$])", data):
        n = int(num) if num else 1
        if tag == "$":
            y += n
            x = 0
        elif tag in "bB.":
            x += n
        else:
            cells.append((y, x, n))
            x += n
    if not cells:
        return uni.empty_node(5)
    H = max(c[0] for c in cells) + 1
    W = max(c[1] + c[2] for c in cells)
    arr = np.zeros((H, W), bool)
    for yy, xx, n in cells:
        arr[yy, xx:xx + n] = True
    return from_array(uni, arr)


 
def load_pattern(uni, path):
    text = _read_text(path)
    if text.lstrip().startswith("[M2]"):
        return load_mc(uni, text)
    return load_rle(uni, text)