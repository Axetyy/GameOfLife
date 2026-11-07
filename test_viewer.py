import sys
import time

import numpy as np

from logic.hashlife import Universe, load_pattern,load_rle,GOSPER_GUN,from_array


def naive_step(a):
    n = sum(np.roll(np.roll(a,dy,0),dx,1) for dy in (-1,0,1) for dx in (-1,0,1) if (dy,dx)!=(0,0))
    return (n==3) | ((a==1) & (n==2))


def check_random():
    rng = np.random.default_rng()
    soup = rng.random((40,40)) < 0.35
    for j in (0,1,2,3,4,5,6):
        u = Universe()
        root = from_array(u,soup)
        want_gens = 1 << j
        got = u.live_cells(u.advance(root,j))
        
        S = 1024
        big = np.zeros((S,S),bool)
        big[S // 2 - 20:S // 2 + 20, S // 2 - 20:S // 2 + 20] = soup
        for _ in range(want_gens):
            big = naive_step(big)
        ys,xs = np.nonzero(big)
        want = {(int(x) - S // 2, int(y) - S // 2) for x, y in zip(xs, ys)}
        
        ox = -20 - (-20)
        assert got == want, f"MISMATCH at j={j}: {len(got)} vs {len(want)}"
        print(f"  j={j} ({want_gens} gens): OK, {len(got)} live cells")
        

def check_composition():
    """advance(j=3) must equal 8 x advance(j=0)."""
    u = Universe()
    root = load_rle(u, GOSPER_GUN)
    a = root
    for _ in range(8):
        a = u.advance(a, 0)
    b = u.advance(root, 3)
    assert u.live_cells(a) == u.live_cells(b)
    c = u.advance(root, 10)
    d = root
    for _ in range(4):
        d = u.advance(d, 8)
    d = u.advance(d, 9) if False else d
    e = u.advance(u.advance(root, 9), 9)
    assert u.live_cells(c) != set() and len(u.live_cells(e)) > len(u.live_cells(root))
    print("  composition OK (glider gun keeps emitting; pop after 2^10 gens =", len(u.live_cells(c)), ")")

def bench_file(path):
    u = Universe()
    t = time.time()
    root = load_pattern(u, path)
    print(f"  loaded {path}: level {u.LVL[root]} (side {2 ** u.LVL[root]:,} cells), "
          f"{len(u):,} nodes, pop {u.POP[root]:,} in {time.time() - t:.2f}s")
    for j in (0, 4, 8, 12, 15):
        t = time.time()
        root = u.advance(root, j)
        print(f"  step 2^{j:<2} = {1 << j:>6} gens: {time.time() - t:7.2f}s, nodes {len(u):,}, pop {u.POP[root]:,}")
        
if __name__ == "__main__":
    print("random soup vs naive simulator")
    check_random()
    print("hashlife self-consistency")
    check_composition()
    if len(sys.argv) > 1:
        print("benchmark")
        bench_file(sys.argv[1])
    print("ALL OK")
        