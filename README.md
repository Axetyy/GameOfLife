# Game of Life

An interactive Conway's Game of Life simulator powered by a from-scratch
HashLife engine. It uses a memoized quadtree to skip through large numbers of
generations efficiently, while a Pygame viewer renders the currently visible
part of the universe.

The project also includes a MacroCell pattern for a small programmable
computer, plus the program used to run that computer.

## Features

- Fast HashLife simulation for large and long-running patterns
- Interactive Pygame viewer with zooming and panning
- Vectorized NumPy rendering
- RLE and MacroCell (`.mc`) pattern loading
- Optional gzip-compressed pattern files (`.gz`)
- Built-in Gosper glider gun, glider, R-pentomino, acorn, pulsar, and
  lightweight spaceship patterns
- Generation, population, step size, universe size, and performance
  information displayed in the viewer
- Screenshot saving from inside the viewer

The engine currently runs the standard Conway rule **B3/S23**.

## Requirements

- Python 3
- NumPy
- Pygame

Install the dependencies with:

```bash
python -m pip install numpy pygame
```

## Running the viewer

Start with the built-in Gosper glider gun:

```bash
python viewer.py
```

Load a pattern from a file:

```bash
python viewer.py patterns/collatz67.mc
```

The viewer also accepts RLE files and gzip-compressed files:

```bash
python viewer.py path/to/pattern.rle.gz
```

Useful command-line options:

```text
--j J              Start with steps of 2^J generations
--size WIDTHxHEIGHT
                   Set the window size (default: 1280x800)
--max-nodes N      Compact the HashLife universe above N nodes
--rate RATE        Playback updates per second; 0 means unlimited
--run              Start running immediately
```

For example:

```bash
python viewer.py patterns/collatz67.mc --j 4 --size 1600x900 --run
```

['Overview'](/screenshots/overview.png)

## Viewer controls

| Input | Action |
| --- | --- |
| `Space` | Run or pause |
| `N` or `Right Arrow` | Advance one step |
| `Up` / `Down` | Increase or decrease the step size |
| `[` / `]` | Decrease or increase playback speed |
| Mouse wheel | Zoom around the cursor |
| Left-drag | Pan the view |
| `F` | Fit the whole pattern |
| `R` | Reset to generation 0 |
| `S` | Save a PNG screenshot |
| `H` | Show or hide help |
| `Esc` or `Q` | Quit |

The step size is shown as `2^j`, so a single viewer step can advance many
generations when `j` is increased.

## Project layout

```text
hashlife.py                 HashLife engine and pattern loaders
viewer.py                   Pygame interface and renderer
patterns/collatz67.mc       MacroCell computer pattern
screenshots/                Example screenshots
test_viewer.py              Viewer-related tests
```

## The programmable computer

The computer architecture used by the pattern was designed by
[Nicolas Loizeau](https://github.com/NicolasLoizeau). The implementation,
simulation, viewer, and the program below were built by me.

This is the program used to program the computer:

```text
write a1 67
write a2 1
write a3 3
write a4 1
write a5 0
write a6 0
goto 7
=0 a5 a1
jump a5
goto 12
and a6 a1 a4
jump a6
goto 16
>> a1 a1
print a1
goto 7
<< a2 a1
+ a1 a2 a1
++ a1 a1
print a1
goto 7
```

The corresponding computer pattern is available at
[`patterns/collatz67.mc`](patterns/collatz67.mc).


The computer can be seen in these screenshots that is calculating the Collatz sequence for 67 correctly:

[![](./screenshots/first_calculation.png)](./screenshots/first_calculation.png)
[![](./screenshots/second_calculation.png)](./screenshots/second_calculation.png)
[![](./screenshots/third_calculation.png)](./screenshots/third_calculation.png)

## Credits

- **Computer architecture:** [Nicolas Loizeau](https://github.com/NicolasLoizeau)
- **Everything else in this repository:** built by me
