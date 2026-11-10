# Box contacts against mesh triangles: phantom deep contacts since #4974

This pack reproduces a contact regression in Newton's `CollisionPipeline`, shows where it started, and carries a
proposed fix with a regression test. Everything runs on public Newton; the scripts import only `newton`, `warp` and
`numpy`.

**In one line:** since `536a3dd7` (*Fix downward triangle-box ground-contact normals*, #4974) a mesh triangle that
dips slightly into a box face can get a contact that is centimetres deep and points sideways. It is on `main`
(`0ced8f84`) and on `release-1.8` (`db915e1c`).

We first saw it through `SolverSteel` on a manipulation scene, so the larger repro steps Steel. The cause is in
`newton/_src/geometry/penetration.py`, not in Steel.

## Contents

| File | What it is |
|---|---|
| `box_triangle_repro.py` | The small repro: one cube mesh resting on one box. CPU, a few seconds. |
| `steel_contact_repro.py` | The scene where we found it: N worlds stepped with `SolverSteel`, with a contact-depth count. Needs a CUDA GPU. |
| `scene/` | The MJCF, its meshes and start state for `steel_contact_repro.py`, with licences (`scene/NOTICE.md`). |
| `fix/fix.diff`, `fix/0001-*.patch` | The proposed fix, its regression test and a changelog fragment, on `main` `0ced8f84`. |
| `runs/` | The JSON output of every run quoted below. |

## Install

```bash
uv venv --python 3.12 venv-old
uv pip install --python venv-old/bin/python \
  "newton[sim] @ git+https://github.com/newton-physics/newton@ac7f7830b64e60c5fa74f57960eadcf11380a9d8" trimesh scipy
uv venv --python 3.12 venv-rel
uv pip install --python venv-rel/bin/python \
  "newton[sim] @ git+https://github.com/newton-physics/newton@db915e1c79e9e4a09f61221f743e1e4fe814d75c" trimesh scipy
uv venv --python 3.12 venv-main
uv pip install --python venv-main/bin/python \
  "newton[sim] @ git+https://github.com/newton-physics/newton@0ced8f8420d4ee1eab9ecfaee59e1354ef670348" trimesh scipy
```

Our runs: one NVIDIA L4 (driver 580.178.04), Ubuntu 22.04, Python 3.12.16, Warp 1.19.0 in every environment.

## 1. The small repro

A 2 cm cube, as a triangle mesh, rests 0.5 mm deep on the top face of a wide static box, tilted by 0.02 rad.

```bash
venv-main/bin/python box_triangle_repro.py
```

| Newton | Deepest contact | Its normal (box to mesh) |
|---|---|---|
| `ac7f7830` (before #4974) | 0.500 mm | (0, 0, 1) |
| `0ced8f84` (main) | **65.198 mm** | (-1, 0, 0) |
| `0ced8f84` + `fix/fix.diff` | 0.500 mm | (0, 0, 1) |

65 mm is the distance from the cube to the far side face of the box.

**Why.** `_box_triangle_depth` now discards every separating direction that leans behind a one-sided triangle. For
a side triangle of the cube, the shallow way out is to lift it by 0.5 mm; with the tilt, that direction leans
0.02 rad toward the triangle's front, so it is discarded, and the next admissible exit crosses the whole box.

## 2. The scene where we found it

Two 6-DoF arms with parallel grippers, 8 blocks, a ledge and a bin (`scene/scene.xml`). The arm links and fingers
are convex-hull meshes; the blocks, ledge, bin and table are boxes. Each world is driven for 2 s with random legal
position targets (fixed seed, 50 Hz). After every solver step the script reads the depth of every contact from the
pipeline, `-(n . (x1 - x0) - (margin0 + margin1))`, and counts the worlds where any contact was deeper than 1 cm.

```bash
venv-old/bin/python steel_contact_repro.py --preset probe --worlds 512
venv-old/bin/python steel_contact_repro.py --preset lift  --worlds 1024
```

Settings: `lift` is the Isaac Lab Lift task's Steel preset (Isaac Lab PR #8940 at `19cf8a81`: 4 ms step, 2 solver
steps per collision pass, 8 iterations, `baumgarte=0.05`, `friction_anchor_beta=0`), with two row buffers raised to
fit two arms and 8 blocks (`articulation_constraint_max` 192 to 1024, `free_body_constraint_max` 64 to 512).
`probe` is the same at a 0.625 ms step with `baumgarte=0.2` and 16 iterations, where the old commit is clean.

Worlds with a contact deeper than 1 cm (seed 100000 unless noted; other seeds with `--seed`):

| Newton | `probe`, 512 worlds | `lift`, 1,024 worlds |
|---|---|---|
| `ac7f7830` | 0 (0.00 %); seed 100001: 0 | 293 (28.6 %); seeds 100001 / 100002: 327 / 302 |
| `db915e1c` (release-1.8) | **193 (37.7 %)**, 70 worlds overflow their rows | **870 (85.0 %)** |
| `0ced8f84` (main) | **196 (38.3 %)**, 71 overflow; seed 100001: 202 (39.5 %) | **875 (85.5 %)** |
| `0ced8f84` with `536a3dd7` reverted | 2 (0.39 %); seed 100001: 3 | 306 (29.9 %); seeds 100001 / 100002: 344 / 307 |
| `0ced8f84` + `fix/fix.diff` | 1 (0.20 %); seed 100001: 1 | 340 (33.2 %); seeds 100001 / 100002: 347 / 340 |

The phantom contacts also multiply the constraint rows Steel asks for: at `probe` the peak per world goes from 181
(`ac7f7830`) to 4,897 (`db915e1c`) against a buffer of 1,024, which is the overflow above
(`constraint_row_watermarks()`, `articulation_raw_high_water`).

One of the first phantom contacts in that scene, rebuilt as a single pair: a fingertip hull resting 0.9 mm deep on
the ledge's top face gets a ninth contact, 62.4 mm deep, normal (-1, 0, 0).

The `lift` preset is too coarse for this scene on every commit (about 30 % before #4974); the table is there to
show the step from about 30 % to 85 %, and back with the fix.

## 3. Bisect

`git bisect` between `ac7f7830` (good) and `0ced8f84` (bad), one `probe` run of 512 worlds per step, bad = more
than 5 % of worlds. Runs on both sides of the first bad commit:

| Commit | Worlds with a deep contact |
|---|---|
| `2e20d82d` Add configurable inertia validation controls (#4971) | 0 / 512 |
| `3ba2f21c` Reject non-finite contact materials at finalization (#4905), the parent | 1 / 512 |
| **`536a3dd7` Fix downward triangle-box ground-contact normals (#4974)** | **196 / 512** |
| `28d8d473` Deprecate rod start frames and clean up VBD options (#4978) | 192 / 512 |

To bisect yourself: `uv pip install -e ".[sim]"` in a checkout, then at each step
`python steel_contact_repro.py --preset probe --worlds 512 --out step.json` and read `deep_share`.

## 4. The proposed fix

`fix/fix.diff` changes `_box_triangle_depth` only (20 lines added, 6 removed):

- A candidate direction that leans behind a one-sided triangle is no longer discarded. It is replaced by its
  projection onto the triangle's plane, at the travel that clears the same axis (`depth / sqrt(1 - side^2)`). The
  box may then slide off an edge of the triangle, which is the shallow exit in the cases above.
- A direction that points mostly through the triangle becomes much longer when projected, so the front-side exit
  still wins there. #4974's own seam test (`test_refinement_triangle_box_seam`) passes unchanged.
- A separating axis (negative depth) is kept whichever side it points to: the pair does not overlap.

The new test `test_refinement_triangle_box_tilted_rest` is the small repro above, in three tilt directions and both
shape orders. It fails on `0ced8f84` (65 mm against 0.5 mm) and passes with the fix.

The diff also applies cleanly to `release-1.8` `db915e1c`.
