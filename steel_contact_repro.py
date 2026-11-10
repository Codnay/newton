"""SolverSteel contact penetration on a two-arm bin scene. Needs only newton, warp and numpy.

The scene (scene/scene.xml): two 6-DoF arms with parallel grippers, 8 free blocks, a ledge and a bin. Each world is
driven for 2 s with random position targets (an Ornstein-Uhlenbeck walk inside the legal box, rate limited, 50 Hz,
fixed seed). After every solver step the script measures the depth of every contact of Newton's CollisionPipeline
and counts the worlds where any contact was deeper than 1 cm.

    python steel_contact_repro.py --preset probe --worlds 512
    python steel_contact_repro.py --preset lift --worlds 1024

Presets: lift = the Isaac Lab Lift task's Steel settings (4 ms step, 8 PGS iterations, baumgarte 0.05, 2 substeps
per collision pass) with the two row buffers raised to fit this scene; tuned = the same at a 0.625 ms step;
probe = tuned with SolverSteel's default baumgarte 0.2 and 16 iterations.
"""
from __future__ import annotations

import argparse
import inspect
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverMuJoCo, SolverSteel

SCENE = Path(__file__).resolve().parent / "scene"
PRESETS = {"lift": {"dt": 0.004, "iterations": 8, "baumgarte": 0.05},
           "tuned": {"dt": 0.000625, "iterations": 8, "baumgarte": 0.05},
           "probe": {"dt": 0.000625, "iterations": 16, "baumgarte": 0.2}}
# Isaac Lab PR #8940 (19cf8a81) Lift `newton_steel` preset over its SteelSolverCfg defaults. Raised here, to fit two
# arms and 8 blocks: articulation_constraint_max 192 -> 1024, free_body_constraint_max 64 -> 512.
# "matrix_free" is the solve mode's name on ac7f7830; later commits map it to "serial" with a deprecation warning.
STEEL = {"solve_mode": "matrix_free", "update_mass_matrix_interval": 2, "use_parallel_streams": False,
         "enable_joint_limits": True, "joint_limit_activation_gap": 0.1, "enable_joint_velocity_limits": False,
         "velocity_limit_activation_fraction": 0.7, "articulation_constraint_max": 1024,
         "free_body_constraint_max": 512, "friction_anchor_beta": 0.0, "contact_friction_gap_max": None,
         "contact_friction_position_iterations": -1, "contact_torsion_device": False, "parallel_tree": False,
         "row_watermark": True}
SHAPE = {"margin": 0.0, "gap": 0.01, "ke": 2.5e3, "kd": 100.0, "mu": 1.0}   # Isaac Lab NewtonShapeCfg defaults
SUBSTEPS = 2            # solver steps per collision pass
CONTACT_MAX = 4_000_000
CONTROL_HZ = 50.0
DEPTH_LIMIT = 0.01      # m
OU_THETA, OU_SIGMA, TRACE_BATCH = 2.0, 0.5, 2048


def last(label) -> str:
    return str(label).rsplit("/", 1)[-1]


def legalize(trace, low, high, max_delta, initial):
    """Clip [W, T, A] targets to the box and rate-limit them from ``initial``; float32 out."""
    boxed = np.clip(np.asarray(trace, np.float64), low, high)
    out = np.empty_like(boxed)
    prev = np.broadcast_to(np.clip(initial, low, high), boxed[:, 0].shape)
    for t in range(boxed.shape[1]):
        out[:, t] = prev + np.clip(boxed[:, t] - prev, -max_delta, max_delta)
        prev = out[:, t]
    return out.astype(np.float32)


def random_targets(space: dict, seed: int, seconds: float) -> np.ndarray:
    """[TRACE_BATCH, ticks, A] legal position targets: an OU walk in units of the half-range, one draw per tick."""
    low, high = np.asarray(space["low"], np.float64), np.asarray(space["high"], np.float64)
    max_delta = np.asarray(space["max_delta_per_tick"], np.float64)
    initial = np.asarray(space["initial"], np.float64)
    ticks, dt = max(1, round(seconds * CONTROL_HZ)), 1.0 / CONTROL_HZ
    rng = np.random.default_rng(seed)
    x = np.zeros((TRACE_BATCH, low.size))
    unit = np.empty((TRACE_BATCH, ticks, low.size))
    for t in range(ticks):
        x += -OU_THETA * x * dt + OU_SIGMA * np.sqrt(2.0 * OU_THETA * dt) * rng.standard_normal(x.shape)
        unit[:, t] = x
    raw = 0.5 * (low + high) + unit * 0.5 * (high - low)
    once = legalize(raw, low, high, max_delta, initial)
    return legalize(once, low, high, max_delta, initial)


def new_builder():
    b = newton.ModelBuilder(up_axis="Z")
    SolverMuJoCo.register_custom_attributes(b)
    SolverSteel.register_custom_attributes(b)
    cfg = b.default_shape_cfg
    for key, value in SHAPE.items():
        setattr(cfg, key, value)
    return b


def add_drives(b, xml: Path) -> list[int]:
    """Each MJCF affine position servo becomes a POSITION joint drive (ke = kp, kd = kv). Returns the joint_q
    coordinate of each actuator, in actuator order."""
    where = {last(lbl): j for j, lbl in enumerate(b.joint_label)}
    coords = []
    for act in ET.parse(xml).getroot().iter("general"):
        kp = float(act.get("gainprm").split()[0])
        kv = -float(act.get("biasprm").split()[2])
        j = where[act.get("joint")]
        d = b.joint_qd_start[j]
        b.joint_target_ke[d], b.joint_target_kd[d] = kp, kv
        b.joint_target_mode[d] = int(newton.JointTargetMode.POSITION)
        coords.append(b.joint_q_start[j])
    return coords


def filter_pairs(b, allowed: set) -> int:
    """Keep only the body pairs MuJoCo collides for this MJCF (contype / conaffinity / parent-child)."""
    names = [last(x) for x in b.body_label]
    owner = [names[i] if i >= 0 else "world" for i in b.shape_body]
    n = 0
    for i in range(len(owner)):
        for j in range(i + 1, len(owner)):
            if tuple(sorted((owner[i], owner[j]))) not in allowed:
                b.add_shape_collision_filter_pair(i, j)
                n += 1
    return n


def start_state(b, data: dict) -> tuple[np.ndarray, np.ndarray]:
    """One world's joint_q / joint_qd from the MuJoCo qpos / qvel in scene.json (free joints wxyz -> xyzw)."""
    q, qd = np.array(b.joint_q, np.float64), np.zeros(len(b.joint_qd))
    where = {last(lbl): j for j, lbl in enumerate(b.joint_label)}
    qpos, qvel = np.asarray(data["initial"]["qpos"]), np.asarray(data["initial"]["qvel"])
    for jt in data["joints"]:
        j = where[jt["name"]]
        qs, ds, a, v = b.joint_q_start[j], b.joint_qd_start[j], jt["qposadr"], jt["dofadr"]
        if jt["type"] == "free":
            p = qpos[a:a + 7]
            q[qs:qs + 7] = [*p[:3], p[4], p[5], p[6], p[3]]
            qd[ds:ds + 6] = qvel[v:v + 6]
        else:
            q[qs], qd[ds] = qpos[a], qvel[v]
    return q.astype(np.float32), qd.astype(np.float32)


def build(worlds: int, device: str):
    """(model, scene data, drive coordinates, coordinates per world, q0, qd0, body names of one world)."""
    data = json.loads((SCENE / "scene.json").read_text())
    t = new_builder()
    t.add_mjcf(str(SCENE / "scene.xml"), ctrl_direct=True, parse_visuals=False, convert_mjc_equality_constraints=True)
    coords = add_drives(t, SCENE / "scene.xml")
    filter_pairs(t, {tuple(sorted(p)) for p in data["contact_body_pairs"]})
    q0, qd0 = start_state(t, data)
    b = new_builder()
    count = "copy_count" if "copy_count" in inspect.signature(b.replicate).parameters else "world_count"
    b.replicate(t, **{count: worlds})
    with wp.ScopedDevice(device):
        model = b.finalize(device=device)
    return model, data, coords, t.joint_coord_count, q0, qd0, [last(x) for x in t.body_label]


@wp.kernel
def max_contact_depth(count: wp.array(dtype=wp.int32), shape0: wp.array(dtype=wp.int32),
                      shape1: wp.array(dtype=wp.int32), point0: wp.array(dtype=wp.vec3),
                      point1: wp.array(dtype=wp.vec3), normal: wp.array(dtype=wp.vec3),
                      margin0: wp.array(dtype=wp.float32), margin1: wp.array(dtype=wp.float32),
                      shape_body: wp.array(dtype=wp.int32), shape_world: wp.array(dtype=wp.int32),
                      body_q: wp.array(dtype=wp.transform), depth: wp.array(dtype=wp.float32)):
    """Per world, the deepest contact so far: -(n . (x1 - x0) - (margin0 + margin1)) at the current body poses."""
    c = wp.tid()
    if c >= count[0]:
        return
    s0 = shape0[c]
    s1 = shape1[c]
    if s0 < 0 or s1 < 0:
        return
    w = wp.max(shape_world[s0], shape_world[s1])
    if w < 0:
        return
    x0 = wp.vec3(point0[c])
    x1 = wp.vec3(point1[c])
    if shape_body[s0] >= 0:
        x0 = wp.transform_point(body_q[shape_body[s0]], x0)
    if shape_body[s1] >= 0:
        x1 = wp.transform_point(body_q[shape_body[s1]], x1)
    wp.atomic_max(depth, w, margin0[c] + margin1[c] - wp.dot(normal[c], x1 - x0))


def snapshot(state, worlds: int) -> tuple[np.ndarray, np.ndarray]:
    """Host copies of body_q [W, B, 7] and body_qd [W, B, 6]."""
    return state.body_q.numpy().reshape(worlds, -1, 7).copy(), state.body_qd.numpy().reshape(worlds, -1, 6).copy()


def steel_kwargs(args) -> dict:
    want = dict(STEEL, iterations=args.iterations, baumgarte=args.baumgarte,
                articulation_constraint_max=args.articulation_constraint_max)
    valid = set(inspect.signature(SolverSteel.__init__).parameters)
    dropped = sorted(set(want) - valid)
    if dropped:
        print("options not in this SolverSteel, dropped:", dropped)
    return {k: v for k, v in want.items() if k in valid}


def run(args) -> dict:
    dev = args.device
    model, data, coords, ncoord, q0, qd0, names = build(args.worlds, dev)
    space = data["action_space"]
    traces = random_targets(space, args.seed, args.seconds)[:args.worlds]          # [W, ticks, A]
    index = np.asarray(space["index"])
    with wp.ScopedDevice(dev):
        model.rigid_contact_max = CONTACT_MAX      # Steel sizes its buffers from the model
        solver = SolverSteel(model, **steel_kwargs(args))
        pipeline = newton.CollisionPipeline(model, broad_phase="explicit", rigid_contact_max=CONTACT_MAX)
        contacts = pipeline.contacts()
        state, control = model.state(), model.control()
        state.joint_q.assign(np.tile(q0, args.worlds))
        state.joint_qd.assign(np.tile(qd0, args.worlds))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        depth = wp.full(args.worlds, -1.0, dtype=wp.float32)
        target = control.joint_target_q.numpy().reshape(args.worlds, ncoord)
        steps_per_tick, k, t0 = round(1.0 / (CONTROL_HZ * args.dt)), 0, time.perf_counter()
        frames = [snapshot(state, args.worlds)] if args.frames else None
        for tick in range(traces.shape[1]):
            target[:, np.asarray(coords)[index]] = traces[:, tick]
            control.joint_target_q.assign(target.reshape(-1))
            for _ in range(steps_per_tick):
                if k % SUBSTEPS == 0:
                    pipeline.collide(state, contacts)
                solver.step(state, state, control, contacts, args.dt)
                state.clear_forces()
                c = contacts
                wp.launch(max_contact_depth, dim=c.rigid_contact_max, inputs=[
                    c.rigid_contact_count, c.rigid_contact_shape0, c.rigid_contact_shape1, c.rigid_contact_point0,
                    c.rigid_contact_point1, c.rigid_contact_normal, c.rigid_contact_margin0, c.rigid_contact_margin1,
                    model.shape_body, model.shape_world, state.body_q, depth])
                k += 1
                if frames is not None:
                    frames.append(snapshot(state, args.worlds))
        wp.synchronize_device(dev)
        wall = time.perf_counter() - t0
        d = depth.numpy()
        over = solver.constraint_overflow.numpy()
        marks = solver.constraint_row_watermarks() if hasattr(solver, "constraint_row_watermarks") else {}
        finite = np.isfinite(state.body_q.numpy()).reshape(args.worlds, -1).all(1)
    if frames is not None:
        np.savez_compressed(args.frames, body_q=np.stack([f[0] for f in frames]), body_qd=np.stack([f[1] for f in frames]),
                            t=np.arange(len(frames)) * args.dt, body_names=np.array(names), world_max_depth=d)
    deep = (d > DEPTH_LIMIT) | ~finite
    return {"worlds": args.worlds, "steps": k, "dt": args.dt, "deep_worlds": int(deep.sum()),
            "deep_share": float(deep.mean()), "nonfinite_worlds": int((~finite).sum()),
            "max_depth_m": float(np.nanmax(d)), "median_world_max_depth_m": float(np.nanmedian(d)),
            "overflow_worlds": int(over[:args.worlds].sum()), "overflow_global": bool(over[args.worlds:].any()),
            "row_watermarks": marks, "wall_s": round(wall, 1), "deep_world_ids": np.nonzero(deep)[0].tolist()}


def versions() -> dict:
    """Newton and Warp versions; the Newton commit from pip's record of a git install, else from the checkout."""
    import importlib.metadata as md
    sha = ""
    try:
        sha = json.loads(md.distribution("newton").read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id", "")
    except (md.PackageNotFoundError, ValueError):
        pass
    if not sha:
        root = Path(newton.__file__).resolve().parent.parent
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"newton": newton.__version__, "newton_commit": sha or "unknown", "warp": wp.__version__,
            "device": str(wp.get_device())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="probe")
    ap.add_argument("--worlds", type=int, default=512)
    ap.add_argument("--dt", type=float)
    ap.add_argument("--iterations", type=int)
    ap.add_argument("--baumgarte", type=float)
    ap.add_argument("--articulation-constraint-max", type=int, default=STEEL["articulation_constraint_max"])
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=100000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, help="write the result as JSON (with the per-world ids)")
    ap.add_argument("--frames", type=Path, help="save body_q / body_qd after every step as npz (large)")
    args = ap.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    wp.init()
    args.device = args.device or str(wp.get_device())
    res = {"preset": args.preset, "iterations": args.iterations, "baumgarte": args.baumgarte,
           "articulation_constraint_max": args.articulation_constraint_max, "seed": args.seed, **versions(), **run(args)}
    if args.out:
        args.out.write_text(json.dumps(res))
    ids = res.pop("deep_world_ids")
    print(json.dumps(res))
    print(f"RESULT {res['newton_commit'][:8]} {args.preset}: {res['deep_worlds']} / {res['worlds']} worlds "
          f"({100 * res['deep_share']:.2f} %) had a contact deeper than 1 cm; constraint overflow in "
          f"{res['overflow_worlds']} worlds; first ids {ids[:8]}")


if __name__ == "__main__":
    main()
