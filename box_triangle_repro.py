"""Box against a triangle mesh: a phantom deep contact. Needs only newton, warp and numpy; runs on the CPU.

A 2 cm cube, as a triangle mesh, rests 0.5 mm deep on the top face of a wide static box, tilted by 0.02 rad.
Every contact of CollisionPipeline should push the cube up by at most 0.5 mm. On the affected commits one more
contact appears: 65 mm deep, pointing sideways (the distance from the cube to the far side face of the box).

    python box_triangle_repro.py
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton

HALF = np.array([0.075, 0.125, 0.04])     # the static box
SIDE, DEPTH, TILT = 0.01, 0.0005, -0.02   # cube half extent [m], true penetration [m], tilt [rad]


def contacts(axis: tuple, device: str) -> list[tuple[float, np.ndarray]]:
    """(depth [m], normal from the box to the mesh) of every contact, for the cube tilted about ``axis``."""
    cube = newton.Mesh.create_box(SIDE, SIDE, SIDE, duplicate_vertices=False, compute_normals=False,
                                  compute_uvs=False, compute_inertia=False)
    rotation = wp.quat_from_axis_angle(wp.vec3(*axis), TILT)
    lowest = min(float(wp.quat_rotate(rotation, wp.vec3(*v))[2]) for v in np.asarray(cube.vertices))
    pose = wp.transform(wp.vec3(-0.02, 0.03, float(HALF[2] - DEPTH - lowest)), rotation)
    builder = newton.ModelBuilder()
    cfg = builder.ShapeConfig(margin=0.0, gap=1.0e-3, density=0.0)
    body = builder.add_body(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3)))
    box = builder.add_shape_box(-1, hx=HALF[0], hy=HALF[1], hz=HALF[2], cfg=cfg)
    builder.add_shape_mesh(body, mesh=cube, cfg=cfg)
    model = builder.finalize(device)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit")
    found = pipeline.contacts()
    pipeline.collide(model.state(), found)
    count = int(found.rigid_contact_count.numpy()[0])
    shape0, normal = found.rigid_contact_shape0.numpy(), found.rigid_contact_normal.numpy()
    point0, point1 = found.rigid_contact_point0.numpy(), found.rigid_contact_point1.numpy()
    out = []
    for i in range(count):
        box_first = shape0[i] == box
        on_box = point0[i] if box_first else point1[i]
        on_mesh = np.asarray(wp.transform_point(pose, wp.vec3(*(point1[i] if box_first else point0[i]))))
        direction = normal[i] if box_first else -normal[i]
        out.append((float((on_box - on_mesh) @ direction), direction))
    return out


def main() -> None:
    wp.config.quiet = True
    wp.init()
    print(f"newton {newton.__version__}, warp {wp.__version__}; true penetration {1e3 * DEPTH:.1f} mm")
    worst = 0.0
    for axis in ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)):
        found = contacts(axis, "cpu")
        deepest = max(found, key=lambda c: c[0])
        worst = max(worst, deepest[0])
        print(f"tilt {TILT} rad about {axis}: {len(found)} contacts, deepest {1e3 * deepest[0]:.3f} mm, "
              f"normal {np.round(deepest[1], 3).tolist()}")
    print("RESULT", "phantom deep contact" if worst > 10.0 * DEPTH else "ok", f"(deepest {1e3 * worst:.3f} mm)")


if __name__ == "__main__":
    main()
