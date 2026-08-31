#!/usr/bin/env python3
"""Generate the village geometry in scene_obstacles.xml.

Houses are built from primitives (walls + a two-slab gable roof + openings) and
injected between the VILLAGE markers, so re-running replaces them cleanly.

They sit 3.2-6.5 m out, well clear of the obstacle course, and face the origin.
Static worldbody geoms: no bodies, no joints, nothing for the solver to
integrate — they only ever appear in collision, which the duck never reaches.

    python3 scripts/make_village.py
"""
import numpy as np

SCENE = "src/mjlab_microduck/robot/microduck/scene_obstacles.xml"
BEGIN, END = "<!-- VILLAGE:BEGIN -->", "<!-- VILLAGE:END -->"

rng = np.random.default_rng(2026)


def house(i: int, x: float, y: float) -> list[str]:
    w = rng.uniform(0.55, 0.95)          # width  (along the ridge)
    d = rng.uniform(0.45, 0.70)          # depth
    h = rng.uniform(0.40, 0.68)          # wall height
    pitch = rng.uniform(0.42, 0.62)      # roof angle, radians
    eave = 0.045
    wall = f"wall{rng.integers(0, 3)}"
    roof = f"roof{rng.integers(0, 3)}"

    # face the origin: the facade is the -y wall in the house's local frame
    px, py = -x, -y
    n = np.hypot(px, py)
    yaw = float(np.arctan2(-px / n, py / n))
    g = [f'        <!-- house {i} -->']

    def rot(pitch_x: float, yaw_z: float) -> str:
        """quat for Rz(yaw) @ Rx(pitch) — world yaw applied after local pitch."""
        cx, sx = np.cos(pitch_x / 2), np.sin(pitch_x / 2)
        cz, sz = np.cos(yaw_z / 2), np.sin(yaw_z / 2)
        # quaternion product qz * qx, in MuJoCo's (w, x, y, z) order
        return (f"{cz*cx:.6f} {cz*sx:.6f} {sz*sx:.6f} {sz*cx:.6f}")

    def geom(name, typ, size, pos, euler=None, mat="wall0"):
        e = f' quat="{euler}"' if euler else ""
        g.append(f'        <geom name="v{i}_{name}" type="{typ}" size="{size}" '
                 f'pos="{pos}"{e} material="{mat}"/>')

    # walls
    geom("body", "box", f"{w/2:.3f} {d/2:.3f} {h/2:.3f}",
         f"{x:.3f} {y:.3f} {h/2:.3f}", rot(0, yaw), wall)

    # gable roof: two slabs meeting at the ridge
    hd = d / 2 + eave
    slab = hd / np.cos(pitch)
    for sgn in (+1, -1):
        cy, cz = sgn * hd / 2, h + hd * np.tan(pitch) / 2
        # rotate the local (0, cy) offset by yaw, same convention as the walls
        lx = x - cy * np.sin(yaw)
        ly = y + cy * np.cos(yaw)
        geom(f"roof{'a' if sgn > 0 else 'b'}", "box",
             f"{w/2+eave:.3f} {slab/2:.3f} 0.022",
             f"{lx:.3f} {ly:.3f} {cz:.3f}",
             rot(-sgn * pitch, yaw), roof)

    # gable end triangles, faked with a thin box under the ridge
    geom("gable", "box", f"{w/2*0.55:.3f} {d/2*0.9:.3f} {hd*np.tan(pitch)/2:.3f}",
         f"{x:.3f} {y:.3f} {h + hd*np.tan(pitch)/2:.3f}", rot(0, yaw), wall)

    # door and windows, pushed just proud of the facade so they don't z-fight
    fx, fy = np.sin(yaw), -np.cos(yaw)   # outward normal of the local -y wall
    off = d / 2 + 0.006
    dw, dh = 0.10, 0.22
    geom("door", "box", f"{dw:.3f} 0.008 {dh:.3f}",
         f"{x+fx*off:.3f} {y+fy*off:.3f} {dh:.3f}", rot(0, yaw), "door")
    for s in (-1, 1):
        wx = x + fx * off - s * 0.20 * np.cos(yaw)
        wy = y + fy * off - s * 0.20 * np.sin(yaw)
        geom(f"win{'l' if s < 0 else 'r'}", "box", "0.070 0.008 0.055",
             f"{wx:.3f} {wy:.3f} {h*0.62:.3f}", rot(0, yaw), "glass")

    # chimney
    cx = x + 0.28 * w * np.cos(yaw)
    cy2 = y + 0.28 * w * np.sin(yaw)
    geom("chim", "box", "0.032 0.032 0.09",
         f"{cx:.3f} {cy2:.3f} {h + hd*np.tan(pitch) + 0.04:.3f}", None, "roof2")
    return g


def tree(i: int, x: float, y: float) -> list[str]:
    th = rng.uniform(0.30, 0.55)
    r = rng.uniform(0.13, 0.22)
    return [
        f'        <geom name="t{i}_trunk" type="cylinder" size="0.026 {th/2:.3f}" '
        f'pos="{x:.3f} {y:.3f} {th/2:.3f}" material="bark"/>',
        f'        <geom name="t{i}_top" type="ellipsoid" size="{r:.3f} {r:.3f} {r*1.35:.3f}" '
        f'pos="{x:.3f} {y:.3f} {th + r*0.95:.3f}" material="leaf"/>',
    ]


def main() -> None:
    out: list[str] = []
    n_house = 14
    for i in range(n_house):
        a = 2 * np.pi * i / n_house + rng.uniform(-0.16, 0.16)
        rad = rng.uniform(3.2, 6.5)
        out += house(i, rad * np.cos(a), rad * np.sin(a))
    for j in range(22):
        a = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(2.4, 8.0)
        out += tree(j, rad * np.cos(a), rad * np.sin(a))

    s = open(SCENE).read()
    body = BEGIN + "\n" + "\n".join(out) + "\n        " + END
    if BEGIN in s:
        pre, rest = s.split(BEGIN, 1)
        _, post = rest.split(END, 1)
        s = pre + body + post
    else:
        anchor = '        <!-- ── Obstacles'
        s = s.replace(anchor, "        " + body + "\n\n" + anchor, 1)
    open(SCENE, "w").write(s)
    print(f"wrote {n_house} houses + 22 trees into {SCENE}")


if __name__ == "__main__":
    main()
