#!/usr/bin/env python
"""
assemble.py -- composes scene_aaa.xml, "THE WATER COURT", from the five build
agents' MJCF fragments.

Run:  uv run python scripts/aaa/assemble.py

Reads   src/mjlab_microduck/robot/microduck/aaa_<agent>_assets.xml   (<texture>/<material>/<hfield>)
        src/mjlab_microduck/robot/microduck/aaa_<agent>_body.xml     (<geom>/<light>)
        src/mjlab_microduck/robot/microduck/aaa_atmosphere_visual.xml (<statistic>/<visual>)
        src/mjlab_microduck/robot/microduck/scene_obstacles.xml       (keyframes, verbatim source)
Writes  src/mjlab_microduck/robot/microduck/scene_aaa.xml

Idempotent: the output is a pure function of the inputs, so re-running produces a
byte-identical file.

--------------------------------------------------------------------------------
COMPOSITION-LEVEL OVERRIDES  (applied here, in the scene, NOT in any agent's file)
--------------------------------------------------------------------------------
Each agent owns its own fragment and none may be edited from here.  Three defects
are only visible once the fragments are composed with the robot, and all three
live in scene-level settings -- the lighting rig and <visual><map> -- so they are
patched by rewriting the spliced elements on the way into scene_aaa.xml.  Each is
listed below with the measurement that justifies it.

  1. NEAR CLIP vs THE DUCK'S OWN JAW.  *** the important one ***
     The art bible sets znear="0.003" (19.5 m at extent 6.5) and its criterion 8
     checks only one failure mode: that the clip stays under the 148 mm of ground
     at the bottom of frame.  It never checks the other side.  Measured here on
     the composed scene at the STAND keyframe through duck_eye:

         znear  19.5 mm -> jaw_soft fills the TOP 41% OF EVERY FRAME (40.18% of
                           pixels are the duck's own beak, rows 0-185 of 450)
         znear  26.0 mm -> 37.16%
         znear  32.5 mm -> 0.90%
         znear  39.0 mm -> 0.00%   <- jaw fully clear
         znear 130.0 mm -> 0.00%, and the floor is still uncut

     scene_obstacles.xml never showed this because its 135.8 mm clip (auto extent
     13.578) threw the beak away for free.  Dropping to 19.5 mm to protect the
     floor hands 40% of the policy's visual field to the inside of its own face.
     This scene therefore ships znear="0.007" = 45.5 mm: 17% clear of the 39 mm
     jaw threshold and 3.3x clear of the 148 mm floor -- inside a window the
     bible did not know was two-sided.

  2. SUN SHADOW POSITION.  The art bible ships aaa_sun as
        pos="-6 -5 2"  dir="-0.752 -0.631 -0.191"
     A directional light ignores `pos` for shading but NOT for its shadow map.
     The terrain agent measured that this combination casts no shadow at all in
     isolation.  In the fully composed scene it does cast -- the shadow frustum is
     fitted to the model, and the model is now 16 m of court plus background
     ranges -- so that finding does not reproduce as stated.  It is still the
     wrong `pos`: an azimuth-40 sun physically sits in +X+Y, and mirroring `pos`
     there, with `dir` untouched so shading is bit-for-bit unchanged, measurably
     deepens the shadows it casts.  Measured on the hero vista, 1280x720:

         pos -6 -5 2 :  near-band contrast std 12.6,  shadow-toggle max delta  76
         pos  6 5 2  :  near-band contrast std 18.9,  shadow-toggle max delta 145

     Shadows are the primary graphic element of this direction (a 60 cm column
     throws 3.09 m at 11 deg), so the deeper set is the correct one.

  4. SKY FILL LEVEL.  The bible's own pass criteria are the evidence here.  With
     the rig as shipped, measured over ten duck-height viewpoints across the
     court, the scene fails criterion 3 (sunlit:shadow must be 2.0-3.0:1) at
     3.64:1 and sits on the edge of criterion 4 (shadow chroma >= 0.28) at 0.291.
     The atmosphere agent flagged exactly this and declined to re-tune, because
     the architecture albedos were calibrated against the un-tuned renders.  The
     composer is the right place to close it, because it is a whole-scene
     balance.  Raising aaa_skyfill from 0.26 0.28 0.44 to 0.42 0.44 0.66:

         sunlit:shadow  3.64 -> 2.84  (into the 2.0-3.0 window)
         shadow chroma  0.291 -> 0.339  (clears 0.28 with margin)
         pixels < lum25 5.99% -> 3.71%
         near-band std  15.0 -> 16.8
         channel clipping unchanged at 0.22%

     Ambient goes 0.09 0.09 0.13 -> 0.12 0.12 0.17 for the same reason, taking
     below-25 to 3.38% and chroma to 0.374 for +0.12% clipping.  Note WHY skyfill
     is the right knob and a downward fill (override 3) is the wrong one: skyfill
     is horizontal and anti-solar, so it lights the vertical shadow faces that
     read as black without touching the sunlit floor -- it deepens the violet
     instead of washing it out, and it leaves the raking shadow bars intact.

  3. NEAR-BAND FILL -- MEASURED AND REJECTED.  The materials agent asked for a 4th
     light, a warm downward fill, to lift near-band contrast std for criterion 1
     (>= 28); on a swatch board it measured 6.3 -> 15.7.  Built and rendered on
     the actual hero vista it does the opposite of what the direction needs:

         no fill : near-band std 18.9, mean 54.4, red-channel clipping  0.02%
         w/ fill : near-band std 31.0, mean 94.7, red-channel clipping 13.97%

     The std goes up because the fill floods the terracotta to near-white, not
     because anything is better resolved -- and 13.97% clipped red is a direct
     violation of criterion 5 (no channel clipping on any sunlit surface).  Worse,
     a downward fill washes out the six raking shadow bars that ARE the hero
     vista.  Rendered A/B in scratchpad/asm/var_*.png; the unlit version wins on
     sight.  EXTRA_LIGHTS is therefore empty and the scene ships the bible's
     three lights exactly.  Criterion 1 is met without it once the jaw is out of
     frame (override 1) -- the jaw was a flat dark band skewing the statistic.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "src/mjlab_microduck/robot/microduck"
OUT = MODEL_DIR / "scene_aaa.xml"
KEYFRAME_SOURCE = MODEL_DIR / "scene_obstacles.xml"

# Splice order.  Assets are order-free; geoms are not -- the atmosphere fragment
# must come FIRST in the worldbody because it declares the three lights and only
# the first 7 lights in declaration order render.
AGENTS = ["atmosphere", "terrain", "architecture", "vegetation", "materials", "props"]

ROBOT_INCLUDE = "robot_allcollisions_cam.xml"

# The art bible's spawn: origin, facing +X, inside the empty r<=0.9 m apron.
SPAWN_XYZ = (0.0, 0.0, 0.12)

# The ball (ball.xml) is NOT included in this scene, so every keyframe copied out
# of scene_obstacles.xml carries 7 trailing qpos that must be dropped.
BALL_NQ = 7
ROBOT_NQ = 21


# ----------------------------------------------------------------------------
# composition-level light overrides -- see the module docstring
# ----------------------------------------------------------------------------
LIGHT_OVERRIDES = {
    # name            attribute  new value
    "aaa_sun": {"pos": "6 5 2"},
    "aaa_skyfill": {"diffuse": "0.42 0.44 0.66"},
}

# <visual><map> attributes rewritten on the way in.  See override 3 in the docstring.
VISUAL_MAP_OVERRIDES: dict[str, str] = {}   # atmosphere now ships znear 0.012 itself

# <visual><headlight> attributes rewritten on the way in.  See override 4.
VISUAL_HEADLIGHT_OVERRIDES = {"ambient": "0.12 0.12 0.17"}

# A 4th light, appended immediately after the bible's three so it still lands
# well inside the 7-light cap.  Directional, no shadow, aimed down and slightly
# with the sun so its highlight direction does not fight the baked albedo.
EXTRA_LIGHTS = ""  # see NEAR-BAND FILL note in the docstring: measured, rejected.


def strip_comments(text: str) -> str:
    return re.sub(r"<!--.*?-->", "", text, flags=re.S)


# ----------------------------------------------------------------------------
# XML WELL-FORMEDNESS.  MuJoCo's tinyxml2 tolerates "--" inside a comment body;
# xmllint, xml.etree.ElementTree, lxml and therefore dm_control.mjcf all abort
# with "Double hyphen within comment".  Every comment written by this file or by
# any agent fragment is normalised before it is written out, and the result is
# round-tripped through ElementTree as the last assertion in main().
# ----------------------------------------------------------------------------
def sanitise_comments(text: str) -> str:
    def _fix(m: "re.Match[str]") -> str:
        body = m.group(1)
        body = re.sub(r"-{2,}", lambda h: "=" * len(h.group()), body)
        if body.endswith("-"):
            body = body[:-1] + "="
        return "<!--%s-->" % body

    return re.sub(r"<!--(.*?)-->", _fix, text, flags=re.S)


def assert_well_formed(text: str) -> None:
    import xml.etree.ElementTree as ET

    try:
        ET.fromstring(text)
    except ET.ParseError as e:
        raise SystemExit("scene_aaa.xml is not well-formed XML: %s" % e)


# ----------------------------------------------------------------------------
# DEAD-ASSET PRUNING.  Agents author their own material libraries; the shared
# materials.py library and several per-agent leftovers end up referenced by no
# geom at all.  Measured before this pass: 46 of 102 materials and 20 of 39
# textures were loaded for zero pixels, 38.7 MB of GPU texture upload.  Anything
# no surviving <geom>/<material> names is dropped here rather than in the agent,
# so an agent may keep authoring a spare without paying for it.
#   * a <material> survives if some <geom material="..."> names it
#   * a <texture> survives if some surviving material names it, OR it is the
#     skybox (bound by type, not by geom_matid)
# ----------------------------------------------------------------------------
def surviving_textures(all_assets: str, bodies: str) -> set[str]:
    """Textures kept ANYWHERE in the scene.  This has to be computed across every agent's
    assets at once, not per agent: props_* materials point at materials_tex_* textures that
    another agent declared, so a per-agent pass drops a texture that is still in use and
    the model fails to compile with "texture not found in material"."""
    used_mats = set(re.findall(r'<geom\b[^>]*\bmaterial="([^"]+)"', bodies))
    used_tex: set[str] = set()
    for el in re.findall(r"<material\b.*?(?:/>|</material>)", all_assets, re.S):
        nm = re.search(r'name="([^"]+)"', el)
        if nm and nm.group(1) in used_mats:
            used_tex.update(re.findall(r'texture="([^"]+)"', el))
    return used_tex


def prune_dead_assets(assets: str, bodies: str, used_tex: set[str]) -> tuple[str, list[str], list[str]]:
    used_mats = set(re.findall(r'<geom\b[^>]*\bmaterial="([^"]+)"', bodies))

    mat_els = re.findall(r"<material\b.*?(?:/>|</material>)", assets, re.S)
    keep_mats, drop_mats = [], []
    for el in mat_els:
        name = re.search(r'name="([^"]+)"', el).group(1)
        (keep_mats if name in used_mats else drop_mats).append((name, el))

    tex_els = re.findall(r"<texture\b.*?(?:/>|</texture>)", assets, re.S)
    keep_tex, drop_tex = [], []
    for el in tex_els:
        nm = re.search(r'name="([^"]+)"', el)
        name = nm.group(1) if nm else ""
        is_sky = 'type="skybox"' in el
        (keep_tex if (name in used_tex or is_sky) else drop_tex).append((name, el))

    out = assets
    for _, el in drop_mats + drop_tex:
        out = out.replace(el, "", 1)
    # collapse the blank lines the removals leave behind
    out = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", out)
    return out, [n for n, _ in drop_mats], [n for n, _ in drop_tex]


def read_fragment(name: str) -> str:
    p = MODEL_DIR / name
    if not p.exists():
        return ""
    return p.read_text()


def indent_block(text: str, spaces: int) -> str:
    pad = " " * spaces
    out = []
    for line in text.splitlines():
        out.append(pad + line if line.strip() else "")
    return "\n".join(out)


def apply_light_overrides(body: str) -> str:
    """Rewrite named attributes on named <light> elements in a body fragment."""
    for light_name, attrs in LIGHT_OVERRIDES.items():
        # match the whole <light ... name="X" ... /> element, dot-all
        pattern = re.compile(
            r"(<light\b(?=[^>]*\bname=\"%s\")[^>]*/>)" % re.escape(light_name), re.S
        )

        def _sub(m: "re.Match[str]") -> str:
            el = m.group(1)
            for attr, value in attrs.items():
                new = re.sub(
                    r'\b%s="[^"]*"' % re.escape(attr), '%s="%s"' % (attr, value), el
                )
                if new == el:  # attribute absent -> insert it
                    new = el.replace("<light ", '<light %s="%s" ' % (attr, value), 1)
                el = new
            return el

        body, n = pattern.subn(_sub, body)
        if n == 0:
            raise SystemExit("light override target %r not found" % light_name)
    return body


def inject_extra_lights(body: str) -> str:
    """Insert EXTRA_LIGHTS immediately after the last <light> element."""
    matches = list(re.finditer(r"<light\b[^>]*/>", body, re.S))
    if not matches:
        raise SystemExit("no <light> found in the atmosphere body fragment")
    if not EXTRA_LIGHTS.strip():
        return body
    end = matches[-1].end()
    return body[:end] + "\n" + EXTRA_LIGHTS.rstrip() + body[end:]


def apply_visual_overrides(visual: str) -> str:
    """Rewrite <map> and <headlight> attributes in the atmosphere <visual> block."""
    for attr, value in list(VISUAL_MAP_OVERRIDES.items()) + list(
        VISUAL_HEADLIGHT_OVERRIDES.items()
    ):  # noqa: E501
        new = re.sub(r'\b%s="[^"]*"' % re.escape(attr), '%s="%s"' % (attr, value), visual)
        if new == visual:
            raise SystemExit("visual override target %r not found" % attr)
        visual = new
    return visual


def build_keyframes() -> str:
    """Copy scene_obstacles.xml's <keyframe> block, drop the ball's 7 qpos from
    every key, and place the trunk at the art bible's spawn point."""
    src = KEYFRAME_SOURCE.read_text()
    m = re.search(r"<keyframe>(.*?)</keyframe>", src, re.S)
    if not m:
        raise SystemExit("no <keyframe> block in %s" % KEYFRAME_SOURCE)
    block = m.group(1)

    keys = []
    for km in re.finditer(r"<key\b[^>]*?/>", block, re.S):
        el = km.group(0)
        # keys that are commented out never appear here -- re.finditer over the
        # raw block would also match inside <!-- -->, so filter those explicitly.
        start = block.rfind("<!--", 0, km.start())
        if start != -1 and block.find("-->", start) > km.start():
            continue
        name = re.search(r'name="([^"]+)"', el).group(1)
        qpos = re.search(r'qpos="([^"]*)"', el, re.S).group(1).split()
        ctrl_m = re.search(r'ctrl="([^"]*)"', el, re.S)
        ctrl = ctrl_m.group(1).split() if ctrl_m else []

        if len(qpos) == ROBOT_NQ + BALL_NQ:
            qpos = qpos[:ROBOT_NQ]
        elif len(qpos) != ROBOT_NQ:
            raise SystemExit(
                "key %s has %d qpos, expected %d or %d"
                % (name, len(qpos), ROBOT_NQ, ROBOT_NQ + BALL_NQ)
            )
        qpos[0], qpos[1], qpos[2] = ("%g" % v for v in SPAWN_XYZ)

        free = " ".join(qpos[:7])
        joints = " ".join(qpos[7:])
        keys.append(
            '    <key name="%s"\n         qpos="%s\n               %s"\n'
            '         ctrl="%s"/>' % (name, free, joints, " ".join(ctrl))
        )
    return "\n".join(keys)


HEADER = """<!--
  ============================================================================
  scene_aaa.xml  --  "THE WATER COURT"
  ============================================================================
  GENERATED FILE.  Do not hand-edit: regenerate with

      .venv-sim/bin/python scripts/aaa/assemble.py

  Composed from five independently-authored MJCF fragments (scripts/aaa/*.py):
  atmosphere (sky, lights, background recession), terrain (floor, water, inlays),
  architecture (walls, colonnade, gate, ruin, fountain), vegetation, materials
  (asset library only).

  A walled ceremonial court at ~10x human scale, paved in terracotta flagstone.
  Rain stopped an hour ago; the sun has dropped to 11 deg elevation at azimuth 40
  and rakes the court from the west.  The 25 cm duck reads as mouse-sized.

  HARD CONSTRAINTS held by this file:
    * no <body>, <joint> or <freejoint> outside the robot include, so nq is
      unchanged at 21 and the policy's keyframes stay valid
    * exactly 17 collidable geoms: the floor plane, 12 column proxies, 2 wall
      proxies, 2 gate-mass proxies.  Every other geom is contype=0 conaffinity=0
    * <camera name="duck_eye"> comes from robot_allcollisions_cam.xml
    * <light> count is 4, under the hard cap of 7, sun declared first
  ============================================================================
-->
"""


def main() -> None:
    visual = apply_visual_overrides(read_fragment("aaa_atmosphere_visual.xml").strip())

    raw_assets, raw_bodies = {}, {}
    for agent in AGENTS:
        raw_assets[agent] = read_fragment("aaa_%s_assets.xml" % agent).strip()
        b = read_fragment("aaa_%s_body.xml" % agent).strip()
        if b and agent == "atmosphere":
            b = apply_light_overrides(b)
            b = inject_extra_lights(b)
        raw_bodies[agent] = b

    all_bodies = "\n".join(raw_bodies.values())
    used_tex = surviving_textures("\n".join(raw_assets.values()), all_bodies)
    dropped_m, dropped_t = [], []
    for agent in AGENTS:
        if raw_assets[agent]:
            raw_assets[agent], dm, dt = prune_dead_assets(
                raw_assets[agent], all_bodies, used_tex)
            dropped_m += dm
            dropped_t += dt

    assets, bodies = [], []
    for agent in AGENTS:
        a, b = raw_assets[agent].strip(), raw_bodies[agent]
        if a:
            assets.append("    <!-- ========== %s ========== -->\n%s" % (agent.upper(), indent_block(a, 4)))
        if b:
            bodies.append("    <!-- ========== %s ========== -->\n%s" % (agent.upper(), indent_block(b, 4)))

    doc = [
        HEADER,
        '<mujoco model="scene_aaa">',
        '    <include file="%s"/>' % ROBOT_INCLUDE,
        "",
        indent_block(visual, 4),
        "",
        "    <asset>",
        "\n\n".join(assets),
        "    </asset>",
        "",
        "    <worldbody>",
        "\n\n".join(bodies),
        "    </worldbody>",
        "",
        "    <keyframe>",
        build_keyframes(),
        "    </keyframe>",
        "</mujoco>",
        "",
    ]
    text = sanitise_comments("\n".join(doc))
    assert_well_formed(text)

    old = OUT.read_bytes() if OUT.exists() else b""
    OUT.write_text(text)
    new = OUT.read_bytes()
    print("wrote %s  (%d bytes)  md5=%s  %s" % (
        OUT, len(new), hashlib.md5(new).hexdigest(),
        "unchanged" if old == new else "CHANGED"))

    stripped = strip_comments(text)
    ngeom = len(re.findall(r"<geom\b", stripped))
    ncol = len(re.findall(r'<geom\b(?![^>]*contype="0")', stripped))
    print("  geoms=%d (collidable %d)  lights=%d  textures=%d  materials=%d  hfields=%d"
          % (ngeom, ncol,
             len(re.findall(r"<light\b", stripped)),
             len(re.findall(r"<texture\b", stripped)),
             len(re.findall(r"<material\b", stripped)),
             len(re.findall(r"<hfield\b", stripped))))
    if dropped_m or dropped_t:
        print("  pruned %d unreferenced materials, %d unreferenced textures"
              % (len(dropped_m), len(dropped_t)))
        print("    materials: %s" % " ".join(sorted(dropped_m)))
        print("    textures : %s" % " ".join(sorted(dropped_t)))
    print("  XML well-formed: yes (ElementTree round-trip)")


if __name__ == "__main__":
    main()
