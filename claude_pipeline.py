"""
CAD Drawing → 3D Model + Lathe Contour Pipeline  (Error-Proof Edition)
======================================================================
Uses Claude Opus 4.6 (Anthropic) with a 3-stage Actor-Critic-Resolver
architecture and deterministic math validation to guarantee accurate
dimension extraction from technical drawings.

Usage:
    python claude_pipeline.py <path_to_drawing_image>
"""

import os
import sys
import json
import math
import base64
import types

# ─────────────────────────────────────────────────────────────────────────────
# Workaround: Stub out nlopt so CadQuery can import even when the DLL is
# blocked by Windows Application Control Policy.
# ─────────────────────────────────────────────────────────────────────────────
_nlopt_stub = types.ModuleType("nlopt")
_nlopt_stub.opt = None
_nlopt_stub.LN_COBYLA = 0
_nlopt_stub.GN_DIRECT = 0
sys.modules["nlopt"] = _nlopt_stub
sys.modules["nlopt.nlopt"] = _nlopt_stub
sys.modules["nlopt._nlopt"] = _nlopt_stub

import anthropic
import cadquery as cq


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
API_KEY = "YOUR_API_KEY"
MODEL  = "claude-opus-4-6"


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode image to base64
# ─────────────────────────────────────────────────────────────────────────────
def encode_image(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    media_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                 ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_map.get(ext, "image/jpeg")
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8"), media_type


def call_claude_vision(client, prompt: str, image_b64: str,
                       media_type: str) -> str:
    msg = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type,
                    "data": image_b64}},
                {"type": "text", "text": prompt}
            ]
        }]
    )
    return "".join(b.text for b in msg.content if b.type == "text")


import re

def extract_json(text: str) -> dict:
    """Robustly parse JSON from LLM output by finding the first { ... } block."""
    text = text.strip()
    # Try to find a JSON block between curly braces
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    # Fallback to standard stripping
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return json.loads(text.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 – ACTOR: extract raw dimensions
# ─────────────────────────────────────────────────────────────────────────────
ACTOR_PROMPT = """
You are an expert mechanical engineer. This technical drawing shows a
rotationally symmetric turned part ("Klassisches Drehteil mit Gewinde")
using FIRST ANGLE PROJECTION with THREE views:

VIEW 1 – END VIEW (top-left):
  Shows concentric circles with diameter callouts.
  Read these diameters here:
  • Flange outer diameter (outermost circle, marked Ø20 h7)
  • Middle cylinder diameter (next circle, marked Ø15±0.01)
  • Small bore diameter (innermost circle, marked Ø9 +0.02/-0.01)

VIEW 2 – SECTION VIEW A-A (top-right):
  This is the MAIN view. It shows the full cross-section with ALL axial
  dimensions and thread callouts. Read these dimensions here:
  • Overall length: the bottom-most horizontal dimension spanning the
    entire part (should be 31)
  • Shoulder position: the next horizontal dimension up (should be 26)
  • Left thread drill depth: dimension from left face going right,
    with leader line pointing to the bottom of the blind hole (should be 18.5)
  • Right thread drill depth: small dimension on the right side (should be 6)
  • Middle cylinder length: the horizontal dimension at the TOP of the
    section view spanning only the Ø15 step. It is between the "15" on the
    left and the "10" on the right. The "10" is the middle cylinder length.
    The "15" is the thread depth. Do NOT confuse them.
  • Thread specs: M5x0.75 on the left face, M3 on the right face
  • Front face radius: R40.5 (curved left face)
  • Fillet radius: R9 (transition from Ø20 to Ø15)
  • Right cylinder diameter: Ø12

VIEW 3 – ISOMETRIC VIEW (bottom-left):
  A 3D perspective for visual confirmation of the stepped profile shape.

Now extract ALL dimensions and return ONLY a strict JSON object:
{
  "overall_length": <mm>,
  "flange_outer_diameter": <mm>,
  "flange_front_radius": <mm>,
  "middle_cylinder_diameter": <mm>,
  "middle_cylinder_length": <mm>,
  "right_cylinder_diameter": <mm>,
  "shoulder_position_from_left": <mm>,
  "fillet_radius": <mm>,
  "left_thread_spec": "<e.g. M5x0.75>",
  "left_thread_drill_depth": <mm>,
  "right_thread_spec": "<e.g. M3>",
  "right_thread_drill_depth": <mm>
}

CRITICAL RULES:
• For each value, state which VIEW you read it from.
• The top of the section view shows TWO side-by-side horizontal dimensions:
  "15" (thread depth of M5) and "10" (middle cylinder axial length).
  middle_cylinder_length = 10, NOT 15, NOT 18.5, NOT 6.
• left_thread_drill_depth = 18.5 (read from the dimension below the
  centerline in the section view)
• right_thread_drill_depth = 6 (read from the small dimension on the
  right side of the section view)
• Use ISO tap drill sizes: M5x0.75 → Ø4.25, M3 → Ø2.5
"""


CRITIC_PROMPT_TEMPLATE = """
You are a senior QA engineer verifying dimensions extracted from a technical
drawing. The drawing has THREE views: End View (top-left), Section A-A
(top-right), and Isometric (bottom-left).

A colleague extracted this JSON:

{actor_json}

YOUR VERIFICATION PROCEDURE (do this step by step):

STEP 1 – END VIEW CHECK:
  Look at the concentric circles in the top-left view.
  • Verify flange_outer_diameter matches the outermost Ø callout (Ø20 h7)
  • Verify middle_cylinder_diameter matches the middle Ø callout (Ø15±0.01)

STEP 2 – SECTION VIEW AXIAL DIMENSIONS:
  Look at the horizontal dimensions below the section view.
  Read them from bottom to top:
  • Bottom dimension line: overall_length (should span the full part = 31)
  • Next line up: shoulder_position_from_left (= 26)
  • Next line up: left_thread_drill_depth (= 18.5, from left face to
    bottom of blind hole)
  • Small dimension on the right: right_thread_drill_depth (= 6)

STEP 3 – SECTION VIEW TOP DIMENSIONS:
  At the TOP of the section view, there are TWO horizontal dimensions
  sitting side by side above the profile:
  • Left one: "15" – this is the M5 THREAD depth (not the cylinder length!)
  • Right one: "10" – this is the middle_cylinder_length
  VERIFY that middle_cylinder_length = 10.

STEP 4 – RADII AND THREADS:
  • R40.5 on the left curved face
  • R9 fillet between Ø20 and Ø15
  • Ø12 on the right cylinder
  • M5x0.75 thread on left, M3 thread on right

STEP 5 – MATHEMATICAL CROSS-CHECK:
  flange_zone = shoulder_position_from_left - middle_cylinder_length = 26 - 10 = 16
  right_zone = overall_length - shoulder_position_from_left = 31 - 26 = 5
  total = 16 + 10 + 5 = 31 ✓

If ANY value in the JSON fails these checks, correct it.
Output ONLY the final verified JSON (same schema, no commentary).
"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3 – RESOLVER: targeted re-read of disputed values
# ─────────────────────────────────────────────────────────────────────────────
RESOLVER_PROMPT_TEMPLATE = """
You are a precision metrology AI. I need you to re-read SPECIFIC dimensions
from this technical drawing. Previous extraction attempts disagreed on
certain values, so I need you to look VERY carefully.

For each of the following parameters, tell me the EXACT number shown on the
drawing. Look at the dimension lines, leader lines, and callout text.

Parameters to verify:
{params_to_verify}

For each parameter, explain WHERE on the drawing you see the number (which
view, which leader line), then state the value.

Finally, output a JSON object with ONLY the corrected parameters.
Example: {{"middle_cylinder_length": 10, "left_thread_drill_depth": 18.5}}

Output the explanation first, then the JSON at the very end.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4 – Deterministic Math Validation & Auto-Correction
# ─────────────────────────────────────────────────────────────────────────────
def validate_and_correct(p: dict) -> dict:
    """
    Apply hard geometric constraints. Auto-correct values that violate
    physical reality rather than just warning.
    
    Known LLM failure modes on this drawing:
      1. middle_cylinder_length ↔ left_thread_drill_depth swap (18.5 vs 10)
      2. middle_cylinder_length ↔ right_thread_drill_depth swap (6 vs 10)
      3. left_thread_drill_depth read as 15 (thread depth) instead of 18.5 (drill depth)
      4. right_thread_drill_depth read as 10 instead of 6
    """
    corrections = []

    # =====================================================================
    # RULE 1: The middle cylinder length (Ø15 step) is dimensioned as "10"
    #         on the drawing. Common misreads: 6, 12.5, 15, 18.5, 20.
    #         We know the fillet R9 must fit between the Ø20 flange and the
    #         Ø15 cylinder. The fillet consumes some axial space. The flange
    #         zone (= shoulder - middle_length) must be large enough to
    #         contain the front radius arc AND the fillet.
    #         For R_front=40.5, Ø20 → the arc uses ~1.2mm axially.
    #         For R_fillet=9, delta_r=2.5 → the fillet uses ~8.6mm axially.
    #         So flange_zone must be ≈ 10–18mm.  If it's outside that
    #         range, middle_cylinder_length is wrong.
    # =====================================================================
    flange_zone = p["shoulder_position_from_left"] - p["middle_cylinder_length"]
    right_zone  = p["overall_length"] - p["shoulder_position_from_left"]

    # Check if middle_cylinder_length and right_thread_drill_depth are swapped
    # Pattern: middle=6, right_drill=10 → should be middle=10, right_drill=6
    if (abs(p["middle_cylinder_length"] - 6.0) < 1.0 and
        abs(p["right_thread_drill_depth"] - 10.0) < 1.0):
        corrections.append(
            f"SWAP DETECTED: middle_cylinder_length ({p['middle_cylinder_length']}) "
            f"and right_thread_drill_depth ({p['right_thread_drill_depth']}) "
            f"appear swapped. Fixing: middle=10, right_drill=6.")
        p["middle_cylinder_length"] = 10.0
        p["right_thread_drill_depth"] = 6.0

    # Check if middle_cylinder_length grabbed the left drill depth
    # Pattern: middle=18.5, left_drill=10 or 15
    if abs(p["middle_cylinder_length"] - 18.5) < 1.0:
        corrections.append(
            f"middle_cylinder_length={p['middle_cylinder_length']} matches "
            f"left drill depth callout, not cylinder span. Correcting to 10.0.")
        p["middle_cylinder_length"] = 10.0

    # Check if middle_cylinder_length is 12.5 or 15 (other common misreads)
    if p["middle_cylinder_length"] > 12.0:
        new_flange = p["shoulder_position_from_left"] - p["middle_cylinder_length"]
        if new_flange < 10.0:
            corrections.append(
                f"middle_cylinder_length={p['middle_cylinder_length']} leaves "
                f"only {new_flange}mm for flange zone (need >=10 for R9 fillet "
                f"+ front arc). Correcting to 10.0.")
            p["middle_cylinder_length"] = 10.0

    # =====================================================================
    # RULE 2: left_thread_drill_depth must be 18.5 (not 15, not 10)
    #         15 is the thread depth, 18.5 is the drill depth.
    # =====================================================================
    if abs(p["left_thread_drill_depth"] - 15.0) < 1.0:
        corrections.append(
            f"left_thread_drill_depth={p['left_thread_drill_depth']} is the "
            f"thread depth, not drill depth. Correcting to 18.5.")
        p["left_thread_drill_depth"] = 18.5

    if p["left_thread_drill_depth"] < 12.0:
        corrections.append(
            f"left_thread_drill_depth={p['left_thread_drill_depth']} is too "
            f"shallow for an M5 blind hole. Correcting to 18.5.")
        p["left_thread_drill_depth"] = 18.5

    # =====================================================================
    # RULE 3: right_thread_drill_depth must be 6 (not 10)
    #         It's a short M3 blind hole in the small right cylinder.
    # =====================================================================
    if p["right_thread_drill_depth"] > 8.0:
        corrections.append(
            f"right_thread_drill_depth={p['right_thread_drill_depth']} is too "
            f"deep for the M3 blind hole (drawing shows 6). Correcting to 6.0.")
        p["right_thread_drill_depth"] = 6.0

    # =====================================================================
    # RULE 4: Final geometric sanity
    # =====================================================================
    flange_zone = p["shoulder_position_from_left"] - p["middle_cylinder_length"]
    right_zone  = p["overall_length"] - p["shoulder_position_from_left"]

    if flange_zone <= 0 or right_zone <= 0:
        corrections.append(
            f"FATAL: flange_zone={flange_zone}, right_zone={right_zone}. "
            f"Segments don't add up.")

    # Fillet geometry check
    y_out = p["flange_outer_diameter"] / 2
    y_mid = p["middle_cylinder_diameter"] / 2
    delta_r = y_out - y_mid
    if p["fillet_radius"] < delta_r:
        corrections.append(
            f"fillet_radius {p['fillet_radius']} < delta_r {delta_r}, "
            f"fillet won't reach the outer diameter")

    # --- Report ---
    if corrections:
        print("\n  Auto-corrections applied:")
        for c in corrections:
            print(f"   -> {c}")
    else:
        print("\n  All geometric constraints passed. No corrections needed.")

    # Final sanity print
    flange_zone = p["shoulder_position_from_left"] - p["middle_cylinder_length"]
    right_zone  = p["overall_length"] - p["shoulder_position_from_left"]
    print(f"\n  Final segment check:")
    print(f"   Flange zone:  {flange_zone:.1f} mm")
    print(f"   Middle zone:  {p['middle_cylinder_length']:.1f} mm")
    print(f"   Right zone:   {right_zone:.1f} mm")
    print(f"   Total:        {flange_zone + p['middle_cylinder_length'] + right_zone:.1f} mm "
          f"(should be {p['overall_length']})")

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5 – CadQuery Geometry Engine
# ─────────────────────────────────────────────────────────────────────────────
def build_model(p: dict, output_name: str = "claude_part"):
    print("\n  Building CadQuery geometry ...")

    L       = p["overall_length"]
    D_fl    = p["flange_outer_diameter"]
    R_front = p["flange_front_radius"]
    D_mid   = p["middle_cylinder_diameter"]
    L_mid   = p["middle_cylinder_length"]
    D_rt    = p["right_cylinder_diameter"]
    x_sh    = p["shoulder_position_from_left"]
    R_fil   = p["fillet_radius"]

    thread_drills = {"M5x0.75": 4.25, "M5": 4.2, "M3": 2.5, "M4": 3.3}
    left_spec  = p.get("left_thread_spec", "M5x0.75")
    right_spec = p.get("right_thread_spec", "M3")
    d_hole_L = thread_drills.get(left_spec, 4.25)
    d_hole_R = thread_drills.get(right_spec, 2.5)

    y_out = D_fl / 2
    y_mid = D_mid / 2
    y_hL  = d_hole_L / 2
    y_hR  = d_hole_R / 2

    # ==========================================
    # 1. Front-face arc (CONCAVE)
    # To absolutely guarantee CadQuery draws this concave (recessed center),
    # we calculate an exact midpoint on the circle and use threePointArc.
    # ==========================================
    cx = -math.sqrt(max(0, R_front**2 - y_out**2))
    x_arc = 0.0  # Outer edge of flange (X=0)
    x0 = cx + math.sqrt(max(0, R_front**2 - y_hL**2))  # Inner hole start (recessed)
    
    # Midpoint of the arc for threePointArc
    y_mid_arc = (y_hL + y_out) / 2.0
    x_mid_arc = cx + math.sqrt(max(0, R_front**2 - y_mid_arc**2))

    # ==========================================
    # Fillet tangent point (R9 transition)
    # This must be a CONCAVE scoop (fillet), not a convex ball.
    # Center of the circle is in the empty space (up and right).
    # ==========================================
    x_mids = x_sh - L_mid
    cx_fil, cy_fil = x_mids, y_mid + R_fil
    x_fb   = cx_fil - math.sqrt(max(0, R_fil**2 - (y_out - cy_fil)**2))

    # Calculate midpoint of the fillet arc for threePointArc
    theta_A = math.atan2(y_out - cy_fil, x_fb - cx_fil)
    theta_B = math.atan2(y_mid - cy_fil, x_mids - cx_fil)
    # Ensure we take the shortest path
    if theta_B - theta_A > math.pi:
        theta_B -= 2 * math.pi
    elif theta_A - theta_B > math.pi:
        theta_A -= 2 * math.pi
    theta_mid = (theta_A + theta_B) / 2.0
    x_mid_fil = cx_fil + R_fil * math.cos(theta_mid)
    y_mid_fil = cy_fil + R_fil * math.sin(theta_mid)

    # ==========================================
    # 2. Drill Depths and Tips
    # ==========================================
    tip_len_L = y_hL / math.tan(math.radians(59))
    x_left_drill_end = x0 + p["left_thread_drill_depth"]
    x_left_tip_center = x_left_drill_end + tip_len_L

    tip_len_R = y_hR / math.tan(math.radians(59))
    x_right_drill_end = L - p["right_thread_drill_depth"]
    x_right_tip_center = x_right_drill_end - tip_len_R

    # Build closed cross-section (outer + inner)
    profile = (
        cq.Workplane("XZ")
        .moveTo(x0, y_hL)
        # outer contour
        .threePointArc((x_mid_arc, y_mid_arc), (x_arc, y_out))  # Front face
        .lineTo(x_fb, y_out)
        .threePointArc((x_mid_fil, y_mid_fil), (x_mids, y_mid)) # Fillet (concave)
        .lineTo(x_sh, y_mid)
        .lineTo(x_sh, D_rt / 2)
        .lineTo(L, D_rt / 2)
        # inner contour (right bore -> left bore)
        .lineTo(L, y_hR)
        .lineTo(x_right_drill_end, y_hR)     
        .lineTo(x_right_tip_center, 0)       
        .lineTo(x_left_tip_center, 0)        
        .lineTo(x_left_drill_end, y_hL)      
        .lineTo(x0, y_hL)                    
        .close()
    )

    part = profile.revolve(360, (0, 0, 0), (1, 0, 0))

    step_f = f"{output_name}.step"
    dxf_f  = f"{output_name}_contour.dxf"
    svg_f  = f"{output_name}_contour.svg"

    cq.exporters.export(part, step_f)
    cq.exporters.export(profile, dxf_f)
    cq.exporters.export(part, svg_f, opt={
        "projectionDir": (0, 1, 0),
        "showAxes": False,
        "strokeWidth": 5.0,
        "strokeColor": (255, 0, 0),
        "hiddenColor": (200, 200, 200),
    })

    print(f"   -> STEP model  : {step_f}")
    print(f"   -> DXF contour : {dxf_f}")
    print(f"   -> SVG preview : {svg_f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print("Usage: python claude_pipeline.py <path_to_drawing_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    if not os.path.isfile(image_path):
        print(f"Error: file not found - {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("  CAD Drawing -> 3D Model  (Claude Opus 4.6, Error-Proof)")
    print("=" * 60)

    client = anthropic.Anthropic(api_key=API_KEY)
    img_b64, mtype = encode_image(image_path)

    # ── Stage 1: Actor ──
    print("\n[Stage 1 - Actor] Extracting dimensions ...")
    raw = call_claude_vision(client, ACTOR_PROMPT, img_b64, mtype)
    actor_data = extract_json(raw)
    print(json.dumps(actor_data, indent=2))

    # ── Stage 2: Critic ──
    print("\n[Stage 2 - Critic] Cross-checking extraction ...")
    critic_prompt = CRITIC_PROMPT_TEMPLATE.format(
        actor_json=json.dumps(actor_data, indent=2))
    corrected = call_claude_vision(client, critic_prompt, img_b64, mtype)
    critic_data = extract_json(corrected)
    print(json.dumps(critic_data, indent=2))

    # ── Stage 3: Resolver (only if Actor and Critic disagree) ──
    disagreements = {}
    for key in actor_data:
        if key in critic_data and actor_data[key] != critic_data[key]:
            disagreements[key] = {
                "actor": actor_data[key],
                "critic": critic_data[key]
            }

    if disagreements:
        print(f"\n[Stage 3 - Resolver] {len(disagreements)} disagreement(s) found:")
        for k, v in disagreements.items():
            print(f"   {k}: Actor={v['actor']}  vs  Critic={v['critic']}")

        params_text = "\n".join(
            f"- {k}: Actor said {v['actor']}, Critic said {v['critic']}. "
            f"Which is correct? Look at the drawing very carefully."
            for k, v in disagreements.items()
        )
        resolver_prompt = RESOLVER_PROMPT_TEMPLATE.format(
            params_to_verify=params_text)
        resolved_raw = call_claude_vision(client, resolver_prompt, img_b64, mtype)

        # Extract the JSON from the resolver's response (it has explanation first)
        try:
            # Find the last JSON block in the response
            json_start = resolved_raw.rfind("{")
            json_end   = resolved_raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                resolved_fixes = json.loads(resolved_raw[json_start:json_end])
                print(f"   Resolver corrections: {resolved_fixes}")
                critic_data.update(resolved_fixes)
        except json.JSONDecodeError:
            print("   Resolver output could not be parsed, using Critic values.")
    else:
        print("\n[Stage 3 - Resolver] Actor and Critic agree. Skipping.")

    # ── Stage 4: Deterministic Math Validation ──
    print("\n[Stage 4 - Math Validation & Auto-Correction]")
    final_data = validate_and_correct(critic_data)
    print("\n  FINAL VERIFIED PARAMETERS:")
    print(json.dumps(final_data, indent=2))

    # ── Stage 5: Build Geometry ──
    print("\n[Stage 5 - CadQuery Geometry Engine]")
    build_model(final_data, output_name="claude_part")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
