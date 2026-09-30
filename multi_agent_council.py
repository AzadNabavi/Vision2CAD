"""
Multi-Agent CAD Council Pipeline — OpenAI GPT-5.6 Sol
=======================================================
Architecture:
  Pass 1 - GPT-5.6-Sol reads Section View LEFT->RIGHT: narrative description
  Pass 2 - GPT-5.6-Sol reads End View: confirms/corrects diameters
  Pass 3 - GPT-5.6-Sol reads Isometric View: confirms overall shape
  Pass 4 - GPT-5.6-Sol acts as Mechanical Designer: proofreads all three views,
            checks geometry, produces authoritative description
  Pass 5 - GPT-5.6-Sol converts authoritative description to JSON
  Pass 6 - CV pipeline runs and produces the highlighted overlay image
"""

import cv2, json, base64, os, subprocess, sys
import numpy as np
from openai import OpenAI

# Fix Windows console unicode printing
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("[Error] Please set OPENAI_API_KEY in the .env file.", file=sys.stderr)
    sys.exit(1)

VISION_MODEL = "gpt-6-astra"
TEXT_MODEL   = "gpt-6-astra"

client = OpenAI(api_key=OPENAI_KEY, timeout=600.0)

JSON_SCHEMA = """{
  "overall_length": <float mm>,
  "outer_profile": [
    List features from RIGHT (start_x=overall_length) to LEFT (end_x=0).
    {"feature": "cylinder|step|fillet|face|chamfer",
     "start_x": <float>, "end_x": <float>,
     "start_y": <float radius>, "end_y": <float radius>,
     "radius": <float, fillet/face only>,
     "type": <"concave"|"convex", fillet/face only>}
  ],
  "internal_features": [
    List features from LEFT (start_x=0) to RIGHT (end_x=overall_length).
    {"feature": "thread|hole",
     "start_x": <float>, "end_x": <float>,
     "diameter": <float full diameter>}
  ]
}"""


def crop(img, y0, y1, x0, x1, path):
    cv2.imwrite(path, img[y0:y1, x0:x1])
    # print(f"  Saved: {path}  shape={img[y0:y1, x0:x1].shape}")
    return path


def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def call_vision(img_path, prompt):
    """Call model with an image + prompt."""
    b64 = img_b64(img_path)
    try:
        r = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        content = r.choices[0].message.content
        if not content:
            return "[GPT Vision Error] Returned empty content or None"
        return content.strip()
    except Exception as e:
        return f"[GPT Vision Error] {e}"


def call_text(prompt):
    """Call GPT-5.6-Luna with text only (proofreading / JSON)."""
    try:
        r = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        return f"[GPT Text Error] {e}"


def extract_json(text):
    for fence in ("```json", "```"):
        if fence in text:
            text = text.split(fence)[1].split("```")[0]
            break
    depth, start = 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i+1]
    return text


def sep(title):
    pass
    # bar = "=" * 64
    # print(f"\n{bar}\n  {title}\n{bar}")


def main():
    img = cv2.imread("technical_drawing.jpg")
    H, W = img.shape[:2]
    os.makedirs("views", exist_ok=True)

    section_path  = crop(img,  30, 400, 380, W-20, "views/section_view.jpg")
    end_view_path = crop(img,  30, 380,  20,  380, "views/end_view.jpg")
    iso_path      = crop(img, 390, H-80, 20,  420, "views/isometric_view.jpg")

    # ------------------------------------------------------------------
    # PASS 0 — Local OCR Extraction (with Cloud Fallback)
    # ------------------------------------------------------------------
    try:
        import easyocr
        print("Initializing Local EasyOCR (PyTorch)...")
        reader = easyocr.Reader(['en'])
        section_text = ", ".join(reader.readtext(section_path, detail=0))
        end_text = ", ".join(reader.readtext(end_view_path, detail=0))
        iso_text = ", ".join(reader.readtext(iso_path, detail=0))
    except Exception as e:
        print(f"\n[Warning] Local EasyOCR failed to load (OS Security/DLL Block): {e}")
        print("[Warning] Gracefully falling back to Cloud Vision OCR...\n")
        ocr_prompt = "Act as an OCR engine. Read every piece of text, dimension, symbol, and number in this image. Return ONLY a comma-separated list of the text you see. Do not add any conversational filler."
        section_text = call_vision(section_path, ocr_prompt)
        end_text = call_vision(end_view_path, ocr_prompt)
        iso_text = call_vision(iso_path, ocr_prompt)

    # ------------------------------------------------------------------
    # PASS 1 — Section View: Left to Right Narrative
    # ------------------------------------------------------------------
    # sep("PASS 1 — Section View: Left to Right Description [GPT-Astra-6.0]")
    p1 = call_vision(section_path,
        "You are reading the SECTION VIEW (cross-section) of a turned lathe part.\n"
        f"OCR HINTS DETECTED IN THIS VIEW: [{section_text}]\n"
        "Warning: The OCR may misread symbols (e.g., reading Ø20 as 520). Use these as hints but verify with your own eyes.\n\n"
        "Starting from the LEFT face and moving continuously to the RIGHT, describe "
        "EVERY feature you encounter. For each feature state:\n\n"
        "  [X position range from left] FEATURE NAME\n"
        "  - Outer diameter: Ø?\n"
        "  - Length: ?mm\n"
        "  - Internal feature directly below (thread spec / clearance bore Ø / solid): ?\n"
        "  - Cross-section hatched (solid metal) or open (cavity): ?\n\n"
        "Cover ALL of these in order:\n"
        "  1. The shape of the left face (flat? curved? what radius?)\n"
        "  2. Each outer cylinder and its diameter\n"
        "  3. Each step between cylinders (stepping up or down in diameter?)\n"
        "  4. Any fillets (what radius? which two diameters do they blend?)\n"
        "  5. Every dimension annotation you can find on the drawing.\n"
        "  6. Thread call-outs: which bore does each belong to?\n"
        "Do not skip anything. Find and be precise with every number you can read."
    )
    # print(p1)

    # ------------------------------------------------------------------
    # PASS 2 — End View: Diameter Confirmation
    # ------------------------------------------------------------------
    # sep("PASS 2 — End View: Diameter Confirmation [GPT-Astra-6.0]")
    p2 = call_vision(end_view_path,
        "This is the END VIEW (A-A) of the same turned part showing concentric circles.\n"
        f"OCR HINTS DETECTED IN THIS VIEW: [{end_text}]\n"
        "Warning: The OCR may misread symbols. Use these as hints but verify visually.\n\n"
        f"Previous section view description:\n{p1[:1000]}\n\n"
        "Read every circle from outermost to innermost. For each circle state:\n"
        "  - Its exact labeled diameter (with tolerance if shown)\n"
        "  - Whether it is an OUTER surface or an INNER bore\n"
        "  - Which feature in the section view it corresponds to\n\n"
        "If any diameter stated in the section view description is wrong, "
        "explicitly state the correction with your reason."
    )
    # print(p2)

    # ------------------------------------------------------------------
    # PASS 3 — Isometric View: Shape Confirmation
    # ------------------------------------------------------------------
    # sep("PASS 3 — Isometric View: Shape Confirmation [GPT-Astra-6.0]")
    p3 = call_vision(iso_path,
        "This is the 3D ISOMETRIC VIEW of the same turned part.\n"
        f"OCR HINTS DETECTED IN THIS VIEW: [{iso_text}]\n"
        "Warning: The OCR may misread symbols. Use these as hints but verify visually.\n\n"
        "Describe what you see and confirm or correct the following:\n"
        "  1. How many distinct outer-diameter sections are visible? List them left to right.\n"
        "  2. Which end (left or right) has the largest diameter?\n"
        "  3. Is the left face flat or curved outward (convex dome)?\n"
        "  4. Is there a visible bore / hole on the left face?\n"
        "  5. Describe the transitions between cylinders (sharp 90° steps or blended fillets?).\n"
        "  6. State any corrections needed to the descriptions from the section and end views."
    )
    # print(p3)

    # ------------------------------------------------------------------
    # PASS 4 — Mechanical Designer Proofreading
    # ------------------------------------------------------------------
    sep("PASS 4 — Mechanical Designer Proofreading [GPT-5.6-Sol]")
    p4_prompt = (
        "You are a senior mechanical designer approving a turned lathe part for manufacturing.\n"
        "You have three view descriptions. Perform a full engineering review:\n\n"
        "STEP A — Resolve contradictions:\n"
        "  List every diameter, length, or feature that differs between the three views.\n"
        "  State which view is authoritative and why.\n\n"
        "STEP B — Engineering checks:\n"
        "  1. List every outer cylinder length. Verify they sum to the overall length found in the drawing.\n"
        "  2. List every internal feature length. Verify they sum to the overall length.\n"
        "  3. For any fillet found: identify the two radii it connects (r1 and r2).\n"
        "     Compute: dx = horizontal span, dy = r1 - r2.\n"
        "     Verify tangency: sqrt(dx^2 + dy^2) <= 2 * fillet_radius. Flag if impossible.\n"
        "  4. The smallest concentric circle in the end view: is it an outer surface or an inner bore?\n"
        "     State your reasoning clearly.\n\n"
        "STEP C — Final authoritative description (use this exact format):\n\n"
        "OUTER PROFILE (Left to Right):\n"
        "  L1: [feature] | Dia=[value] | Length=[value]mm | x=[start] to x=[end]\n"
        "  ...\n\n"
        "INTERNAL PROFILE (Left to Right):\n"
        "  I1: [feature] | Dia=[value] | x=[start] to x=[end] | Length=[value]mm\n"
        "  ...\n\n"
        "GEOMETRIC VERIFICATION:\n"
        "  Outer sum: [each segment + total]\n"
        "  Inner sum: [each segment + total]\n"
        "  R9 fillet: dx=? dy=? dist=? <= 18? [pass/fail]\n"
        "  Issues: [list or None]\n\n"
        f"=== SECTION VIEW ===\n{p1}\n\n"
        f"=== END VIEW ===\n{p2}\n\n"
        f"=== ISOMETRIC VIEW ===\n{p3}\n"
    )
    p4 = call_text(p4_prompt)
    # print(p4)

    # ------------------------------------------------------------------
    # PASS 5 — JSON Generation
    # ------------------------------------------------------------------
    sep("PASS 5 — JSON Generation [GPT-5.6-Sol]")
    p5_prompt = (
        "Convert the following authoritative mechanical description to a JSON object.\n"
        f"Schema (output ONLY raw JSON, no markdown):\n{JSON_SCHEMA}\n\n"
        "CONVERSION RULES:\n"
        "  - outer_profile: RIGHT to LEFT (start_x decreases from 31 to 0).\n"
        "  - internal_features: LEFT to RIGHT (start_x increases from 0 to 31).\n"
        "  - start_y and end_y are RADII (= diameter / 2).\n"
        "  - step: start_x == end_x; start_y = old radius; end_y = new radius.\n"
        "  - fillet: start_y = radius at the RIGHT tangent point; end_y = radius at LEFT tangent.\n"
        "  - Thread diameters: use full minor diameter (M5x0.75 minor=4.2mm, M3 minor=2.5mm).\n"
        "  - Clearance bore: use actual bore diameter (Dia9 = 9.0).\n\n"
        f"=== AUTHORITATIVE DESCRIPTION ===\n{p4}\n"
    )
    p5_raw = call_text(p5_prompt)
    json_str = extract_json(p5_raw)

    try:
        cad_json = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[JSON parse ERROR] {e}", file=sys.stderr)
        print(f"Raw: {json_str[:500]}", file=sys.stderr)
        return

    # ------------------------------------------------------------------
    # PASS 5.5 — AI Geometric Resolution Layer
    # ------------------------------------------------------------------
    sep("PASS 5.5 — AI Geometric Resolution Layer [GPT-5.6-Sol]")
    p55_prompt = (
        "You are an expert computational geometry AI.\n"
        "I am providing you a JSON representation of a turned CAD part. Some coordinates are marked as `null`.\n"
        "These `null` values represent implicit intersections that must be calculated mathematically based on the adjacent explicit features.\n\n"
        "YOUR TASK:\n"
        "1. Identify every `null` value in the JSON.\n"
        "2. Look at the neighboring features to determine the geometric constraint (e.g., a convex face of radius R intersecting a cylinder of radius Y).\n"
        "3. Use standard trigonometry and algebra (e.g., the circle equation `(x - x_c)^2 + (y - y_c)^2 = R^2`) to calculate the exact coordinate.\n"
        "4. Replace the `null` with your calculated float value (rounded to 6 decimal places).\n"
        "5. Output ONLY the final, fully resolved raw JSON. No markdown formatting, no explanations.\n"
        "CRITICAL RULE: You are strictly forbidden from outputting error messages or complaining about 'missing constraints'. You MUST make reasonable standard geometric assumptions (e.g., assume tangency, assume circles are centered appropriately on the axis) to mathematically force a solution for every single `null` value. Return ONLY the completed JSON matching the original schema.\n\n"
        f"=== INCOMPLETE JSON ===\n{json.dumps(cad_json, indent=2)}\n"
    )
    
    p55_raw = call_text(p55_prompt)
    final_json_str = extract_json(p55_raw)
    
    try:
        final_cad_json = json.loads(final_json_str)
    except json.JSONDecodeError as e:
        print(f"[JSON parse ERROR in AI Geometric Layer] {e}", file=sys.stderr)
        final_cad_json = cad_json

    with open("final_council_extracted_parameters.json", "w") as f:
        json.dump(final_cad_json, f, indent=2)
    print(json.dumps(final_cad_json, indent=2))

    # ------------------------------------------------------------------
    # PASS 6 — CAD Generation (OpenSCAD)
    # ------------------------------------------------------------------
    sep("PASS 6 — 3D CAD Generation [OpenSCAD]")
    out = subprocess.run(
        ["python", "json_to_scad.py"],
        capture_output=True, text=True, cwd=os.getcwd()
    )
    if out.stderr:
        print("STDERR:", out.stderr[:500], file=sys.stderr)
    print("\n>>> Pipeline complete. Result saved to generated_part.scad")


if __name__ == "__main__":
    main()
