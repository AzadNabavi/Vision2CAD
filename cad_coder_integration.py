"""
CAD-Coder Integration — cad_coder_integration.py
==================================================
Integrates the open-source CAD-Coder model (CADCODER/CAD-Coder on HuggingFace)
into the existing multi-agent pipeline.

CAD-Coder takes a CAD drawing image and directly generates CadQuery Python code
that builds the 3D part as a solid model.

This replaces Passes 1-5 of our GPT pipeline with a single specialized model.
The GPT pipeline still runs in parallel for comparison and JSON output.
"""

import os
import sys
import json
import base64
import subprocess
import textwrap
from pathlib import Path

# Fix Windows console unicode printing
sys.stdout.reconfigure(encoding="utf-8")

# ─── Configuration ─────────────────────────────────────────────────────────────
DRAWING_PATH   = "technical_drawing.jpg"   # The original input image
OUTPUT_DIR     = "cadcoder_output"          # Directory for all outputs
HF_MODEL_ID    = "CADCODER/CAD-Coder"      # HuggingFace model identifier
DEVICE         = "auto"                     # "cuda", "cpu", or "auto"


# ─── Helpers ───────────────────────────────────────────────────────────────────
def sep(title):
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


# Path to the Python interpreter in our isolated CAD-Coder environment
CADCODER_PYTHON = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cadcoder_env", "Scripts", "python.exe"
)
CADCODER_RUNNER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "cadcoder_runner.py"
)


def run_cad_coder(image_path):
    """
    Calls cadcoder_runner.py inside the isolated Python 3.10 virtual environment.

    This approach keeps CAD-Coder's old transformers==4.37 dependency
    completely separate from our main Python environment.

    Returns the generated CadQuery code string.
    """
    sep("Running CAD-Coder Inference")

    if not os.path.exists(CADCODER_PYTHON):
        print("  [Error] cadcoder_env not found.")
        print("  Run: py -3.10 -m venv cadcoder_env && cadcoder_env\\Scripts\\python.exe -m pip install transformers==4.37.2 torch torchvision Pillow accelerate git+https://github.com/haotian-liu/LLaVA.git")
        return None

    print(f"  Image   : {image_path}")
    print(f"  Runner  : {CADCODER_RUNNER}")
    print("  Calling cadcoder_runner.py in Python 3.10 venv (model download ~7GB on first run)...")

    result = subprocess.run(
        [CADCODER_PYTHON, CADCODER_RUNNER, image_path],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    # Progress/status messages go to stderr in the runner
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            print(f"  {line}")

    if result.returncode != 0:
        print(f"\n  [Error] cadcoder_runner.py exited with code {result.returncode}")
        print("  Stderr:", result.stderr[-500:])
        return None

    code = result.stdout.strip()
    if not code:
        print("  [Error] No output generated.")
        return None

    print("\n  === Generated CadQuery Code ===")
    for line in code.split("\n"):
        print(f"  {line}")
    return code


# ─── Step 3: Save and Execute the CadQuery Code ────────────────────────────────
def execute_cadquery(code, output_dir):
    """
    Saves the generated CadQuery Python code to a file and executes it.
    If CadQuery is installed, this produces a .STEP file (usable in any CAD software).
    
    We wrap the code to auto-export a STEP file named 'cad_coder_result.step'.
    """
    sep("Executing Generated CadQuery Code")
    ensure_dir(output_dir)

    # Inject STEP export at the end if not already present
    export_snippet = textwrap.dedent(f"""

    # Auto-export to STEP (injected by cad_coder_integration.py)
    import os
    _step_path = os.path.join(r"{output_dir}", "cad_coder_result.step")
    try:
        result.val().exportStep(_step_path)
        print(f"[STEP Export] Saved to {{_step_path}}")
    except Exception as _e:
        print(f"[STEP Export] Could not export STEP file: {{_e}}")
    """)

    # Write the code file
    code_path = os.path.join(output_dir, "generated_cadquery.py")
    with open(code_path, "w", encoding="utf-8") as f:
        f.write(code)
        f.write(export_snippet)
    print(f"  Code saved to: {code_path}")

    # Try to execute it
    print("  Executing code...")
    result = subprocess.run(
        [sys.executable, code_path],
        capture_output=True, text=True, encoding="utf-8"
    )

    if result.returncode == 0:
        print(f"  [OK] Execution successful!")
        print("  Output:", result.stdout[:500])
    else:
        print(f"  [Warning] Execution failed (CadQuery may not be installed).")
        print("  Error:", result.stderr[:500])
        print("  Install CadQuery with: conda install -c conda-forge cadquery")

    return code_path


# ─── Step 4: Compare with GPT Pipeline ─────────────────────────────────────────
def run_gpt_comparison(drawing_path, openai_key):
    """
    Runs our existing multi_agent_council.py pipeline alongside CAD-Coder.
    This gives us both:
      - CAD-Coder output: executable CadQuery 3D model
      - GPT output: structured JSON parameters + highlighted image
    Comparing both lets us validate which approach is more accurate.
    """
    sep("Running GPT Multi-Agent Pipeline for Comparison")
    if not openai_key:
        print("  No OpenAI key provided — skipping GPT comparison.")
        return None

    # Temporarily set the key in env so multi_agent_council can use it
    env = os.environ.copy()
    env["OPENAI_KEY"] = openai_key

    result = subprocess.run(
        [sys.executable, "multi_agent_council.py"],
        capture_output=True, text=True, encoding="utf-8", env=env
    )
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("  GPT Pipeline error:", result.stderr[:500])
    return result.stdout


# ─── Main Entry Point ──────────────────────────────────────────────────────────
def main():
    sep("CAD-Coder + Multi-Agent Pipeline")
    print("  This pipeline combines two approaches:")
    print("  1. CAD-Coder (open-source VLM): image → CadQuery 3D code")
    print("  2. GPT Multi-Agent: image → JSON parameters + highlighted image\n")

    openai_key = input("Enter your OpenAI API Key (leave blank to skip GPT comparison): ").strip()
    drawing    = input(f"Enter path to CAD drawing image [{DRAWING_PATH}]: ").strip()
    if not drawing:
        drawing = DRAWING_PATH

    if not os.path.exists(drawing):
        print(f"Error: File '{drawing}' not found.")
        sys.exit(1)

    ensure_dir(OUTPUT_DIR)

    # ── CAD-Coder Pass ──────────────────────────────────────────────────────────
    cadquery_code = run_cad_coder(drawing)
    if cadquery_code:
        code_path = execute_cadquery(cadquery_code, OUTPUT_DIR)

        summary = {
            "model": HF_MODEL_ID,
            "input_image": drawing,
            "cadquery_code_path": code_path,
            "step_file": os.path.join(OUTPUT_DIR, "cad_coder_result.step"),
        }
        with open(os.path.join(OUTPUT_DIR, "cadcoder_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Summary saved to {OUTPUT_DIR}/cadcoder_summary.json")
    else:
        print("\n  [Skipped] CAD-Coder produced no output.")

    # ── GPT Comparison Pass ─────────────────────────────────────────────────────
    if openai_key:
        run_gpt_comparison(drawing, openai_key)

    sep("Complete")
    print(f"  Outputs in   : {OUTPUT_DIR}/")
    print(f"  CadQuery code: {OUTPUT_DIR}/generated_cadquery.py")
    print(f"  STEP file    : {OUTPUT_DIR}/cad_coder_result.step  (if CadQuery installed)")
    print(f"  GPT JSON     : final_council_extracted_parameters.json")
    print(f"  GPT Highlight: agentic_highlighted_result_v10.jpg")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
