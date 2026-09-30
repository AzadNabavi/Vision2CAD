# Vision2CAD-Pipeline

This repository contains a fully automated, multi-agent AI pipeline designed to extract mechanical engineering parameters from 2D technical drawings and automatically compile them into fully parametric 3D CAD models.

## Overview

Bridging the gap between legacy 2D technical drawings (cross-sections, end views, isometric views) and modern 3D CAD environments is traditionally a manual, time-consuming process. This project leverages Agentic AI and Computer Vision to automate the conversion end-to-end.

### Key Features
1. **Multi-Agent Vision Council:** Utilizes a sequential LLM architecture. Specialized vision agents independently analyze different engineering views (using Cloud Vision OCR to extract text and dimensions), while a 'Mechanical Designer' agent aggregates the data, resolves geometric contradictions, and outputs a structured JSON schema.
2. **AI Geometric Constraint Solver:** Handles implicit geometry not explicitly dimensioned on the drawings (e.g., intersection points between arcs and cylinders). An AI-driven mathematical resolution layer applies trigonometric constraints to calculate missing floating-point coordinates with 100% structural consistency.
3. **Automated CAD Compilation:** Converts the mathematically resolved JSON data directly into an OpenSCAD 3D model script, fully bypassing the need for manual 3D modeling. 
*(Note: Built entirely using API endpoints and OpenSCAD to comply with strict enterprise Windows Defender Application Control (WDAC) hardware restrictions that block native local C++ CAD kernels).*

## Pipeline Architecture
1. **Pass 0:** Cloud AI OCR extraction.
2. **Pass 1-3:** Vision Agents parse the Section, End, and Isometric views.
3. **Pass 4:** Mechanical Designer Agent validates physics, tolerances, and resolves contradictions.
4. **Pass 5:** Generates structural JSON (with `null` for unknown implicit intersections).
5. **Pass 5.5:** AI Geometric Resolution Agent performs trigonometry to mathematically solve all `null` coordinates.
6. **Pass 6:** Auto-compiles the JSON into a `generated_part.scad` 3D CAD file.

## Setup

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment file and add your OpenAI API key:
   ```bash
   cp .env.example .env
   ```
4. Place your technical drawing in the root folder as `technical_drawing.jpg`.

## Usage

Run the main pipeline:
```bash
python multi_agent_council.py
```

The pipeline will execute the LLM passes and output two files:
* `final_council_extracted_parameters.json` - The fully extracted and mathematically resolved parameters.
* `generated_part.scad` - The parametric 3D model (open in OpenSCAD to view or export to STL).
