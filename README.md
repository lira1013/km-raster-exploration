# km-raster-exploration
this is a project as the curve digitization step of KM TO ipd agent 
# KM Raster Digitization Exploration

**Status:** Experimental & exploratory – not production-ready

This repository documents the exploration of raster-based Kaplan–Meier (KM) curve digitization for individual patient data (IPD) reconstruction. The project was initiated to support the **MarkerMind agent** with an automated KM2IPD pipeline and to identify feasible technical approaches under real-world constraints.

## Purpose

The goal of this work is to:

- Investigate the technical feasibility of extracting KM curves from raster images (PNG/JPG/HEIC) and vector graphics (SVG/PDF).
- Explore algorithmic solutions for full automation while preserving accuracy for downstream IPD reconstruction.
- Document the challenges, design decisions, and trade-offs encountered during the project.

This repository **serves as a reference for lessons learned, failure analysis, and solution exploration**, rather than a fully deployable system.

## Project Overview

### Problem Definition

IPD reconstruction from published KM curves is constrained by the input format:

- **Raster images**: Pixel-based; require computer vision and OCR for curve extraction. Results are inherently approximate due to noise, artifacts, and lack of absolute coordinate semantics. LLMs/VLMs can assist in structure recognition but cannot guarantee precise numeric extraction.
- **Vector graphics**: Path-based; provide exact coordinates and can be parsed directly. However, access is limited, and AI models cannot directly “see” vectors—they must parse XML or path data.

| Feature                | Raster (PNG/JPG/HEIC)                          | Vector (SVG/PDF)                     |
|------------------------|-----------------------------------------------|-------------------------------------|
| Data Nature            | Pixel matrix, no coordinate semantics          | Path data, absolute coordinates      |
| Acquisition Difficulty | Easy: screenshots, scans, web preview         | Hard: limited to databases/papers   |
| Visualization          | May blur when zoomed                           | Infinite zoom clarity               |
| Extraction Method       | CV + OCR + curve fitting                        | Path parsing or regex               |
| Determinacy            | Probabilistic                                  | Deterministic                        |
| LLM/VLM Usage           | Assist structure recognition                  | Assist code/path parsing             |
| Challenges              | Noise, artifacts                               | Large source files, token limits    |
| Best Practice           | Algorithm + OCR + manual verification          | Direct parsing from structured source |

### Core Approach

- **Pipeline decomposition**: 7 modular steps from ROI cropping, enhancement, skeletonization, coordinate mapping, to KM statistical reconstruction.
- **Algorithm design & validation**: Explored edge detection, skeletonization, and color/statistical modeling. Key insight: **topology-preserving skeletonization + statistical reconstruction** is more robust for raster KM curves than simple edge detection.
- **Synthetic benchmarks**: Programmatically generated ~300 KM images based on cBioPortal clinical data to validate algorithms and measure error.
- **Hybrid stack strategy**: Combining R (SurvdigitizeR + Guyot) as algorithmic core with Python as orchestration layer, balancing scientific accuracy, automation potential, and system integration.

### Lessons Learned

- Purely automatic raster-based digitization is extremely challenging due to noise, overlapping curves, and image artifacts.
- Hybrid approaches allow **rapid feasibility verification** while maintaining scientific reliability.
- Building a pipeline with modular components facilitates error attribution, algorithm replacement, and ablation studies.
- The exploration highlights the importance of **business understanding**, decision-making under constraints, and systematic problem-solving.

### Project Status

- **Vector path (cbioPortal)**: Implemented and verified; extraction is accurate for validated KM curves.
- **Raster path (Python pipeline)**: Pipeline implemented and partially validated; requires further algorithm refinement to handle noise and complex curve overlap.
- **Hybrid stack (R + Python)**: Selected as short-term optimal solution; Python implementation planned for future full automation once algorithms are validated.

---

**Disclaimer:** This repository reflects experimental work, solution exploration, and technical decision-making. It is **not intended for production use**.
