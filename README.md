# Perception Stack - Grounding VLMs with Structured Physical Perception

## Project Overview

Vision-language models (VLMs) excel at object recognition and semantic understanding but fail at physical reasoning tasks such as predicting trajectories, inferring collisions, and answering counterfactual questions about object dynamics. The root cause is missing physical grounding: a VLM given a single video frame has no access to depth, motion, or collision geometry.

Perception Stack addresses this by extracting structured physical measurements from video using four specialist perception models and injecting them as natural language context into a VLM before it reasons. No fine-tuning. No new architecture. Pure context injection.

**Pipeline stages:**
- Stage 1: Object detection via CLEVRER Mask R-CNN proposal files
- Stage 2: Object tracking via SAM 2 (centroids, trajectories, mask overlaps)
- Stage 3: Depth estimation via Depth Anything V2 Small (collision-plane classification)
- Stage 4: Optical flow and velocity via RAFT-small (velocity vectors, direction)
- VLM: LLaVA-1.5 7B with 4-bit NF4 quantization (baseline and enhanced conditions)

Evaluated on clips from the CLEVRER benchmark across four question types: descriptive, explanatory, predictive, and counterfactual.

---

## Notebooks Guide

Each notebook is self-contained and designed to run on Google Colab. Run them in order:

| Notebook | Runtime | Description |
|---|---|---|
| 00_data_download | CPU | Downloads clips, annotations, and extracts keyframes |
| 01_LLaVA_baseline | A100 | Runs baseline LLaVA inference on all 70 clips |
| 02a_sam2 | A100 | Runs SAM 2 object tracking on all 70 clips |
| 02b_depth | T4 | Runs Depth Anything V2 on overlap and final frames |
| 02c_RAFT | A100 | Runs RAFT optical flow on frame pairs |
| 03_perception_enhanced | A100 | Runs all 4 enhanced conditions (full, no_sam2, no_depth, no_raft) |
| 04_Analysis | T4 / CPU | Aggregates results and generates all report figures |

**Important:** Each notebook has a clear cell at the top that deletes previous outputs before a fresh run. All loops are checkpointed: re-running a notebook will skip already-processed clips.

---

## Dependencies

All notebooks install their own dependencies at runtime. Key libraries:

- `transformers`, `bitsandbytes`, `accelerate` (LLaVA)
- `segment-anything-2` (SAM 2, installed from GitHub)
- `torch`, `torchvision` (RAFT)
- `opencv-python-headless`, `Pillow`, `numpy`, `matplotlib`
- `pycocotools` (RLE mask decoding for proposal files)

---

## Contact

Harsh Vijay Mamania
mamania.h@northeastern.edu
Northeastern University, Boston MA
