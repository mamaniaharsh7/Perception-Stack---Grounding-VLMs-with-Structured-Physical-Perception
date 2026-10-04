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
