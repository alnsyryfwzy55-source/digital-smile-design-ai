# Task: run the smile-generation proof of concept and return the results

## What this experiment answers
Can an open, free image model (Stable Diffusion 1.5 inpainting + ControlNet) draw realistic upper front
teeth **exactly inside a geometric tooth design**, without changing the lips, the lower teeth or the face?
The design is computed first by `dsd_gen.py`; the model only renders it; `check_generated()` then measures
whether the rendered teeth match the design (IoU against the design outlines).

**Success criterion:** `pass_rate >= 0.80` in `summary.json` (automatic check), then a blind realism
rating by a dentist using `results/blind_test/` (not part of your task).

## Contents
| File | Purpose |
|---|---|
| `run_poc.py` | the experiment (command line) |
| `dsd_gen.py` | face crop, tooth segmentation (ONNX), geometric design, generator inputs, automatic check |
| `models/` | `teeth_unet_v2.onnx` (our U-Net, 7 classes) and `face_landmarker.task` (MediaPipe). If absent, `run_poc.py` downloads both from the project repository on GitHub |
| `generation_poc.ipynb` | the same experiment as a Google Colab notebook (alternative) |
| `requirements.txt` | Python packages |

## Steps
1. **Hardware.** A CUDA GPU is needed for the full run (a T4 with 16 GB is enough). Report what you have.
   If there is no GPU, run `--quick` on CPU and report the time per image; do not substitute another model.
2. **Photos.** Put 15–20 frontal photos of smiling people with the upper teeth visible in `photos/`, from Pexels or
   Unsplash (any will do; a list of their URLs is enough). Choose them before running and do not swap them after.
3. Install: `pip install -r requirements.txt`
4. Pipeline check without the diffusion model:
   `python run_poc.py --photos photos --out results_dry --dry-run`
5. Full run: `python run_poc.py --photos photos --out results`
   (about 25–45 minutes on a T4; it downloads about 6 GB of model weights from Hugging Face on first run).
   With no GPU: `python run_poc.py --photos photos --out results_quick --quick`
6. Save the complete console output to `results/log.txt`.

## Rules
- Do not change the design, the thresholds in `check_generated()`, the photo selection or the seed
  selection after seeing results. Report results as they are, including failures.
- If a bug or a package version stops the run, make the smallest fix, and list every change as a diff in your report.
- Do not drop photos because their results look bad. Photos skipped by the script are listed in `summary.json`.
- Do not publish the generated images anywhere; they are edits of real people's faces, for internal evaluation only.

## Return
One zip containing:
- `results/` complete (summary.json, metrics.csv, sweep.csv, all case_* images, blind_test/, blind_key_DO_NOT_SHARE.csv, log.txt)
- `results_dry/summary.json`
- `photos/SOURCES.csv` (not the photos themselves)
- `REPORT.md`: hardware, package versions, run time, every problem met and every code change (as a diff),
  and your own observations on the after-images (realism, artefacts, lips, lower teeth), with case names.
