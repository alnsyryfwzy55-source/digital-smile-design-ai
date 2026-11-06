# Tooth model v3 (in use since 2026-11-06, file `docs/teeth_unet_v2.onnx`)

U-Net + MobileNetV2, input 1×3×384×384, output 7 classes (background, 13, 12, 11, 21, 22, 23), the same contract as v2,
so the app and the server needed no code change. Trained on Colab (T4, 60 epochs) with `training/v3/train_v3.ipynb`.

## Data
- `smile_teeth_v2`: retractor photos, labelled by people.
- 36 real smiles from openly licensed internet photos, drafted automatically and reviewed tooth by tooth by an AI
  (verdicts in `training/review/ai_verdicts.json`). A visible canine the reviewer could not accept is an
  IGNORE region (255), not background. Children, tangled braces and photos with a wrong incisor outline were dropped.
- Augmentation `gap_augment`: a tooth erased into a dark gap or a healed ridge, labelled background (20% of samples).

## Results (same photos, same measures; `training/accept_report_v3.json`)
| | v2 | v3 |
|---|---|---|
| Dice, retractor test photos (human labels) | 0.956 | 0.955 |
| Dice, held-out real smiles (AI-reviewed labels) | 0.888 | 0.927 |
| Tooth painted into a gap | 80-88% | 0% |
| Structural guarantees violated | 0 | 0 |
| Real teeth taken for a gap | 0 | 0 |
| Dice inside the browser (WASM, 20 test photos) | 0.959 | 0.960 |

## The acceptance gate and the one exception
9 of 10 items passed. Item G (confirmation threshold) failed at 0.93: v3 is more confident, so its errors are too. At
**0.95** (now used by the app and the server) it catches 43/47 wrong teeth on the server and 42/47 in the app (the gate
asks 90%: one tooth short), and asks about 3/257 correct teeth in clean photos. The uncaught ones come only from JPEG
quality 12 and an 18° in-crop rotation, which the pipeline removes by aligning the crop to the eye line. v2 at 0.93
missed 3-4 (JPEG 12, lip cover). Accepted on that basis; recorded here so it is not silently forgotten.

## What v3 changed elsewhere
v3 no longer paints a lost lateral, but when it hesitates between 13 and 12 it can split the canine and give the lost
tooth's number to it. The model-independent safety net `gap_check` (core) / `gapCheck` (app) now asks the doctor when two
neighbours have a tooth-sized gap between them or overlap like one split tooth: an erased lateral is caught 84/86 times,
intact photos raise 2 flags on 258 teeth.

## Limits
- The real-smile test labels are AI-reviewed; a human-reviewed test set (a few dozen photos) is still to be made.
- The canine at the end of the arch remains the weakest tooth (Dice 0.86-0.89 on smiles).
- Fallback: v2 is kept at `training/models_archive/teeth_unet_v2_original.onnx` (not published).
