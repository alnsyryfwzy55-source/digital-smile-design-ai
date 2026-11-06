# Task (round 2): more draft labels from internet smile photos

Round 1 kept 19 of 577 photos. The skip reasons showed where good photos were lost, and this version fixes that:
- **Several faces:** the largest face is used.
- **Tilted head:** a tilt of up to 25° is accepted, because the crop is rotated to the eye line.
- **Small mouths:** a mouth of 120 px or more is accepted.
- **Close-up mouth photos (no face found):** these are now labelled from the lips that face parsing finds.
- **Failed downloads:** the download resumes and retries on 504 and connection errors.

It also adds four queries for damaged teeth: broken, chipped, braces and dentures.

## Steps
1. `pip install -r requirements.txt`
2. If you still have last round's `drafts/downloaded/` folder (577 photos + `SOURCES.csv`), copy it to
   `run2/downloaded/`. This way only the missing photos are downloaded. If you don't have it, skip this step and everything is downloaded again.
3. `python bootstrap.py --openverse 1500 --out run2`
   If Openverse is overloaded, wait and run the **same command** again. It continues where it stopped.
4. Start the full run; no separate check run is needed this time. CPU is fine.

## Rules (unchanged)
- Do not change `FILTERS`, the models or the post-processing. If something breaks, make the smallest fix and show it as a diff.
- Do not publish the photos or labels.

## Return
One zip with `run2/drafts.zip`, `run2/report.json`, `run2/log.txt` and a short `REPORT.md`:
- the run time;
- problems you hit and any diffs;
- a look at 10 random drafts, including at least 3 with `"mode": "closeup"` in their meta. Are the teeth, lips and
  numbering right?
