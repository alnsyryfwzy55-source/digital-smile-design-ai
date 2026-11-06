---
title: Smile Design Analysis
emoji: 🦷
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# Smile design analysis server

The analysis server for the clinic's smile-design app. It receives a smile photo and the face points found on the phone. It returns:

- the teeth, with structural guarantees and the gap and lower-tooth checks;
- the measurements;
- a validated design;
- a lab file at 1:1 mm.

Photos are analysed in memory and are not saved or logged. Only the clinic app's address (`DSD_ORIGINS`) may call it from a browser.

`GET /api/health` · `POST /api/analyze`
