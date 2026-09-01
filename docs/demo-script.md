# Demo script

1. Start the backend and frontend using the commands in `RUN_INSTRUCTIONS.md`.
2. Open `http://localhost:3000`; the checked-in demo scene loads automatically.
3. Rotate the model and show the colored, georeferenced mesh.
4. Click two visible surface points; the HUD displays the metric distance.
5. Toggle **Show low-confidence** to reveal flagged occlusion points.
6. Upload a short training video and optional CSV/JSON telemetry; wait for the
   pipeline status to reach `complete`, then inspect the generated flight.

State the limitation clearly: v1 flags unseen/low-confidence surfaces rather
than hallucinating occluded geometry, and GPU inference/training is optional.
