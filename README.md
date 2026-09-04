# mildew-detector

Upload a single plant leaf photo and get an instant Healthy/Infected reading, powered by an on-device ONNX model - runs entirely in the browser, no server required.

**Live site:** https://sites.google.com/view/leafvision-mildewdetector/home
**Source / hosting repo:** https://github.com/samvadajainm/mildew-detector

## How it works

1. User uploads one image via the upload widget.
2. The image is resized to 224×224 and converted to a raw (unnormalized) RGB float32 tensor — matching the preprocessing used in the companion Android app.
3. `onnxruntime-web` runs the model fully client-side (no image ever leaves the browser).
4. The model's output score is scaled to a **0–10 rating**:
   - **0–4 → Healthy**
   - **Above 4 → Infected**

## Files

| `mildew-detector.html` | The full app - upload widget, preprocessing, inference, and result display |
| `mildew_detector.onnx` | The model, converted from the original `.tflite` file (verified numerically identical output) |

## Deploying changes

This page is hosted as a static site via **GitHub Pages**. To update anything (styling, thresholds, wording):

1. Edit `mildew-detector.html` locally.
2. Upload the new version to this repo (same filename, same folder as `mildew_detector.onnx`).
3. GitHub Pages auto-redeploys within about a minute.
4. No changes are needed on the Google Sites side — it just embeds the GitHub Pages URL via an iframe, so it always reflects the latest deployed version.

## Notes

- Threshold and rating logic live in `mildew-detector.html` under `RATING_THRESHOLD`.
- Model input: `[1, 224, 224, 3]` float32, NHWC, no normalization.
- Model output: `[1, 1]` float32 sigmoid score (0 = healthy, 1 = infected).
