"""Reference JSON API for stage 1. Flask, single process, CPU.

    pip install -r requirements-inference.txt
    python serve_example.py            ->  http://127.0.0.1:8000

    GET  /health                 model version + weights hash
    GET  /                       upload page - shows the three panels, for eyeballing
    POST /segment                multipart 'file', or the raw image bytes as the body
         ?renders=1              also return outline / isolated / mask / panel as PNG data URIs
         ?render_width=900       long edge of those renders

This is a reference, not a production server: one model in one process, no auth,
no rate limit, no queue. Put it behind gunicorn/waitress with 1-2 workers (the
model is ~24 MB of weights plus torch per worker).
"""
import base64
import json

import cv2
from flask import Flask, Response, request

import plantseg

app = Flask(__name__)
MAX_BYTES = 25 * 1024 * 1024


@app.get("/health")
def health():
    plantseg.load_model()
    return {"ok": True, "model": "plantseg", "version": plantseg.MODEL_VERSION,
            "weights_sha256": plantseg._weights_sha, "device": "cpu",
            "stage": "1 of 3 - segmentation only, no disease score"}


@app.post("/segment")
def segment():
    f = request.files.get("file")
    data = f.read() if f else request.get_data()
    if not data:
        return {"ok": False, "error": "no image supplied"}, 400
    if len(data) > MAX_BYTES:
        return {"ok": False, "error": "image larger than 25 MB"}, 413
    try:
        res = plantseg.segment(data)
    except Exception as e:                      # bad/corrupt upload, not our bug
        return {"ok": False, "error": str(e)}, 400

    want = request.args.get("renders") in ("1", "true", "yes")
    width = int(request.args.get("render_width", 900))
    body = plantseg.to_json(res, renders=want, render_width=width)
    return Response(json.dumps(body), mimetype="application/json")


PAGE = """<!doctype html><meta charset=utf-8><title>plantseg stage 1</title>
<style>body{font:15px system-ui;margin:2rem;max-width:1100px}
img{max-width:100%;border:1px solid #ccc;margin:.5rem 0}
pre{background:#f4f4f4;padding:.8rem;overflow:auto;font-size:13px}</style>
<h2>plantseg - stage 1 (segmentation only)</h2>
<p>Upload a plant photo. Returns the outline it found and the isolated plant.
No disease score - that stage does not exist yet.</p>
<form method=post enctype=multipart/form-data action="/demo">
<input type=file name=file accept="image/*" required> <button>Segment</button></form>
"""


@app.get("/")
def index():
    return PAGE


@app.post("/demo")
def demo():
    f = request.files.get("file")
    if not f:
        return PAGE
    res = plantseg.segment(f.read())
    panel = plantseg.render_panel(res, width=1500)
    uri = plantseg.to_json(res, renders=False)
    png = cv2.imencode(".png", panel)[1].tobytes()
    b64 = base64.b64encode(png).decode()
    return (PAGE + '<img src="data:image/png;base64,%s">' % b64
            + "<pre>%s</pre>" % json.dumps(uri, indent=2))


if __name__ == "__main__":
    plantseg.load_model()          # pay the load cost before the first request
    app.run(host="127.0.0.1", port=8000, debug=False)
