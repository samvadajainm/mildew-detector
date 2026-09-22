"""Does this machine reproduce our segmentation?

Run once after installing. It segments the bundled photos and compares the masks
against the ones we got, at IoU 0.98. A pass means your install is equivalent to
ours; a fail means torch/ultralytics/opencv differ enough to matter and the
numbers in HANDOFF.md are not yours.

    python verify_install.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

import plantseg

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "samples", "golden")
TOL_IOU = 0.98


def main(write=False):
    os.makedirs(GOLD, exist_ok=True)
    inputs = sorted(glob.glob(os.path.join(HERE, "samples", "inputs", "*.jpg")))
    if not inputs:
        print("no samples/inputs/*.jpg found")
        return 1

    meta_path = os.path.join(GOLD, "golden.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    rows, ok = [], True

    for path in inputs:
        stem = os.path.splitext(os.path.basename(path))[0]
        res = plantseg.segment(path)
        m = cv2.resize(res["mask"], (640, int(640 * res["mask"].shape[0] / res["mask"].shape[1])),
                       interpolation=cv2.INTER_NEAREST) > 127
        gp = os.path.join(GOLD, stem + ".png")

        if write:
            cv2.imwrite(gp, m.astype(np.uint8) * 255)
            meta[stem] = {"area_fraction": res["plant_area_fraction"],
                          "confidence": res["confidence"]}
            rows.append((stem, 1.0, res["timing_ms"]))
            continue

        if not os.path.exists(gp):
            print("  missing golden mask for", stem)
            ok = False
            continue
        g = np.squeeze(cv2.imread(gp, 0)) > 127
        iou = (g & m).sum() / max(1, (g | m).sum())
        ok &= iou >= TOL_IOU
        rows.append((stem, iou, res["timing_ms"]))

    if write:
        meta["_weights_sha256"] = plantseg._weights_sha
        json.dump(meta, open(meta_path, "w"), indent=2)
        print("wrote %d golden masks" % len(rows))
        return 0

    sha_ok = meta.get("_weights_sha256") == plantseg._weights_sha
    print("weights sha256 %s" % ("MATCH" if sha_ok else "DIFFERENT - wrong .pt file"))
    for stem, iou, ms in rows:
        print("  %-34s IoU vs reference %.4f  %5d ms  %s"
              % (stem, iou, ms, "ok" if iou >= TOL_IOU else "FAIL"))
    print("\n%s" % ("PASS - your install matches ours" if (ok and sha_ok) else "FAIL - see above"))
    return 0 if (ok and sha_ok) else 1


if __name__ == "__main__":
    sys.exit(main(write="--write" in sys.argv))
