/**
 * normalise.js — port of normalise.py (felt-referenced exposure
 * normalisation) for OpenCV.js.
 *
 * CANON_RGB is CANON_BGR from the Python source reversed to RGB order:
 * Python (62.0, 60.0, 63.0) in B,G,R  ->  here [63.0, 60.0, 62.0] in R,G,B.
 */
import { channelHistogram, percentileFromHistogram } from './percentile.js';

export const CANON_RGB = [63.0, 60.0, 62.0];

/** Background cloth only: not leaf, not card, away from the leaf edge.
 * @param {cv.Mat} rgb CV_8UC3, RGB order
 * @param {cv.Mat} maskMat CV_8UC1 leaf mask (0/255)
 * @returns {cv.Mat} CV_8UC1 felt mask — caller must .delete()
 */
export function feltMask(rgb, maskMat) {
  const hsv = new cv.Mat();
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);
  const rows = rgb.rows, cols = rgb.cols;

  const big = new cv.Mat();
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(61, 61));
  cv.dilate(maskMat, big, k);
  k.delete();

  const felt = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const bigData = big.data, hsvData = hsv.data, feltData = felt.data;
  let count = 0;
  for (let i = 0; i < rows * cols; i++) {
    if (bigData[i] === 0 && hsvData[i * 3 + 2] < 150 && hsvData[i * 3 + 1] < 60) {
      feltData[i] = 255;
      count++;
    }
  }
  hsv.delete();

  if (count < 5000) {
    cv.bitwise_not(big, felt); // fall back to "everything outside the dilated leaf"
  }
  big.delete();
  return felt;
}

function medianColour(rgb, feltMat) {
  const out = [0, 0, 0];
  for (let c = 0; c < 3; c++) {
    const { hist, n } = channelHistogram(rgb, c, feltMat);
    out[c] = n > 0 ? percentileFromHistogram(hist, n, 50) : 0;
  }
  return out; // [R, G, B]
}

export function feltRGB(rgb, maskMat) {
  const felt = feltMask(rgb, maskMat);
  const c = medianColour(rgb, felt);
  felt.delete();
  return c;
}

function toLinear(x) {
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}
function toSrgb(x) {
  x = Math.min(1, Math.max(0, x));
  return x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055;
}

/**
 * Scale each channel so this frame's felt matches the canonical felt.
 * Exposure is linear-light, so the gain is applied after undoing the sRGB
 * transfer function, exactly as in the Python version.
 * @returns {{normalised: cv.Mat, gain: [number, number, number]}}
 *   normalised is CV_8UC3 RGB — caller must .delete() it.
 */
export function normalise(rgb, maskMat, canon = CANON_RGB) {
  const ref = feltRGB(rgb, maskMat).map(v => Math.max(v, 1.0));
  const linRef = ref.map(v => toLinear(v / 255));
  const linCanon = canon.map(v => toLinear(v / 255));
  const gain = linCanon.map((v, i) => v / linRef[i]);

  const rows = rgb.rows, cols = rgb.cols;
  const out = new cv.Mat(rows, cols, cv.CV_8UC3);
  const srcData = rgb.data, outData = out.data;
  for (let i = 0, n = rows * cols; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const lin = toLinear(srcData[i * 3 + c] / 255) * gain[c];
      outData[i * 3 + c] = Math.round(toSrgb(lin) * 255);
    }
  }
  return { normalised: out, gain };
}
