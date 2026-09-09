/**
 * score.js — port of leafscore.py for OpenCV.js.
 *
 * Usage:
 *   import { LeafScorer, overlay, disposeResult } from './score.js';
 *   const scorer = await LeafScorer.load('model_v2.json');
 *   const rgba = cv.imread(canvasOrImgElement);       // CV_8UC4
 *   const result = scorer.score(rgba);                // plain JSON-able dict
 *   const view = overlay(rgba, result);                // CV_8UC3 RGB, for display
 *   disposeResult(result);                              // free intermediate Mats
 *   rgba.delete(); view.delete();
 */
import { segment } from './segment.js';
import { normalise, feltMask } from './normalise.js';
import { classify } from './classify.js';
import { channelHistogram, percentileFromHistogram } from './percentile.js';

const MIN_LEAF_FRAC = 0.010;   // smallest training leaf was 1.1% of the frame
const MAX_LEAF_FRAC = 0.400;   // largest was 14.8%; 40% allows a much closer crop
const MIN_BG_FRAC = 0.45;      // training frames were 78-88% background
const MAX_FELT_SD = 45.0;      // background brightness spread; training max was ~30
const MIN_SIDE = 400;          // below this, morphology radii stop making sense

function rgbFromRgba(rgba) {
  const rgb = new cv.Mat();
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
  return rgb;
}

function meanMasked(mat, channelIdx, maskMat, predicate) {
  const data = mat.data, mData = maskMat.data;
  const rows = mat.rows, cols = mat.cols, ch = mat.channels();
  let sum = 0, n = 0;
  for (let i = 0; i < rows * cols; i++) {
    if (!mData[i]) continue;
    n++;
    if (predicate(data[i * ch + channelIdx])) sum++;
  }
  return n > 0 ? sum / n : 0;
}

function stdMasked(mat, channelIdx, maskMat) {
  const data = mat.data, mData = maskMat.data;
  const rows = mat.rows, cols = mat.cols, ch = mat.channels();
  let sum = 0, n = 0;
  for (let i = 0; i < rows * cols; i++) if (mData[i]) { sum += data[i * ch + channelIdx]; n++; }
  if (n === 0) return 0;
  const mean = sum / n;
  let sq = 0;
  for (let i = 0; i < rows * cols; i++) if (mData[i]) { const d = data[i * ch + channelIdx] - mean; sq += d * d; }
  return Math.sqrt(sq / n);
}

export class LeafScorer {
  constructor(model) {
    this.feats = model.feats;
    this.mu = model.mu;
    this.sd = model.sd;
    this.coef = model.coef;
    this.intercept = model.intercept;
    this.regCoef = model.reg_coef;
    this.regIntercept = model.reg_intercept;
  }

  /** @param {string} url path to model_v2.json */
  static async load(url = 'model_v2.json') {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`could not load ${url}: ${res.status}`);
    return new LeafScorer(await res.json());
  }

  _analyse(rgba) {
    const rgb = rgbFromRgba(rgba);
    const { mask, interior } = segment(rgb);
    if (cv.countNonZero(interior) < 500) {
      rgb.delete(); mask.delete(); interior.delete();
      return null;
    }

    const { normalised, gain } = normalise(rgb, mask);
    rgb.delete();

    const hsv = new cv.Mat();
    cv.cvtColor(normalised, hsv, cv.COLOR_RGB2HSV);

    const tissue = classify(interior, hsv);

    const n = cv.countNonZero(interior);
    const hHist = channelHistogram(hsv, 0, interior);
    const sHist = channelHistogram(hsv, 1, interior);

    const f = {
      h_med: percentileFromHistogram(hHist.hist, hHist.n, 50),
      h_p10: percentileFromHistogram(hHist.hist, hHist.n, 10),
      h_p90: percentileFromHistogram(hHist.hist, hHist.n, 90),
      green_frac: meanMasked(hsv, 0, interior, v => v >= 38),
      s_med: percentileFromHistogram(sHist.hist, sHist.n, 50),
      chlorotic_frac: cv.countNonZero(tissue.chlor) / n,
      bloom_frac: cv.countNonZero(tissue.bloom) / n,
      bloom_rel_frac: cv.countNonZero(tissue.bRel) / n,
      bloom_local_frac: cv.countNonZero(tissue.bLoc) / n,
    };

    return { f, mask, interior, normalised, gain, hsv, tissue };
  }

  _checkQuality(rgb, mask, interior, gain) {
    const warn = [];
    const h = rgb.rows, w = rgb.cols;
    if (Math.min(h, w) < MIN_SIDE) warn.push(`image is small (${w}x${h}); results may be unreliable`);

    const leafFrac = cv.countNonZero(mask) / (h * w);
    if (leafFrac < MIN_LEAF_FRAC) warn.push('detected leaf is very small in frame');
    else if (leafFrac > MAX_LEAF_FRAC) warn.push('detected region covers much of the frame; the background may not be separating correctly');

    const fm = feltMask(rgb, mask);
    const bgFrac = cv.countNonZero(fm) / (h * w);
    if (bgFrac < MIN_BG_FRAC) warn.push('not enough plain background detected; exposure normalisation needs a large uniform neutral backdrop');

    if (cv.countNonZero(fm) > 100) {
      const hsvFull = new cv.Mat();
      cv.cvtColor(rgb, hsvFull, cv.COLOR_RGB2HSV);
      const sd = stdMasked(hsvFull, 2, fm);
      hsvFull.delete();
      if (sd > MAX_FELT_SD) warn.push('background is uneven (shadows or clutter); expected a plain, evenly lit backdrop');
    }
    fm.delete();

    const maskData = mask.data;
    let minY = h, maxY = -1, minX = w, maxX = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (maskData[y * w + x]) {
          if (y < minY) minY = y; if (y > maxY) maxY = y;
          if (x < minX) minX = x; if (x > maxX) maxX = x;
        }
      }
    }
    const margin = maxY >= 0 ? Math.min(minY, minX, h - 1 - maxY, w - 1 - maxX) : 0;
    if (margin <= 3) warn.push('leaf touches the edge of the frame and may be cut off');

    const g = (gain[0] + gain[1] + gain[2]) / 3;
    if (g > 1.05) warn.push('image is darker than the reference; brightening it risks clipping and the result may be distorted');

    return {
      usable: warn.length === 0,
      warnings: warn,
      leaf_frac: Math.round(leafFrac * 10000) / 10000,
      background_frac: Math.round(bgFrac * 1000) / 1000,
      exposure_gain: Math.round(g * 1000) / 1000,
    };
  }

  /**
   * @param {cv.Mat} rgba CV_8UC4 (e.g. from cv.imread on a canvas). Caller
   *   keeps ownership of `rgba` — this does not delete it.
   * @returns plain, JSON-serialisable result dict. If `ok`, a `_debug` field
   *   carries the intermediate cv.Mat objects (for overlay/panel display) —
   *   pass the whole result to `disposeResult()` when finished with them.
   */
  score(rgba) {
    const got = this._analyse(rgba);
    if (!got) {
      return {
        ok: false,
        error: 'no leaf found',
        detail: 'Could not separate a leaf from the background. Photograph a single flat leaf on a plain dark backdrop, lit evenly, from directly above.',
        quality: { usable: false, warnings: ['segmentation failed'] },
      };
    }
    const { f, mask, interior, normalised, gain, hsv, tissue } = got;

    const x = this.feats.map((k, i) => (f[k] - this.mu[i]) / this.sd[i]);
    const z = x.reduce((s, v, i) => s + v * this.coef[i], this.intercept);
    const p = 1 / (1 + Math.exp(-z));
    let sev = x.reduce((s, v, i) => s + v * this.regCoef[i], this.regIntercept);
    sev = Math.max(0, Math.min(10, sev));

    const n = cv.countNonZero(interior);
    const rgbFull = rgbFromRgba(rgba);
    const quality = this._checkQuality(rgbFull, mask, interior, gain);
    rgbFull.delete();

    return {
      ok: true,
      verdict: p >= 0.5 ? 'DISEASED' : 'HEALTHY',
      p_diseased: Math.round(p * 1000) / 1000,
      confidence: Math.round(Math.max(p, 1 - p) * 1000) / 1000,
      severity_0_10: Math.round(sev * 10) / 10,
      tissue: {
        healthy_pct: Math.round((cv.countNonZero(tissue.healthy) / n) * 1000) / 10,
        chlorotic_pct: Math.round((cv.countNonZero(tissue.chlor) / n) * 1000) / 10,
        bloom_pct: Math.round((cv.countNonZero(tissue.bloom) / n) * 1000) / 10,
      },
      measurements: {
        median_hue: Math.round(f.h_med * 10) / 10,
        median_saturation: Math.round(f.s_med * 10) / 10,
        leaf_pixels: cv.countNonZero(mask),
      },
      quality,
      // DEBUG: raw feature vector + per-channel gain, for comparing directly
      // against leafscore.py's own output on the same image. Not part of the
      // original Python API's return shape — safe to ignore/strip in prod.
      _debug_features: f,
      _debug_gain_rgb: gain,
      _debug_leaf_frac: cv.countNonZero(mask) / (rgba.rows * rgba.cols),
      _debug_interior_px: n,
      _debug: { mask, interior, normalised, hsv, tissue },
    };
  }
}

/** Exposure-corrected leaf with the tissue map blended over it (RGB).
 *  @param {number} alpha 0-1 blend strength; use 1.0 for a pure tissue-map view.
 *  @returns {cv.Mat} CV_8UC3 RGB — caller must .delete() */
export function overlay(rgba, result, alpha = 0.55) {
  if (!result.ok) {
    const rgb = new cv.Mat();
    cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);
    return rgb;
  }
  const { normalised, mask, tissue } = result._debug;
  const rows = normalised.rows, cols = normalised.cols;
  const out = normalised.clone();
  const outData = out.data;
  const hData = tissue.healthy.data, cData = tissue.chlor.data, bData = tissue.bloom.data;
  for (let i = 0; i < rows * cols; i++) {
    let tint = null;
    if (hData[i]) tint = [0, 200, 0];
    else if (cData[i]) tint = [255, 220, 0];
    else if (bData[i]) tint = [255, 0, 255];
    if (tint) {
      for (let c = 0; c < 3; c++) {
        outData[i * 3 + c] = Math.round(outData[i * 3 + c] * (1 - alpha) + tint[c] * alpha);
      }
    }
  }
  const contours = new cv.MatVector();
  const hierarchy = new cv.Mat();
  cv.findContours(mask, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
  cv.drawContours(out, contours, -1, new cv.Scalar(255, 0, 255), 2);
  contours.delete(); hierarchy.delete();
  return out;
}

/** Free every intermediate cv.Mat carried in a successful score() result. */
export function disposeResult(result) {
  if (!result.ok || !result._debug) return;
  const { mask, interior, normalised, hsv, tissue } = result._debug;
  mask.delete(); interior.delete(); normalised.delete(); hsv.delete();
  tissue.healthy.delete(); tissue.chlor.delete(); tissue.bloom.delete();
  tissue.bRel.delete(); tissue.bLoc.delete();
}
