/**
 * classify.js — port of classify_v2.py (tissue classification in the
 * felt-normalised colour space) for OpenCV.js.
 *
 * All thresholds are unchanged from the Python source and must stay in
 * lock-step with it — see the Python docstring for why each was tuned to
 * its current value.
 */
import { channelHistogram, percentileFromHistogram } from './percentile.js';

export const CHLOROTIC_H = 35;
export const CHLOR_OPEN = 5;
export const REL_REF_Q = 75;
export const REL_FRAC = 0.65;
export const LOCAL_WIN = 101;
export const LOCAL_DROP = 25;
export const BLOOM_OPEN = 3;
export const BLOOM_MAX_ELONG = 3.0;
export const BLOOM_MIN_HALFWIDTH = 2.0;

export const FEATS = ["h_med", "green_frac", "chlorotic_frac", "s_med", "h_p10", "h_p90",
  "bloom_frac", "bloom_rel_frac", "bloom_local_frac"];

function openMask(m, r) {
  if (r <= 0) return m.clone();
  const k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(r * 2 + 1, r * 2 + 1));
  const out = new cv.Mat();
  cv.morphologyEx(m, out, cv.MORPH_OPEN, k);
  k.delete();
  return out;
}

/** Delete whole components that are long and thin (e.g. the midrib); keep
 * compact blobs (candidate bloom colonies). */
function dropRidges(m) {
  if (cv.countNonZero(m) === 0) return m.clone();
  const labels = new cv.Mat(), statsMat = new cv.Mat(), centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(m, labels, statsMat, centroids, 8, cv.CV_32S);
  if (n <= 1) { labels.delete(); statsMat.delete(); centroids.delete(); return m.clone(); }

  const dt = new cv.Mat();
  cv.distanceTransform(m, dt, cv.DIST_L2, 5);

  const rows = m.rows, cols = m.cols;
  const lblData = labels.data32S;
  const dtData = dt.data32F;
  const statsData = statsMat.data32S;

  const maxR = new Float32Array(n);
  for (let p = 0; p < rows * cols; p++) {
    const l = lblData[p];
    if (l === 0) continue;
    if (dtData[p] > maxR[l]) maxR[l] = dtData[p];
  }

  const keep = new Uint8Array(n);
  for (let i = 1; i < n; i++) {
    const r = maxR[i];
    if (r < BLOOM_MIN_HALFWIDTH) continue;
    const area = statsData[i * 5 + 4];
    const elong = area / (Math.PI * r * r);
    if (elong <= BLOOM_MAX_ELONG) keep[i] = 1;
  }

  const out = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const outData = out.data;
  for (let p = 0; p < rows * cols; p++) {
    const l = lblData[p];
    if (l !== 0 && keep[l]) outData[p] = 255;
  }

  dt.delete(); labels.delete(); statsMat.delete(); centroids.delete();
  return out;
}

/**
 * healthy / chlorotic (yellow) / bloom (desaturated patch) tissue masks.
 * "bloom" flags tissue whose saturation is low relative to the rest of the
 * same leaf — a patchy-pigmentation signal, not fungal-structure detection.
 * @param {cv.Mat} interior CV_8UC1 mask (0/255)
 * @param {cv.Mat} hsv CV_8UC3, from cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV)
 * @returns {{healthy: cv.Mat, chlor: cv.Mat, bloom: cv.Mat, bRel: cv.Mat, bLoc: cv.Mat}}
 *   all CV_8UC1 — caller must .delete() every one.
 */
export function classify(interior, hsv) {
  const rows = hsv.rows, cols = hsv.cols;
  const src = hsv.data;
  const hData = new Uint8Array(rows * cols);
  const sData = new Uint8Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) { hData[i] = src[i * 3]; sData[i] = src[i * 3 + 1]; }

  const { hist, n } = channelHistogram(hsv, 1, interior);
  const ref = percentileFromHistogram(hist, n, REL_REF_Q);
  const relThresh = ref * REL_FRAC;

  const intData = interior.data;

  // --- relative-saturation bloom candidate ---
  let relRaw = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const relRawData = relRaw.data;
  for (let i = 0; i < rows * cols; i++) if (intData[i] && sData[i] < relThresh) relRawData[i] = 255;
  let opened = openMask(relRaw, BLOOM_OPEN);
  relRaw.delete();
  const bRel = dropRidges(opened);
  opened.delete();

  // --- local-drop bloom candidate (median-blurred S minus S) ---
  const sMat = cv.matFromArray(rows, cols, cv.CV_8UC1, sData);
  const blur = new cv.Mat();
  cv.medianBlur(sMat, blur, LOCAL_WIN);
  sMat.delete();
  const blurData = blur.data;
  let locRaw = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const locRawData = locRaw.data;
  for (let i = 0; i < rows * cols; i++) {
    if (intData[i] && (blurData[i] - sData[i]) > LOCAL_DROP) locRawData[i] = 255;
  }
  blur.delete();
  let opened2 = openMask(locRaw, BLOOM_OPEN);
  locRaw.delete();
  const bLoc = dropRidges(opened2);
  opened2.delete();

  const bloom = new cv.Mat();
  cv.bitwise_or(bRel, bLoc, bloom);

  // --- chlorotic: yellow hue, outside bloom ---
  let chlorRaw = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const bloomData = bloom.data, chlorRawData = chlorRaw.data;
  for (let i = 0; i < rows * cols; i++) {
    if (intData[i] && !bloomData[i] && hData[i] < CHLOROTIC_H) chlorRawData[i] = 255;
  }
  const chlor = openMask(chlorRaw, CHLOR_OPEN);
  chlorRaw.delete();

  // --- healthy: everything else inside the interior ---
  const healthy = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  const chlorData = chlor.data, healthyData = healthy.data;
  for (let i = 0; i < rows * cols; i++) {
    if (intData[i] && !bloomData[i] && !chlorData[i]) healthyData[i] = 255;
  }

  return { healthy, chlor, bloom, bRel, bLoc };
}
