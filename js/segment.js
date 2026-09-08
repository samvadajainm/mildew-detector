/**
 * segment.js — port of segment_v5.py (background-relative chromaticity
 * segmentation) for OpenCV.js.
 *
 * IMPORTANT: browsers give RGB(A) pixel order off a <canvas>, not BGR, so
 * every cv2.COLOR_BGR2* call in the Python source becomes cv.COLOR_RGB2* /
 * cv.COLOR_RGBA2* here. All numeric constants are unchanged from the Python
 * version and must stay in lock-step with it.
 *
 * All exported/internal functions take and return cv.Mat objects. Caller is
 * responsible for eventually .delete()-ing the `mask` and `interior` Mats
 * returned by segment().
 */
import { channelHistogram, percentileFromHistogram, channelMedian } from './percentile.js';

export const MIN_BLOB_FRAC = 0.08;
export const MIN_BLOB_PX = 1500;
export const DIST_FLOOR = 6.0;
export const BORDER_PX = 3;
export const MAX_BORDER_CONTACT = 150;

/**
 * Chromatic distance of every pixel from the estimated background colour
 * (median a*/b* of the whole frame, since felt is ~80% of every image).
 * @param {cv.Mat} rgb CV_8UC3, RGB order
 * @returns {{dist: Float32Array, bg: [number, number]}}
 */
function bgDistance(rgb) {
  const lab = new cv.Mat();
  cv.cvtColor(rgb, lab, cv.COLOR_RGB2Lab);

  const bgA = channelMedian(lab, 1);
  const bgB = channelMedian(lab, 2);

  const rows = lab.rows, cols = lab.cols;
  const dist = new Float32Array(rows * cols);
  const data = lab.data;
  for (let i = 0, n = rows * cols; i < n; i++) {
    const da = data[i * 3 + 1] - bgA;
    const db = data[i * 3 + 2] - bgB;
    dist[i] = Math.sqrt(da * da + db * db);
  }
  lab.delete();
  return { dist, bg: [bgA, bgB] };
}

function distToU8Mat(dist, rows, cols) {
  const m = new cv.Mat(rows, cols, cv.CV_8UC1);
  const d = m.data;
  for (let i = 0; i < dist.length; i++) {
    let v = dist[i];
    if (v < 0) v = 0; else if (v > 255) v = 255;
    d[i] = Math.round(v);
  }
  return m;
}

/**
 * Fill enclosed holes only where they look like leaf, not like background.
 * A hole is filled if it's tiny (threshold noise) or clearly brighter than
 * the background (real leaf/bloom), matched against the Python version's
 * brightness-midpoint cut.
 */
function fillBrightHoles(mask, hsv) {
  if (cv.countNonZero(mask) < 100) return mask.clone();

  const inv = new cv.Mat();
  cv.bitwise_not(mask, inv);
  const labels = new cv.Mat(), statsMat = new cv.Mat(), centroids = new cv.Mat();
  const n = cv.connectedComponentsWithStats(inv, labels, statsMat, centroids, 4, cv.CV_32S);
  inv.delete();
  if (n <= 1) {
    labels.delete(); statsMat.delete(); centroids.delete();
    return mask.clone();
  }

  const rows = mask.rows, cols = mask.cols;
  const lblData = labels.data32S;
  const outerLabel = lblData[0]; // component touching (0,0) is outer background

  const leafHist = channelHistogram(hsv, 2, mask);
  const leafV = percentileFromHistogram(leafHist.hist, leafHist.n, 50);

  // per-label V histograms in a single pass
  const vData = hsv.data;
  const hists = Array.from({ length: n }, () => new Uint32Array(256));
  const counts = new Uint32Array(n);
  for (let p = 0; p < rows * cols; p++) {
    const l = lblData[p];
    hists[l][vData[p * 3 + 2]]++;
    counts[l]++;
  }
  const bgV = counts[outerLabel] > 0
    ? percentileFromHistogram(hists[outerLabel], counts[outerLabel], 50) : 0;
  const cut = bgV + 0.5 * (leafV - bgV);

  const statsData = statsMat.data32S; // 5 cols/label: LEFT, TOP, WIDTH, HEIGHT, AREA
  const fillLabel = new Uint8Array(n);
  for (let i = 1; i < n; i++) {
    if (i === outerLabel) continue;
    const area = statsData[i * 5 + 4];
    if (area < 200) { fillLabel[i] = 1; continue; } // tiny specks: always fill
    const medV = counts[i] > 0 ? percentileFromHistogram(hists[i], counts[i], 50) : 0;
    if (medV > cut) fillLabel[i] = 1;
  }

  const out = mask.clone();
  const outData = out.data;
  for (let p = 0; p < rows * cols; p++) {
    const l = lblData[p];
    if (l !== 0 && fillLabel[l]) outData[p] = 255;
  }

  labels.delete(); statsMat.delete(); centroids.delete();
  return out;
}

/** How many border-strip columns/rows a label touches, matching the
 *  Python version's per-edge "any" counting (not raw pixel count). */
function borderContact(lblData, rows, cols, label) {
  let top = 0, bottom = 0, left = 0, right = 0;
  for (let x = 0; x < cols; x++) {
    let found = false;
    for (let y = 0; y < BORDER_PX; y++) if (lblData[y * cols + x] === label) { found = true; break; }
    if (found) top++;
    found = false;
    for (let y = rows - BORDER_PX; y < rows; y++) if (lblData[y * cols + x] === label) { found = true; break; }
    if (found) bottom++;
  }
  for (let y = 0; y < rows; y++) {
    let found = false;
    for (let x = 0; x < BORDER_PX; x++) if (lblData[y * cols + x] === label) { found = true; break; }
    if (found) left++;
    found = false;
    for (let x = cols - BORDER_PX; x < cols; x++) if (lblData[y * cols + x] === label) { found = true; break; }
    if (found) right++;
  }
  return top + bottom + left + right;
}

/**
 * @param {cv.Mat} rgb CV_8UC3, RGB order
 * @returns {{mask: cv.Mat, interior: cv.Mat}} both CV_8UC1 (0/255) — caller
 *   must .delete() both when done.
 */
export function segment(rgb) {
  const rows = rgb.rows, cols = rgb.cols;
  const hsv = new cv.Mat();
  cv.cvtColor(rgb, hsv, cv.COLOR_RGB2HSV);

  const { dist } = bgDistance(rgb);
  const d8 = distToU8Mat(dist, rows, cols);

  const tmpThresh = new cv.Mat();
  let t = cv.threshold(d8, tmpThresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
  tmpThresh.delete();
  t = Math.max(t, DIST_FLOOR);

  let m = new cv.Mat();
  cv.threshold(d8, m, t, 255, cv.THRESH_BINARY);
  d8.delete();

  let k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(7, 7));
  cv.morphologyEx(m, m, cv.MORPH_OPEN, k);
  k.delete();
  k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(11, 11));
  cv.morphologyEx(m, m, cv.MORPH_CLOSE, k);
  k.delete();

  let labels = new cv.Mat(), statsMat = new cv.Mat(), centroids = new cv.Mat();
  let n = cv.connectedComponentsWithStats(m, labels, statsMat, centroids, 8, cv.CV_32S);
  if (n <= 1) {
    m.delete(); labels.delete(); statsMat.delete(); centroids.delete(); hsv.delete();
    const z = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
    return { mask: z, interior: z.clone() };
  }

  let lblData = labels.data32S;
  let statsData = statsMat.data32S;

  const cand = [];
  for (let i = 1; i < n; i++) {
    const area = statsData[i * 5 + 4];
    if (area < MIN_BLOB_PX) continue;
    const contact = borderContact(lblData, rows, cols, i);
    if (contact <= MAX_BORDER_CONTACT) cand.push([area, i]);
  }
  let candidates = cand.length ? cand
    : Array.from({ length: n - 1 }, (_, idx) => [statsData[(idx + 1) * 5 + 4], idx + 1]);

  const big = Math.max(...candidates.map(([a]) => a));
  const keep = new Set(candidates.filter(([a]) => a >= Math.max(MIN_BLOB_FRAC * big, MIN_BLOB_PX))
    .map(([, i]) => i));

  let mask = new cv.Mat(rows, cols, cv.CV_8UC1, new cv.Scalar(0));
  let maskData = mask.data;
  for (let p = 0; p < rows * cols; p++) if (keep.has(lblData[p])) maskData[p] = 255;

  m.delete(); labels.delete(); statsMat.delete(); centroids.delete();

  const filled = fillBrightHoles(mask, hsv);
  mask.delete();
  mask = filled;
  maskData = mask.data;

  // Evict anything that still looks like felt: background-coloured AND dark.
  if (cv.countNonZero(mask) > 0) {
    const leafV = channelHistogram(hsv, 2, mask);
    const leafVMed = percentileFromHistogram(leafV.hist, leafV.n, 50);
    const notMask = new cv.Mat();
    cv.bitwise_not(mask, notMask);
    const bgVHist = channelHistogram(hsv, 2, notMask);
    notMask.delete();
    const bgVMed = percentileFromHistogram(bgVHist.hist, bgVHist.n, 50);
    const cut = bgVMed + 0.5 * (leafVMed - bgVMed);

    const vData = hsv.data;
    for (let p = 0; p < rows * cols; p++) {
      if (maskData[p] && dist[p] < t && vData[p * 3 + 2] < cut) maskData[p] = 0;
    }

    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(5, 5));
    cv.morphologyEx(mask, mask, cv.MORPH_OPEN, k);
    k.delete();

    const l2 = new cv.Mat(), s2 = new cv.Mat(), c2 = new cv.Mat();
    const n2 = cv.connectedComponentsWithStats(mask, l2, s2, c2, 8, cv.CV_32S);
    if (n2 > 1) {
      const l2Data = l2.data32S, s2Data = s2.data32S;
      let maxA = 0;
      for (let i = 1; i < n2; i++) if (s2Data[i * 5 + 4] > maxA) maxA = s2Data[i * 5 + 4];
      const keep2 = new Set();
      for (let i = 1; i < n2; i++) {
        if (s2Data[i * 5 + 4] >= Math.max(MIN_BLOB_FRAC * maxA, MIN_BLOB_PX)) keep2.add(i);
      }
      for (let p = 0; p < rows * cols; p++) maskData[p] = keep2.has(l2Data[p]) ? 255 : 0;
    }
    l2.delete(); s2.delete(); c2.delete();
  }

  const kErode = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(13, 13));
  let interior = new cv.Mat();
  cv.erode(mask, interior, kErode);
  kErode.delete();
  if (cv.countNonZero(interior) < 500) {
    interior.delete();
    interior = mask.clone();
  }

  hsv.delete();
  return { mask, interior };
}
