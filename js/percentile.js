/**
 * percentile.js — histogram-based percentile/median for 8-bit single-channel
 * data, optionally restricted to a mask. OpenCV.js has no np.percentile
 * equivalent, so every _med / _p10 / _p90 / REL_REF_Q call in the Python
 * source is rebuilt on top of this.
 *
 * Because source data is always CV_8U (0-255), a 256-bin histogram gives an
 * EXACT order statistic (not an approximation) — only the two candidate
 * values either side of the fractional position need interpolating, exactly
 * as numpy's default 'linear' percentile method does.
 */

/**
 * Build a 256-bin histogram of one channel of `mat`, optionally restricted
 * to pixels where `maskMat` is non-zero.
 *
 * @param {cv.Mat} mat CV_8U*, any channel count
 * @param {number} channelIdx which channel to histogram
 * @param {cv.Mat|null} maskMat CV_8UC1 (0/255), or null for "every pixel"
 * @returns {{hist: Uint32Array(256), n: number}}
 */
export function channelHistogram(mat, channelIdx, maskMat = null) {
  const rows = mat.rows, cols = mat.cols, ch = mat.channels();
  const data = mat.data;
  const hist = new Uint32Array(256);
  let n = 0;
  if (maskMat) {
    const mData = maskMat.data;
    for (let i = 0; i < rows * cols; i++) {
      if (mData[i]) {
        hist[data[i * ch + channelIdx]]++;
        n++;
      }
    }
  } else {
    for (let i = 0; i < rows * cols; i++) {
      hist[data[i * ch + channelIdx]]++;
      n++;
    }
  }
  return { hist, n };
}

/**
 * numpy-compatible ('linear' method) percentile from a 256-bin histogram.
 * @param {Uint32Array} hist length-256 counts
 * @param {number} n total count (sum of hist)
 * @param {number} q percentile in [0, 100]
 */
export function percentileFromHistogram(hist, n, q) {
  if (n === 0) return 0;
  const pos = (q / 100) * (n - 1);
  const lo = Math.floor(pos), hi = Math.ceil(pos), frac = pos - lo;

  let cum = 0, vLo = -1, vHi = -1;
  for (let v = 0; v < 256; v++) {
    const c = hist[v];
    if (c === 0) continue;
    const start = cum, end = cum + c - 1;
    if (vLo === -1 && lo >= start && lo <= end) vLo = v;
    if (vHi === -1 && hi >= start && hi <= end) vHi = v;
    cum += c;
    if (vLo !== -1 && vHi !== -1) break;
  }
  if (vLo === -1) vLo = 0;
  if (vHi === -1) vHi = vLo;
  return vLo + (vHi - vLo) * frac;
}

/** Convenience: median (50th percentile) of one channel, optionally masked. */
export function channelMedian(mat, channelIdx, maskMat = null) {
  const { hist, n } = channelHistogram(mat, channelIdx, maskMat);
  return percentileFromHistogram(hist, n, 50);
}
