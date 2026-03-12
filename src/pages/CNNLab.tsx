import { useState, useEffect, useRef, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════
   CNN MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function conv2D(inp, ker, stride = 1, padding = 0) {
  let padded = inp;
  if (padding > 0) {
    const pH = inp.length + 2 * padding, pW = inp[0].length + 2 * padding;
    padded = Array(pH).fill(0).map(() => Array(pW).fill(0));
    for (let i = 0; i < inp.length; i++) for (let j = 0; j < inp[0].length; j++) padded[i + padding][j + padding] = inp[i][j];
  }
  const k = ker.length, oH = Math.floor((padded.length - k) / stride) + 1, oW = Math.floor((padded[0].length - k) / stride) + 1;
  const out = [];
  for (let i = 0; i < oH; i++) { const row = []; for (let j = 0; j < oW; j++) { let s = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += (padded[i * stride + a]?.[j * stride + b] || 0) * ker[a][b]; row.push(s); } out.push(row); }
  return out;
}
function conv3D(channels, kernels3D, stride = 1, padding = 0) {
  // kernels3D is array of 2D kernels, one per channel, same size
  let summed = null;
  for (let c = 0; c < channels.length; c++) {
    const ker = kernels3D[c % kernels3D.length];
    const result = conv2D(channels[c], ker, stride, padding);
    if (!summed) summed = result;
    else for (let i = 0; i < result.length; i++) for (let j = 0; j < result[0].length; j++) summed[i][j] += result[i][j];
  }
  return summed;
}
function maxP(inp, ps = 2, st = 2) {
  const oH = Math.floor((inp.length - ps) / st) + 1, oW = Math.floor((inp[0].length - ps) / st) + 1, out = [];
  for (let i = 0; i < oH; i++) { const row = []; for (let j = 0; j < oW; j++) { let mx = -Infinity; for (let a = 0; a < ps; a++) for (let b = 0; b < ps; b++) { const v = inp[i * st + a]?.[j * st + b] ?? 0; if (v > mx) mx = v; } row.push(mx); } out.push(row); }
  return out;
}
function avgP(inp, ps = 2, st = 2) {
  const oH = Math.floor((inp.length - ps) / st) + 1, oW = Math.floor((inp[0].length - ps) / st) + 1, out = [];
  for (let i = 0; i < oH; i++) { const row = []; for (let j = 0; j < oW; j++) { let s = 0; for (let a = 0; a < ps; a++) for (let b = 0; b < ps; b++) s += inp[i * st + a]?.[j * st + b] ?? 0; row.push(s / (ps * ps)); } out.push(row); }
  return out;
}
function relu(m) { return m.map(r => r.map(v => Math.max(0, v))); }
function flattenM(chs) { const f = []; chs.forEach(c => { if (Array.isArray(c[0])) c.forEach(r => r.forEach(v => f.push(v))); else c.forEach(v => f.push(v)); }); return f; }
function softmaxFn(a) { const mx = Math.max(...a), e = a.map(v => Math.exp(v - mx)), s = e.reduce((x, y) => x + y, 0); return e.map(v => v / s); }

/* ═══════════════════════════════════════════════════════════
   SAMPLE IMAGES — 8×8 grids, Grayscale + RGB
   ═══════════════════════════════════════════════════════════ */
function mkImg(type, sz = 8) {
  const m = () => Array(sz).fill(0).map(() => Array(sz).fill(0));
  if (type === "vert") { const g = m(); for (let i = 0; i < sz; i++) { g[i][3] = 1; g[i][4] = 1; } return [g]; }
  if (type === "horiz") { const g = m(); for (let j = 0; j < sz; j++) { g[3][j] = 1; g[4][j] = 1; } return [g]; }
  if (type === "diag") { const g = m(); for (let i = 0; i < sz; i++) g[i][i] = 1; return [g]; }
  if (type === "cross") { const g = m(); for (let i = 0; i < sz; i++) { g[i][sz >> 1] = 1; g[sz >> 1][i] = 1; } return [g]; }
  if (type === "circle") { const g = m(), c = sz / 2 - .5, r = sz / 3; for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) if (Math.abs(Math.sqrt((i - c) ** 2 + (j - c) ** 2) - r) < 1) g[i][j] = 1; return [g]; }
  if (type === "box") { const g = m(); for (let i = 1; i < sz - 1; i++) { g[1][i] = 1; g[sz - 2][i] = 1; g[i][1] = 1; g[i][sz - 2] = 1; } return [g]; }
  if (type === "rgb_redsq") { const r = m(), g = m(), b = m(); for (let i = 2; i < 6; i++) for (let j = 2; j < 6; j++) { r[i][j] = 1; g[i][j] = .1; b[i][j] = .1; } return [r, g, b]; }
  if (type === "rgb_bluecircle") { const r = m(), g = m(), b = m(), cx = sz / 2 - .5, rad = sz / 3; for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) { const d = Math.sqrt((i - cx) ** 2 + (j - cx) ** 2); if (d < rad) { r[i][j] = .1; g[i][j] = .3; b[i][j] = 1; } } return [r, g, b]; }
  if (type === "rgb_greentriangle") { const r = m(), g = m(), b = m(); for (let i = 2; i < 7; i++) { const w = (i - 2) * 1.2, l = Math.floor(4 - w / 2), ri = Math.floor(4 + w / 2); for (let j = l; j <= ri; j++) if (j >= 0 && j < sz) { r[i][j] = .1; g[i][j] = .9; b[i][j] = .2; } } return [r, g, b]; }
  if (type === "rgb_sunset") { const r = m(), g = m(), b = m(); for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) { const t = i / (sz - 1); r[i][j] = 1 - t * .3; g[i][j] = .5 - t * .4; b[i][j] = .2 + t * .6; } return [r, g, b]; }
  if (type === "rgb_flag") { const r = m(), g = m(), b = m(); for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) { if (i < 3) { r[i][j] = 1; g[i][j] = .5; b[i][j] = 0; } else if (i < 5) { r[i][j] = 1; g[i][j] = 1; b[i][j] = 1; } else { r[i][j] = .1; g[i][j] = .6; b[i][j] = .2; } } return [r, g, b]; }
  if (type === "rgb_gradient") { const r = m(), g = m(), b = m(); for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) { r[i][j] = i / (sz - 1); g[i][j] = j / (sz - 1); b[i][j] = 1 - (i + j) / (2 * sz - 2); } return [r, g, b]; }
  return [m()];
}
const IMAGES = [
  { id: "vert", name: "Vertical Line", rgb: false }, { id: "horiz", name: "Horizontal Line", rgb: false },
  { id: "diag", name: "Diagonal", rgb: false }, { id: "cross", name: "Cross (+)", rgb: false },
  { id: "circle", name: "Circle", rgb: false }, { id: "box", name: "Box", rgb: false },
  { id: "rgb_redsq", name: "🟥 Red Square (RGB)", rgb: true }, { id: "rgb_bluecircle", name: "🔵 Blue Circle (RGB)", rgb: true },
  { id: "rgb_greentriangle", name: "🟢 Green Triangle (RGB)", rgb: true }, { id: "rgb_sunset", name: "🌅 Sunset Gradient (RGB)", rgb: true },
  { id: "rgb_flag", name: "🇮🇳 Tricolor Flag (RGB)", rgb: true }, { id: "rgb_gradient", name: "🎨 RGB Gradient (RGB)", rgb: true },
];
const PRESETS = {
  edge: { n: "Edge Detection", k: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]] },
  vert_edge: { n: "Vertical Edge", k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  horiz_edge: { n: "Horizontal Edge", k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  sharpen: { n: "Sharpen", k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] },
  blur: { n: "Box Blur", k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => +(v / 9).toFixed(3))) },
  emboss: { n: "Emboss", k: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]] },
  identity: { n: "Identity", k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]] },
};
function kernelFor(id) { const seed = id.split('').reduce((a, c) => a + c.charCodeAt(0), 0); const pool = Object.values(PRESETS); return pool[seed % pool.length].k; }

/* ═══════════════════════════════════════════════════════════
   FORWARD PASS — handles RGB multi-channel properly
   ═══════════════════════════════════════════════════════════ */
function fwdPass(layers, img) {
  const outs = new Map(); let cur = img;
  for (const L of layers) {
    try {
      if (L.type === "conv2d") {
        const ker = kernelFor(L.id), nf = Math.min(L.cfg.filters, 6), chs = [];
        const inChs = (Array.isArray(cur[0]) && Array.isArray(cur[0][0])) ? cur : [cur];
        const pad = L.cfg.padding === "same" ? Math.floor(ker.length / 2) : 0;
        for (let f = 0; f < nf; f++) {
          const kers3D = inChs.map(() => ker); // same kernel per channel (simplified)
          const result = conv3D(inChs, kers3D, L.cfg.stride, pad);
          chs.push(L.cfg.activation === "relu" ? relu(result) : result);
        }
        cur = chs; outs.set(L.id, { t: "3d", d: cur });
      } else if (L.type === "maxpool") { cur = cur.map(c => maxP(c, L.cfg.ps, L.cfg.st)); outs.set(L.id, { t: "3d", d: cur }); }
      else if (L.type === "avgpool") { cur = cur.map(c => avgP(c, L.cfg.ps, L.cfg.st)); outs.set(L.id, { t: "3d", d: cur }); }
      else if (L.type === "flatten") { const f = flattenM(cur); cur = [f]; outs.set(L.id, { t: "1d", d: f }); }
      else if (L.type === "dense") {
        const inp = Array.isArray(cur[0]) && Array.isArray(cur[0][0]) ? flattenM(cur) : cur.flat ? cur.flat() : cur[0] || [];
        const r = []; for (let i = 0; i < L.cfg.units; i++) { let s = 0; for (let j = 0; j < Math.min(inp.length, 20); j++) s += inp[j] * (Math.sin(i * 13.7 + j * 7.3) * 0.3); r.push(L.cfg.activation === "relu" ? Math.max(0, s) : s); }
        cur = [r]; outs.set(L.id, { t: "1d", d: r });
      } else if (L.type === "softmax") {
        const inp = Array.isArray(cur[0]) && Array.isArray(cur[0][0]) ? flattenM(cur) : cur.flat ? cur.flat() : cur[0] || [];
        const sm = softmaxFn(inp.slice(0, 10)); cur = [sm]; outs.set(L.id, { t: "1d", d: sm });
      }
    } catch (e) { outs.set(L.id, { t: "err", d: null }); }
  }
  return outs;
}

/* ═══════════════════════════════════════════════════════════
   RENDERING COMPONENTS
   ═══════════════════════════════════════════════════════════ */
const S = { mono: "'IBM Plex Mono', monospace", sans: "'IBM Plex Sans', system-ui, sans-serif" };
const CHN_COLORS = ["#ef4444", "#22c55e", "#3b82f6"]; // R, G, B
const CHN_NAMES = ["Red (R)", "Green (G)", "Blue (B)"];

function VisualGrid({ data, cellSize = 36, highlight, highlightColor = "#facc15", activeCell, label, maxCells = 10, channelColor }) {
  if (!data?.length || !data[0]?.length) return null;
  const rows = data.length, cols = data[0].length;
  const show = rows <= maxCells && cols <= maxCells;
  let mn = Infinity, mx = -Infinity;
  data.forEach(r => r.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; }));
  const range = mx - mn || 1;
  return (
    <div>
      {label && <p style={{ fontSize: 10, fontWeight: 700, color: channelColor || "#94a3b8", marginBottom: 4, fontFamily: S.mono }}>{label}</p>}
      <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: 1, background: "#1e293b", padding: 1, borderRadius: 6, border: `1px solid ${channelColor ? channelColor + "44" : "#334155"}` }}>
        {data.map((row, ri) => row.map((val, ci) => {
          const inHL = highlight && ri >= highlight.r && ri < highlight.r + highlight.h && ci >= highlight.c && ci < highlight.c + highlight.w;
          const isActive = activeCell && ri === activeCell.r && ci === activeCell.c;
          const norm = (val - mn) / range;
          const baseBg = channelColor ? `${channelColor}${Math.round(norm * 200 + 20).toString(16).padStart(2, '0')}` : `hsl(210,15%,${10 + norm * 50}%)`;
          const bg = inHL ? highlightColor + "55" : isActive ? "#22c55e" : baseBg;
          const border = inHL ? `2px solid ${highlightColor}` : isActive ? "2px solid #22c55e" : "1px solid transparent";
          return (
            <div key={`${ri}-${ci}`} style={{
              width: cellSize, height: cellSize, display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: cellSize < 30 ? 7 : cellSize < 40 ? 9 : 11, fontWeight: 600, fontFamily: S.mono,
              background: bg, border, color: isActive || norm > 0.5 ? "#fff" : "#94a3b8",
              borderRadius: 2, transition: "all 0.2s",
              boxShadow: isActive ? "0 0 8px rgba(34,197,94,0.5)" : inHL ? `0 0 6px ${highlightColor}44` : "none",
            }}>
              {show ? val.toFixed(1) : ""}
            </div>
          );
        }))}
      </div>
      <p style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>{rows}×{cols}</p>
    </div>
  );
}

function KernelGrid({ kernel, cellSize = 42, label, channelColor }) {
  const k = kernel.length;
  return (
    <div>
      {label && <p style={{ fontSize: 10, fontWeight: 700, color: channelColor || "#f59e0b", marginBottom: 4, fontFamily: S.mono }}>{label}</p>}
      <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${k}, ${cellSize}px)`, gap: 2 }}>
        {kernel.flat().map((v, i) => (
          <div key={i} style={{
            width: cellSize, height: cellSize, display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 11, fontWeight: 700, fontFamily: S.mono,
            background: v > 0 ? `rgba(250,204,21,${Math.min(Math.abs(v) / 5 + .15, .9)})` : v < 0 ? `rgba(239,68,68,${Math.min(Math.abs(v) / 5 + .15, .9)})` : "rgba(100,116,139,.25)",
            color: "#fff", borderRadius: 4, border: v > 0 ? "2px solid #eab308" : v < 0 ? "2px solid #dc2626" : "2px solid #475569",
          }}>{v.toFixed(1)}</div>
        ))}
      </div>
    </div>
  );
}

function RGBCanvas({ channels, size = 160 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !channels || channels.length < 3) return;
    const R = channels[0], G = channels[1], B = channels[2], rows = R.length, cols = R[0].length;
    c.width = cols; c.height = rows; const ctx = c.getContext("2d"), img = ctx.createImageData(cols, rows);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const idx = (i * cols + j) * 4;
      img.data[idx] = Math.round(Math.min(1, Math.max(0, R[i][j])) * 255);
      img.data[idx + 1] = Math.round(Math.min(1, Math.max(0, G[i][j])) * 255);
      img.data[idx + 2] = Math.round(Math.min(1, Math.max(0, B[i][j])) * 255);
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [channels]);
  if (!channels || channels.length < 3) return null;
  return <canvas ref={ref} style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 8, border: "1px solid #334155" }} />;
}

function FeatureCanvas({ data, size = 72 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !data?.length || !data[0]?.length) return;
    const ctx = c.getContext("2d"), rows = data.length, cols = data[0].length; c.width = cols; c.height = rows;
    let mn = Infinity, mx = -Infinity; data.forEach(r => r.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; })); const rng = mx - mn || 1;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const g = Math.round(((data[i][j] - mn) / rng) * 255); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(j, i, 1, 1); }
  }, [data]);
  return <canvas ref={ref} style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 6, border: "1px solid #334155" }} />;
}

function OutputImagePreview({ channels, size = 80 }) {
  const ref = useRef(null);
  const multi = channels?.length >= 3;
  useEffect(() => {
    const c = ref.current; if (!c || !channels?.length) return;
    const ch0 = channels[0]; if (!ch0?.length || !ch0[0]?.length) return;
    const rows = ch0.length, cols = ch0[0].length; c.width = cols; c.height = rows; const ctx = c.getContext("2d");
    const norm = ch => { let mn = Infinity, mx = -Infinity; ch.forEach(r => r.forEach(v => { if (v < mn) mn = v; if (v > mx) mx = v; })); const rng = mx - mn || 1; return ch.map(r => r.map(v => (v - mn) / rng)); };
    if (multi) {
      const R = norm(channels[0]), G = norm(channels[1]), B = norm(channels[2]);
      const img = ctx.createImageData(cols, rows);
      for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const idx = (i * cols + j) * 4; img.data[idx] = Math.round(R[i][j] * 255); img.data[idx + 1] = Math.round(G[i][j] * 255); img.data[idx + 2] = Math.round(B[i][j] * 255); img.data[idx + 3] = 255; }
      ctx.putImageData(img, 0, 0);
    } else { const G = norm(channels[0]); for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) { const g = Math.round(G[i][j] * 255); ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(j, i, 1, 1); } }
  }, [channels, multi]);
  if (!channels?.length || !channels[0]?.length) return null;
  return <canvas ref={ref} style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 6, border: "1px solid #334155" }} />;
}

function TensorShape({ shape, color = "#60a5fa" }) {
  return <span style={{ fontSize: 11, fontFamily: S.mono, fontWeight: 700, padding: "3px 8px", borderRadius: 6, background: `${color}15`, border: `1px solid ${color}33`, color }}>[{shape.join(" × ")}]</span>;
}

/* ═══════════════════════════════════════════════════════════
   🌟 THE MAIN EVENT: RGB-AWARE CONVOLUTION STEP-BY-STEP MODULE
   ═══════════════════════════════════════════════════════════ */
function ConvolutionModule({ inputChannels, onParamChange }) {
  const isRGB = inputChannels.length === 3;
  const nCh = inputChannels.length;
  const sz = inputChannels[0].length;
  // User-adjustable parameters
  const [kernelPreset, setKernelPreset] = useState("sharpen");
  const [stride, setStride] = useState(1);
  const [padding, setPadding] = useState(0);
  const [numFilters, setNumFilters] = useState(1);
  const [activation, setActivation] = useState("relu");
  const [step, setStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);
  // Animation position
  const [pos, setPos] = useState({ r: 0, c: 0 });
  const [animating, setAnimating] = useState(false);

  const kernel = PRESETS[kernelPreset]?.k || PRESETS.sharpen.k;
  const kSize = kernel.length;

  // Compute per-channel convolutions and total
  const perChannelOutputs = useMemo(() => inputChannels.map(ch => conv2D(ch, kernel, stride, padding)), [inputChannels, kernel, stride, padding]);
  const summedOutput = useMemo(() => {
    const o = perChannelOutputs[0].map(r => [...r]);
    for (let c = 1; c < perChannelOutputs.length; c++) for (let i = 0; i < o.length; i++) for (let j = 0; j < o[0].length; j++) o[i][j] += perChannelOutputs[c][i][j];
    return o;
  }, [perChannelOutputs]);
  const activatedOutput = useMemo(() => activation === "relu" ? relu(summedOutput) : summedOutput, [summedOutput, activation]);

  const outH = activatedOutput.length, outW = activatedOutput[0]?.length || 0;
  const padH = sz + 2 * padding, padW = sz + 2 * padding;

  // Padded input for display
  const paddedChannels = useMemo(() => {
    if (padding === 0) return inputChannels;
    return inputChannels.map(ch => {
      const p = Array(padH).fill(0).map(() => Array(padW).fill(0));
      for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) p[i + padding][j + padding] = ch[i][j];
      return p;
    });
  }, [inputChannels, padding, padH, padW, sz]);

  // Step definitions
  const STEPS = [
    { t: "📷 Color Image Input", d: `The image arrives as ${nCh} channel${nCh > 1 ? "s" : ""} (${isRGB ? "R, G, B" : "Grayscale"}) of ${sz}×${sz} pixels.` },
    ...(isRGB ? [{ t: "🔴🟢🔵 Split Into RGB Channels", d: `Each color channel is a separate ${sz}×${sz} matrix of intensity values (0.0–1.0).` }] : []),
    ...(padding > 0 ? [{ t: `📐 Apply Padding (p=${padding})`, d: `${padding}px of zeros added around each channel. Size: ${sz}×${sz} → ${padH}×${padW}. Tensor: [${nCh}, ${padH}, ${padW}]` }] : []),
    { t: `🎯 3D Kernel: ${kSize}×${kSize}×${nCh}`, d: `The filter has ${nCh} slice${nCh > 1 ? "s" : ""}, one per channel. Each slice is ${kSize}×${kSize}. Total weights: ${kSize * kSize * nCh}.` },
    { t: "✖️ Per-Channel Convolution", d: `Each kernel slice convolves with its matching channel independently. Stride=${stride}.` },
    { t: "➕ Sum Across Channels", d: `The ${nCh} per-channel results are summed element-wise to produce one output feature map.` },
    { t: `⚡ Activation: ${activation.toUpperCase()}`, d: activation === "relu" ? "ReLU(x) = max(0, x). Negative values become 0." : "Linear pass-through (no activation)." },
    { t: "🎬 Animated Sliding Window", d: "Watch the kernel slide across the padded input, producing each output value." },
    ...(numFilters > 1 ? [{ t: `🎨 Multiple Filters (${numFilters})`, d: `Each filter detects a different feature. ${numFilters} filters → ${numFilters} output channels. Output tensor: [${numFilters}, ${outH}, ${outW}].` }] : []),
    { t: "📊 Final Output Tensor", d: `Output shape: [${numFilters}, ${outH}, ${outW}]. Total values: ${numFilters * outH * outW}.` },
  ];
  const totalSteps = STEPS.length;

  useEffect(() => {
    if (autoPlay && step < totalSteps - 1) { const t = setTimeout(() => setStep(p => p + 1), 3500); return () => clearTimeout(t); }
    else setAutoPlay(false);
  }, [autoPlay, step, totalSteps]);

  // Sliding window animation
  useEffect(() => {
    if (!animating) return;
    const t = setTimeout(() => {
      let { r, c } = pos; c++; if (c >= outW) { c = 0; r++; } if (r >= outH) { setAnimating(false); return; }
      setPos({ r, c });
    }, 250);
    return () => clearTimeout(t);
  }, [animating, pos, outH, outW]);

  // Compute current position math
  const iR = pos.r * stride, iC = pos.c * stride;
  const channelProducts = paddedChannels.map((ch, ci) => {
    const prods = [];
    let sum = 0;
    for (let a = 0; a < kSize; a++) for (let b = 0; b < kSize; b++) {
      const iv = ch[iR + a]?.[iC + b] ?? 0, kv = kernel[a][b], p = iv * kv;
      prods.push({ iv, kv, p }); sum += p;
    }
    return { prods, sum };
  });
  const totalSum = channelProducts.reduce((a, c) => a + c.sum, 0);
  const activatedVal = activation === "relu" ? Math.max(0, totalSum) : totalSum;

  // Determine which step content to render
  const stepIdx = step;
  let stepOffset = 0;
  const isStepRGBSplit = isRGB && stepIdx === 1;
  const isStepPadding = padding > 0 && stepIdx === (isRGB ? 2 : 1);
  const isStep3DKernel = stepIdx === (isRGB ? 1 : 0) + (padding > 0 ? 1 : 0) + 1;
  const isStepPerCh = stepIdx === (isRGB ? 1 : 0) + (padding > 0 ? 1 : 0) + 2;
  const isStepSum = stepIdx === (isRGB ? 1 : 0) + (padding > 0 ? 1 : 0) + 3;
  const isStepAct = stepIdx === (isRGB ? 1 : 0) + (padding > 0 ? 1 : 0) + 4;
  const isStepAnimate = stepIdx === (isRGB ? 1 : 0) + (padding > 0 ? 1 : 0) + 5;
  const isStepMultiFilter = numFilters > 1 && stepIdx === totalSteps - 2;
  const isStepFinal = stepIdx === totalSteps - 1;

  return (
    <div>
      {/* Parameter Controls */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16, padding: 12, background: "#0f172a", borderRadius: 10, border: "1px solid #1e293b" }}>
        <label style={{ fontSize: 11, color: "#64748b" }}>Filter:
          <select value={kernelPreset} onChange={e => setKernelPreset(e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 6, fontSize: 11, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155" }}>
            {Object.entries(PRESETS).map(([k, v]) => <option key={k} value={k}>{v.n}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 11, color: "#64748b" }}>Stride:
          <select value={stride} onChange={e => setStride(+e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 6, fontSize: 11, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155" }}>
            {[1, 2, 3].map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 11, color: "#64748b" }}>Padding:
          <select value={padding} onChange={e => setPadding(+e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 6, fontSize: 11, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155" }}>
            {[0, 1, 2].map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 11, color: "#64748b" }}>Filters:
          <select value={numFilters} onChange={e => setNumFilters(+e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 6, fontSize: 11, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155" }}>
            {[1, 2, 3, 4].map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <label style={{ fontSize: 11, color: "#64748b" }}>Activation:
          <select value={activation} onChange={e => setActivation(e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 6, fontSize: 11, background: "#1e293b", color: "#e2e8f0", border: "1px solid #334155" }}>
            <option value="relu">ReLU</option><option value="none">None</option>
          </select>
        </label>
        <div style={{ marginLeft: "auto", fontSize: 10, color: "#475569" }}>
          Input: <TensorShape shape={[nCh, sz, sz]} color="#60a5fa" /> → Output: <TensorShape shape={[numFilters, outH, outW]} color="#22c55e" />
        </div>
      </div>

      {/* Step Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", gap: 2 }}>
          {STEPS.map((_, i) => <button key={i} onClick={() => { setStep(i); setAutoPlay(false); }} style={{ width: 24, height: 6, borderRadius: 3, background: i <= step ? "#3b82f6" : "#1e293b", border: "none", cursor: "pointer" }} />)}
        </div>
        <div style={{ display: "flex", gap: 6 }}>
          <button onClick={() => { setStep(0); setAutoPlay(false); setPos({ r: 0, c: 0 }); setAnimating(false); }} style={{ padding: "3px 8px", borderRadius: 6, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
          <button onClick={() => setAutoPlay(!autoPlay)} style={{ padding: "3px 10px", borderRadius: 6, fontSize: 10, fontWeight: 700, background: autoPlay ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{autoPlay ? "⏸" : "▶ Auto"}</button>
        </div>
      </div>

      {/* Step Title */}
      <div style={{ padding: "10px 14px", borderRadius: 10, background: "#3b82f611", border: "1px solid #3b82f633", marginBottom: 14, display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, fontWeight: 800, background: "#3b82f622", color: "#3b82f6", flexShrink: 0 }}>{step + 1}</div>
        <div><h5 style={{ fontSize: 14, fontWeight: 700, color: "#fff", margin: 0 }}>{STEPS[step]?.t}</h5><p style={{ fontSize: 11, color: "#94a3b8", margin: 0 }}>{STEPS[step]?.d}</p></div>
      </div>

      {/* Step Visual Content */}
      <div style={{ minHeight: 250, padding: 16, background: "rgba(15,23,42,0.5)", borderRadius: 12, border: "1px solid #1e293b" }}>
        {/* Step 0: Show original image */}
        {step === 0 && (
          <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
            {isRGB ? <RGBCanvas channels={inputChannels} size={180} /> : <VisualGrid data={inputChannels[0]} cellSize={38} />}
            <div style={{ fontSize: 13, color: "#94a3b8", maxWidth: 300 }}>
              <p style={{ marginBottom: 8 }}>{isRGB ? "This is a color image with 3 channels: Red, Green, and Blue." : "This is a grayscale image with 1 channel."}</p>
              <p>Tensor shape: <TensorShape shape={[nCh, sz, sz]} /></p>
              <p style={{ marginTop: 8, fontSize: 12 }}>Total pixel values: <b style={{ color: "#fff" }}>{nCh * sz * sz}</b></p>
            </div>
          </div>
        )}

        {/* RGB Split */}
        {isStepRGBSplit && (
          <div>
            <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap", marginBottom: 16 }}>
              <div style={{ textAlign: "center" }}>
                <RGBCanvas channels={inputChannels} size={120} />
                <p style={{ fontSize: 10, color: "#94a3b8", marginTop: 4 }}>Original RGB</p>
              </div>
              <div style={{ fontSize: 28, color: "#facc15", fontWeight: 800, alignSelf: "center" }}>=</div>
              {inputChannels.map((ch, ci) => (
                <div key={ci}>
                  <VisualGrid data={ch} cellSize={30} label={CHN_NAMES[ci]} channelColor={CHN_COLORS[ci]} maxCells={10} />
                </div>
              ))}
            </div>
            <p style={{ fontSize: 11, color: "#94a3b8" }}>Each channel is an independent {sz}×{sz} matrix. The CNN processes them <b style={{ color: "#fff" }}>simultaneously</b> with a 3D kernel.</p>
          </div>
        )}

        {/* Padding */}
        {isStepPadding && (
          <div style={{ display: "flex", gap: 24, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VisualGrid data={inputChannels[0]} cellSize={32} label={`Original ${sz}×${sz}`} />
            <div style={{ fontSize: 24, color: "#f59e0b", fontWeight: 800, alignSelf: "center" }}>→</div>
            <VisualGrid data={paddedChannels[0]} cellSize={32} label={`Padded ${padH}×${padW}`} />
            <div style={{ fontSize: 12, color: "#94a3b8", maxWidth: 220, alignSelf: "center" }}>
              <p>{padding}px of <b style={{ color: "#facc15" }}>zeros</b> added around each channel.</p>
              <p style={{ marginTop: 6 }}>This preserves spatial dimensions in the output.</p>
              <p style={{ marginTop: 6 }}>Per channel: <TensorShape shape={[padH, padW]} color="#f59e0b" /></p>
            </div>
          </div>
        )}

        {/* 3D Kernel */}
        {isStep3DKernel && (
          <div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap", marginBottom: 16 }}>
              {inputChannels.map((_, ci) => (
                <KernelGrid key={ci} kernel={kernel} cellSize={38} label={isRGB ? `Kernel slice: ${CHN_NAMES[ci]}` : "Kernel"} channelColor={isRGB ? CHN_COLORS[ci] : "#f59e0b"} />
              ))}
            </div>
            <p style={{ fontSize: 12, color: "#94a3b8" }}>
              Total kernel shape: <TensorShape shape={[kSize, kSize, nCh]} color="#f59e0b" /> = <b style={{ color: "#fff" }}>{kSize * kSize * nCh} weights</b>.
              {isRGB && " Each color channel has its own kernel slice."}
            </p>
          </div>
        )}

        {/* Per-channel convolution */}
        {isStepPerCh && (
          <div>
            {inputChannels.map((ch, ci) => {
              const demo = paddedChannels[ci].slice(0, 5).map(r => r.slice(0, 5));
              const prods = [];
              for (let a = 0; a < kSize; a++) for (let b = 0; b < kSize; b++) prods.push((demo[a]?.[b] ?? 0) * kernel[a][b]);
              const sum = prods.reduce((a, b) => a + b, 0);
              return (
                <div key={ci} style={{ display: "flex", gap: 12, alignItems: "center", marginBottom: 12, padding: 10, borderRadius: 8, background: "#0f172a", border: `1px solid ${isRGB ? CHN_COLORS[ci] + "33" : "#334155"}` }}>
                  <div style={{ width: 8, height: 40, borderRadius: 4, background: isRGB ? CHN_COLORS[ci] : "#f59e0b", flexShrink: 0 }} />
                  <VisualGrid data={demo} cellSize={28} highlight={{ r: 0, c: 0, h: kSize, w: kSize }} highlightColor={isRGB ? CHN_COLORS[ci] : "#facc15"} maxCells={8} />
                  <span style={{ fontSize: 16, color: "#facc15", fontWeight: 800 }}>⊛</span>
                  <KernelGrid kernel={kernel} cellSize={28} />
                  <span style={{ fontSize: 14, color: "#22c55e", fontWeight: 800 }}>=</span>
                  <span style={{ fontSize: 14, fontFamily: S.mono, fontWeight: 800, color: sum >= 0 ? "#22c55e" : "#ef4444" }}>{sum.toFixed(2)}</span>
                </div>
              );
            })}
            <p style={{ fontSize: 11, color: "#94a3b8" }}>Each channel convolved independently. Stride={stride}. Output per channel: <TensorShape shape={[outH, outW]} /></p>
          </div>
        )}

        {/* Sum across channels */}
        {isStepSum && (
          <div>
            <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
              {perChannelOutputs.map((ch, ci) => (
                <VisualGrid key={ci} data={ch} cellSize={28} label={isRGB ? CHN_NAMES[ci] + " output" : `Ch ${ci + 1}`} channelColor={isRGB ? CHN_COLORS[ci] : undefined} maxCells={10} />
              ))}
              <div style={{ fontSize: 24, fontWeight: 800, color: "#22c55e", alignSelf: "center" }}>Σ →</div>
              <VisualGrid data={summedOutput} cellSize={28} label="Summed output" maxCells={10} />
            </div>
            <div style={{ marginTop: 12, padding: "8px 16px", background: "#064e3b", borderRadius: 8, border: "1px solid #22c55e", display: "inline-block" }}>
              <span style={{ fontSize: 12, fontFamily: S.mono, color: "#22c55e" }}>
                {isRGB ? "R_out + G_out + B_out" : "Summed"} = Feature Map <TensorShape shape={[1, outH, outW]} color="#22c55e" />
              </span>
            </div>
          </div>
        )}

        {/* Activation */}
        {isStepAct && (
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VisualGrid data={summedOutput} cellSize={30} label="Before activation" maxCells={10} />
            <div style={{ fontSize: 24, color: "#f59e0b", fontWeight: 800, alignSelf: "center" }}>→ {activation === "relu" ? "ReLU" : "Linear"} →</div>
            <VisualGrid data={activatedOutput} cellSize={30} label="After activation" maxCells={10} />
            <div style={{ alignSelf: "center" }}>
              <FeatureCanvas data={activatedOutput} size={80} />
              <p style={{ fontSize: 9, color: "#22c55e", textAlign: "center", marginTop: 4 }}>Visual output</p>
            </div>
          </div>
        )}

        {/* Animated sliding window */}
        {isStepAnimate && (
          <div>
            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
              <span style={{ fontSize: 11, color: "#64748b" }}>Position ({pos.r},{pos.c}) · Output: <b style={{ color: "#22c55e" }}>{activatedVal.toFixed(3)}</b></span>
              <div style={{ display: "flex", gap: 4 }}>
                <button onClick={() => { setPos({ r: 0, c: 0 }); setAnimating(false); }} style={{ padding: "2px 8px", borderRadius: 4, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
                <button onClick={() => setAnimating(!animating)} style={{ padding: "2px 10px", borderRadius: 4, fontSize: 10, fontWeight: 700, background: animating ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{animating ? "⏸" : "▶ Play"}</button>
              </div>
            </div>
            <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
              {/* Per-channel views with highlight */}
              {paddedChannels.map((ch, ci) => (
                <VisualGrid key={ci} data={ch} cellSize={24} highlight={{ r: iR, c: iC, h: kSize, w: kSize }} highlightColor={isRGB ? CHN_COLORS[ci] : "#facc15"} label={isRGB ? CHN_NAMES[ci].split(" ")[0] : "Input"} channelColor={isRGB ? CHN_COLORS[ci] : undefined} maxCells={12} />
              ))}
              <div style={{ alignSelf: "center", textAlign: "center" }}>
                <KernelGrid kernel={kernel} cellSize={24} />
                <p style={{ fontSize: 8, color: "#f59e0b", marginTop: 2 }}>Kernel</p>
              </div>
              <VisualGrid data={activatedOutput} cellSize={24} activeCell={{ r: pos.r, c: pos.c }} label="Output" maxCells={12} />
            </div>
            {/* Per-channel math */}
            <div style={{ marginTop: 10, padding: 8, background: "#0f172a", borderRadius: 8, border: "1px solid #1e293b", fontSize: 10, fontFamily: S.mono }}>
              {channelProducts.map((cp, ci) => (
                <div key={ci} style={{ marginBottom: 4 }}>
                  <span style={{ color: isRGB ? CHN_COLORS[ci] : "#60a5fa", fontWeight: 700 }}>{isRGB ? ["R", "G", "B"][ci] : "Ch"}: </span>
                  {cp.prods.slice(0, 5).map((p, i) => <span key={i} style={{ color: "#94a3b8" }}>{p.iv.toFixed(1)}×{p.kv.toFixed(1)} </span>)}
                  <span style={{ color: "#94a3b8" }}>{cp.prods.length > 5 ? "... " : ""}</span>
                  <span style={{ color: cp.sum >= 0 ? "#22c55e" : "#ef4444", fontWeight: 700 }}>= {cp.sum.toFixed(2)}</span>
                </div>
              ))}
              <div style={{ borderTop: "1px solid #334155", paddingTop: 4, marginTop: 4, fontWeight: 700 }}>
                <span style={{ color: "#f59e0b" }}>Sum: {totalSum.toFixed(3)}</span>
                <span style={{ color: "#22c55e", marginLeft: 12 }}>→ {activation}({totalSum.toFixed(3)}) = {activatedVal.toFixed(3)}</span>
              </div>
            </div>
          </div>
        )}

        {/* Multiple filters */}
        {isStepMultiFilter && (
          <div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              {Array.from({ length: numFilters }).map((_, fi) => {
                const kIdx = fi % Object.keys(PRESETS).length;
                const kName = Object.values(PRESETS)[kIdx].n;
                return (
                  <div key={fi} style={{ padding: 10, background: "#0f172a", borderRadius: 8, border: "1px solid #334155", textAlign: "center" }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: "#f59e0b", marginBottom: 4 }}>Filter {fi + 1}: {kName}</p>
                    <FeatureCanvas data={activatedOutput} size={64} />
                    <p style={{ fontSize: 9, color: "#475569", marginTop: 2 }}>{outH}×{outW}</p>
                  </div>
                );
              })}
            </div>
            <p style={{ fontSize: 11, color: "#94a3b8", marginTop: 10 }}>
              {numFilters} filters → {numFilters} output channels. Output tensor: <TensorShape shape={[numFilters, outH, outW]} color="#22c55e" />
            </p>
          </div>
        )}

        {/* Final output */}
        {isStepFinal && (
          <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
            <div style={{ textAlign: "center" }}>
              {isRGB ? <RGBCanvas channels={inputChannels} size={100} /> : <FeatureCanvas data={inputChannels[0]} size={100} />}
              <p style={{ fontSize: 10, color: "#60a5fa", marginTop: 4, fontWeight: 700 }}>Input <TensorShape shape={[nCh, sz, sz]} /></p>
            </div>
            <div style={{ fontSize: 28, color: "#3b82f6", fontWeight: 800 }}>→</div>
            <div style={{ textAlign: "center" }}>
              <FeatureCanvas data={activatedOutput} size={100} />
              <p style={{ fontSize: 10, color: "#22c55e", marginTop: 4, fontWeight: 700 }}>Output <TensorShape shape={[numFilters, outH, outW]} color="#22c55e" /></p>
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", maxWidth: 250 }}>
              <p>Filter: <b style={{ color: "#fff" }}>{PRESETS[kernelPreset]?.n}</b></p>
              <p>Stride: <b style={{ color: "#fff" }}>{stride}</b>, Padding: <b style={{ color: "#fff" }}>{padding}</b></p>
              <p>Activation: <b style={{ color: "#fff" }}>{activation}</b></p>
              <p>Params: <b style={{ color: "#fff" }}>{kSize * kSize * nCh * numFilters + numFilters}</b> ({kSize}×{kSize}×{nCh}×{numFilters} + {numFilters} bias)</p>
            </div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
        <button onClick={() => { setStep(Math.max(0, step - 1)); setAutoPlay(false); }} disabled={step === 0} style={{ padding: "5px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === 0 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === 0 ? "not-allowed" : "pointer" }}>← Previous</button>
        <span style={{ fontSize: 10, color: "#475569" }}>{step + 1} / {totalSteps}</span>
        <button onClick={() => { setStep(Math.min(totalSteps - 1, step + 1)); setAutoPlay(false); }} disabled={step === totalSteps - 1} style={{ padding: "5px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === totalSteps - 1 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === totalSteps - 1 ? "not-allowed" : "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   POOLING STEP MODULE (kept concise)
   ═══════════════════════════════════════════════════════════ */
function PoolingModule({ inputData, poolSize, poolType }) {
  const demo = inputData.slice(0, 6).map(r => r.slice(0, 6));
  const output = poolType === "max" ? maxP(demo, poolSize, poolSize) : avgP(demo, poolSize, poolSize);
  const [pos, setPos] = useState({ r: 0, c: 0 });
  const [playing, setPlaying] = useState(false);
  const outH = output.length, outW = output[0]?.length || 0;
  useEffect(() => { if (!playing) return; const t = setTimeout(() => { let { r, c } = pos; c++; if (c >= outW) { c = 0; r++; } if (r >= outH) { setPlaying(false); return; } setPos({ r, c }); }, 400); return () => clearTimeout(t); }, [playing, pos, outH, outW]);
  const iR = pos.r * poolSize, iC = pos.c * poolSize;
  const vals = []; for (let a = 0; a < poolSize; a++) for (let b = 0; b < poolSize; b++) vals.push(demo[iR + a]?.[iC + b] ?? 0);
  const result = poolType === "max" ? Math.max(...vals) : vals.reduce((a, b) => a + b, 0) / vals.length;
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
        <span style={{ fontSize: 11, color: "#64748b" }}>({pos.r},{pos.c}) · {poolType}: <b style={{ color: "#22c55e" }}>{result.toFixed(3)}</b></span>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => { setPos({ r: 0, c: 0 }); setPlaying(false); }} style={{ padding: "2px 8px", borderRadius: 4, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
          <button onClick={() => setPlaying(!playing)} style={{ padding: "2px 10px", borderRadius: 4, fontSize: 10, fontWeight: 700, background: playing ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{playing ? "⏸" : "▶"}</button>
        </div>
      </div>
      <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
        <VisualGrid data={demo} cellSize={36} highlight={{ r: iR, c: iC, h: poolSize, w: poolSize }} highlightColor="#f59e0b" label="Input" />
        <VisualGrid data={output} cellSize={40} activeCell={{ r: pos.r, c: pos.c }} label={`${poolType} Pooled`} />
      </div>
      <div style={{ marginTop: 8, padding: 8, background: "#0f172a", borderRadius: 6, border: "1px solid #1e293b" }}>
        <span style={{ fontSize: 11, fontFamily: S.mono, color: "#f59e0b" }}>{poolType}([{vals.map(v => v.toFixed(1)).join(",")}]) = </span>
        <b style={{ fontSize: 12, fontFamily: S.mono, color: "#22c55e" }}>{result.toFixed(3)}</b>
      </div>
    </div>
  );
}

function Output1D({ data, type }) {
  if (!data?.length) return null;
  const mx = Math.max(...data.map(Math.abs));
  const isSM = type === "softmax";
  return (
    <div style={{ maxHeight: 200, overflowY: "auto" }}>
      {data.slice(0, 20).map((v, i) => (
        <div key={i} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 3 }}>
          <span style={{ fontSize: 10, color: "#64748b", width: 40, textAlign: "right", fontFamily: S.mono }}>{isSM ? `C${i}` : `U${i}`}</span>
          <div style={{ flex: 1, height: 16, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
            <div style={{ height: "100%", width: `${isSM ? v * 100 : Math.min(Math.abs(v) / (mx || 1) * 100, 100)}%`, background: isSM && v === Math.max(...data) ? "#22c55e" : "#3b82f6", borderRadius: 4 }} />
          </div>
          <span style={{ fontSize: 10, fontFamily: S.mono, color: "#94a3b8", width: 50, textAlign: "right" }}>{v.toFixed(4)}</span>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   FULLSCREEN MODAL — uses ConvolutionModule for conv layers
   ═══════════════════════════════════════════════════════════ */
const INFO = {
  conv2d: { icon: "🔍", c: "#3b82f6", t: "Convolutional Layer", d: "Detects features using sliding filters" },
  maxpool: { icon: "⬇️", c: "#f59e0b", t: "Max Pooling", d: "Reduces size keeping max values" },
  avgpool: { icon: "📊", c: "#f59e0b", t: "Average Pooling", d: "Reduces size with averages" },
  flatten: { icon: "➡️", c: "#8b5cf6", t: "Flatten Layer", d: "Reshapes 3D → 1D vector" },
  dense: { icon: "🧠", c: "#ec4899", t: "Dense Layer", d: "Fully connected neurons" },
  softmax: { icon: "📈", c: "#ef4444", t: "Softmax", d: "Scores → Probabilities" },
};

function FullModal({ layer, idx, prevOut, layerOut, onClose }) {
  const info = INFO[layer.type] || { icon: "⚙️", c: "#64748b", t: layer.type, d: "" };
  const [tab, setTab] = useState("explain");
  const is3D = prevOut && Array.isArray(prevOut) && Array.isArray(prevOut[0]) && Array.isArray(prevOut[0][0]);
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(2,6,23,0.95)", backdropFilter: "blur(16px)", overflow: "auto" }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, background: `${info.c}22` }}>{info.icon}</div>
            <div><h2 style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>{info.t}</h2><p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>Layer #{idx + 1} · {info.d}</p></div>
          </div>
          <button onClick={onClose} style={{ width: 36, height: 36, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, color: "#64748b", background: "#1e293b", border: "1px solid #334155", cursor: "pointer" }}>×</button>
        </div>

        <div style={{ background: "rgba(15,23,42,0.6)", borderRadius: 14, border: "1px solid #1e293b", padding: 20, minHeight: 400 }}>
          {layer.type === "conv2d" && is3D && <ConvolutionModule inputChannels={prevOut} />}
          {(layer.type === "maxpool" || layer.type === "avgpool") && is3D && <PoolingModule inputData={prevOut[0]} poolSize={layer.cfg.ps || 2} poolType={layer.type === "maxpool" ? "max" : "average"} />}
          {layer.type === "flatten" && is3D && (
            <div>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>{prevOut.slice(0, 4).map((ch, i) => <VisualGrid key={i} data={ch} cellSize={30} label={`Ch ${i}`} maxCells={10} />)}</div>
              <div style={{ textAlign: "center", fontSize: 24, color: "#8b5cf6", margin: "12px 0" }}>↓ Flatten ↓</div>
              {layerOut && <Output1D data={layerOut.d} type="flatten" />}
            </div>
          )}
          {(layer.type === "dense" || layer.type === "softmax") && layerOut && <Output1D data={layerOut.d} type={layer.type} />}
          {!is3D && !["dense", "softmax", "flatten"].includes(layer.type) && <p style={{ color: "#64748b" }}>Add an input image and run forward pass to see visualization.</p>}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LAYER CARD — shows Input→Output image pipeline
   ═══════════════════════════════════════════════════════════ */
function LCard({ layer, idx, prevOut, layerOut, onRemove, onExpand }) {
  const [exp, setExp] = useState(false);
  const info = INFO[layer.type] || { icon: "⚙️", c: "#64748b" };
  const is3DPrev = prevOut && Array.isArray(prevOut) && Array.isArray(prevOut[0]) && Array.isArray(prevOut[0][0]);
  const is3DOut = layerOut?.t === "3d" && layerOut.d?.length;
  const is1DOut = layerOut?.t === "1d" && layerOut.d?.length;
  return (
    <div style={{ borderRadius: 12, overflow: "hidden", background: "rgba(15,23,42,0.7)", border: `1px solid ${exp ? info.c + "66" : "#1e293b"}`, transition: "all 0.2s" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", cursor: "pointer" }} onClick={() => setExp(!exp)}>
        <div style={{ width: 32, height: 32, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, background: `${info.c}22`, flexShrink: 0 }}>{info.icon}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: 12, fontWeight: 700, color: "#fff" }}>{layer.name}</span>
            <span style={{ fontSize: 9, padding: "1px 5px", borderRadius: 8, background: `${info.c}22`, color: info.c, fontWeight: 600 }}>#{idx + 1}</span>
            {is3DOut && <TensorShape shape={[layerOut.d.length, layerOut.d[0]?.length, layerOut.d[0]?.[0]?.length]} color="#22c55e" />}
            {is1DOut && <TensorShape shape={[layerOut.d.length]} color="#22c55e" />}
          </div>
          <p style={{ fontSize: 10, color: "#475569", margin: 0 }}>{getSummary(layer)}</p>
        </div>
        {/* Collapsed preview: In → Out */}
        {(is3DOut || is3DPrev) && !exp && (
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            {is3DPrev && <OutputImagePreview channels={prevOut} size={36} />}
            <span style={{ fontSize: 12, color: info.c }}>→</span>
            {is3DOut && <OutputImagePreview channels={layerOut.d} size={36} />}
          </div>
        )}
        <button onClick={e => { e.stopPropagation(); onExpand(); }} title="Fullscreen" style={{ width: 26, height: 26, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", background: "transparent", border: "1px solid #334155", color: "#64748b", cursor: "pointer", fontSize: 12 }}>⛶</button>
        <button onClick={e => { e.stopPropagation(); onRemove(); }} style={{ width: 26, height: 26, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", background: "transparent", border: "1px solid #334155", color: "#64748b", cursor: "pointer", fontSize: 10 }}>✕</button>
        <span style={{ color: "#475569", fontSize: 10 }}>{exp ? "▲" : "▼"}</span>
      </div>
      {exp && (
        <div style={{ borderTop: "1px solid #1e293b", padding: 14 }}>
          {/* Input → Output pipeline */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, padding: 12, background: "#0f172a", borderRadius: 10, border: "1px solid #1e293b", overflowX: "auto" }}>
            {is3DPrev && <div style={{ textAlign: "center", flexShrink: 0 }}><OutputImagePreview channels={prevOut} size={72} /><p style={{ fontSize: 8, color: "#60a5fa", marginTop: 2 }}>IN {prevOut.length}ch</p></div>}
            <div style={{ fontSize: 20, color: info.c, fontWeight: 800 }}>→</div>
            {is3DOut && (
              <div style={{ display: "flex", gap: 6, flexShrink: 0 }}>
                {layerOut.d.slice(0, 6).map((ch, i) => <div key={i} style={{ textAlign: "center" }}><FeatureCanvas data={ch} size={52} /><p style={{ fontSize: 7, color: "#475569" }}>F{i + 1}</p></div>)}
              </div>
            )}
            {is1DOut && <div style={{ display: "flex", gap: 1, height: 40, alignItems: "end" }}>
              {layerOut.d.slice(0, 20).map((v, i) => <div key={i} style={{ width: 6, background: layer.type === "softmax" && v === Math.max(...layerOut.d) ? "#22c55e" : "#3b82f6", height: `${Math.max(Math.abs(v) / (Math.max(...layerOut.d.map(Math.abs)) || 1) * 100, 6)}%`, borderRadius: "2px 2px 0 0", minHeight: 2 }} />)}
            </div>}
          </div>
          <button onClick={onExpand} style={{ width: "100%", padding: "7px 0", borderRadius: 8, fontSize: 11, fontWeight: 700, background: `${info.c}15`, color: info.c, border: `1px solid ${info.c}33`, cursor: "pointer" }}>⛶ Fullscreen · Interactive Step-by-Step</button>
        </div>
      )}
    </div>
  );
}

function getSummary(L) {
  if (L.type === "conv2d") return `${L.cfg.filters}f, ${L.cfg.kernelSize}×${L.cfg.kernelSize}, s${L.cfg.stride}, ${L.cfg.activation}`;
  if (L.type === "maxpool" || L.type === "avgpool") return `${L.cfg.ps}×${L.cfg.ps}, s${L.cfg.st}`;
  if (L.type === "dense") return `${L.cfg.units} units, ${L.cfg.activation}`;
  if (L.type === "flatten") return "→ 1D"; if (L.type === "softmax") return "probabilities"; return "";
}

/* ═══════════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════════ */
const LTPLS = {
  conv2d: { type: "conv2d", name: "CONV2D", cfg: { filters: 32, kernelSize: 3, stride: 1, padding: "same", activation: "relu" } },
  maxpool: { type: "maxpool", name: "MAXPOOL", cfg: { ps: 2, st: 2 } },
  avgpool: { type: "avgpool", name: "AVGPOOL", cfg: { ps: 2, st: 2 } },
  flatten: { type: "flatten", name: "FLATTEN", cfg: {} },
  dense: { type: "dense", name: "DENSE", cfg: { units: 10, activation: "relu" } },
  softmax: { type: "softmax", name: "SOFTMAX", cfg: {} },
};

export default function App() {
  const [layers, setLayers] = useState([]);
  const [imgId, setImgId] = useState("rgb_sunset");
  const [outs, setOuts] = useState(new Map());
  const [modal, setModal] = useState(null);
  const imgData = useMemo(() => mkImg(imgId), [imgId]);
  const add = type => { const t = LTPLS[type]; if (t) setLayers(p => [...p, { ...t, id: `${type}-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`, cfg: { ...t.cfg } }]); };
  const remove = id => setLayers(p => p.filter(l => l.id !== id));

  useEffect(() => {
    if (layers.length > 0) { const t = setTimeout(() => setOuts(fwdPass(layers, imgData)), 150); return () => clearTimeout(t); }
    else setOuts(new Map());
  }, [layers, imgData]);

  const prevOut = idx => { if (idx === 0) return imgData; const p = layers[idx - 1]; return outs.get(p?.id)?.d || imgData; };

  const preset = name => {
    const C = (id, f = 32, k = 3, s = 1, act = "relu") => ({ ...LTPLS.conv2d, id, cfg: { filters: f, kernelSize: k, stride: s, padding: "same", activation: act } });
    const MP = (id) => ({ ...LTPLS.maxpool, id, cfg: { ps: 2, st: 2 } });
    const AP = (id) => ({ ...LTPLS.avgpool, id, cfg: { ps: 2, st: 2 } });
    const FL = (id) => ({ ...LTPLS.flatten, id, cfg: {} });
    const DN = (id, u = 10) => ({ ...LTPLS.dense, id, cfg: { units: u, activation: "relu" } });
    const SM = (id) => ({ ...LTPLS.softmax, id, cfg: {} });
    const ps = {
      simple: [C("c1", 8), MP("p1"), FL("f1"), DN("d1", 10), SM("s1")],
      lenet: [C("le1", 6, 5), AP("ap1"), C("le2", 16, 5), AP("ap2"), FL("lf"), DN("ld1", 120), DN("ld2", 84), DN("ld3", 10), SM("ls")],
      resnet: [C("r1", 16), C("r2", 16), C("r3", 16), MP("rp1"), C("r4", 32), C("r5", 32), MP("rp2"), C("r7", 64), AP("rap"), FL("rf"), DN("rd1", 64), DN("rd2", 10), SM("rs")],
      deep: [C("c1", 32), C("c2", 32), MP("p1"), C("c3", 64), C("c4", 64), MP("p2"), FL("f1"), DN("d1", 128), DN("d2", 10), SM("s1")],
      edge: [C("c1", 4)],
    };
    setLayers(ps[name] || []);
  };

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #0c1222 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#3b82f6,#06b6d4)" }}>🧠</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>CNN Lab</h1>
        <span style={{ fontSize: 10 }}>Interactive RGB Convolution Simulator</span>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 14 }}>
          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>📷 Input Image</h3>
              <select value={imgId} onChange={e => setImgId(e.target.value)} style={{ width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}>
                <optgroup label="Grayscale">{IMAGES.filter(i => !i.rgb).map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</optgroup>
                <optgroup label="RGB Color">{IMAGES.filter(i => i.rgb).map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</optgroup>
              </select>
              <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
                {imgData.length === 3 ? <RGBCanvas channels={imgData} size={160} /> : <VisualGrid data={imgData[0]} cellSize={24} />}
              </div>
              <p style={{ fontSize: 9, color: "#475569", textAlign: "center", marginTop: 4 }}>
                <TensorShape shape={[imgData.length, imgData[0].length, imgData[0][0].length]} />
              </p>
            </div>
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>🚀 Templates</h3>
              {[{ id: "simple", n: "Simple CNN", c: "#3b82f6" }, { id: "lenet", n: "LeNet-5", c: "#8b5cf6" }, { id: "resnet", n: "ResNet-style", c: "#f59e0b" }, { id: "deep", n: "Deep CNN", c: "#06b6d4" }, { id: "edge", n: "Edge Detector", c: "#22c55e" }].map(t => (
                <button key={t.id} onClick={() => preset(t.id)} style={{ display: "block", width: "100%", padding: "6px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#0f172a", border: `1px solid ${t.c}33`, color: "#94a3b8", cursor: "pointer", marginBottom: 3, textAlign: "left" }}><span style={{ color: "#fff" }}>{t.n}</span></button>
              ))}
            </div>
          </div>

          {/* Main */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ borderRadius: 12, padding: "8px 12px", background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b", display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b" }}>Add:</span>
              {Object.entries(LTPLS).map(([k, t]) => { const i = INFO[k]; return (<button key={k} onClick={() => add(k)} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: `${i?.c || "#475569"}18`, color: i?.c || "#94a3b8", border: `1px solid ${i?.c || "#475569"}33`, cursor: "pointer" }}>+ {t.name}</button>); })}
              <button onClick={() => { setLayers([]); setOuts(new Map()); }} style={{ marginLeft: "auto", padding: "3px 8px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>Clear</button>
            </div>

            {/* Pipeline strip */}
            {layers.length > 0 && outs.size > 0 && (
              <div style={{ borderRadius: 12, padding: 10, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b", overflowX: "auto" }}>
                <p style={{ fontSize: 9, fontWeight: 700, color: "#475569", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🔬 Image Pipeline</p>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <div style={{ textAlign: "center", flexShrink: 0 }}>
                    {imgData.length === 3 ? <RGBCanvas channels={imgData} size={48} /> : <FeatureCanvas data={imgData[0]} size={48} />}
                    <p style={{ fontSize: 7, color: "#22c55e", fontWeight: 700, marginTop: 1 }}>Input</p>
                  </div>
                  {layers.map((L) => { const lo = outs.get(L.id), inf = INFO[L.type]; return (
                    <div key={L.id} style={{ display: "flex", alignItems: "center", gap: 6, flexShrink: 0 }}>
                      <span style={{ fontSize: 10, color: inf?.c }}>→</span>
                      <div style={{ textAlign: "center" }}>
                        {lo?.t === "3d" && lo.d?.length ? <OutputImagePreview channels={lo.d} size={48} /> :
                          lo?.t === "1d" ? <div style={{ width: 48, height: 48, display: "flex", alignItems: "end", gap: 1, background: "#1e293b", borderRadius: 4, padding: 2, border: "1px solid #334155" }}>
                            {lo.d.slice(0, 6).map((v, j) => <div key={j} style={{ flex: 1, background: "#3b82f6", height: `${Math.max(Math.abs(v) / (Math.max(...lo.d.map(Math.abs)) || 1) * 100, 8)}%`, borderRadius: "1px 1px 0 0", minHeight: 2 }} />)}</div> :
                            <div style={{ width: 48, height: 48, background: "#1e293b", borderRadius: 4, border: "1px solid #334155" }} />}
                        <p style={{ fontSize: 6, color: inf?.c, fontWeight: 700, marginTop: 1 }}>{L.name}</p>
                      </div>
                    </div>
                  ); })}
                </div>
              </div>
            )}

            {layers.length === 0 ? (
              <div style={{ borderRadius: 12, padding: "50px 20px", textAlign: "center", background: "rgba(15,23,42,0.3)", border: "1px dashed #334155" }}>
                <div style={{ fontSize: 36, marginBottom: 6 }}>🏗️</div>
                <h3 style={{ fontSize: 16, fontWeight: 800, color: "#fff" }}>Build Your CNN</h3>
                <p style={{ fontSize: 12, color: "#475569" }}>Select an <b style={{ color: "#f59e0b" }}>RGB color image</b>, add layers, and click ⛶ to see interactive step-by-step convolution with R/G/B channel splitting.</p>
              </div>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 6, background: "#22c55e08", border: "1px solid #22c55e22" }}>
                  <span style={{ width: 5, height: 5, borderRadius: 3, background: "#22c55e" }} /><span style={{ fontSize: 10, fontWeight: 700, color: "#22c55e" }}>INPUT</span>
                  <TensorShape shape={[imgData.length, imgData[0].length, imgData[0][0].length]} />
                  {imgData.length === 3 && <div style={{ display: "flex", gap: 2 }}>{CHN_COLORS.map((c, i) => <div key={i} style={{ width: 6, height: 6, borderRadius: 3, background: c }} />)}</div>}
                </div>
                {layers.map((L, i) => (
                  <div key={L.id}>
                    <div style={{ display: "flex", justifyContent: "center" }}><div style={{ width: 1.5, height: 8, background: "#334155" }} /></div>
                    <LCard layer={L} idx={i} prevOut={prevOut(i)} layerOut={outs.get(L.id)} onRemove={() => remove(L.id)} onExpand={() => setModal({ layer: L, idx: i })} />
                  </div>
                ))}
                {outs.size > 0 && (
                  <div style={{ display: "flex", alignItems: "center", gap: 6, padding: "4px 10px", borderRadius: 6, background: "#22c55e08", border: "1px solid #22c55e22" }}>
                    <span style={{ width: 5, height: 5, borderRadius: 3, background: "#22c55e" }} /><span style={{ fontSize: 10, fontWeight: 700, color: "#22c55e" }}>✓ COMPLETE · {layers.length} layers</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
      {modal && <FullModal layer={modal.layer} idx={modal.idx} prevOut={prevOut(modal.idx)} layerOut={outs.get(modal.layer.id)} onClose={() => setModal(null)} />}
    </div>
  );
}
