import { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════
   VAE MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
function reluFn(x) { return Math.max(0, x); }
function tanhFn(x) { return Math.tanh(x); }
function randMat(r, c) { const s = Math.sqrt(2 / c); return Array(r).fill(0).map(() => Array(c).fill(0).map(() => (Math.random() - 0.5) * s * 2)); }
function randVec(n) { return Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.1); }
function randomNormal() { return Math.sqrt(-2 * Math.log(Math.max(1e-10, Math.random()))) * Math.cos(2 * Math.PI * Math.random()); }

function denseLayer(input, weights, biases, activation) {
  const z = weights.map((row, i) => row.reduce((s, w, j) => s + w * (input[j] || 0), biases[i]));
  if (activation === "relu") return z.map(reluFn);
  if (activation === "sigmoid") return z.map(sigmoid);
  if (activation === "tanh") return z.map(tanhFn);
  return z;
}

function buildVAE(inputSize, hiddenSizes, latentDim) {
  // Encoder layers
  const encLayers = [];
  let prevSize = inputSize;
  for (const hs of hiddenSizes) {
    encLayers.push({ w: randMat(hs, prevSize), b: randVec(hs), act: "relu", inSize: prevSize, outSize: hs });
    prevSize = hs;
  }
  // Mean and logvar heads
  const muLayer = { w: randMat(latentDim, prevSize), b: randVec(latentDim), act: "linear", inSize: prevSize, outSize: latentDim };
  const lvLayer = { w: randMat(latentDim, prevSize), b: randVec(latentDim), act: "linear", inSize: prevSize, outSize: latentDim };

  // Decoder layers (mirror)
  const decLayers = [];
  prevSize = latentDim;
  for (const hs of [...hiddenSizes].reverse()) {
    decLayers.push({ w: randMat(hs, prevSize), b: randVec(hs), act: "relu", inSize: prevSize, outSize: hs });
    prevSize = hs;
  }
  decLayers.push({ w: randMat(inputSize, prevSize), b: randVec(inputSize), act: "sigmoid", inSize: prevSize, outSize: inputSize });

  return { encLayers, muLayer, lvLayer, decLayers, latentDim, inputSize };
}

function vaeForward(input, vae) {
  // Encode
  const encActs = [input];
  let h = input;
  for (const l of vae.encLayers) {
    h = denseLayer(h, l.w, l.b, l.act);
    encActs.push(h);
  }
  const mu = denseLayer(h, vae.muLayer.w, vae.muLayer.b, "linear");
  const logVar = denseLayer(h, vae.lvLayer.w, vae.lvLayer.b, "linear");

  // Reparameterize
  const std = logVar.map(x => Math.exp(0.5 * x));
  const epsilon = std.map(() => randomNormal());
  const z = mu.map((m, i) => m + std[i] * epsilon[i]);

  // Decode
  const decActs = [z];
  let d = z;
  for (const l of vae.decLayers) {
    d = denseLayer(d, l.w, l.b, l.act);
    decActs.push(d);
  }

  // Losses
  const reconLoss = input.reduce((s, v, i) => s + (v - d[i]) ** 2, 0) / input.length;
  const klDiv = -0.5 * mu.reduce((s, m, i) => s + 1 + logVar[i] - m ** 2 - Math.exp(logVar[i]), 0) / mu.length;

  return { mu, logVar, std, epsilon, z, reconstruction: d, encActs, decActs, reconLoss, klDiv, totalLoss: reconLoss + klDiv };
}

function decodeFromZ(z, vae) {
  let d = z;
  for (const l of vae.decLayers) d = denseLayer(d, l.w, l.b, l.act);
  return d;
}

/* ═══════════════════════════════════════════════════════════
   INPUT DATA — 8×8 patterns flattened to 64-dim vectors
   ═══════════════════════════════════════════════════════════ */
function mkPattern(type, sz = 8) {
  const m = Array(sz).fill(0).map(() => Array(sz).fill(0));
  if (type === "vert") { for (let i = 0; i < sz; i++) { m[i][3] = 1; m[i][4] = 1; } }
  else if (type === "horiz") { for (let j = 0; j < sz; j++) { m[3][j] = 1; m[4][j] = 1; } }
  else if (type === "cross") { for (let i = 0; i < sz; i++) { m[i][sz >> 1] = 1; m[sz >> 1][i] = 1; } }
  else if (type === "diag") { for (let i = 0; i < sz; i++) m[i][i] = 1; }
  else if (type === "circle") { const c = sz / 2 - .5, r = sz / 3; for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) if (Math.abs(Math.sqrt((i - c) ** 2 + (j - c) ** 2) - r) < 1.2) m[i][j] = 1; }
  else if (type === "box") { for (let i = 1; i < sz - 1; i++) { m[1][i] = 1; m[sz - 2][i] = 1; m[i][1] = 1; m[i][sz - 2] = 1; } }
  else if (type === "dot") { m[3][3] = 1; m[3][4] = 1; m[4][3] = 1; m[4][4] = 1; }
  else if (type === "corner") { for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) m[i][j] = 1; }
  else if (type === "checker") { for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) m[i][j] = (i + j) % 2 ? 1 : 0; }
  else if (type === "random") { for (let i = 0; i < sz; i++) for (let j = 0; j < sz; j++) m[i][j] = Math.random() > 0.55 ? 1 : 0; }
  return m;
}
function flatten8x8(m) { return m.flat(); }
function unflatten8x8(v) { const m = []; for (let i = 0; i < 8; i++) m.push(v.slice(i * 8, (i + 1) * 8)); return m; }

const PATTERNS = [
  { id: "vert", name: "Vertical Line" }, { id: "horiz", name: "Horizontal Line" },
  { id: "cross", name: "Cross (+)" }, { id: "diag", name: "Diagonal" },
  { id: "circle", name: "Circle" }, { id: "box", name: "Box" },
  { id: "dot", name: "Dot" }, { id: "corner", name: "Corner" },
  { id: "checker", name: "Checkerboard" }, { id: "random", name: "Random Noise" },
];

const S = { mono: "'IBM Plex Mono', monospace", sans: "'IBM Plex Sans', system-ui, sans-serif" };

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS
   ═══════════════════════════════════════════════════════════ */
function GridImage({ data, size = 100, label, border }) {
  // data is 8x8 matrix
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !data?.length) return;
    const rows = data.length, cols = data[0].length; c.width = cols; c.height = rows;
    const ctx = c.getContext("2d");
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = Math.max(0, Math.min(1, data[i][j]));
      const g = Math.round(v * 255);
      ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(j, i, 1, 1);
    }
  }, [data]);
  return (
    <div style={{ textAlign: "center" }}>
      {label && <p style={{ fontSize: 9, fontWeight: 700, color: "#94a3b8", marginBottom: 3, fontFamily: S.mono }}>{label}</p>}
      <canvas ref={ref} style={{ width: size, height: size, imageRendering: "pixelated", borderRadius: 6, border: border || "1px solid #334155" }} />
    </div>
  );
}

function VectorBar({ values, label, color = "#3b82f6", maxShow = 16, height = 50 }) {
  if (!values?.length) return null;
  const mx = Math.max(...values.map(Math.abs), 0.01);
  return (
    <div>
      {label && <p style={{ fontSize: 9, fontWeight: 700, color, marginBottom: 3, fontFamily: S.mono }}>{label} [{values.length}]</p>}
      <div style={{ display: "flex", gap: 1, height, alignItems: "center" }}>
        {values.slice(0, maxShow).map((v, i) => {
          const h = Math.abs(v) / mx * (height - 4);
          return <div key={i} style={{ width: 10, display: "flex", flexDirection: "column", justifyContent: v >= 0 ? "flex-end" : "flex-start", height: "100%" }}>
            <div style={{ width: 10, height: Math.max(h, 2), background: v >= 0 ? color : "#ef4444", borderRadius: 2, transition: "height 0.3s" }} />
          </div>;
        })}
        {values.length > maxShow && <span style={{ fontSize: 7, color: "#475569" }}>+{values.length - maxShow}</span>}
      </div>
    </div>
  );
}

function VectorDisplay({ values, label, color = "#3b82f6", maxShow = 8 }) {
  if (!values?.length) return null;
  return (
    <div>
      {label && <p style={{ fontSize: 9, fontWeight: 700, color, marginBottom: 3, fontFamily: S.mono }}>{label} [{values.length}]</p>}
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {values.slice(0, maxShow).map((v, i) => (
          <div key={i} style={{ minWidth: 36, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, fontWeight: 600, fontFamily: S.mono, borderRadius: 3, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 160 + 40).toString(16).padStart(2, '0')}`, color: "#fff", border: `1px solid ${color}44`, padding: "0 3px" }}>
            {v.toFixed(2)}
          </div>
        ))}
        {values.length > maxShow && <span style={{ fontSize: 8, color: "#475569", alignSelf: "center" }}>+{values.length - maxShow}</span>}
      </div>
    </div>
  );
}

function TensorShape({ shape, color = "#60a5fa" }) {
  return <span style={{ fontSize: 10, fontFamily: S.mono, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: `${color}15`, border: `1px solid ${color}33`, color }}>[{shape.join("×")}]</span>;
}

/* ═══════════════════════════════════════════════════════════
   2D LATENT SPACE EXPLORER
   ═══════════════════════════════════════════════════════════ */
function LatentSpaceExplorer({ vae, allResults, onSelectZ }) {
  const ref = useRef(null);
  const [hoverZ, setHoverZ] = useState(null);
  const [hoverImg, setHoverImg] = useState(null);
  const size = 300;

  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = size; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);

    // Background — decode a grid of z values
    if (vae.latentDim === 2) {
      const res = 20;
      for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
        const z1 = (i / res) * 6 - 3, z2 = (j / res) * 6 - 3;
        const dec = decodeFromZ([z1, z2], vae);
        const avg = dec.reduce((s, v) => s + v, 0) / dec.length;
        const g = Math.round(avg * 200 + 30);
        ctx.fillStyle = `rgb(${g},${g},${g})`; ctx.fillRect(i * (size / res), j * (size / res), size / res + 1, size / res + 1);
      }
    }

    // Axes
    ctx.strokeStyle = "#47569955"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2); ctx.stroke();

    // Plot encoded points
    allResults.forEach(r => {
      if (!r.mu || r.mu.length < 2) return;
      const px = (r.mu[0] + 3) / 6 * size, py = (r.mu[1] + 3) / 6 * size;
      ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = r.color || "#3b82f6"; ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
      // Variance ellipse
      if (r.std) {
        ctx.beginPath();
        ctx.ellipse(px, py, Math.abs(r.std[0]) * size / 6 * 0.5, Math.abs(r.std[1]) * size / 6 * 0.5, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `${r.color || "#3b82f6"}44`; ctx.lineWidth = 1; ctx.stroke();
      }
    });

    // Labels
    ctx.fillStyle = "#64748b"; ctx.font = "9px 'IBM Plex Mono'"; ctx.textAlign = "center";
    ctx.fillText("z₁", size / 2, size - 4);
    ctx.save(); ctx.translate(10, size / 2); ctx.rotate(-Math.PI / 2); ctx.fillText("z₂", 0, 0); ctx.restore();
  }, [vae, allResults, size]);

  const handleClick = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width, y = (e.clientY - rect.top) / rect.height;
    const z1 = x * 6 - 3, z2 = y * 6 - 3;
    const z = vae.latentDim === 2 ? [z1, z2] : Array(vae.latentDim).fill(0).map((_, i) => i === 0 ? z1 : i === 1 ? z2 : 0);
    const dec = decodeFromZ(z, vae);
    setHoverZ([z1.toFixed(2), z2.toFixed(2)]);
    setHoverImg(unflatten8x8(dec));
    if (onSelectZ) onSelectZ(z);
  };

  const handleMove = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width, y = (e.clientY - rect.top) / rect.height;
    const z1 = x * 6 - 3, z2 = y * 6 - 3;
    if (vae.latentDim >= 2) {
      const z = [z1, z2, ...Array(Math.max(0, vae.latentDim - 2)).fill(0)];
      const dec = decodeFromZ(z, vae);
      setHoverZ([z1.toFixed(2), z2.toFixed(2)]);
      setHoverImg(unflatten8x8(dec));
    }
  };

  return (
    <div>
      <p style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 6, fontFamily: S.mono }}>2D Latent Space — click to generate</p>
      <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
        <canvas ref={ref} onClick={handleClick} onMouseMove={handleMove} style={{ width: size, height: size, borderRadius: 10, border: "1px solid #334155", cursor: "crosshair" }} />
        <div>
          {hoverZ && <p style={{ fontSize: 10, fontFamily: S.mono, color: "#f59e0b", marginBottom: 4 }}>z = [{hoverZ.join(", ")}]</p>}
          {hoverImg && <GridImage data={hoverImg} size={90} label="Decoded output" border="2px solid #f59e0b" />}
          <p style={{ fontSize: 9, color: "#475569", marginTop: 6, maxWidth: 120 }}>Move your mouse across the latent space to see what the decoder generates at each point.</p>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   INTERPOLATION PANEL
   ═══════════════════════════════════════════════════════════ */
function InterpolationPanel({ vae, resultA, resultB }) {
  const [t, setT] = useState(0.5);
  if (!resultA?.z || !resultB?.z) return <p style={{ fontSize: 11, color: "#475569" }}>Select two different patterns to interpolate between them.</p>;
  const steps = 7;
  const interps = Array.from({ length: steps }).map((_, i) => {
    const alpha = i / (steps - 1);
    const z = resultA.z.map((za, j) => za * (1 - alpha) + (resultB.z[j] || 0) * alpha);
    return { alpha, z, img: unflatten8x8(decodeFromZ(z, vae)) };
  });
  const curZ = resultA.z.map((za, j) => za * (1 - t) + (resultB.z[j] || 0) * t);
  const curImg = unflatten8x8(decodeFromZ(curZ, vae));

  return (
    <div>
      <p style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 8, fontFamily: S.mono }}>Latent Space Interpolation</p>
      <div style={{ display: "flex", gap: 6, alignItems: "flex-end", marginBottom: 10, overflowX: "auto", paddingBottom: 4 }}>
        {interps.map((ip, i) => (
          <div key={i} style={{ textAlign: "center", flexShrink: 0 }}>
            <GridImage data={ip.img} size={52} border={i === 0 || i === steps - 1 ? "2px solid #a855f7" : "1px solid #334155"} />
            <p style={{ fontSize: 7, color: "#475569", marginTop: 2 }}>α={ip.alpha.toFixed(1)}</p>
          </div>
        ))}
      </div>
      <label style={{ fontSize: 10, color: "#64748b", display: "block" }}>Interpolation: α = {t.toFixed(2)}
        <input type="range" min={0} max={1} step={0.01} value={t} onChange={e => setT(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
      </label>
      <div style={{ display: "flex", gap: 12, marginTop: 8, alignItems: "center" }}>
        <GridImage data={curImg} size={72} label={`α=${t.toFixed(2)}`} border="2px solid #f59e0b" />
        <VectorDisplay values={curZ} label="Interpolated z" color="#f59e0b" maxShow={6} />
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   STEP-BY-STEP VAE FORWARD PASS MODULE
   ═══════════════════════════════════════════════════════════ */
function VAEStepModule({ input, vae, result }) {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);

  const inputImg = useMemo(() => unflatten8x8(input), [input]);
  const reconImg = useMemo(() => unflatten8x8(result.reconstruction), [result]);

  const STEPS = [
    { t: "📥 Input Image", d: `An 8×8 image is flattened into a ${input.length}-dimensional vector.`, c: "#22c55e" },
    { t: "🔻 Encoder Network", d: `${vae.encLayers.length} dense layer${vae.encLayers.length > 1 ? "s" : ""} compress the input into a compact representation.`, c: "#3b82f6" },
    { t: "📊 μ (Mean) and log(σ²) (Log-Variance)", d: `Two separate heads produce the parameters of the latent distribution. Each has ${vae.latentDim} values.`, c: "#a855f7" },
    { t: "🎲 Reparameterization Trick", d: `z = μ + σ × ε, where ε ~ N(0,1). This allows gradients to flow through the sampling step.`, c: "#f59e0b" },
    { t: "🔺 Decoder Network", d: `${vae.decLayers.length} dense layer${vae.decLayers.length > 1 ? "s" : ""} reconstruct the image from the latent code z.`, c: "#ec4899" },
    { t: "📤 Reconstruction + Loss", d: "Original vs reconstructed image. Loss = Reconstruction Loss + KL Divergence.", c: "#ef4444" },
  ];
  const totalSteps = STEPS.length;

  useEffect(() => {
    if (auto && step < totalSteps - 1) { const t = setTimeout(() => setStep(p => p + 1), 3000); return () => clearTimeout(t); }
    else setAuto(false);
  }, [auto, step, totalSteps]);

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ display: "flex", gap: 2 }}>{STEPS.map((_, i) => <button key={i} onClick={() => { setStep(i); setAuto(false); }} style={{ flex: "0 0 auto", width: 32, height: 6, borderRadius: 3, background: i <= step ? STEPS[i].c : "#1e293b", border: "none", cursor: "pointer" }} />)}</div>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => { setStep(0); setAuto(false); }} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
          <button onClick={() => setAuto(!auto)} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: auto ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{auto ? "⏸" : "▶ Auto"}</button>
        </div>
      </div>

      {/* Title */}
      <div style={{ padding: "8px 12px", borderRadius: 8, background: `${STEPS[step].c}11`, border: `1px solid ${STEPS[step].c}33`, marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 24, height: 24, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, background: `${STEPS[step].c}22`, color: STEPS[step].c, flexShrink: 0 }}>{step + 1}</div>
        <div><h5 style={{ fontSize: 12, fontWeight: 700, color: "#fff", margin: 0 }}>{STEPS[step].t}</h5><p style={{ fontSize: 10, color: "#94a3b8", margin: 0 }}>{STEPS[step].d}</p></div>
      </div>

      {/* Visual */}
      <div style={{ padding: 14, background: "rgba(15,23,42,0.5)", borderRadius: 10, border: "1px solid #1e293b", minHeight: 200 }}>
        {step === 0 && (
          <div style={{ display: "flex", gap: 20, alignItems: "center", flexWrap: "wrap" }}>
            <GridImage data={inputImg} size={120} label="8×8 Input" border="2px solid #22c55e" />
            <div style={{ fontSize: 20, color: "#22c55e", fontWeight: 800 }}>→ flatten →</div>
            <div>
              <VectorBar values={input} label="Flattened vector" color="#22c55e" maxShow={32} height={60} />
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Shape: <TensorShape shape={[input.length]} color="#22c55e" /></p>
            </div>
          </div>
        )}

        {step === 1 && (
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              <TensorShape shape={[input.length]} color="#22c55e" />
              {vae.encLayers.map((l, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 14, color: "#3b82f6" }}>→</span>
                  <div style={{ padding: "4px 10px", borderRadius: 6, background: "#3b82f622", border: "1px solid #3b82f644" }}>
                    <span style={{ fontSize: 10, fontFamily: S.mono, color: "#3b82f6", fontWeight: 700 }}>Dense({l.outSize}, {l.act})</span>
                  </div>
                </div>
              ))}
              <span style={{ fontSize: 14, color: "#3b82f6" }}>→</span>
              <TensorShape shape={[vae.encLayers[vae.encLayers.length - 1]?.outSize || "?"]} color="#3b82f6" />
            </div>
            {result.encActs.slice(1).map((act, i) => (
              <div key={i} style={{ marginBottom: 6 }}>
                <VectorBar values={act} label={`Layer ${i + 1} output`} color="#3b82f6" maxShow={20} height={35} />
              </div>
            ))}
            <p style={{ fontSize: 10, color: "#94a3b8", marginTop: 8 }}>The encoder compresses {input.length} dims → {vae.encLayers[vae.encLayers.length - 1]?.outSize} dims, extracting key features.</p>
          </div>
        )}

        {step === 2 && (
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
            <div>
              <VectorDisplay values={result.mu} label="μ (mean)" color="#a855f7" />
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Where in latent space this input "should" map to</p>
            </div>
            <div>
              <VectorDisplay values={result.logVar} label="log(σ²) (log-variance)" color="#ec4899" />
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>How "spread out" the encoding should be</p>
            </div>
            <div>
              <VectorDisplay values={result.std} label="σ (std dev)" color="#f59e0b" />
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>σ = exp(0.5 × log(σ²))</p>
            </div>
          </div>
        )}

        {step === 3 && (
          <div>
            <p style={{ fontSize: 13, fontFamily: S.mono, fontWeight: 700, color: "#f59e0b", marginBottom: 12 }}>z = μ + σ × ε</p>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              {result.mu.map((m, i) => (
                <div key={i} style={{ padding: "6px 8px", borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b", textAlign: "center" }}>
                  <p style={{ fontSize: 8, color: "#a855f7", fontFamily: S.mono }}>μ={m.toFixed(2)}</p>
                  <p style={{ fontSize: 8, color: "#f59e0b", fontFamily: S.mono }}>σ={result.std[i].toFixed(2)}</p>
                  <p style={{ fontSize: 8, color: "#64748b", fontFamily: S.mono }}>ε={result.epsilon[i].toFixed(2)}</p>
                  <div style={{ borderTop: "1px solid #334155", marginTop: 3, paddingTop: 3 }}>
                    <p style={{ fontSize: 9, color: "#22c55e", fontFamily: S.mono, fontWeight: 700 }}>z={result.z[i].toFixed(2)}</p>
                  </div>
                </div>
              ))}
            </div>
            <div style={{ padding: "8px 14px", borderRadius: 8, background: "#78350f33", border: "1px solid #f59e0b55" }}>
              <p style={{ fontSize: 10, color: "#f59e0b" }}><b>Why the trick?</b> We can't backpropagate through random sampling. By making z = μ + σ×ε, the randomness (ε) is external — gradients flow through μ and σ.</p>
            </div>
          </div>
        )}

        {step === 4 && (
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              <TensorShape shape={[vae.latentDim]} color="#f59e0b" />
              {vae.decLayers.map((l, i) => (
                <div key={i} style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontSize: 14, color: "#ec4899" }}>→</span>
                  <div style={{ padding: "4px 10px", borderRadius: 6, background: "#ec489922", border: "1px solid #ec489944" }}>
                    <span style={{ fontSize: 10, fontFamily: S.mono, color: "#ec4899", fontWeight: 700 }}>Dense({l.outSize}, {l.act})</span>
                  </div>
                </div>
              ))}
              <span style={{ fontSize: 14, color: "#ec4899" }}>→</span>
              <TensorShape shape={[input.length]} color="#ec4899" />
            </div>
            {result.decActs.map((act, i) => (
              <div key={i} style={{ marginBottom: 6 }}>
                <VectorBar values={act} label={i === 0 ? "Latent z" : `Decoder L${i}`} color="#ec4899" maxShow={20} height={30} />
              </div>
            ))}
            <p style={{ fontSize: 10, color: "#94a3b8", marginTop: 8 }}>The decoder expands {vae.latentDim} dims → {input.length} dims, reconstructing the image.</p>
          </div>
        )}

        {step === 5 && (
          <div>
            <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap", marginBottom: 16 }}>
              <GridImage data={inputImg} size={100} label="Original" border="2px solid #22c55e" />
              <div style={{ fontSize: 24, color: "#64748b" }}>vs</div>
              <GridImage data={reconImg} size={100} label="Reconstructed" border="2px solid #ec4899" />
            </div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
              <div style={{ padding: "8px 14px", borderRadius: 8, background: "#3b82f611", border: "1px solid #3b82f633" }}>
                <p style={{ fontSize: 9, color: "#64748b" }}>Reconstruction Loss (MSE)</p>
                <p style={{ fontSize: 16, fontFamily: S.mono, fontWeight: 800, color: "#3b82f6" }}>{result.reconLoss.toFixed(4)}</p>
              </div>
              <div style={{ padding: "8px 14px", borderRadius: 8, background: "#a855f711", border: "1px solid #a855f733" }}>
                <p style={{ fontSize: 9, color: "#64748b" }}>KL Divergence</p>
                <p style={{ fontSize: 16, fontFamily: S.mono, fontWeight: 800, color: "#a855f7" }}>{result.klDiv.toFixed(4)}</p>
              </div>
              <div style={{ padding: "8px 14px", borderRadius: 8, background: "#ef444411", border: "1px solid #ef444433" }}>
                <p style={{ fontSize: 9, color: "#64748b" }}>Total Loss</p>
                <p style={{ fontSize: 16, fontFamily: S.mono, fontWeight: 800, color: "#ef4444" }}>{result.totalLoss.toFixed(4)}</p>
              </div>
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8", marginTop: 10 }}><b style={{ color: "#3b82f6" }}>Recon Loss</b> = how well the image is reproduced. <b style={{ color: "#a855f7" }}>KL Div</b> = how close the latent distribution is to N(0,1).</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
        <button onClick={() => { setStep(Math.max(0, step - 1)); setAuto(false); }} disabled={step === 0} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === 0 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === 0 ? "not-allowed" : "pointer" }}>← Prev</button>
        <span style={{ fontSize: 10, color: "#475569" }}>{step + 1}/{totalSteps}</span>
        <button onClick={() => { setStep(Math.min(totalSteps - 1, step + 1)); setAuto(false); }} disabled={step === totalSteps - 1} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === totalSteps - 1 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === totalSteps - 1 ? "not-allowed" : "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   FULLSCREEN MODAL
   ═══════════════════════════════════════════════════════════ */
function FullModal({ input, vae, result, onClose }) {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(2,6,23,0.95)", backdropFilter: "blur(16px)", overflow: "auto" }}>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, background: "#a855f722" }}>✨</div>
            <div><h2 style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>VAE Forward Pass — Step by Step</h2><p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>Encode → Reparameterize → Decode</p></div>
          </div>
          <button onClick={onClose} style={{ width: 36, height: 36, borderRadius: 8, fontSize: 18, color: "#64748b", background: "#1e293b", border: "1px solid #334155", cursor: "pointer" }}>×</button>
        </div>
        <div style={{ background: "rgba(15,23,42,0.6)", borderRadius: 14, border: "1px solid #1e293b", padding: 20 }}>
          <VAEStepModule input={input} vae={vae} result={result} />
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ARCHITECTURE PRESETS
   ═══════════════════════════════════════════════════════════ */
const ARCH_PRESETS = {
  small: { name: "Small (64→16→2→16→64)", hidden: [16], latent: 2 },
  medium: { name: "Medium (64→32→16→2→16→32→64)", hidden: [32, 16], latent: 2 },
  deep: { name: "Deep (64→32→16→8→2→...→64)", hidden: [32, 16, 8], latent: 2 },
  wide: { name: "Wide latent (64→32→8→32→64)", hidden: [32], latent: 8 },
  tiny: { name: "Tiny (64→8→1→8→64)", hidden: [8], latent: 1 },
};

/* ═══════════════════════════════════════════════════════════
   MAIN VAE LAB
   ═══════════════════════════════════════════════════════════ */
export default function VAELab() {
  const [patternId, setPatternId] = useState("cross");
  const [patternIdB, setPatternIdB] = useState("circle");
  const [hiddenSizes, setHiddenSizes] = useState([32, 16]);
  const [latentDim, setLatentDim] = useState(2);
  const [modal, setModal] = useState(false);
  const [tab, setTab] = useState("overview"); // overview, latent, interpolate

  const inputSize = 64; // 8×8
  const vae = useMemo(() => buildVAE(inputSize, hiddenSizes, latentDim), [inputSize, hiddenSizes, latentDim]);

  const input = useMemo(() => flatten8x8(mkPattern(patternId)), [patternId]);
  const inputImg = useMemo(() => mkPattern(patternId), [patternId]);
  const result = useMemo(() => vaeForward(input, vae), [input, vae]);
  const reconImg = useMemo(() => unflatten8x8(result.reconstruction), [result]);

  const inputB = useMemo(() => flatten8x8(mkPattern(patternIdB)), [patternIdB]);
  const resultB = useMemo(() => vaeForward(inputB, vae), [inputB, vae]);

  // All patterns encoded for latent space plot
  const PAT_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#f59e0b", "#a855f7", "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#64748b"];
  const allResults = useMemo(() => PATTERNS.map((p, i) => {
    const inp = flatten8x8(mkPattern(p.id));
    const res = vaeForward(inp, vae);
    return { ...res, name: p.name, id: p.id, color: PAT_COLORS[i % PAT_COLORS.length] };
  }), [vae]);

  const loadPreset = (key) => {
    const p = ARCH_PRESETS[key];
    setHiddenSizes([...p.hidden]);
    setLatentDim(p.latent);
  };

  const totalParams = useMemo(() => {
    let p = 0;
    vae.encLayers.forEach(l => p += l.outSize * l.inSize + l.outSize);
    p += vae.muLayer.outSize * vae.muLayer.inSize + vae.muLayer.outSize;
    p += vae.lvLayer.outSize * vae.lvLayer.inSize + vae.lvLayer.outSize;
    vae.decLayers.forEach(l => p += l.outSize * l.inSize + l.outSize);
    return p;
  }, [vae]);

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #100820 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      {/* Lab title + tabs */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#a855f7,#ec4899)" }}>✨</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>VAE Lab</h1>
        <span style={{ fontSize: 10 }}>Variational Autoencoder Simulator</span>
        <div style={{ marginLeft: "auto", display: "flex", gap: 4 }}>
          {["overview", "latent", "interpolate"].map(t => (
            <button key={t} onClick={() => setTab(t)} style={{ padding: "4px 10px", borderRadius: 6, fontSize: 10, fontWeight: 700, background: tab === t ? "#a855f7" : "transparent", color: tab === t ? "#fff" : "#64748b", border: "none", cursor: "pointer" }}>
              {t === "overview" ? "🏗 Overview" : t === "latent" ? "🗺 Latent Space" : "🔀 Interpolation"}
            </button>
          ))}
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 14 }}>
          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Input pattern */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>📷 Input Pattern</h3>
              <select value={patternId} onChange={e => setPatternId(e.target.value)} style={{ width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}>
                {PATTERNS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <div style={{ display: "flex", gap: 12, marginTop: 8, justifyContent: "center" }}>
                <GridImage data={inputImg} size={90} label="Original" border="2px solid #22c55e" />
                <GridImage data={reconImg} size={90} label="Reconstructed" border="2px solid #ec4899" />
              </div>
              <p style={{ fontSize: 9, color: "#475569", textAlign: "center", marginTop: 4 }}>
                Loss: <span style={{ color: "#ef4444", fontWeight: 700 }}>{result.totalLoss.toFixed(3)}</span>
              </p>
            </div>

            {/* Architecture */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>🏗 Architecture</h3>
              {Object.entries(ARCH_PRESETS).map(([k, v]) => (
                <button key={k} onClick={() => loadPreset(k)} style={{ display: "block", width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600, background: "#0f172a", border: "1px solid #1e293b", color: "#94a3b8", cursor: "pointer", marginBottom: 3, textAlign: "left" }}>
                  <span style={{ color: "#fff" }}>{v.name}</span>
                </button>
              ))}
              <div style={{ marginTop: 8 }}>
                <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 4 }}>Latent Dimension: <b style={{ color: "#fff" }}>{latentDim}</b>
                  <input type="range" min={1} max={8} value={latentDim} onChange={e => setLatentDim(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
                </label>
              </div>
              <div style={{ marginTop: 6, padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 9, color: "#64748b" }}>Encoder: {inputSize} → {hiddenSizes.join(" → ")} → {latentDim}</p>
                <p style={{ fontSize: 9, color: "#64748b" }}>Decoder: {latentDim} → {[...hiddenSizes].reverse().join(" → ")} → {inputSize}</p>
                <p style={{ fontSize: 9, color: "#64748b" }}>Params: <b style={{ color: "#fff" }}>{totalParams.toLocaleString()}</b></p>
              </div>
            </div>

            {/* Step-by-step button */}
            <button onClick={() => setModal(true)} style={{ width: "100%", padding: "10px 0", borderRadius: 10, fontSize: 12, fontWeight: 700, background: "#a855f718", color: "#a855f7", border: "1px solid #a855f733", cursor: "pointer" }}>
              ⛶ Step-by-Step VAE Forward Pass
            </button>
          </div>

          {/* Main area */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Architecture pipeline */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 9, fontWeight: 700, color: "#475569", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🔬 VAE Pipeline</p>
              <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
                <div style={{ textAlign: "center" }}><GridImage data={inputImg} size={48} /><p style={{ fontSize: 7, color: "#22c55e", fontWeight: 700 }}>Input</p></div>
                <span style={{ color: "#3b82f6" }}>→</span>
                <div style={{ padding: "4px 8px", borderRadius: 6, background: "#3b82f622", border: "1px solid #3b82f644" }}><span style={{ fontSize: 9, color: "#3b82f6", fontFamily: S.mono, fontWeight: 700 }}>Encoder</span></div>
                <span style={{ color: "#a855f7" }}>→</span>
                <div style={{ padding: "4px 8px", borderRadius: 6, background: "#a855f722", border: "1px solid #a855f744" }}><span style={{ fontSize: 9, color: "#a855f7", fontFamily: S.mono, fontWeight: 700 }}>μ, σ²</span></div>
                <span style={{ color: "#f59e0b" }}>→</span>
                <div style={{ padding: "4px 8px", borderRadius: 6, background: "#f59e0b22", border: "1px solid #f59e0b44" }}><span style={{ fontSize: 9, color: "#f59e0b", fontFamily: S.mono, fontWeight: 700 }}>z = μ+σε</span></div>
                <span style={{ color: "#ec4899" }}>→</span>
                <div style={{ padding: "4px 8px", borderRadius: 6, background: "#ec489922", border: "1px solid #ec489944" }}><span style={{ fontSize: 9, color: "#ec4899", fontFamily: S.mono, fontWeight: 700 }}>Decoder</span></div>
                <span style={{ color: "#ec4899" }}>→</span>
                <div style={{ textAlign: "center" }}><GridImage data={reconImg} size={48} /><p style={{ fontSize: 7, color: "#ec4899", fontWeight: 700 }}>Recon</p></div>
              </div>
            </div>

            {tab === "overview" && (
              <>
                {/* Encoder / Decoder details */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #3b82f633" }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: "#3b82f6", marginBottom: 8, fontFamily: S.mono }}>🔻 ENCODER</p>
                    {result.encActs.map((act, i) => <VectorBar key={i} values={act} label={i === 0 ? `Input [${act.length}]` : `Hidden ${i} [${act.length}]`} color="#3b82f6" maxShow={24} height={28} />)}
                    <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                      <VectorDisplay values={result.mu} label="μ" color="#a855f7" maxShow={latentDim} />
                      <VectorDisplay values={result.logVar} label="log(σ²)" color="#ec4899" maxShow={latentDim} />
                    </div>
                  </div>
                  <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #ec489933" }}>
                    <p style={{ fontSize: 10, fontWeight: 700, color: "#ec4899", marginBottom: 8, fontFamily: S.mono }}>🔺 DECODER</p>
                    {result.decActs.map((act, i) => <VectorBar key={i} values={act} label={i === 0 ? `z [${act.length}]` : i === result.decActs.length - 1 ? `Output [${act.length}]` : `Hidden ${i} [${act.length}]`} color="#ec4899" maxShow={24} height={28} />)}
                  </div>
                </div>

                {/* Reparameterization */}
                <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #f59e0b33" }}>
                  <p style={{ fontSize: 10, fontWeight: 700, color: "#f59e0b", marginBottom: 8, fontFamily: S.mono }}>🎲 REPARAMETERIZATION: z = μ + σ × ε</p>
                  <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                    {result.z.map((zv, i) => (
                      <div key={i} style={{ padding: "6px 10px", borderRadius: 8, background: "#0f172a", border: "1px solid #1e293b", textAlign: "center" }}>
                        <span style={{ fontSize: 8, color: "#a855f7", fontFamily: S.mono }}>μ={result.mu[i].toFixed(2)}</span>
                        <span style={{ fontSize: 8, color: "#64748b" }}> + </span>
                        <span style={{ fontSize: 8, color: "#f59e0b", fontFamily: S.mono }}>σ={result.std[i].toFixed(2)}</span>
                        <span style={{ fontSize: 8, color: "#64748b" }}> × </span>
                        <span style={{ fontSize: 8, color: "#94a3b8", fontFamily: S.mono }}>ε={result.epsilon[i].toFixed(2)}</span>
                        <p style={{ fontSize: 10, color: "#22c55e", fontWeight: 700, fontFamily: S.mono, marginTop: 2 }}>z={zv.toFixed(3)}</p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Losses */}
                <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #ef444433" }}>
                  <p style={{ fontSize: 10, fontWeight: 700, color: "#ef4444", marginBottom: 8, fontFamily: S.mono }}>📊 LOSSES</p>
                  <div style={{ display: "flex", gap: 12 }}>
                    <div style={{ flex: 1, padding: "10px 14px", borderRadius: 8, background: "#3b82f611", border: "1px solid #3b82f633", textAlign: "center" }}>
                      <p style={{ fontSize: 9, color: "#64748b" }}>Reconstruction (MSE)</p>
                      <p style={{ fontSize: 20, fontFamily: S.mono, fontWeight: 800, color: "#3b82f6" }}>{result.reconLoss.toFixed(4)}</p>
                      <p style={{ fontSize: 8, color: "#475569" }}>How well was the input reproduced?</p>
                    </div>
                    <div style={{ flex: 1, padding: "10px 14px", borderRadius: 8, background: "#a855f711", border: "1px solid #a855f733", textAlign: "center" }}>
                      <p style={{ fontSize: 9, color: "#64748b" }}>KL Divergence</p>
                      <p style={{ fontSize: 20, fontFamily: S.mono, fontWeight: 800, color: "#a855f7" }}>{result.klDiv.toFixed(4)}</p>
                      <p style={{ fontSize: 8, color: "#475569" }}>How close is q(z|x) to N(0,1)?</p>
                    </div>
                    <div style={{ flex: 1, padding: "10px 14px", borderRadius: 8, background: "#ef444411", border: "1px solid #ef444433", textAlign: "center" }}>
                      <p style={{ fontSize: 9, color: "#64748b" }}>Total = Recon + KL</p>
                      <p style={{ fontSize: 20, fontFamily: S.mono, fontWeight: 800, color: "#ef4444" }}>{result.totalLoss.toFixed(4)}</p>
                    </div>
                  </div>
                </div>
              </>
            )}

            {tab === "latent" && (
              <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
                {latentDim >= 2 ? (
                  <LatentSpaceExplorer vae={vae} allResults={allResults} />
                ) : (
                  <div>
                    <p style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>Latent dimension = 1. Showing all patterns on a 1D number line:</p>
                    <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                      {allResults.map((r, i) => (
                        <div key={i} style={{ textAlign: "center" }}>
                          <GridImage data={mkPattern(r.id)} size={40} />
                          <p style={{ fontSize: 8, fontFamily: S.mono, color: r.color }}>{r.mu[0]?.toFixed(2)}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
                {/* Encoded positions legend */}
                <div style={{ marginTop: 12, display: "flex", gap: 6, flexWrap: "wrap" }}>
                  {allResults.map((r, i) => (
                    <span key={i} style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: `${r.color}22`, color: r.color, fontWeight: 600 }}>● {r.name}</span>
                  ))}
                </div>
              </div>
            )}

            {tab === "interpolate" && (
              <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
                <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                  <label style={{ fontSize: 10, color: "#64748b" }}>Pattern A:
                    <select value={patternId} onChange={e => setPatternId(e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 4, fontSize: 10, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155" }}>
                      {PATTERNS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                    </select>
                  </label>
                  <label style={{ fontSize: 10, color: "#64748b" }}>Pattern B:
                    <select value={patternIdB} onChange={e => setPatternIdB(e.target.value)} style={{ marginLeft: 4, padding: "3px 6px", borderRadius: 4, fontSize: 10, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155" }}>
                      {PATTERNS.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}
                    </select>
                  </label>
                </div>
                <InterpolationPanel vae={vae} resultA={result} resultB={resultB} />
              </div>
            )}
          </div>
        </div>
      </div>

      {modal && <FullModal input={input} vae={vae} result={result} onClose={() => setModal(false)} />}
    </div>
  );
}
