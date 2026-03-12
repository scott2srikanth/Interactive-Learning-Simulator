import { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════
   ANN MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
function relu(x) { return Math.max(0, x); }
function tanhFn(x) { return Math.tanh(x); }
function softmaxArr(arr) { const mx = Math.max(...arr), e = arr.map(v => Math.exp(v - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); }

function activate(x, fn) {
  if (fn === "sigmoid") return sigmoid(x);
  if (fn === "relu") return relu(x);
  if (fn === "tanh") return tanhFn(x);
  return x; // linear
}
function activateName(fn) { return fn === "sigmoid" ? "σ" : fn === "relu" ? "ReLU" : fn === "tanh" ? "tanh" : "linear"; }

function initWeights(rows, cols) {
  const scale = Math.sqrt(2.0 / cols);
  return Array(rows).fill(0).map(() => Array(cols).fill(0).map(() => (Math.random() * 2 - 1) * scale));
}
function initBiases(n) { return Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.1); }

function forwardPass(input, layers) {
  // layers: [{neurons, activation, weights[][], biases[]}]
  // Returns activations at each layer including input
  const acts = [input];
  let cur = input;
  for (let l = 0; l < layers.length; l++) {
    const { neurons, activation, weights, biases } = layers[l];
    const out = [];
    for (let j = 0; j < neurons; j++) {
      let z = biases[j];
      for (let i = 0; i < cur.length; i++) z += cur[i] * weights[j][i];
      out.push(activate(z, activation));
    }
    if (activation === "softmax") {
      const sm = softmaxArr(out);
      acts.push(sm);
      cur = sm;
    } else {
      acts.push(out);
      cur = out;
    }
  }
  return acts;
}

// Simple backprop + SGD training
function trainStep(input, target, layers, lr) {
  // Forward
  const acts = [input];
  const zs = [null];
  let cur = input;
  for (let l = 0; l < layers.length; l++) {
    const { neurons, activation, weights, biases } = layers[l];
    const z = [], a = [];
    for (let j = 0; j < neurons; j++) {
      let zj = biases[j];
      for (let i = 0; i < cur.length; i++) zj += cur[i] * weights[j][i];
      z.push(zj);
      a.push(activate(zj, activation));
    }
    zs.push(z);
    acts.push(a);
    cur = a;
  }
  // Backward
  const deltas = new Array(layers.length + 1).fill(null);
  // Output layer delta
  const outIdx = layers.length;
  const outAct = acts[outIdx];
  deltas[outIdx] = outAct.map((a, i) => a - (target[i] || 0)); // MSE derivative

  for (let l = layers.length - 1; l >= 0; l--) {
    const { neurons, activation, weights } = layers[l];
    if (l === layers.length - 1) {
      // Already computed
    }
    if (l > 0) {
      // Compute delta for layer l
      const prevNeurons = layers[l - 1].neurons;
      const prevDelta = [];
      for (let i = 0; i < prevNeurons; i++) {
        let err = 0;
        for (let j = 0; j < neurons; j++) err += deltas[l + 1][j] * weights[j][i];
        const z = zs[l][i];
        const deriv = layers[l - 1].activation === "sigmoid" ? sigmoid(z) * (1 - sigmoid(z))
          : layers[l - 1].activation === "relu" ? (z > 0 ? 1 : 0)
          : layers[l - 1].activation === "tanh" ? (1 - Math.tanh(z) ** 2) : 1;
        prevDelta.push(err * deriv);
      }
      deltas[l] = prevDelta;
    }
  }
  // Update weights
  for (let l = 0; l < layers.length; l++) {
    const { neurons, weights, biases } = layers[l];
    const prevAct = acts[l];
    for (let j = 0; j < neurons; j++) {
      for (let i = 0; i < prevAct.length; i++) {
        weights[j][i] -= lr * deltas[l + 1][j] * prevAct[i];
      }
      biases[j] -= lr * deltas[l + 1][j];
    }
  }
  // Loss (MSE)
  const loss = acts[outIdx].reduce((s, a, i) => s + (a - (target[i] || 0)) ** 2, 0) / acts[outIdx].length;
  return loss;
}

function predict(input, layers) {
  const acts = forwardPass(input, layers);
  return acts[acts.length - 1];
}

/* ═══════════════════════════════════════════════════════════
   DATASETS — 2D classification problems
   ═══════════════════════════════════════════════════════════ */
function generateDataset(type, n = 200) {
  const pts = [];
  if (type === "xor") {
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
      const label = (x > 0) !== (y > 0) ? 1 : 0;
      pts.push({ x, y, label });
    }
  } else if (type === "circle") {
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
      const label = Math.sqrt(x * x + y * y) < 0.6 ? 1 : 0;
      pts.push({ x, y, label });
    }
  } else if (type === "moons") {
    for (let i = 0; i < n; i++) {
      const half = i < n / 2;
      const angle = half ? Math.PI * Math.random() : Math.PI + Math.PI * Math.random();
      const r = 0.5 + (Math.random() - 0.5) * 0.3;
      const x = r * Math.cos(angle) + (half ? 0 : 0.5);
      const y = r * Math.sin(angle) + (half ? 0 : -0.3);
      pts.push({ x: x * 1.2 - 0.3, y: y * 1.2, label: half ? 0 : 1 });
    }
  } else if (type === "spiral") {
    for (let i = 0; i < n; i++) {
      const half = i < n / 2;
      const r = (i % (n / 2)) / (n / 2) * 0.8 + 0.1;
      const angle = r * 4 + (half ? 0 : Math.PI);
      const noise = (Math.random() - 0.5) * 0.15;
      pts.push({ x: r * Math.cos(angle) + noise, y: r * Math.sin(angle) + noise, label: half ? 0 : 1 });
    }
  } else if (type === "linear") {
    for (let i = 0; i < n; i++) {
      const x = Math.random() * 2 - 1, y = Math.random() * 2 - 1;
      pts.push({ x, y, label: x + y > 0 ? 1 : 0 });
    }
  } else if (type === "gaussian") {
    for (let i = 0; i < n; i++) {
      const half = i < n / 2;
      const cx = half ? -0.4 : 0.4, cy = half ? -0.4 : 0.4;
      pts.push({ x: cx + (Math.random() - 0.5) * 0.6, y: cy + (Math.random() - 0.5) * 0.6, label: half ? 0 : 1 });
    }
  }
  return pts;
}
const DATASETS = [
  { id: "xor", name: "XOR Problem", desc: "Non-linearly separable" },
  { id: "circle", name: "Circle", desc: "Concentric classes" },
  { id: "moons", name: "Two Moons", desc: "Crescent shapes" },
  { id: "spiral", name: "Spiral", desc: "Interleaved spirals" },
  { id: "linear", name: "Linear", desc: "Linearly separable" },
  { id: "gaussian", name: "Gaussian Blobs", desc: "Overlapping clusters" },
];

const ACTIVATIONS = ["relu", "sigmoid", "tanh"];
const S = { mono: "'IBM Plex Mono', monospace", sans: "'IBM Plex Sans', system-ui, sans-serif" };

/* ═══════════════════════════════════════════════════════════
   SCATTER PLOT — shows 2D data + decision boundary
   ═══════════════════════════════════════════════════════════ */
function ScatterPlot({ data, layers, size = 280, showBoundary = true }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext("2d"); c.width = size; c.height = size;
    ctx.clearRect(0, 0, size, size);

    // Decision boundary
    if (showBoundary && layers?.length) {
      const res = 40;
      for (let i = 0; i < res; i++) for (let j = 0; j < res; j++) {
        const x = (i / res) * 2 - 1, y = (j / res) * 2 - 1;
        const out = predict([x, y], layers);
        const p = out.length > 1 ? out[1] : out[0];
        const r = Math.round(255 * (1 - p) * 0.4 + 100);
        const g = Math.round(100);
        const b = Math.round(255 * p * 0.4 + 100);
        ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
        ctx.fillRect(i * (size / res), j * (size / res), size / res + 1, size / res + 1);
      }
    }

    // Grid lines
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2); ctx.stroke();

    // Data points
    data.forEach(p => {
      const px = (p.x + 1) / 2 * size, py = (p.y + 1) / 2 * size;
      ctx.beginPath(); ctx.arc(px, py, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? "#3b82f6" : "#ef4444";
      ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 0.8; ctx.stroke();
    });
  }, [data, layers, size, showBoundary]);
  return <canvas ref={canvasRef} style={{ width: size, height: size, borderRadius: 10, border: "1px solid #334155", background: "#0f172a" }} />;
}

/* ═══════════════════════════════════════════════════════════
   NEURAL NETWORK DIAGRAM — visual neurons + connections
   ═══════════════════════════════════════════════════════════ */
function NetworkDiagram({ layerSizes, activations, weights, currentActivations, highlightLayer, size = 400 }) {
  const canvasRef = useRef(null);
  const maxNeurons = Math.max(...layerSizes, 1);
  const nLayers = layerSizes.length;

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext("2d");
    const W = size, H = Math.max(280, maxNeurons * 40 + 40);
    c.width = W; c.height = H;
    ctx.clearRect(0, 0, W, H);

    const gapX = W / (nLayers + 1);
    const positions = layerSizes.map((n, li) => {
      const gapY = H / (n + 1);
      return Array(n).fill(0).map((_, ni) => ({ x: gapX * (li + 1), y: gapY * (ni + 1) }));
    });

    // Draw connections
    for (let l = 1; l < nLayers; l++) {
      const w = weights?.[l - 1];
      for (let j = 0; j < layerSizes[l]; j++) {
        for (let i = 0; i < layerSizes[l - 1]; i++) {
          const from = positions[l - 1][i], to = positions[l][j];
          const wVal = w?.[j]?.[i] ?? 0;
          const intensity = Math.min(Math.abs(wVal) * 2, 1);
          ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.lineTo(to.x, to.y);
          ctx.strokeStyle = wVal >= 0 ? `rgba(59,130,246,${intensity * 0.6 + 0.05})` : `rgba(239,68,68,${intensity * 0.6 + 0.05})`;
          ctx.lineWidth = intensity * 2 + 0.3;
          ctx.stroke();
        }
      }
    }

    // Draw neurons
    for (let l = 0; l < nLayers; l++) {
      const isHL = highlightLayer === l;
      for (let n = 0; n < layerSizes[l]; n++) {
        const { x, y } = positions[l][n];
        const act = currentActivations?.[l]?.[n];
        const hasAct = act !== undefined && act !== null;
        const norm = hasAct ? Math.min(Math.abs(act), 1) : 0;

        // Glow
        if (isHL) {
          ctx.beginPath(); ctx.arc(x, y, 18, 0, Math.PI * 2);
          ctx.fillStyle = "rgba(250,204,21,0.15)"; ctx.fill();
        }

        // Neuron circle
        ctx.beginPath(); ctx.arc(x, y, 14, 0, Math.PI * 2);
        const color = l === 0 ? `rgba(34,197,94,${norm * 0.7 + 0.3})` :
          l === nLayers - 1 ? `rgba(168,85,247,${norm * 0.7 + 0.3})` :
            `rgba(59,130,246,${norm * 0.7 + 0.3})`;
        ctx.fillStyle = color; ctx.fill();
        ctx.strokeStyle = isHL ? "#facc15" : "#475569"; ctx.lineWidth = isHL ? 2 : 1; ctx.stroke();

        // Value text
        if (hasAct) {
          ctx.fillStyle = "#fff"; ctx.font = "bold 9px 'IBM Plex Mono'"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
          ctx.fillText(act.toFixed(2), x, y);
        }
      }

      // Layer label
      const cx = positions[l][0]?.x || gapX * (l + 1);
      ctx.fillStyle = "#64748b"; ctx.font = "bold 10px 'IBM Plex Sans'"; ctx.textAlign = "center";
      const label = l === 0 ? "Input" : l === nLayers - 1 ? "Output" : `Hidden ${l}`;
      ctx.fillText(label, cx, H - 10);
      ctx.fillStyle = "#475569"; ctx.font = "9px 'IBM Plex Mono'";
      ctx.fillText(`${layerSizes[l]}n · ${activations?.[l] || "—"}`, cx, H - 22);
    }
  }, [layerSizes, activations, weights, currentActivations, highlightLayer, size, maxNeurons, nLayers]);

  const H = Math.max(280, maxNeurons * 40 + 40);
  return <canvas ref={canvasRef} style={{ width: size, height: H, borderRadius: 10, border: "1px solid #1e293b", background: "#020617" }} />;
}

/* ═══════════════════════════════════════════════════════════
   ACTIVATION FUNCTION PLOT
   ═══════════════════════════════════════════════════════════ */
function ActivationPlot({ fn, size = 120 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = size; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, size / 2); ctx.lineTo(size, size / 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(size / 2, 0); ctx.lineTo(size / 2, size); ctx.stroke();
    ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2; ctx.beginPath();
    for (let px = 0; px < size; px++) {
      const x = (px / size) * 6 - 3;
      const y = activate(x, fn);
      const py = size - ((y + 1) / 2) * size * (fn === "relu" ? 0.3 : 1);
      if (px === 0) ctx.moveTo(px, Math.max(0, Math.min(size, py)));
      else ctx.lineTo(px, Math.max(0, Math.min(size, py)));
    }
    ctx.stroke();
    ctx.fillStyle = "#94a3b8"; ctx.font = "bold 10px 'IBM Plex Mono'"; ctx.textAlign = "center";
    ctx.fillText(activateName(fn), size / 2, 14);
  }, [fn, size]);
  return <canvas ref={ref} style={{ width: size, height: size, borderRadius: 6, border: "1px solid #334155", background: "#0f172a" }} />;
}

/* ═══════════════════════════════════════════════════════════
   TRAINING CHART
   ═══════════════════════════════════════════════════════════ */
function LossChart({ history, size = 300 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = 150; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, 150);
    if (!history.length) return;
    const maxL = Math.max(...history.map(h => h.loss), 0.01);
    ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 1.5; ctx.beginPath();
    history.forEach((h, i) => {
      const x = (i / Math.max(history.length - 1, 1)) * size;
      const y = 150 - (h.loss / maxL) * 140 - 5;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // Accuracy
    ctx.strokeStyle = "#22c55e"; ctx.lineWidth = 1.5; ctx.beginPath();
    history.forEach((h, i) => {
      const x = (i / Math.max(history.length - 1, 1)) * size;
      const y = 150 - h.acc * 140 - 5;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = "#ef4444"; ctx.font = "9px 'IBM Plex Mono'"; ctx.fillText(`Loss: ${history[history.length - 1].loss.toFixed(4)}`, 4, 12);
    ctx.fillStyle = "#22c55e"; ctx.fillText(`Acc: ${(history[history.length - 1].acc * 100).toFixed(1)}%`, 4, 24);
  }, [history, size]);
  return <canvas ref={ref} style={{ width: size, height: 150, borderRadius: 8, border: "1px solid #1e293b", background: "#0f172a" }} />;
}

/* ═══════════════════════════════════════════════════════════
   STEP-BY-STEP FORWARD PROPAGATION MODULE
   ═══════════════════════════════════════════════════════════ */
function ForwardPropModule({ input, layers, layerSizes, actNames }) {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const allActs = useMemo(() => forwardPass(input, layers), [input, layers]);
  const totalSteps = layers.length + 1; // input + each layer

  useEffect(() => {
    if (auto && step < totalSteps - 1) { const t = setTimeout(() => setStep(p => p + 1), 2500); return () => clearTimeout(t); }
    else setAuto(false);
  }, [auto, step, totalSteps]);

  const STEPS = [
    { t: "📥 Input Layer", d: `Input vector [${input.map(v => v.toFixed(2)).join(", ")}] enters the network. These are the raw feature values (x, y coordinates).` },
    ...layers.map((l, i) => ({
      t: `${i === layers.length - 1 ? "📤" : "🧠"} ${i === layers.length - 1 ? "Output" : "Hidden"} Layer ${i + 1} — ${l.neurons} neurons, ${activateName(l.activation)}`,
      d: `Each neuron computes: z = Σ(w_i × input_i) + bias, then applies ${activateName(l.activation)} activation.`
    })),
  ];

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <div style={{ display: "flex", gap: 2 }}>{STEPS.map((_, i) => <button key={i} onClick={() => { setStep(i); setAuto(false); }} style={{ width: 28, height: 6, borderRadius: 3, background: i <= step ? "#3b82f6" : "#1e293b", border: "none", cursor: "pointer" }} />)}</div>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => { setStep(0); setAuto(false); }} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
          <button onClick={() => setAuto(!auto)} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: auto ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{auto ? "⏸" : "▶ Auto"}</button>
        </div>
      </div>

      {/* Step title */}
      <div style={{ padding: "10px 14px", borderRadius: 10, background: "#3b82f611", border: "1px solid #3b82f633", marginBottom: 14, display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 28, height: 28, borderRadius: 7, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 13, fontWeight: 800, background: "#3b82f622", color: "#3b82f6", flexShrink: 0 }}>{step + 1}</div>
        <div><h5 style={{ fontSize: 13, fontWeight: 700, color: "#fff", margin: 0 }}>{STEPS[step]?.t}</h5><p style={{ fontSize: 11, color: "#94a3b8", margin: 0 }}>{STEPS[step]?.d}</p></div>
      </div>

      {/* Visual content */}
      <div style={{ minHeight: 220, padding: 16, background: "rgba(15,23,42,0.5)", borderRadius: 10, border: "1px solid #1e293b" }}>
        {step === 0 && (
          <div style={{ display: "flex", gap: 24, alignItems: "center", flexWrap: "wrap" }}>
            <div>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#22c55e", marginBottom: 6, fontFamily: S.mono }}>Input Vector</p>
              <div style={{ display: "flex", gap: 6 }}>
                {input.map((v, i) => (
                  <div key={i} style={{ width: 52, height: 52, borderRadius: 26, display: "flex", alignItems: "center", justifyContent: "center", background: `rgba(34,197,94,${Math.abs(v) * 0.5 + 0.2})`, border: "2px solid #22c55e", color: "#fff", fontSize: 12, fontWeight: 700, fontFamily: S.mono }}>{v.toFixed(2)}</div>
                ))}
              </div>
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Shape: [{input.length}]</p>
            </div>
            <div style={{ fontSize: 24, color: "#22c55e", fontWeight: 800 }}>→</div>
            <NetworkDiagram layerSizes={layerSizes} activations={actNames} weights={layers.map(l => l.weights)} currentActivations={[input]} highlightLayer={0} size={300} />
          </div>
        )}

        {step > 0 && step <= layers.length && (() => {
          const li = step - 1;
          const l = layers[li];
          const prevAct = allActs[li];
          const curAct = allActs[li + 1];
          return (
            <div>
              {/* Neuron computations */}
              <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
                <div style={{ flex: 1, minWidth: 280 }}>
                  <p style={{ fontSize: 10, fontWeight: 700, color: "#60a5fa", marginBottom: 8, fontFamily: S.mono }}>Per-neuron computation:</p>
                  <div style={{ maxHeight: 180, overflowY: "auto", space: "y-2" }}>
                    {curAct.slice(0, 8).map((a, ni) => {
                      let z = l.biases[ni];
                      const terms = prevAct.map((pa, pi) => { const w = l.weights[ni][pi]; z; return { pa, w, prod: pa * w }; });
                      const zVal = l.biases[ni] + terms.reduce((s, t) => s + t.prod, 0);
                      return (
                        <div key={ni} style={{ marginBottom: 8, padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 4 }}>
                            <div style={{ width: 24, height: 24, borderRadius: 12, background: li === layers.length - 1 ? "#7c3aed44" : "#3b82f644", border: `1px solid ${li === layers.length - 1 ? "#a855f7" : "#3b82f6"}`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, color: "#fff" }}>n{ni}</div>
                            <span style={{ fontSize: 10, fontFamily: S.mono, color: "#94a3b8" }}>z = </span>
                            <div style={{ display: "flex", gap: 3, flexWrap: "wrap", alignItems: "center" }}>
                              {terms.slice(0, 4).map((t, ti) => (
                                <span key={ti} style={{ fontSize: 9, fontFamily: S.mono, color: "#cbd5e1" }}>
                                  <span style={{ color: "#22c55e" }}>{t.pa.toFixed(2)}</span>
                                  <span style={{ color: "#64748b" }}>×</span>
                                  <span style={{ color: t.w >= 0 ? "#60a5fa" : "#ef4444" }}>{t.w.toFixed(2)}</span>
                                  {ti < Math.min(terms.length, 4) - 1 && <span style={{ color: "#64748b" }}> + </span>}
                                </span>
                              ))}
                              {terms.length > 4 && <span style={{ fontSize: 9, color: "#475569" }}>+...</span>}
                              <span style={{ fontSize: 9, color: "#64748b", fontFamily: S.mono }}> + b({l.biases[ni].toFixed(2)})</span>
                            </div>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                            <span style={{ fontSize: 10, fontFamily: S.mono, color: "#f59e0b" }}>z = {zVal.toFixed(4)}</span>
                            <span style={{ color: "#475569" }}>→</span>
                            <span style={{ fontSize: 10, fontFamily: S.mono, color: "#22c55e", fontWeight: 700 }}>{activateName(l.activation)}(z) = {a.toFixed(4)}</span>
                          </div>
                        </div>
                      );
                    })}
                    {curAct.length > 8 && <p style={{ fontSize: 9, color: "#475569" }}>...and {curAct.length - 8} more neurons</p>}
                  </div>
                </div>

                <div>
                  <NetworkDiagram layerSizes={layerSizes} activations={actNames} weights={layers.map(l2 => l2.weights)} currentActivations={allActs.slice(0, li + 2)} highlightLayer={li + 1} size={260} />
                  <div style={{ display: "flex", gap: 6, marginTop: 8, justifyContent: "center" }}>
                    <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 4, background: "#22c55e22", color: "#22c55e", fontFamily: S.mono }}>Input: [{prevAct.length}]</span>
                    <span style={{ fontSize: 9, color: "#475569" }}>→</span>
                    <span style={{ fontSize: 9, padding: "2px 8px", borderRadius: 4, background: "#3b82f622", color: "#3b82f6", fontFamily: S.mono }}>Output: [{curAct.length}]</span>
                  </div>
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      {/* Nav */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
        <button onClick={() => { setStep(Math.max(0, step - 1)); setAuto(false); }} disabled={step === 0} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === 0 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === 0 ? "not-allowed" : "pointer" }}>← Prev</button>
        <span style={{ fontSize: 10, color: "#475569" }}>{step + 1}/{totalSteps}</span>
        <button onClick={() => { setStep(Math.min(totalSteps - 1, step + 1)); setAuto(false); }} disabled={step === totalSteps - 1} style={{ padding: "4px 12px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#1e293b", color: step === totalSteps - 1 ? "#334155" : "#94a3b8", border: "1px solid #334155", cursor: step === totalSteps - 1 ? "not-allowed" : "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LAYER CARD
   ═══════════════════════════════════════════════════════════ */
const LINFO = {
  input: { icon: "📥", c: "#22c55e", t: "Input Layer" },
  hidden: { icon: "🧠", c: "#3b82f6", t: "Hidden Layer" },
  output: { icon: "📤", c: "#a855f7", t: "Output Layer" },
};

function LayerCard({ layer, idx, total, onRemove, onUpdate }) {
  const [exp, setExp] = useState(false);
  const info = LINFO[layer.type] || LINFO.hidden;
  const isRemovable = layer.type === "hidden";
  return (
    <div style={{ borderRadius: 10, background: "rgba(15,23,42,0.7)", border: `1px solid ${exp ? info.c + "66" : "#1e293b"}`, transition: "all 0.2s" }}>
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "10px 14px", cursor: "pointer" }} onClick={() => setExp(!exp)}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, background: `${info.c}22`, flexShrink: 0 }}>{info.icon}</div>
        <div style={{ flex: 1 }}>
          <span style={{ fontSize: 12, fontWeight: 700, color: "#fff" }}>{info.t}</span>
          <span style={{ fontSize: 9, marginLeft: 6, padding: "1px 5px", borderRadius: 8, background: `${info.c}22`, color: info.c }}>{layer.neurons}n · {layer.activation || "—"}</span>
        </div>
        {isRemovable && <button onClick={e => { e.stopPropagation(); onRemove(); }} style={{ width: 24, height: 24, borderRadius: 6, background: "transparent", border: "1px solid #334155", color: "#64748b", cursor: "pointer", fontSize: 10 }}>✕</button>}
        <span style={{ color: "#475569", fontSize: 10 }}>{exp ? "▲" : "▼"}</span>
      </div>
      {exp && (
        <div style={{ borderTop: "1px solid #1e293b", padding: "10px 14px" }}>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
            <label style={{ fontSize: 10, color: "#64748b" }}>Neurons:
              <input type="number" min={1} max={32} value={layer.neurons} onChange={e => onUpdate({ neurons: Math.max(1, Math.min(32, +e.target.value)) })} style={{ width: 50, marginLeft: 4, padding: "2px 6px", borderRadius: 4, fontSize: 11, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155" }} />
            </label>
            {layer.type !== "input" && (
              <label style={{ fontSize: 10, color: "#64748b" }}>Activation:
                <select value={layer.activation} onChange={e => onUpdate({ activation: e.target.value })} style={{ marginLeft: 4, padding: "2px 6px", borderRadius: 4, fontSize: 10, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155" }}>
                  {ACTIVATIONS.map(a => <option key={a} value={a}>{a}</option>)}
                  <option value="softmax">softmax</option>
                </select>
              </label>
            )}
          </div>
          {layer.type !== "input" && <ActivationPlot fn={layer.activation} size={100} />}
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ARCHITECTURE PRESETS
   ═══════════════════════════════════════════════════════════ */
const ARCH_PRESETS = {
  simple: { name: "Simple (2→4→1)", layers: [{ type: "input", neurons: 2 }, { type: "hidden", neurons: 4, activation: "relu" }, { type: "output", neurons: 1, activation: "sigmoid" }] },
  medium: { name: "Medium (2→8→4→1)", layers: [{ type: "input", neurons: 2 }, { type: "hidden", neurons: 8, activation: "relu" }, { type: "hidden", neurons: 4, activation: "relu" }, { type: "output", neurons: 1, activation: "sigmoid" }] },
  deep: { name: "Deep (2→16→8→4→1)", layers: [{ type: "input", neurons: 2 }, { type: "hidden", neurons: 16, activation: "relu" }, { type: "hidden", neurons: 8, activation: "relu" }, { type: "hidden", neurons: 4, activation: "relu" }, { type: "output", neurons: 1, activation: "sigmoid" }] },
  wide: { name: "Wide (2→32→1)", layers: [{ type: "input", neurons: 2 }, { type: "hidden", neurons: 32, activation: "tanh" }, { type: "output", neurons: 1, activation: "sigmoid" }] },
  multiclass: { name: "Multi-class (2→8→4→3)", layers: [{ type: "input", neurons: 2 }, { type: "hidden", neurons: 8, activation: "relu" }, { type: "hidden", neurons: 4, activation: "relu" }, { type: "output", neurons: 3, activation: "softmax" }] },
};

/* ═══════════════════════════════════════════════════════════
   FULLSCREEN MODAL
   ═══════════════════════════════════════════════════════════ */
function FullModal({ data, layerDefs, builtLayers, onClose }) {
  const samplePt = data[0] || { x: 0, y: 0 };
  const input = [samplePt.x, samplePt.y];
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(2,6,23,0.95)", backdropFilter: "blur(16px)", overflow: "auto" }}>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, background: "#3b82f622" }}>🧠</div>
            <div><h2 style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>Forward Propagation</h2><p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>Step-by-step through the network for input [{input.map(v => v.toFixed(2)).join(", ")}]</p></div>
          </div>
          <button onClick={onClose} style={{ width: 36, height: 36, borderRadius: 8, fontSize: 18, color: "#64748b", background: "#1e293b", border: "1px solid #334155", cursor: "pointer" }}>×</button>
        </div>
        <div style={{ background: "rgba(15,23,42,0.6)", borderRadius: 14, border: "1px solid #1e293b", padding: 20 }}>
          <ForwardPropModule input={input} layers={builtLayers} layerSizes={layerDefs.map(l => l.neurons)} actNames={layerDefs.map(l => l.activation || "—")} />
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN ANN LAB
   ═══════════════════════════════════════════════════════════ */
export default function ANNLab() {
  const [datasetId, setDatasetId] = useState("xor");
  const [data, setData] = useState(() => generateDataset("xor"));
  const [layerDefs, setLayerDefs] = useState(ARCH_PRESETS.simple.layers.map(l => ({ ...l })));
  const [lr, setLr] = useState(0.1);
  const [epochs, setEpochs] = useState(0);
  const [trainHistory, setTrainHistory] = useState([]);
  const [isTraining, setIsTraining] = useState(false);
  const [modal, setModal] = useState(false);
  const trainRef = useRef(null);

  // Build layers with weights
  const builtLayers = useMemo(() => {
    const built = [];
    for (let i = 1; i < layerDefs.length; i++) {
      const prevN = layerDefs[i - 1].neurons;
      const curN = layerDefs[i].neurons;
      built.push({
        neurons: curN,
        activation: layerDefs[i].activation || "relu",
        weights: initWeights(curN, prevN),
        biases: initBiases(curN),
      });
    }
    return built;
  }, [layerDefs]);

  // Persistent ref for training
  const layersRef = useRef(builtLayers);
  useEffect(() => { layersRef.current = builtLayers; }, [builtLayers]);

  const changeDataset = (id) => {
    setDatasetId(id);
    setData(generateDataset(id));
    setTrainHistory([]); setEpochs(0);
  };

  const regenerateData = () => {
    setData(generateDataset(datasetId));
    setTrainHistory([]); setEpochs(0);
  };

  const resetWeights = () => {
    // Force rebuild
    setLayerDefs(prev => prev.map(l => ({ ...l })));
    setTrainHistory([]); setEpochs(0);
  };

  const addHiddenLayer = () => {
    setLayerDefs(prev => {
      const out = prev[prev.length - 1];
      const rest = prev.slice(0, -1);
      return [...rest, { type: "hidden", neurons: 4, activation: "relu" }, out];
    });
    setTrainHistory([]); setEpochs(0);
  };

  const removeHiddenLayer = (idx) => {
    setLayerDefs(prev => prev.filter((_, i) => i !== idx));
    setTrainHistory([]); setEpochs(0);
  };

  const updateLayer = (idx, updates) => {
    setLayerDefs(prev => prev.map((l, i) => i === idx ? { ...l, ...updates } : l));
    setTrainHistory([]); setEpochs(0);
  };

  const loadPreset = (key) => {
    setLayerDefs(ARCH_PRESETS[key].layers.map(l => ({ ...l })));
    setTrainHistory([]); setEpochs(0);
  };

  // Training loop
  const trainBatch = useCallback(() => {
    const layers = layersRef.current;
    let totalLoss = 0, correct = 0;
    data.forEach(pt => {
      const target = layers[layers.length - 1].neurons > 1
        ? Array(layers[layers.length - 1].neurons).fill(0).map((_, i) => i === pt.label ? 1 : 0)
        : [pt.label];
      const loss = trainStep([pt.x, pt.y], target, layers, lr);
      totalLoss += loss;
      const out = predict([pt.x, pt.y], layers);
      const pred = out.length > 1 ? out.indexOf(Math.max(...out)) : (out[0] > 0.5 ? 1 : 0);
      if (pred === pt.label) correct++;
    });
    return { loss: totalLoss / data.length, acc: correct / data.length };
  }, [data, lr]);

  useEffect(() => {
    if (!isTraining) { if (trainRef.current) clearInterval(trainRef.current); return; }
    trainRef.current = setInterval(() => {
      const result = trainBatch();
      setEpochs(p => p + 1);
      setTrainHistory(p => [...p, result]);
    }, 50);
    return () => clearInterval(trainRef.current);
  }, [isTraining, trainBatch]);

  const layerSizes = layerDefs.map(l => l.neurons);
  const actNames = layerDefs.map(l => l.activation || "—");
  const sampleInput = data[0] ? [data[0].x, data[0].y] : [0, 0];
  const sampleActs = useMemo(() => {
    try { return forwardPass(sampleInput, builtLayers); } catch { return [sampleInput]; }
  }, [sampleInput, builtLayers]);

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #0a1628 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      {/* Lab title */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#3b82f6,#a855f7)" }}>🧠</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>ANN Lab</h1>
        <span style={{ fontSize: 10 }}>Interactive Neural Network Simulator</span>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 14 }}>
          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Dataset */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>📊 Dataset</h3>
              <select value={datasetId} onChange={e => changeDataset(e.target.value)} style={{ width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}>
                {DATASETS.map(d => <option key={d.id} value={d.id}>{d.name} — {d.desc}</option>)}
              </select>
              <div style={{ display: "flex", justifyContent: "center", marginTop: 8 }}>
                <ScatterPlot data={data} layers={epochs > 0 ? builtLayers : null} size={200} showBoundary={epochs > 0} />
              </div>
              <div style={{ display: "flex", gap: 4, marginTop: 6 }}>
                <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: "#ef444422", color: "#ef4444" }}>● Class 0</span>
                <span style={{ fontSize: 9, padding: "2px 6px", borderRadius: 4, background: "#3b82f622", color: "#3b82f6" }}>● Class 1</span>
                <span style={{ fontSize: 9, color: "#475569", marginLeft: "auto" }}>{data.length} pts</span>
              </div>
              <button onClick={regenerateData} style={{ width: "100%", marginTop: 6, padding: "4px 0", borderRadius: 6, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↻ Regenerate Data</button>
            </div>

            {/* Templates */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>🚀 Architecture Templates</h3>
              {Object.entries(ARCH_PRESETS).map(([k, v]) => (
                <button key={k} onClick={() => loadPreset(k)} style={{ display: "block", width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, fontWeight: 600, background: "#0f172a", border: "1px solid #1e293b", color: "#94a3b8", cursor: "pointer", marginBottom: 3, textAlign: "left" }}>
                  <span style={{ color: "#fff" }}>{v.name}</span>
                </button>
              ))}
            </div>

            {/* Training controls */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>⚡ Training</h3>
              <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 6 }}>Learning Rate:
                <select value={lr} onChange={e => setLr(+e.target.value)} style={{ marginLeft: 4, padding: "2px 6px", borderRadius: 4, fontSize: 10, background: "#0f172a", color: "#e2e8f0", border: "1px solid #334155" }}>
                  {[0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0].map(v => <option key={v} value={v}>{v}</option>)}
                </select>
              </label>
              <div style={{ display: "flex", gap: 4 }}>
                <button onClick={() => setIsTraining(!isTraining)} style={{ flex: 1, padding: "6px 0", borderRadius: 6, fontSize: 11, fontWeight: 700, background: isTraining ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>
                  {isTraining ? "⏸ Stop" : "▶ Train"}
                </button>
                <button onClick={resetWeights} style={{ padding: "6px 10px", borderRadius: 6, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺ Reset</button>
              </div>
              <p style={{ fontSize: 10, color: "#475569", marginTop: 6 }}>Epoch: <b style={{ color: "#fff" }}>{epochs}</b></p>
              {trainHistory.length > 0 && <LossChart history={trainHistory} size={210} />}
            </div>
          </div>

          {/* Main area */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Network builder bar */}
            <div style={{ borderRadius: 12, padding: "8px 12px", background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b", display: "flex", flexWrap: "wrap", gap: 6, alignItems: "center" }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b" }}>Network:</span>
              <span style={{ fontSize: 11, fontFamily: S.mono, color: "#60a5fa" }}>{layerSizes.join(" → ")}</span>
              <button onClick={addHiddenLayer} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: "#3b82f618", color: "#3b82f6", border: "1px solid #3b82f633", cursor: "pointer" }}>+ Hidden Layer</button>
              <button onClick={() => setModal(true)} style={{ marginLeft: "auto", padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: "#f59e0b18", color: "#f59e0b", border: "1px solid #f59e0b33", cursor: "pointer" }}>⛶ Step-by-Step Forward Pass</button>
            </div>

            {/* Network Diagram */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b", textAlign: "center" }}>
              <NetworkDiagram layerSizes={layerSizes} activations={actNames} weights={builtLayers.map(l => l.weights)} currentActivations={sampleActs} size={Math.min(700, layerSizes.length * 140)} />
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Network: {layerSizes.map((n, i) => `${n}${actNames[i] !== "—" ? `(${actNames[i]})` : ""}`).join(" → ")} · Total params: {builtLayers.reduce((s, l) => s + l.neurons * (l.weights[0]?.length || 0) + l.neurons, 0)}</p>
            </div>

            {/* Layer cards */}
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {layerDefs.map((l, i) => (
                <div key={i}>
                  {i > 0 && <div style={{ display: "flex", justifyContent: "center" }}><div style={{ width: 1.5, height: 8, background: "#334155" }} /></div>}
                  <LayerCard layer={l} idx={i} total={layerDefs.length} onRemove={() => removeHiddenLayer(i)} onUpdate={updates => updateLayer(i, updates)} />
                </div>
              ))}
            </div>

            {/* Decision boundary + scatter (large) */}
            {epochs > 0 && (
              <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 11, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🎯 Decision Boundary</p>
                <div style={{ display: "flex", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
                  <ScatterPlot data={data} layers={builtLayers} size={320} showBoundary={true} />
                  <div style={{ fontSize: 12, color: "#94a3b8" }}>
                    <p style={{ marginBottom: 6 }}>Epoch: <b style={{ color: "#fff" }}>{epochs}</b></p>
                    {trainHistory.length > 0 && <>
                      <p>Loss: <b style={{ color: "#ef4444" }}>{trainHistory[trainHistory.length - 1].loss.toFixed(4)}</b></p>
                      <p>Accuracy: <b style={{ color: "#22c55e" }}>{(trainHistory[trainHistory.length - 1].acc * 100).toFixed(1)}%</b></p>
                    </>}
                    <p style={{ marginTop: 8, fontSize: 11, color: "#475569" }}>The colored background shows what the network would predict for each point in the 2D space.</p>
                    <p style={{ fontSize: 11, color: "#475569" }}><span style={{ color: "#ef4444" }}>Red</span> = Class 0, <span style={{ color: "#3b82f6" }}>Blue</span> = Class 1</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {modal && <FullModal data={data} layerDefs={layerDefs} builtLayers={builtLayers} onClose={() => setModal(false)} />}
    </div>
  );
}
