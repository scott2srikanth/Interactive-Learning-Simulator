import { useState, useEffect, useRef, useMemo, useCallback } from "react";

/* ═══════════════════════════════════════════════════════════
   RNN / LSTM MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function sigmoid(x) { return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x)))); }
function tanhFn(x) { return Math.tanh(x); }
function randMat(r, c) { return Array(r).fill(0).map(() => Array(c).fill(0).map(() => (Math.random() - 0.5) * 0.4)); }
function randVec(n) { return Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.1); }
function matVecMul(m, v) { return m.map(row => row.reduce((s, w, i) => s + w * (v[i] || 0), 0)); }
function vecAdd(a, b) { return a.map((v, i) => v + (b[i] || 0)); }
function vecMul(a, b) { return a.map((v, i) => v * (b[i] || 0)); }
function vecSigmoid(v) { return v.map(sigmoid); }
function vecTanh(v) { return v.map(tanhFn); }

function rnnForward(sequence, Wxh, Whh, Why, bh, by, h0) {
  const steps = [];
  let h = h0;
  for (let t = 0; t < sequence.length; t++) {
    const x = sequence[t];
    const xPart = matVecMul(Wxh, x);
    const hPart = matVecMul(Whh, h);
    const zH = vecAdd(vecAdd(xPart, hPart), bh);
    const hNew = vecTanh(zH);
    const zY = vecAdd(matVecMul(Why, hNew), by);
    steps.push({ t, x, hPrev: h, xPart, hPart, zH, hNew, zY, output: zY });
    h = hNew;
  }
  return steps;
}

function lstmForward(sequence, Wf, Wi, Wo, Wc, bf, bi, bo, bc, Why, by, h0, c0) {
  const steps = [];
  let h = h0, c = c0;
  for (let t = 0; t < sequence.length; t++) {
    const x = sequence[t];
    const combined = [...x, ...h];
    const fGateZ = vecAdd(matVecMul(Wf, combined), bf);
    const fGate = vecSigmoid(fGateZ);
    const iGateZ = vecAdd(matVecMul(Wi, combined), bi);
    const iGate = vecSigmoid(iGateZ);
    const oGateZ = vecAdd(matVecMul(Wo, combined), bo);
    const oGate = vecSigmoid(oGateZ);
    const cCandZ = vecAdd(matVecMul(Wc, combined), bc);
    const cCand = vecTanh(cCandZ);
    const cNew = vecAdd(vecMul(fGate, c), vecMul(iGate, cCand));
    const hNew = vecMul(oGate, vecTanh(cNew));
    const zY = vecAdd(matVecMul(Why, hNew), by);
    steps.push({ t, x, hPrev: h, cPrev: c, combined, fGate, iGate, oGate, cCand, cNew, hNew, zY, output: zY });
    h = hNew; c = cNew;
  }
  return steps;
}

/* ═══════════════════════════════════════════════════════════
   SEQUENCE DATASETS
   ═══════════════════════════════════════════════════════════ */
function generateSequence(type, len = 12) {
  const seq = [];
  if (type === "sine") { for (let i = 0; i < len; i++) seq.push([Math.sin(i * 0.5)]); }
  else if (type === "cosine") { for (let i = 0; i < len; i++) seq.push([Math.cos(i * 0.5)]); }
  else if (type === "sawtooth") { for (let i = 0; i < len; i++) seq.push([(i % 5) / 4]); }
  else if (type === "square") { for (let i = 0; i < len; i++) seq.push([i % 6 < 3 ? 1 : -1]); }
  else if (type === "triangle") { for (let i = 0; i < len; i++) { const p = i % 8; seq.push([p < 4 ? p / 4 : (8 - p) / 4]); } }
  else if (type === "random_walk") { let v = 0; for (let i = 0; i < len; i++) { v += (Math.random() - 0.5) * 0.3; seq.push([Math.max(-1, Math.min(1, v))]); } }
  else if (type === "binary") { for (let i = 0; i < len; i++) seq.push([Math.random() > 0.5 ? 1 : 0]); }
  else if (type === "text_hello") {
    const chars = "hello world!";
    const vocab = [...new Set(chars)];
    for (let i = 0; i < Math.min(len, chars.length); i++) {
      const onehot = vocab.map(c => c === chars[i] ? 1 : 0);
      seq.push(onehot);
    }
  }
  return seq;
}
const SEQUENCES = [
  { id: "sine", name: "Sine Wave", desc: "Smooth periodic signal", inputSize: 1 },
  { id: "cosine", name: "Cosine Wave", desc: "Phase-shifted sine", inputSize: 1 },
  { id: "sawtooth", name: "Sawtooth", desc: "Rising ramp pattern", inputSize: 1 },
  { id: "square", name: "Square Wave", desc: "Binary oscillation", inputSize: 1 },
  { id: "triangle", name: "Triangle Wave", desc: "Rising/falling ramp", inputSize: 1 },
  { id: "random_walk", name: "Random Walk", desc: "Cumulative random steps", inputSize: 1 },
  { id: "binary", name: "Binary Random", desc: "Random 0s and 1s", inputSize: 1 },
  { id: "text_hello", name: "\"hello world!\"", desc: "Character-level encoding", inputSize: 8 },
];

const S = { mono: "'IBM Plex Mono', monospace", sans: "'IBM Plex Sans', system-ui, sans-serif" };
const GATE_COLORS = { forget: "#ef4444", input: "#22c55e", output: "#3b82f6", cell: "#f59e0b" };

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS
   ═══════════════════════════════════════════════════════════ */
function SequencePlot({ data, highlights, size = 280, height = 100, label }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = height; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, height);
    ctx.strokeStyle = "#334155"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, height / 2); ctx.lineTo(size, height / 2); ctx.stroke();
    if (!data?.length) return;
    const vals = data.map(d => d[0] ?? 0);
    const mn = Math.min(...vals, -1), mx = Math.max(...vals, 1), rng = mx - mn || 1;
    // Line
    ctx.strokeStyle = "#3b82f6"; ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = height - ((v - mn) / rng) * (height - 20) - 10; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    // Points
    vals.forEach((v, i) => {
      const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = height - ((v - mn) / rng) * (height - 20) - 10;
      ctx.beginPath(); ctx.arc(x, y, highlights?.includes(i) ? 5 : 3, 0, Math.PI * 2);
      ctx.fillStyle = highlights?.includes(i) ? "#facc15" : "#60a5fa"; ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 0.5; ctx.stroke();
    });
  }, [data, highlights, size, height]);
  return (
    <div>
      {label && <p style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4, fontFamily: S.mono }}>{label}</p>}
      <canvas ref={ref} style={{ width: size, height, borderRadius: 8, border: "1px solid #1e293b", background: "#0f172a" }} />
    </div>
  );
}

function VectorDisplay({ values, label, color = "#3b82f6", maxShow = 8 }) {
  if (!values?.length) return null;
  return (
    <div>
      {label && <p style={{ fontSize: 9, fontWeight: 700, color: color, marginBottom: 3, fontFamily: S.mono }}>{label} [{values.length}]</p>}
      <div style={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
        {values.slice(0, maxShow).map((v, i) => (
          <div key={i} style={{ width: 38, height: 26, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, fontWeight: 600, fontFamily: S.mono, borderRadius: 4, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 180 + 40).toString(16).padStart(2, '0')}`, color: "#fff", border: `1px solid ${color}66` }}>
            {v.toFixed(2)}
          </div>
        ))}
        {values.length > maxShow && <span style={{ fontSize: 8, color: "#475569", alignSelf: "center" }}>+{values.length - maxShow}</span>}
      </div>
    </div>
  );
}

function GateDisplay({ name, values, color }) {
  if (!values?.length) return null;
  return (
    <div style={{ padding: 6, borderRadius: 6, background: `${color}11`, border: `1px solid ${color}33` }}>
      <p style={{ fontSize: 9, fontWeight: 700, color, marginBottom: 3, fontFamily: S.mono }}>{name}</p>
      <div style={{ display: "flex", gap: 2 }}>
        {values.slice(0, 6).map((v, i) => (
          <div key={i} style={{ width: 28, height: 20, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 700, fontFamily: S.mono, borderRadius: 3, background: `${color}${Math.round(v * 200 + 30).toString(16).padStart(2, '0')}`, color: "#fff" }}>
            {v.toFixed(2)}
          </div>
        ))}
      </div>
    </div>
  );
}

function HiddenStateTimeline({ steps, hiddenIdx = 0, size = 300 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = 100; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, 100);
    if (!steps?.length) return;
    const vals = steps.map(s => s.hNew[hiddenIdx] ?? 0);
    const mn = Math.min(...vals, -1), mx = Math.max(...vals, 1), rng = mx - mn || 1;
    ctx.strokeStyle = "#a855f7"; ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = 100 - ((v - mn) / rng) * 80 - 10; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = 100 - ((v - mn) / rng) * 80 - 10; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fillStyle = "#c084fc"; ctx.fill(); });
    ctx.fillStyle = "#64748b"; ctx.font = "9px 'IBM Plex Mono'"; ctx.fillText(`h[${hiddenIdx}] over time`, 4, 12);
  }, [steps, hiddenIdx, size]);
  return <canvas ref={ref} style={{ width: size, height: 100, borderRadius: 8, border: "1px solid #1e293b", background: "#0f172a" }} />;
}

function CellStateTimeline({ steps, cellIdx = 0, size = 300 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = 100; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, 100);
    if (!steps?.length || !steps[0].cNew) return;
    const vals = steps.map(s => s.cNew[cellIdx] ?? 0);
    const mn = Math.min(...vals, -1), mx = Math.max(...vals, 1), rng = mx - mn || 1;
    ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = 2; ctx.beginPath();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = 100 - ((v - mn) / rng) * 80 - 10; i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
    ctx.stroke();
    vals.forEach((v, i) => { const x = (i / (vals.length - 1 || 1)) * (size - 20) + 10, y = 100 - ((v - mn) / rng) * 80 - 10; ctx.beginPath(); ctx.arc(x, y, 3, 0, Math.PI * 2); ctx.fillStyle = "#fbbf24"; ctx.fill(); });
    ctx.fillStyle = "#64748b"; ctx.font = "9px 'IBM Plex Mono'"; ctx.fillText(`cell[${cellIdx}] over time`, 4, 12);
  }, [steps, cellIdx, size]);
  return <canvas ref={ref} style={{ width: size, height: 100, borderRadius: 8, border: "1px solid #1e293b", background: "#0f172a" }} />;
}

function TensorShape({ shape, color = "#60a5fa" }) {
  return <span style={{ fontSize: 10, fontFamily: S.mono, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: `${color}15`, border: `1px solid ${color}33`, color }}>[{shape.join("×")}]</span>;
}

/* ═══════════════════════════════════════════════════════════
   RNN STEP-BY-STEP MODULE
   ═══════════════════════════════════════════════════════════ */
function RNNStepModule({ sequence, hiddenSize, steps: rnnSteps }) {
  const [curT, setCurT] = useState(0);
  const [auto, setAuto] = useState(false);
  const [subStep, setSubStep] = useState(0);

  useEffect(() => { if (auto) { const t = setTimeout(() => { if (subStep < 3) setSubStep(p => p + 1); else { if (curT < rnnSteps.length - 1) { setCurT(p => p + 1); setSubStep(0); } else setAuto(false); } }, 1800); return () => clearTimeout(t); } }, [auto, curT, subStep, rnnSteps.length]);

  const step = rnnSteps[curT];
  if (!step) return null;

  const SUB_STEPS = [
    { t: `📥 Timestep ${curT}: Input x_${curT}`, d: `Input vector arrives. Hidden state h_${curT - 1 >= 0 ? curT - 1 : "init"} is carried from previous step.` },
    { t: "✖️ Weighted Sums", d: "Compute W_xh · x_t (input contribution) and W_hh · h_{t-1} (recurrent contribution)." },
    { t: "➕ Combine & Activate", d: "z = W_xh·x_t + W_hh·h_{t-1} + bias → h_t = tanh(z). The new hidden state captures both current input and past context." },
    { t: "📤 Output", d: "y_t = W_hy · h_t + b_y. Hidden state is passed to the next timestep AND used to produce output." },
  ];

  return (
    <div>
      {/* Timeline navigation */}
      <div style={{ display: "flex", gap: 3, marginBottom: 12 }}>
        {rnnSteps.map((_, t) => (
          <button key={t} onClick={() => { setCurT(t); setSubStep(0); setAuto(false); }} style={{ flex: 1, height: 24, borderRadius: 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 9, fontWeight: 700, background: t === curT ? "#3b82f6" : t < curT ? "#3b82f633" : "#1e293b", color: t === curT ? "#fff" : "#64748b", border: `1px solid ${t === curT ? "#3b82f6" : "#334155"}`, cursor: "pointer" }}>
            t={t}
          </button>
        ))}
        <button onClick={() => setAuto(!auto)} style={{ padding: "0 10px", borderRadius: 4, fontSize: 10, fontWeight: 700, background: auto ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{auto ? "⏸" : "▶"}</button>
      </div>

      {/* Sub-step progress */}
      <div style={{ display: "flex", gap: 2, marginBottom: 10 }}>
        {SUB_STEPS.map((_, i) => <div key={i} style={{ flex: 1, height: 4, borderRadius: 2, background: i <= subStep ? "#3b82f6" : "#1e293b", cursor: "pointer" }} onClick={() => { setSubStep(i); setAuto(false); }} />)}
      </div>

      {/* Step title */}
      <div style={{ padding: "8px 12px", borderRadius: 8, background: "#3b82f611", border: "1px solid #3b82f633", marginBottom: 12, display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 24, height: 24, borderRadius: 6, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 11, fontWeight: 800, background: "#3b82f622", color: "#3b82f6", flexShrink: 0 }}>{subStep + 1}</div>
        <div><h5 style={{ fontSize: 12, fontWeight: 700, color: "#fff", margin: 0 }}>{SUB_STEPS[subStep].t}</h5><p style={{ fontSize: 10, color: "#94a3b8", margin: 0 }}>{SUB_STEPS[subStep].d}</p></div>
      </div>

      {/* Visual */}
      <div style={{ padding: 14, background: "rgba(15,23,42,0.5)", borderRadius: 10, border: "1px solid #1e293b", minHeight: 180 }}>
        {subStep === 0 && (
          <div style={{ display: "flex", gap: 20, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VectorDisplay values={step.x} label={`x_${curT} (input)`} color="#22c55e" />
            <div style={{ fontSize: 20, color: "#f59e0b", fontWeight: 800, alignSelf: "center" }}>+</div>
            <VectorDisplay values={step.hPrev} label={`h_${curT > 0 ? curT - 1 : "init"} (hidden)`} color="#a855f7" />
            <div style={{ alignSelf: "center" }}>
              <SequencePlot data={sequence} highlights={[curT]} size={200} height={70} label="Sequence" />
            </div>
          </div>
        )}
        {subStep === 1 && (
          <div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              <div style={{ padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 9, color: "#22c55e", fontWeight: 700, fontFamily: S.mono }}>W_xh · x_t =</p>
                <VectorDisplay values={step.xPart} color="#22c55e" />
              </div>
              <div style={{ fontSize: 16, color: "#f59e0b", fontWeight: 800 }}>+</div>
              <div style={{ padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 9, color: "#a855f7", fontWeight: 700, fontFamily: S.mono }}>W_hh · h_{"{t-1}"} =</p>
                <VectorDisplay values={step.hPart} color="#a855f7" />
              </div>
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8" }}>Input contribution + recurrent contribution from memory.</p>
          </div>
        )}
        {subStep === 2 && (
          <div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              <VectorDisplay values={step.zH} label="z (pre-activation)" color="#f59e0b" />
              <div style={{ fontSize: 18, color: "#f59e0b", fontWeight: 800 }}>→ tanh →</div>
              <VectorDisplay values={step.hNew} label={`h_${curT} (new hidden)`} color="#a855f7" />
            </div>
            <div style={{ marginTop: 8, padding: "6px 12px", borderRadius: 6, background: "#064e3b", border: "1px solid #22c55e", display: "inline-block" }}>
              <span style={{ fontSize: 11, fontFamily: S.mono, color: "#22c55e", fontWeight: 700 }}>h_{curT} = tanh(z) — this IS the memory</span>
            </div>
            <HiddenStateTimeline steps={rnnSteps.slice(0, curT + 1)} size={280} />
          </div>
        )}
        {subStep === 3 && (
          <div style={{ display: "flex", gap: 16, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VectorDisplay values={step.hNew} label={`h_${curT}`} color="#a855f7" />
            <div style={{ fontSize: 18, color: "#3b82f6", fontWeight: 800, alignSelf: "center" }}>→ W_hy →</div>
            <VectorDisplay values={step.output} label={`y_${curT} (output)`} color="#3b82f6" />
            <div style={{ fontSize: 11, color: "#94a3b8", maxWidth: 200, alignSelf: "center" }}>
              <p>h_{curT} passes <b style={{ color: "#a855f7" }}>right →</b> to next timestep</p>
              <p style={{ marginTop: 4 }}>h_{curT} also produces <b style={{ color: "#3b82f6" }}>output y_{curT}</b></p>
            </div>
          </div>
        )}
      </div>

      {/* Nav */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
        <button onClick={() => { if (subStep > 0) { setSubStep(subStep - 1); } else if (curT > 0) { setCurT(curT - 1); setSubStep(3); } setAuto(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>← Prev</button>
        <span style={{ fontSize: 9, color: "#475569" }}>t={curT}, step {subStep + 1}/4</span>
        <button onClick={() => { if (subStep < 3) setSubStep(subStep + 1); else if (curT < rnnSteps.length - 1) { setCurT(curT + 1); setSubStep(0); } setAuto(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LSTM STEP-BY-STEP MODULE
   ═══════════════════════════════════════════════════════════ */
function LSTMStepModule({ sequence, hiddenSize, steps: lstmSteps }) {
  const [curT, setCurT] = useState(0);
  const [auto, setAuto] = useState(false);
  const [subStep, setSubStep] = useState(0);

  useEffect(() => { if (auto) { const t = setTimeout(() => { if (subStep < 5) setSubStep(p => p + 1); else { if (curT < lstmSteps.length - 1) { setCurT(p => p + 1); setSubStep(0); } else setAuto(false); } }, 2000); return () => clearTimeout(t); } }, [auto, curT, subStep, lstmSteps.length]);

  const step = lstmSteps[curT];
  if (!step) return null;

  const SUB = [
    { t: `📥 Timestep ${curT}: Concatenate [x_t, h_{curT > 0 ? curT - 1 : "init"}]`, d: "Input and previous hidden state are concatenated into one vector." },
    { t: "🚪 Forget Gate (f_t)", d: "σ(W_f · [h_{t-1}, x_t] + b_f). Decides what to REMOVE from cell state. Values near 0 = forget, near 1 = keep." },
    { t: "🚪 Input Gate (i_t) + Candidate (c̃_t)", d: "i_t = σ(...) decides what to UPDATE. c̃_t = tanh(...) creates new candidate values." },
    { t: "🧬 Update Cell State", d: "c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t. Old cell state is filtered by forget gate, new info added via input gate." },
    { t: "🚪 Output Gate (o_t)", d: "o_t = σ(W_o · [h_{t-1}, x_t] + b_o). Controls what part of cell state to OUTPUT." },
    { t: "📤 New Hidden State", d: "h_t = o_t ⊙ tanh(c_t). Cell state filtered through output gate becomes the hidden state." },
  ];

  return (
    <div>
      {/* Timeline */}
      <div style={{ display: "flex", gap: 3, marginBottom: 10 }}>
        {lstmSteps.map((_, t) => (
          <button key={t} onClick={() => { setCurT(t); setSubStep(0); setAuto(false); }} style={{ flex: 1, height: 22, borderRadius: 4, fontSize: 8, fontWeight: 700, background: t === curT ? "#f59e0b" : t < curT ? "#f59e0b33" : "#1e293b", color: t === curT ? "#000" : "#64748b", border: `1px solid ${t === curT ? "#f59e0b" : "#334155"}`, cursor: "pointer" }}>t={t}</button>
        ))}
        <button onClick={() => setAuto(!auto)} style={{ padding: "0 8px", borderRadius: 4, fontSize: 9, fontWeight: 700, background: auto ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{auto ? "⏸" : "▶"}</button>
      </div>

      {/* Sub-step progress */}
      <div style={{ display: "flex", gap: 2, marginBottom: 8 }}>{SUB.map((_, i) => <div key={i} style={{ flex: 1, height: 4, borderRadius: 2, background: i <= subStep ? "#f59e0b" : "#1e293b", cursor: "pointer" }} onClick={() => { setSubStep(i); setAuto(false); }} />)}</div>

      {/* Title */}
      <div style={{ padding: "8px 12px", borderRadius: 8, background: "#f59e0b11", border: "1px solid #f59e0b33", marginBottom: 10, display: "flex", alignItems: "center", gap: 8 }}>
        <div style={{ width: 22, height: 22, borderRadius: 5, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 10, fontWeight: 800, background: "#f59e0b22", color: "#f59e0b", flexShrink: 0 }}>{subStep + 1}</div>
        <div><h5 style={{ fontSize: 11, fontWeight: 700, color: "#fff", margin: 0 }}>{SUB[subStep].t}</h5><p style={{ fontSize: 10, color: "#94a3b8", margin: 0 }}>{SUB[subStep].d}</p></div>
      </div>

      {/* Visual */}
      <div style={{ padding: 12, background: "rgba(15,23,42,0.5)", borderRadius: 10, border: "1px solid #1e293b", minHeight: 160 }}>
        {subStep === 0 && (
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VectorDisplay values={step.x} label={`x_${curT}`} color="#22c55e" />
            <div style={{ fontSize: 14, color: "#f59e0b", fontWeight: 800, alignSelf: "center" }}>⊕</div>
            <VectorDisplay values={step.hPrev} label={`h_${curT > 0 ? curT - 1 : "0"}`} color="#a855f7" />
            <div style={{ fontSize: 14, color: "#64748b", fontWeight: 800, alignSelf: "center" }}>=</div>
            <VectorDisplay values={step.combined} label="combined" color="#f59e0b" />
          </div>
        )}
        {subStep === 1 && (
          <div>
            <GateDisplay name="Forget Gate (σ)" values={step.fGate} color={GATE_COLORS.forget} />
            <p style={{ fontSize: 10, color: "#94a3b8", marginTop: 8 }}>Values near <b style={{ color: "#ef4444" }}>0</b> = forget this, near <b style={{ color: "#22c55e" }}>1</b> = keep this.</p>
            <div style={{ display: "flex", gap: 2, marginTop: 6 }}>{step.fGate.slice(0, 6).map((v, i) => <div key={i} style={{ width: 40, height: 20, borderRadius: 3, background: `rgba(239,68,68,${v * 0.8 + 0.1})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 8, color: "#fff", fontFamily: S.mono, fontWeight: 700 }}>{v < 0.5 ? "FORGET" : "KEEP"}</div>)}</div>
          </div>
        )}
        {subStep === 2 && (
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
            <GateDisplay name="Input Gate (σ)" values={step.iGate} color={GATE_COLORS.input} />
            <GateDisplay name="Cell Candidate (tanh)" values={step.cCand} color={GATE_COLORS.cell} />
            <p style={{ fontSize: 10, color: "#94a3b8", width: "100%", marginTop: 4 }}>Input gate decides WHAT to add; candidate provides the new VALUES to add.</p>
          </div>
        )}
        {subStep === 3 && (
          <div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap", marginBottom: 8 }}>
              <VectorDisplay values={step.cPrev} label={`c_${curT > 0 ? curT - 1 : "0"}`} color="#64748b" maxShow={4} />
              <span style={{ fontSize: 10, color: "#ef4444", fontFamily: S.mono, fontWeight: 700 }}>⊙ f_t</span>
              <span style={{ fontSize: 12, color: "#f59e0b", fontWeight: 800 }}>+</span>
              <span style={{ fontSize: 10, color: "#22c55e", fontFamily: S.mono, fontWeight: 700 }}>i_t ⊙ c̃_t</span>
              <span style={{ fontSize: 12, color: "#64748b" }}>=</span>
              <VectorDisplay values={step.cNew} label={`c_${curT}`} color="#f59e0b" maxShow={4} />
            </div>
            <div style={{ marginTop: 6, padding: "6px 10px", borderRadius: 6, background: "#78350f44", border: "1px solid #f59e0b55", display: "inline-block" }}>
              <span style={{ fontSize: 10, fontFamily: S.mono, color: "#f59e0b" }}>Cell state = conveyor belt carrying long-term memory</span>
            </div>
            <div style={{ marginTop: 8 }}><CellStateTimeline steps={lstmSteps.slice(0, curT + 1)} size={280} /></div>
          </div>
        )}
        {subStep === 4 && <GateDisplay name="Output Gate (σ)" values={step.oGate} color={GATE_COLORS.output} />}
        {subStep === 5 && (
          <div style={{ display: "flex", gap: 12, alignItems: "flex-start", flexWrap: "wrap" }}>
            <VectorDisplay values={step.hNew} label={`h_${curT} (hidden)`} color="#a855f7" />
            <VectorDisplay values={step.output} label={`y_${curT} (output)`} color="#3b82f6" />
            <div style={{ marginTop: 6 }}>
              <HiddenStateTimeline steps={lstmSteps.slice(0, curT + 1)} size={220} />
              <CellStateTimeline steps={lstmSteps.slice(0, curT + 1)} size={220} />
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 8 }}>
        <button onClick={() => { if (subStep > 0) setSubStep(subStep - 1); else if (curT > 0) { setCurT(curT - 1); setSubStep(5); } setAuto(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>← Prev</button>
        <span style={{ fontSize: 9, color: "#475569" }}>t={curT}, step {subStep + 1}/6</span>
        <button onClick={() => { if (subStep < 5) setSubStep(subStep + 1); else if (curT < lstmSteps.length - 1) { setCurT(curT + 1); setSubStep(0); } setAuto(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   FULLSCREEN MODAL
   ═══════════════════════════════════════════════════════════ */
function FullModal({ cellType, sequence, hiddenSize, onClose }) {
  const inputSize = sequence[0]?.length || 1;
  const h0 = randVec(hiddenSize).map(() => 0);
  const c0 = randVec(hiddenSize).map(() => 0);

  const rnnData = useMemo(() => {
    if (cellType === "rnn") {
      const Wxh = randMat(hiddenSize, inputSize), Whh = randMat(hiddenSize, hiddenSize), Why = randMat(inputSize, hiddenSize);
      return rnnForward(sequence, Wxh, Whh, Why, randVec(hiddenSize), randVec(inputSize), h0);
    } else {
      const cs = inputSize + hiddenSize;
      const Wf = randMat(hiddenSize, cs), Wi = randMat(hiddenSize, cs), Wo = randMat(hiddenSize, cs), Wc = randMat(hiddenSize, cs);
      const Why = randMat(inputSize, hiddenSize);
      return lstmForward(sequence, Wf, Wi, Wo, Wc, randVec(hiddenSize), randVec(hiddenSize), randVec(hiddenSize), randVec(hiddenSize), Why, randVec(inputSize), h0, c0);
    }
  }, [cellType, sequence, hiddenSize, inputSize]);

  const info = cellType === "rnn" ? { icon: "🔄", c: "#3b82f6", t: "Simple RNN Cell" } : { icon: "🧬", c: "#f59e0b", t: "LSTM Cell" };

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(2,6,23,0.95)", backdropFilter: "blur(16px)", overflow: "auto" }}>
      <div style={{ maxWidth: 1000, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, background: `${info.c}22` }}>{info.icon}</div>
            <div>
              <h2 style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>{info.t} — Step-by-Step</h2>
              <p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>{sequence.length} timesteps · Hidden: {hiddenSize} · Input: {inputSize}</p>
            </div>
          </div>
          <button onClick={onClose} style={{ width: 36, height: 36, borderRadius: 8, fontSize: 18, color: "#64748b", background: "#1e293b", border: "1px solid #334155", cursor: "pointer" }}>×</button>
        </div>
        <div style={{ background: "rgba(15,23,42,0.6)", borderRadius: 14, border: "1px solid #1e293b", padding: 20 }}>
          {cellType === "rnn" ? <RNNStepModule sequence={sequence} hiddenSize={hiddenSize} steps={rnnData} /> : <LSTMStepModule sequence={sequence} hiddenSize={hiddenSize} steps={rnnData} />}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN RNN/LSTM LAB
   ═══════════════════════════════════════════════════════════ */
export default function RNNLab() {
  const [seqId, setSeqId] = useState("sine");
  const [seqLen, setSeqLen] = useState(10);
  const [cellType, setCellType] = useState("rnn");
  const [hiddenSize, setHiddenSize] = useState(4);
  const [modal, setModal] = useState(false);

  const sequence = useMemo(() => generateSequence(seqId, seqLen), [seqId, seqLen]);
  const inputSize = sequence[0]?.length || 1;

  // Run forward pass
  const h0 = useMemo(() => Array(hiddenSize).fill(0), [hiddenSize]);
  const c0 = useMemo(() => Array(hiddenSize).fill(0), [hiddenSize]);
  const { steps, weights } = useMemo(() => {
    const Wxh = randMat(hiddenSize, inputSize), Whh = randMat(hiddenSize, hiddenSize), Why = randMat(inputSize, hiddenSize);
    if (cellType === "rnn") {
      return { steps: rnnForward(sequence, Wxh, Whh, Why, randVec(hiddenSize), randVec(inputSize), h0), weights: { Wxh, Whh, Why } };
    } else {
      const cs = inputSize + hiddenSize;
      const Wf = randMat(hiddenSize, cs), Wi = randMat(hiddenSize, cs), Wo = randMat(hiddenSize, cs), Wc = randMat(hiddenSize, cs);
      return { steps: lstmForward(sequence, Wf, Wi, Wo, Wc, randVec(hiddenSize), randVec(hiddenSize), randVec(hiddenSize), randVec(hiddenSize), Why, randVec(inputSize), h0, c0), weights: { Wf, Wi, Wo, Wc, Why } };
    }
  }, [cellType, sequence, hiddenSize, inputSize, h0, c0]);

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #0a1628 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#f59e0b,#ef4444)" }}>🔄</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>RNN / LSTM Lab</h1>
        <span style={{ fontSize: 10 }}>Interactive Recurrent Network Simulator</span>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 14 }}>
          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Sequence selector */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>📊 Input Sequence</h3>
              <select value={seqId} onChange={e => setSeqId(e.target.value)} style={{ width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}>
                {SEQUENCES.map(s => <option key={s.id} value={s.id}>{s.name} — {s.desc}</option>)}
              </select>
              <SequencePlot data={sequence} size={210} height={80} />
              <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
                <span style={{ fontSize: 9, color: "#475569" }}>{seqLen} timesteps</span>
                <TensorShape shape={[seqLen, inputSize]} color="#22c55e" />
              </div>
              <label style={{ fontSize: 10, color: "#64748b", display: "block", marginTop: 6 }}>Length:
                <input type="range" min={4} max={20} value={seqLen} onChange={e => setSeqLen(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
              </label>
            </div>

            {/* Cell type + params */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>⚙️ Cell Configuration</h3>
              <div style={{ display: "flex", gap: 4, marginBottom: 8 }}>
                {["rnn", "lstm"].map(ct => (
                  <button key={ct} onClick={() => setCellType(ct)} style={{ flex: 1, padding: "6px 0", borderRadius: 6, fontSize: 11, fontWeight: 700, background: cellType === ct ? (ct === "rnn" ? "#3b82f6" : "#f59e0b") : "#0f172a", color: cellType === ct ? "#fff" : "#64748b", border: `1px solid ${cellType === ct ? (ct === "rnn" ? "#3b82f6" : "#f59e0b") : "#334155"}`, cursor: "pointer" }}>
                    {ct === "rnn" ? "🔄 Simple RNN" : "🧬 LSTM"}
                  </button>
                ))}
              </div>
              <label style={{ fontSize: 10, color: "#64748b", display: "block" }}>Hidden Size: <b style={{ color: "#fff" }}>{hiddenSize}</b>
                <input type="range" min={2} max={16} value={hiddenSize} onChange={e => setHiddenSize(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
              </label>
              <div style={{ marginTop: 8, padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 9, color: "#64748b" }}>Input: <TensorShape shape={[seqLen, inputSize]} color="#22c55e" /></p>
                <p style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>Hidden: <TensorShape shape={[hiddenSize]} color="#a855f7" /></p>
                {cellType === "lstm" && <p style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>Cell: <TensorShape shape={[hiddenSize]} color="#f59e0b" /></p>}
                <p style={{ fontSize: 9, color: "#64748b", marginTop: 2 }}>Output: <TensorShape shape={[seqLen, inputSize]} color="#3b82f6" /></p>
                <p style={{ fontSize: 9, color: "#64748b", marginTop: 4 }}>Params: <b style={{ color: "#fff" }}>{cellType === "rnn" ? hiddenSize * (inputSize + hiddenSize + 1) + inputSize * (hiddenSize + 1) : 4 * hiddenSize * (inputSize + hiddenSize + 1) + inputSize * (hiddenSize + 1)}</b></p>
              </div>
            </div>

            {/* Cell type info */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>{cellType === "rnn" ? "🔄 Simple RNN" : "🧬 LSTM"}</h3>
              {cellType === "rnn" ? (
                <div style={{ fontSize: 10, color: "#94a3b8" }}>
                  <p style={{ marginBottom: 4 }}><b style={{ color: "#fff" }}>h_t = tanh(W_xh·x_t + W_hh·h_{"{t-1}"} + b)</b></p>
                  <p>Single hidden state carries all memory. Simple but suffers from vanishing gradients on long sequences.</p>
                </div>
              ) : (
                <div style={{ fontSize: 10, color: "#94a3b8" }}>
                  <p style={{ marginBottom: 4 }}><b style={{ color: "#ef4444" }}>Forget:</b> f_t = σ(W_f·[h,x] + b_f)</p>
                  <p style={{ marginBottom: 4 }}><b style={{ color: "#22c55e" }}>Input:</b> i_t = σ(W_i·[h,x] + b_i)</p>
                  <p style={{ marginBottom: 4 }}><b style={{ color: "#f59e0b" }}>Cell:</b> c_t = f_t⊙c + i_t⊙tanh(...)</p>
                  <p><b style={{ color: "#3b82f6" }}>Output:</b> o_t = σ(W_o·[h,x] + b_o)</p>
                </div>
              )}
            </div>
          </div>

          {/* Main area */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Action bar */}
            <div style={{ borderRadius: 12, padding: "8px 12px", background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b", display: "flex", alignItems: "center", gap: 8 }}>
              <span style={{ fontSize: 11, fontWeight: 600, color: "#64748b" }}>{cellType === "rnn" ? "🔄 RNN" : "🧬 LSTM"} · {seqLen} steps · h={hiddenSize}</span>
              <button onClick={() => setModal(true)} style={{ marginLeft: "auto", padding: "4px 12px", borderRadius: 6, fontSize: 11, fontWeight: 700, background: cellType === "rnn" ? "#3b82f618" : "#f59e0b18", color: cellType === "rnn" ? "#3b82f6" : "#f59e0b", border: `1px solid ${cellType === "rnn" ? "#3b82f633" : "#f59e0b33"}`, cursor: "pointer" }}>⛶ Step-by-Step {cellType.toUpperCase()} Forward Pass</button>
            </div>

            {/* Sequence + outputs overview */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>📈 Sequence Processing Overview</p>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                <SequencePlot data={sequence} size={280} height={90} label="Input Sequence x_t" />
                <SequencePlot data={steps.map(s => s.output)} size={280} height={90} label="Output Predictions y_t" />
              </div>
            </div>

            {/* Hidden state evolution */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🧠 Hidden State Evolution</p>
              <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                {Array.from({ length: Math.min(hiddenSize, 4) }).map((_, i) => (
                  <HiddenStateTimeline key={i} steps={steps} hiddenIdx={i} size={220} />
                ))}
              </div>
            </div>

            {/* Cell state (LSTM only) */}
            {cellType === "lstm" && (
              <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🧬 Cell State (Long-Term Memory)</p>
                <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
                  {Array.from({ length: Math.min(hiddenSize, 4) }).map((_, i) => (
                    <CellStateTimeline key={i} steps={steps} cellIdx={i} size={220} />
                  ))}
                </div>
              </div>
            )}

            {/* Per-timestep cards */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>⏱️ Timestep Details</p>
              <div style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 6 }}>
                {steps.map((st, t) => (
                  <div key={t} style={{ flexShrink: 0, width: 140, padding: 8, borderRadius: 8, background: "#0f172a", border: "1px solid #1e293b" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span style={{ fontSize: 10, fontWeight: 700, color: cellType === "rnn" ? "#3b82f6" : "#f59e0b" }}>t={t}</span>
                      <TensorShape shape={[hiddenSize]} color="#a855f7" />
                    </div>
                    <VectorDisplay values={st.x} label="x" color="#22c55e" maxShow={3} />
                    <VectorDisplay values={st.hNew} label="h" color="#a855f7" maxShow={3} />
                    {cellType === "lstm" && st.cNew && <VectorDisplay values={st.cNew} label="c" color="#f59e0b" maxShow={3} />}
                    {cellType === "lstm" && (
                      <div style={{ display: "flex", gap: 2, marginTop: 4 }}>
                        <div style={{ width: 8, height: 8, borderRadius: 4, background: GATE_COLORS.forget, opacity: st.fGate ? Math.max(...st.fGate) : 0.5 }} title="forget" />
                        <div style={{ width: 8, height: 8, borderRadius: 4, background: GATE_COLORS.input, opacity: st.iGate ? Math.max(...st.iGate) : 0.5 }} title="input" />
                        <div style={{ width: 8, height: 8, borderRadius: 4, background: GATE_COLORS.output, opacity: st.oGate ? Math.max(...st.oGate) : 0.5 }} title="output" />
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>

      {modal && <FullModal cellType={cellType} sequence={sequence} hiddenSize={hiddenSize} onClose={() => setModal(false)} />}
    </div>
  );
}
