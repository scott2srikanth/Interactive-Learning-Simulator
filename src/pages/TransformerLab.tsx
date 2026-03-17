import { useState, useEffect, useRef, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════
   MATH
   ═══════════════════════════════════════════════════════════ */
function softmaxRow(v) { if(!v?.length) return []; const mx = Math.max(...v); const e = v.map(x => Math.exp(x - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / (s || 1)); }
function matMul(a, b) { if(!a?.length || !b?.length) return []; return a.map(ar => b[0].map((_, j) => ar.reduce((s, v, k) => s + v * (b[k]?.[j] || 0), 0))); }
function transpose(m) { if(!m?.length || !m[0]?.length) return []; return m[0].map((_, i) => m.map(r => r[i])); }
function randMat(r, c, s = 0.3) { return Array(r).fill(0).map(() => Array(c).fill(0).map(() => +((Math.random() - 0.5) * s * 2).toFixed(3))); }
function layerNorm(m) { return m.map(row => { const mu = row.reduce((s, v) => s + v, 0) / row.length; const va = row.reduce((s, v) => s + (v - mu) ** 2, 0) / row.length; return row.map(v => +((v - mu) / Math.sqrt(va + 1e-5)).toFixed(3)); }); }
function addMat(a, b) { return a.map((r, i) => r.map((v, j) => +(v + (b[i]?.[j] || 0)).toFixed(3))); }
function attention(Q, K, V) { const dk = K[0]?.length || 1; const sc = matMul(Q, transpose(K)).map(r => r.map(v => +(v / Math.sqrt(dk)).toFixed(3))); const w = sc.map(softmaxRow); return { scores: sc, weights: w, output: matMul(w, V) }; }
function posEnc(len, d) { return Array.from({ length: len }).map((_, p) => Array(d).fill(0).map((_, i) => +(i % 2 === 0 ? Math.sin(p / Math.pow(10000, i / d)) : Math.cos(p / Math.pow(10000, (i - 1) / d))).toFixed(3))); }

const DATASET = [
  { src: "I love cats", tgt: "J'aime les chats", sT: ["I", "love", "cats"], tT: ["J'", "aime", "les", "chats"] },
  { src: "The sun is bright", tgt: "Le soleil est brillant", sT: ["The", "sun", "is", "bright"], tT: ["Le", "soleil", "est", "brillant"] },
  { src: "She reads books", tgt: "Elle lit des livres", sT: ["She", "reads", "books"], tT: ["Elle", "lit", "des", "livres"] },
];
const S = { mono: "'JetBrains Mono', monospace", sans: "'DM Sans', system-ui, sans-serif" };
const dModel = 6;

/* ═══════════════════════════════════════════════════════════
   SMALL VISUAL HELPERS
   ═══════════════════════════════════════════════════════════ */
function MatrixView({ data, label, color, rowLabels, cellSize = 28 }) {
  if (!data?.length) return null;
  const rows = data.length, cols = data[0]?.length || 0;
  return (
    <div style={{ marginBottom: 8 }}>
      {label && <p style={{ fontSize: 9, fontFamily: S.mono, fontWeight: 700, color, marginBottom: 3 }}>{label} [{rows}×{cols}]</p>}
      <div style={{ display: "inline-grid", gridTemplateColumns: rowLabels ? `34px repeat(${cols}, ${cellSize}px)` : `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.map((row, i) => (
          <>{rowLabels && <span style={{ fontSize: 7, fontFamily: S.mono, color: "#64748b", alignSelf: "center", textAlign: "right", paddingRight: 2 }}>{rowLabels[i]?.slice(0, 4)}</span>}
          {row.map((v, j) => (
            <div key={`${i}-${j}`} style={{ width: cellSize, height: cellSize - 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 700, fontFamily: S.mono, borderRadius: 3, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 150 + 30).toString(16).padStart(2, "0")}`, color: "#fff" }}>{v.toFixed(2)}</div>
          ))}</>
        ))}
      </div>
    </div>
  );
}

function Heatmap({ weights, rL, cL, size = 180, label }) {
  const ref = useRef(null);
  const n = rL?.length || 0, m = cL?.length || 0;
  useEffect(() => {
    const c = ref.current; if (!c || !n || !m) return;
    c.width = size; c.height = size; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    const off = 34, cw = (size - off) / m, ch = (size - off) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const v = weights[i]?.[j] || 0;
      ctx.fillStyle = `rgba(168,85,247,${v * 0.85 + 0.05})`;
      ctx.fillRect(off + j * cw, off + i * ch, cw - 1, ch - 1);
      if (n <= 6) { ctx.fillStyle = v > 0.2 ? "#fff" : "#64748b"; ctx.font = `bold ${Math.min(9, cw * 0.3)}px monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText((v * 100).toFixed(0) + "%", off + j * cw + cw / 2, off + i * ch + ch / 2); }
    }
    ctx.fillStyle = "#94a3b8"; ctx.font = "bold 7px monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) ctx.fillText(rL[i].slice(0, 4), off - 2, off + i * ch + ch / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    for (let j = 0; j < m; j++) { ctx.save(); ctx.translate(off + j * cw + cw / 2, off - 2); ctx.rotate(-0.4); ctx.fillText(cL[j].slice(0, 4), 0, 0); ctx.restore(); }
  }, [weights, n, m, size, rL, cL]);
  return <div>{label && <p style={{ fontSize: 9, fontFamily: S.mono, fontWeight: 700, color: "#a855f7", marginBottom: 3 }}>{label}</p>}<canvas ref={ref} style={{ width: size, height: size, borderRadius: 8, border: "1px solid #334155" }} /></div>;
}

/* ═══════════════════════════════════════════════════════════
   BLOCK DETAIL — animated step-by-step
   ═══════════════════════════════════════════════════════════ */
function BlockDetail({ blockType, blockName, tokens, emb, crossTokens, crossEmb }) {
  const [sub, setSub] = useState(0);
  const [auto, setAuto] = useState(false);
  const Wq = useMemo(() => randMat(dModel, 3), []);
  const Wk = useMemo(() => randMat(dModel, 3), []);
  const Wv = useMemo(() => randMat(dModel, 3), []);
  const Q = matMul(emb, Wq), sK = matMul(emb, Wk), sV = matMul(emb, Wv);
  const selfA = attention(Q, sK, sV);
  const norm1 = layerNorm(addMat(emb, selfA.output));
  const cK = crossEmb ? matMul(crossEmb, Wk) : null;
  const cV = crossEmb ? matMul(crossEmb, Wv) : null;
  const crossA = cK ? attention(matMul(norm1, Wq), cK, cV) : null;
  const norm2 = crossA ? layerNorm(addMat(norm1, crossA.output)) : norm1;
  const ffW1 = useMemo(() => randMat(dModel, dModel * 2), []);
  const ffW2 = useMemo(() => randMat(dModel * 2, dModel), []);
  const ffH = matMul(norm2, ffW1).map(r => r.map(v => Math.max(0, v)));
  const ffO = matMul(ffH, ffW2);
  const outM = layerNorm(addMat(norm2, ffO));

  const isDec = blockType === "decoder";
  const steps = isDec ? [
    { t: "① Masked Multi-Head Attention", d: "Target tokens attend only to past tokens (causal mask prevents seeing the future).", c: "#f59e0b" },
    { t: "② Add & Norm", d: "Residual: masked_attn(x) + x → Layer Normalize.", c: "#eab308" },
    { t: "③ Multi-Head Attention (Cross)", d: "Decoder Queries attend to Encoder Keys & Values. This connects encoder output to decoder!", c: "#f97316" },
    { t: "④ Add & Norm", d: "Residual: cross_attn(x) + x → Layer Normalize.", c: "#eab308" },
    { t: "⑤ Feed Forward", d: `Two linear layers: ${dModel}→${dModel * 2} (ReLU) →${dModel}. Per-token.`, c: "#60a5fa" },
    { t: "⑥ Add & Norm → Output", d: "Final residual + norm. Output goes to next decoder block or to Linear→Softmax.", c: "#22c55e" },
  ] : [
    { t: "① Multi-Head Attention", d: "Each source token attends to ALL source tokens (bidirectional, no mask).", c: "#f59e0b" },
    { t: "② Add & Norm", d: "Residual: self_attn(x) + x → Layer Normalize.", c: "#eab308" },
    { t: "③ Feed Forward", d: `Two linear layers: ${dModel}→${dModel * 2} (ReLU) →${dModel}. Per-token.`, c: "#60a5fa" },
    { t: "④ Add & Norm → Output", d: "Final residual + norm. Output sent to next encoder block or to decoder.", c: "#22c55e" },
  ];

  useEffect(() => { if (auto && sub < steps.length - 1) { const t = setTimeout(() => setSub(p => p + 1), 2500); return () => clearTimeout(t); } else setAuto(false); }, [auto, sub, steps.length]);

  return (
    <div style={{ padding: 14, borderRadius: 14, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <h3 style={{ fontSize: 14, fontWeight: 800, color: "#fff", margin: 0 }}>{blockName}</h3>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => setAuto(!auto)} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: auto ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{auto ? "⏸" : "▶"}</button>
          <button onClick={() => setSub(0)} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
        </div>
      </div>
      <div style={{ display: "flex", gap: 2, marginBottom: 8 }}>{steps.map((s, i) => <button key={i} onClick={() => { setSub(i); setAuto(false); }} style={{ flex: 1, height: 5, borderRadius: 3, background: i <= sub ? s.c : "#1e293b", border: "none", cursor: "pointer" }} />)}</div>
      <div style={{ padding: "8px 12px", borderRadius: 8, background: `${steps[sub].c}15`, border: `1px solid ${steps[sub].c}33`, marginBottom: 10 }}>
        <p style={{ fontSize: 12, fontWeight: 700, color: steps[sub].c, margin: 0 }}>{steps[sub].t}</p>
        <p style={{ fontSize: 10, color: "#94a3b8", margin: 0 }}>{steps[sub].d}</p>
      </div>
      <div style={{ minHeight: 180 }}>
        {/* Encoder steps */}
        {!isDec && sub === 0 && <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}><Heatmap weights={selfA.weights} rL={tokens} cL={tokens} size={170} label="Self-Attention" /><div><MatrixView data={Q} label="Q" color="#ef4444" rowLabels={tokens} cellSize={24} /><MatrixView data={sK} label="K" color="#22c55e" rowLabels={tokens} cellSize={24} /></div></div>}
        {!isDec && sub === 1 && <MatrixView data={norm1} label="After Add & LayerNorm" color="#eab308" rowLabels={tokens} />}
        {!isDec && sub === 2 && <div><MatrixView data={ffH.map(r => r.slice(0, 6))} label="FFN Hidden (ReLU)" color="#60a5fa" rowLabels={tokens} cellSize={24} /><MatrixView data={ffO} label="FFN Output" color="#60a5fa" rowLabels={tokens} /></div>}
        {!isDec && sub === 3 && <div><MatrixView data={outM} label="Encoder Output" color="#22c55e" rowLabels={tokens} /><p style={{ fontSize: 10, color: "#22c55e", marginTop: 8, fontWeight: 700 }}>✓ Sent to Decoder cross-attention</p></div>}
        {/* Decoder steps */}
        {isDec && sub === 0 && <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}><Heatmap weights={selfA.weights} rL={tokens} cL={tokens} size={170} label="Masked Self-Attention" /><p style={{ fontSize: 10, color: "#94a3b8", maxWidth: 180 }}>Each token attends only to itself and <b style={{ color: "#f59e0b" }}>tokens before it</b>. Future tokens are masked to prevent cheating.</p></div>}
        {isDec && sub === 1 && <MatrixView data={norm1} label="After Masked Attn + Residual + Norm" color="#eab308" rowLabels={tokens} />}
        {isDec && sub === 2 && crossA && <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}><Heatmap weights={crossA.weights} rL={tokens} cL={crossTokens || []} size={170} label="Cross-Attention (Dec→Enc)" /><div style={{ fontSize: 10, color: "#94a3b8", maxWidth: 200 }}><p><b style={{ color: "#f97316" }}>Translation happens here!</b></p><p style={{ marginTop: 4 }}>Decoder <b style={{ color: "#ef4444" }}>Q</b> attends to Encoder <b style={{ color: "#22c55e" }}>K/V</b>.</p><p style={{ marginTop: 4 }}>e.g. "chats" → "cats"</p></div></div>}
        {isDec && sub === 3 && <MatrixView data={norm2} label="After Cross-Attn + Residual + Norm" color="#eab308" rowLabels={tokens} />}
        {isDec && sub === 4 && <MatrixView data={ffO} label="FFN Output" color="#60a5fa" rowLabels={tokens} />}
        {isDec && sub === 5 && <div><MatrixView data={outM} label="Decoder Output" color="#22c55e" rowLabels={tokens} /><p style={{ fontSize: 10, color: "#22c55e", marginTop: 8, fontWeight: 700 }}>→ Linear → Softmax → Predicted tokens</p></div>}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ARCHITECTURE SVG — "Attention Is All You Need" diagram
   ═══════════════════════════════════════════════════════════ */
function ArchBlock({ x, y, w, h, label, color, selected, onClick }) {
  return (
    <g onClick={onClick} style={{ cursor: "pointer" }}>
      <rect x={x} y={y} width={w} height={h} rx={6} fill={selected ? `${color}` : `${color}44`} stroke={selected ? "#fff" : color} strokeWidth={selected ? 2.5 : 1.5} />
      <text x={x + w / 2} y={y + h / 2} textAnchor="middle" dominantBaseline="central" fill={selected ? "#fff" : "#e2e8f0"} fontSize={10} fontWeight={700} fontFamily="DM Sans, sans-serif">{label}</text>
    </g>
  );
}
function Arrow({ x1, y1, x2, y2 }) {
  return <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#475569" strokeWidth={1.5} markerEnd="url(#arrowhead)" />;
}
function CurvedArrow({ x1, y1, x2, y2, cx, cy }) {
  return <path d={`M${x1},${y1} Q${cx},${cy} ${x2},${y2}`} fill="none" stroke="#475569" strokeWidth={1.5} markerEnd="url(#arrowhead)" />;
}
function ResidualArrow({ x1, y1, x2, y2, side = "right", bw }) {
  const dx = side === "right" ? bw / 2 + 14 : -(bw / 2 + 14);
  return <path d={`M${x1 + dx},${y1} L${x1 + dx + (side === "right" ? 10 : -10)},${y1} L${x2 + dx + (side === "right" ? 10 : -10)},${y2} L${x2 + dx},${y2}`} fill="none" stroke="#64748b" strokeWidth={1} strokeDasharray="3,2" markerEnd="url(#arrowhead)" />;
}

function ArchDiagram({ selected, onSelect }) {
  const bw = 130, bh = 30, gap = 6;
  // Encoder column (left)
  const ex = 30, ey0 = 420;
  // Decoder column (right)
  const dx = 220, dy0 = 420;

  const encBlocks = [
    { id: "src_embed", label: "Input Embedding", y: ey0, c: "#fca5a5" },
    { id: "enc_mha", label: "Multi-Head Attention", y: ey0 - 80, c: "#fdba74" },
    { id: "enc_an1", label: "Add & Norm", y: ey0 - 120, c: "#fde68a" },
    { id: "enc_ff", label: "Feed Forward", y: ey0 - 160, c: "#93c5fd" },
    { id: "enc_an2", label: "Add & Norm", y: ey0 - 200, c: "#fde68a" },
  ];
  const decBlocks = [
    { id: "tgt_embed", label: "Output Embedding", y: dy0, c: "#fca5a5" },
    { id: "dec_mmha", label: "Masked Multi-Head Attn", y: dy0 - 80, c: "#fdba74" },
    { id: "dec_an1", label: "Add & Norm", y: dy0 - 120, c: "#fde68a" },
    { id: "dec_mha", label: "Multi-Head Attention", y: dy0 - 160, c: "#fdba74" },
    { id: "dec_an2", label: "Add & Norm", y: dy0 - 200, c: "#fde68a" },
    { id: "dec_ff", label: "Feed Forward", y: dy0 - 240, c: "#93c5fd" },
    { id: "dec_an3", label: "Add & Norm", y: dy0 - 280, c: "#fde68a" },
    { id: "dec_linear", label: "Linear", y: dy0 - 320, c: "#c4b5fd" },
    { id: "dec_softmax", label: "Softmax", y: dy0 - 356, c: "#86efac" },
  ];

  return (
    <svg viewBox="0 0 400 500" style={{ width: "100%", maxWidth: 400, height: "auto" }}>
      <defs><marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><polygon points="0 0, 8 3, 0 6" fill="#475569" /></marker></defs>

      {/* Labels */}
      <text x={ex + bw / 2} y={468} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={700}>Inputs</text>
      <text x={dx + bw / 2} y={468} textAnchor="middle" fill="#94a3b8" fontSize={10} fontWeight={700}>Outputs (shifted right)</text>

      {/* Positional encoding circles */}
      <circle cx={ex + bw / 2 - 15} cy={ey0 - 30} r={8} fill="none" stroke="#94a3b8" strokeWidth={1} />
      <text x={ex + bw / 2 - 15} y={ey0 - 27} textAnchor="middle" fill="#94a3b8" fontSize={7}>~</text>
      <circle cx={ex + bw / 2 + 5} cy={ey0 - 30} r={8} fill="none" stroke="#94a3b8" strokeWidth={1} />
      <text x={ex + bw / 2 + 5} y={ey0 - 27} textAnchor="middle" fill="#94a3b8" fontSize={9}>⊕</text>
      <text x={ex - 8} y={ey0 - 27} fill="#94a3b8" fontSize={7} textAnchor="end">Positional</text>
      <text x={ex - 8} y={ey0 - 18} fill="#94a3b8" fontSize={7} textAnchor="end">Encoding</text>

      <circle cx={dx + bw / 2 + 15} cy={dy0 - 30} r={8} fill="none" stroke="#94a3b8" strokeWidth={1} />
      <text x={dx + bw / 2 + 15} y={dy0 - 27} textAnchor="middle" fill="#94a3b8" fontSize={7}>~</text>
      <circle cx={dx + bw / 2 + 35} cy={dy0 - 30} r={8} fill="none" stroke="#94a3b8" strokeWidth={1} />
      <text x={dx + bw / 2 + 35} y={dy0 - 27} textAnchor="middle" fill="#94a3b8" fontSize={9}>⊕</text>
      <text x={dx + bw + 48} y={dy0 - 27} fill="#94a3b8" fontSize={7}>Positional</text>
      <text x={dx + bw + 48} y={dy0 - 18} fill="#94a3b8" fontSize={7}>Encoding</text>

      {/* Encoder Nx box */}
      <rect x={ex - 8} y={ey0 - 210} width={bw + 16} height={150} rx={8} fill="none" stroke="#64748b" strokeWidth={1.5} strokeDasharray="4,3" />
      <text x={ex - 14} y={ey0 - 130} fill="#94a3b8" fontSize={11} fontWeight={700} textAnchor="end">N×</text>

      {/* Decoder Nx box */}
      <rect x={dx - 8} y={dy0 - 296} width={bw + 16} height={230} rx={8} fill="none" stroke="#64748b" strokeWidth={1.5} strokeDasharray="4,3" />
      <text x={dx + bw + 24} y={dy0 - 175} fill="#94a3b8" fontSize={11} fontWeight={700}>N×</text>

      {/* Input arrows */}
      <Arrow x1={ex + bw / 2} y1={458} x2={ex + bw / 2} y2={ey0 + bh} />
      <Arrow x1={dx + bw / 2} y1={458} x2={dx + bw / 2} y2={dy0 + bh} />

      {/* Embedding → PE arrows */}
      <Arrow x1={ex + bw / 2} y1={ey0} x2={ex + bw / 2} y2={ey0 - 40} />
      <Arrow x1={dx + bw / 2} y1={dy0} x2={dx + bw / 2} y2={dy0 - 40} />
      {/* PE → MHA arrows */}
      <Arrow x1={ex + bw / 2} y1={ey0 - 45} x2={ex + bw / 2} y2={ey0 - 80 + bh} />
      <Arrow x1={dx + bw / 2} y1={dy0 - 45} x2={dx + bw / 2} y2={dy0 - 80 + bh} />

      {/* Encoder vertical arrows */}
      {[0, 1, 2].map(i => <Arrow key={i} x1={ex + bw / 2} y1={encBlocks[i + 1].y} x2={ex + bw / 2} y2={encBlocks[i + 2].y + bh} />)}
      {/* Encoder residual arrows */}
      <ResidualArrow x1={ex + bw / 2} y1={ey0 - 45} x2={ex + bw / 2} y2={encBlocks[2].y + bh / 2} bw={bw} />
      <ResidualArrow x1={ex + bw / 2} y1={encBlocks[2].y} x2={ex + bw / 2} y2={encBlocks[4].y + bh / 2} bw={bw} />

      {/* Decoder vertical arrows */}
      {[0, 1, 2, 3, 4, 5, 6].map(i => decBlocks[i + 1] && <Arrow key={i} x1={dx + bw / 2} y1={decBlocks[i].y - (i === 0 ? 40 : 0)} x2={dx + bw / 2} y2={(decBlocks[i + 1] || decBlocks[i]).y + bh} />)}

      {/* Encoder → Decoder cross-attention arrow */}
      <path d={`M${ex + bw},${encBlocks[4].y + bh / 2} L${dx},${decBlocks[3].y + bh / 2}`} fill="none" stroke="#f59e0b" strokeWidth={2} markerEnd="url(#arrowhead)" />
      <text x={(ex + bw + dx) / 2} y={decBlocks[3].y + bh / 2 - 6} textAnchor="middle" fill="#f59e0b" fontSize={7} fontWeight={700}>K, V from Encoder</text>

      {/* Output probs */}
      <text x={dx + bw / 2} y={decBlocks[8].y - 12} textAnchor="middle" fill="#22c55e" fontSize={10} fontWeight={700}>Output Probabilities</text>
      <Arrow x1={dx + bw / 2} y1={decBlocks[7].y} x2={dx + bw / 2} y2={decBlocks[8].y + bh} />

      {/* Encoder blocks */}
      {encBlocks.map(b => <ArchBlock key={b.id} x={ex} y={b.y} w={bw} h={bh} label={b.label} color={b.c} selected={selected === b.id} onClick={() => onSelect(b.id)} />)}

      {/* Decoder blocks */}
      {decBlocks.map(b => <ArchBlock key={b.id} x={dx} y={b.y} w={bw} h={bh} label={b.label} color={b.c} selected={selected === b.id} onClick={() => onSelect(b.id)} />)}
    </svg>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN LAB
   ═══════════════════════════════════════════════════════════ */
export default function TransformerLab() {
  const [dIdx, setDIdx] = useState(0);
  const [sel, setSel] = useState(null);
  const d = DATASET[dIdx];
  const sE = useMemo(() => addMat(d.sT.map((t, i) => Array(dModel).fill(0).map((_, dd) => +(Math.sin(t.charCodeAt(0) * 0.25 + dd * 1.4 + i * 0.9) * 0.6).toFixed(3))), posEnc(d.sT.length, dModel)), [d]);
  const tE = useMemo(() => addMat(d.tT.map((t, i) => Array(dModel).fill(0).map((_, dd) => +(Math.sin(t.charCodeAt(0) * 0.3 + dd * 1.2 + i * 0.7) * 0.6).toFixed(3))), posEnc(d.tT.length, dModel)), [d]);

  const renderDetail = () => {
    if (!sel) return <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", minHeight: 300 }}><div style={{ textAlign: "center" }}><p style={{ fontSize: 36, marginBottom: 8 }}>👈</p><p style={{ fontSize: 13, color: "#64748b", fontWeight: 600 }}>Click any block in the diagram</p><p style={{ fontSize: 10, color: "#475569" }}>to see its internals step-by-step</p></div></div>;

    if (sel === "src_embed") return <div style={{ padding: 14, borderRadius: 14, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}><h3 style={{ fontSize: 14, fontWeight: 800, color: "#fca5a5", marginBottom: 10 }}>📥 Input Embedding + Positional Encoding</h3><MatrixView data={d.sT.map((t, i) => Array(dModel).fill(0).map((_, dd) => +(Math.sin(t.charCodeAt(0) * 0.25 + dd * 1.4 + i * 0.9) * 0.6).toFixed(3)))} label="Token Embeddings" color="#fca5a5" rowLabels={d.sT} /><MatrixView data={posEnc(d.sT.length, dModel)} label="+ Positional Encoding" color="#06b6d4" rowLabels={d.sT} /><MatrixView data={sE} label="= Final Input" color="#f59e0b" rowLabels={d.sT} /></div>;

    if (sel === "tgt_embed") return <div style={{ padding: 14, borderRadius: 14, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}><h3 style={{ fontSize: 14, fontWeight: 800, color: "#fca5a5", marginBottom: 10 }}>📥 Output Embedding + Positional Encoding</h3><MatrixView data={tE} label="Final Target Input" color="#3b82f6" rowLabels={d.tT} /><p style={{ fontSize: 10, color: "#94a3b8", marginTop: 6 }}>Target tokens are shifted right: during training, the decoder sees the previous tokens to predict the next one.</p></div>;

    if (sel === "enc_mha" || sel === "enc_an1" || sel === "enc_ff" || sel === "enc_an2")
      return <BlockDetail blockType="encoder" blockName="Encoder Block" tokens={d.sT} emb={sE} />;

    if (sel === "dec_mmha" || sel === "dec_an1" || sel === "dec_mha" || sel === "dec_an2" || sel === "dec_ff" || sel === "dec_an3")
      return <BlockDetail blockType="decoder" blockName="Decoder Block" tokens={d.tT} emb={tE} crossTokens={d.sT} crossEmb={sE} />;

    if (sel === "dec_linear") return <div style={{ padding: 14, borderRadius: 14, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}><h3 style={{ fontSize: 14, fontWeight: 800, color: "#c4b5fd", marginBottom: 10 }}>Linear Projection</h3><p style={{ fontSize: 11, color: "#94a3b8" }}>Projects each decoder output vector from d_model={dModel} to vocabulary size (50,257 for GPT-2). This creates a score for every possible next token.</p><MatrixView data={d.tT.map(() => Array(6).fill(0).map(() => +(Math.random() * 4 - 2).toFixed(2)))} label="Logits (showing 6 of vocab)" color="#c4b5fd" rowLabels={d.tT} /></div>;

    if (sel === "dec_softmax") return <div style={{ padding: 14, borderRadius: 14, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}><h3 style={{ fontSize: 14, fontWeight: 800, color: "#86efac", marginBottom: 10 }}>Softmax → Output Probabilities</h3>{d.tT.map((t, i) => <div key={i} style={{ marginBottom: 6, padding: 6, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b" }}><div style={{ display: "flex", justifyContent: "space-between" }}><span style={{ fontSize: 11, fontFamily: S.mono, color: "#a855f7" }}>Position {i}</span><span style={{ fontSize: 12, fontFamily: S.mono, color: "#22c55e", fontWeight: 800 }}>→ "{t}" ✓</span></div><div style={{ marginTop: 3, height: 6, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}><div style={{ width: `${75 + i * 5}%`, height: "100%", background: "#22c55e", borderRadius: 3 }} /></div><span style={{ fontSize: 8, fontFamily: S.mono, color: "#22c55e" }}>{75 + i * 5}%</span></div>)}</div>;

    return null;
  };

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #1a0a2e 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#f59e0b,#a855f7)" }}>⚡</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>Transformer Lab</h1>
        <span style={{ fontSize: 10, color: "#94a3b8" }}>Encoder-Decoder · Language Translation</span>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px" }}>
        {/* Dataset */}
        <div style={{ borderRadius: 10, padding: 10, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b", marginBottom: 10 }}>
          <div style={{ display: "flex", gap: 6 }}>{DATASET.map((dd, i) => (
            <button key={i} onClick={() => { setDIdx(i); setSel(null); }} style={{ flex: 1, padding: "6px 8px", borderRadius: 6, fontSize: 10, fontWeight: 600, textAlign: "left", background: i === dIdx ? "#f59e0b15" : "#0f172a", border: `1.5px solid ${i === dIdx ? "#f59e0b" : "#1e293b"}`, color: i === dIdx ? "#f59e0b" : "#94a3b8", cursor: "pointer" }}>
              <span style={{ color: "#22c55e" }}>EN:</span> {dd.src} · <span style={{ color: "#3b82f6" }}>FR:</span> {dd.tgt}
            </button>
          ))}</div>
        </div>

        {/* Main: diagram + detail */}
        <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: 12 }}>
          <div style={{ borderRadius: 12, padding: 10, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
            <ArchDiagram selected={sel} onSelect={setSel} />
          </div>
          <div>{renderDetail()}</div>
        </div>
      </div>
    </div>
  );
}
