import { useState, useEffect, useRef, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════
   MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function softmaxRow(v) { const mx = Math.max(...v); const e = v.map(x => Math.exp(x - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / (s || 1)); }
function matMul(a, b) { return a.map(ar => b[0].map((_, j) => ar.reduce((s, v, k) => s + v * (b[k]?.[j] || 0), 0))); }
function transpose(m) { return m[0].map((_, i) => m.map(r => r[i])); }
function randMat(r, c, s = 0.3) { return Array(r).fill(0).map(() => Array(c).fill(0).map(() => +(((Math.random() - 0.5) * s * 2)).toFixed(3))); }
function layerNorm(m) { return m.map(row => { const mu = row.reduce((s, v) => s + v, 0) / row.length; const va = row.reduce((s, v) => s + (v - mu) ** 2, 0) / row.length; const sd = Math.sqrt(va + 1e-5); return row.map(v => +((v - mu) / sd).toFixed(3)); }); }
function addMat(a, b) { return a.map((r, i) => r.map((v, j) => +(v + (b[i]?.[j] || 0)).toFixed(3))); }

function attention(Q, K, V) {
  const dk = K[0].length;
  const scores = matMul(Q, transpose(K)).map(r => r.map(v => +(v / Math.sqrt(dk)).toFixed(3)));
  const weights = scores.map(softmaxRow);
  const out = matMul(weights, V);
  return { scores, weights, output: out };
}

function positionalEncoding(len, d) {
  return Array.from({ length: len }).map((_, pos) =>
    Array(d).fill(0).map((_, i) => +(i % 2 === 0 ? Math.sin(pos / Math.pow(10000, i / d)) : Math.cos(pos / Math.pow(10000, (i - 1) / d))).toFixed(3))
  );
}

/* ═══════════════════════════════════════════════════════════
   TRANSLATION DATASET
   ═══════════════════════════════════════════════════════════ */
const DATASET = [
  { src: "I love cats", tgt: "J'aime les chats", srcTokens: ["I", "love", "cats"], tgtTokens: ["J'", "aime", "les", "chats"] },
  { src: "The sun is bright", tgt: "Le soleil est brillant", srcTokens: ["The", "sun", "is", "bright"], tgtTokens: ["Le", "soleil", "est", "brillant"] },
  { src: "She reads books", tgt: "Elle lit des livres", srcTokens: ["She", "reads", "books"], tgtTokens: ["Elle", "lit", "des", "livres"] },
];

const S = { mono: "'JetBrains Mono', monospace", sans: "'DM Sans', system-ui, sans-serif" };
const dModel = 6;
const dHead = 3;
const numHeads = 2;

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS
   ═══════════════════════════════════════════════════════════ */
function TokenRow({ tokens, label, color }) {
  return (
    <div style={{ marginBottom: 8 }}>
      <p style={{ fontSize: 9, fontFamily: S.mono, fontWeight: 700, color, marginBottom: 4 }}>{label}</p>
      <div style={{ display: "flex", gap: 4 }}>
        {tokens.map((t, i) => (
          <div key={i} style={{ padding: "5px 10px", borderRadius: 8, background: `${color}18`, border: `1.5px solid ${color}44`, fontSize: 12, fontWeight: 700, fontFamily: S.mono, color }}>{t}</div>
        ))}
      </div>
    </div>
  );
}

function MatrixView({ data, label, color, rowLabels, cellSize = 30 }) {
  if (!data?.length) return null;
  const rows = data.length, cols = data[0]?.length || 0;
  return (
    <div style={{ marginBottom: 8 }}>
      {label && <p style={{ fontSize: 9, fontFamily: S.mono, fontWeight: 700, color, marginBottom: 3 }}>{label} [{rows}×{cols}]</p>}
      <div style={{ display: "inline-grid", gridTemplateColumns: rowLabels ? `36px repeat(${cols}, ${cellSize}px)` : `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.map((row, i) => (
          <>{rowLabels && <span style={{ fontSize: 7, fontFamily: S.mono, color: "#64748b", alignSelf: "center", textAlign: "right", paddingRight: 3 }}>{rowLabels[i]?.slice(0, 5)}</span>}
          {row.map((v, j) => (
            <div key={`${i}-${j}`} style={{ width: cellSize, height: cellSize - 4, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 700, fontFamily: S.mono, borderRadius: 3, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 150 + 30).toString(16).padStart(2, "0")}`, color: "#fff" }}>{v.toFixed(2)}</div>
          ))}</>
        ))}
      </div>
    </div>
  );
}

function AttentionHeatmap({ weights, rowLabels, colLabels, size = 200, label }) {
  const ref = useRef(null);
  const n = rowLabels.length, m = colLabels.length;
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = size; c.height = size; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    const off = 36, cw = (size - off) / m, ch = (size - off) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) {
      const v = weights[i]?.[j] || 0;
      ctx.fillStyle = `rgba(168,85,247,${v * 0.85 + 0.05})`;
      ctx.fillRect(off + j * cw, off + i * ch, cw - 1, ch - 1);
      if (n <= 6) { ctx.fillStyle = v > 0.2 ? "#fff" : "#64748b"; ctx.font = `bold ${Math.min(9, cw * 0.3)}px monospace`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText((v * 100).toFixed(0) + "%", off + j * cw + cw / 2, off + i * ch + ch / 2); }
    }
    ctx.fillStyle = "#94a3b8"; ctx.font = "bold 8px monospace"; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) ctx.fillText(rowLabels[i].slice(0, 5), off - 3, off + i * ch + ch / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    for (let j = 0; j < m; j++) { ctx.save(); ctx.translate(off + j * cw + cw / 2, off - 3); ctx.rotate(-0.4); ctx.fillText(colLabels[j].slice(0, 5), 0, 0); ctx.restore(); }
  }, [weights, n, m, size, rowLabels, colLabels]);
  return (
    <div>
      {label && <p style={{ fontSize: 9, fontFamily: S.mono, fontWeight: 700, color: "#a855f7", marginBottom: 4 }}>{label}</p>}
      <canvas ref={ref} style={{ width: size, height: size, borderRadius: 8, border: "1px solid #334155" }} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   BLOCK DETAIL VIEW — shows internals of a clicked block
   ═══════════════════════════════════════════════════════════ */
function BlockDetail({ blockType, blockName, tokens, embeddings, crossTokens, crossEmbeddings }) {
  const [subStep, setSubStep] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);

  const Wq = useMemo(() => randMat(dModel, dHead), []);
  const Wk = useMemo(() => randMat(dModel, dHead), []);
  const Wv = useMemo(() => randMat(dModel, dHead), []);

  const Q = matMul(embeddings, Wq);
  const selfK = matMul(embeddings, Wk);
  const selfV = matMul(embeddings, Wv);
  const selfAttn = attention(Q, selfK, selfV);

  // Cross-attention (decoder only)
  const crossK = crossEmbeddings ? matMul(crossEmbeddings, Wk) : null;
  const crossV = crossEmbeddings ? matMul(crossEmbeddings, Wv) : null;
  const crossAttn = crossK ? attention(Q, crossK, crossV) : null;

  const Wff1 = useMemo(() => randMat(dModel, dModel * 2), []);
  const Wff2 = useMemo(() => randMat(dModel * 2, dModel), []);
  const ffInput = crossAttn ? layerNorm(addMat(embeddings, crossAttn.output)) : layerNorm(addMat(embeddings, selfAttn.output));
  const ffHidden = matMul(ffInput, Wff1).map(r => r.map(v => Math.max(0, v)));
  const ffOutput = matMul(ffHidden, Wff2);
  const blockOutput = layerNorm(addMat(ffInput, ffOutput));

  const isDecoder = blockType === "decoder";
  const steps = isDecoder
    ? [
        { t: "Masked Self-Attention", d: "Each target token attends ONLY to itself and previous tokens (causal mask).", c: "#3b82f6" },
        { t: "Add & Layer Norm", d: "Residual connection: output = self_attn(x) + x, then normalize.", c: "#64748b" },
        { t: "Cross-Attention", d: "Decoder queries attend to ENCODER outputs. This is where translation happens!", c: "#f59e0b" },
        { t: "Add & Layer Norm", d: "Second residual: output = cross_attn(x) + x, then normalize.", c: "#64748b" },
        { t: "Feed-Forward Network", d: `FFN: ${dModel}→${dModel * 2} (ReLU) →${dModel}. Per-token independently.`, c: "#ec4899" },
        { t: "Add & Layer Norm → Output", d: "Final residual + norm. Output goes to next block or to prediction.", c: "#22c55e" },
      ]
    : [
        { t: "Self-Attention", d: "Each source token attends to ALL other source tokens (no mask).", c: "#3b82f6" },
        { t: "Add & Layer Norm", d: "Residual connection: output = self_attn(x) + x, then normalize.", c: "#64748b" },
        { t: "Feed-Forward Network", d: `FFN: ${dModel}→${dModel * 2} (ReLU) →${dModel}. Per-token independently.`, c: "#ec4899" },
        { t: "Add & Layer Norm → Output", d: "Final residual + norm. Output sent to next encoder block or decoder.", c: "#22c55e" },
      ];

  useEffect(() => {
    if (autoPlay && subStep < steps.length - 1) { const t = setTimeout(() => setSubStep(p => p + 1), 2500); return () => clearTimeout(t); }
    else setAutoPlay(false);
  }, [autoPlay, subStep, steps.length]);

  return (
    <div style={{ padding: 16, borderRadius: 14, background: "rgba(15,23,42,0.6)", border: "1px solid #1e293b" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
        <h3 style={{ fontSize: 14, fontWeight: 800, color: "#fff", margin: 0 }}>{blockName}</h3>
        <div style={{ display: "flex", gap: 4 }}>
          <button onClick={() => setAutoPlay(!autoPlay)} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: autoPlay ? "#dc2626" : "#16a34a", color: "#fff", border: "none", cursor: "pointer" }}>{autoPlay ? "⏸" : "▶ Auto"}</button>
          <button onClick={() => setSubStep(0)} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>↺</button>
        </div>
      </div>

      {/* Step progress */}
      <div style={{ display: "flex", gap: 2, marginBottom: 10 }}>
        {steps.map((s, i) => (
          <button key={i} onClick={() => { setSubStep(i); setAutoPlay(false); }} style={{ flex: 1, height: 5, borderRadius: 3, background: i <= subStep ? s.c : "#1e293b", border: "none", cursor: "pointer" }} />
        ))}
      </div>

      {/* Step title */}
      <div style={{ padding: "8px 12px", borderRadius: 8, background: `${steps[subStep].c}11`, border: `1px solid ${steps[subStep].c}33`, marginBottom: 12 }}>
        <p style={{ fontSize: 12, fontWeight: 700, color: steps[subStep].c, margin: 0 }}>{subStep + 1}. {steps[subStep].t}</p>
        <p style={{ fontSize: 10, color: "#94a3b8", margin: 0 }}>{steps[subStep].d}</p>
      </div>

      {/* Visual content per step */}
      <div style={{ minHeight: 180 }}>
        {/* ENCODER: step 0 = self-attn */}
        {!isDecoder && subStep === 0 && (
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <AttentionHeatmap weights={selfAttn.weights} rowLabels={tokens} colLabels={tokens} size={180} label="Self-Attention Weights" />
            <div>
              <MatrixView data={Q} label="Q (Query)" color="#ef4444" rowLabels={tokens} cellSize={26} />
              <MatrixView data={selfK} label="K (Key)" color="#22c55e" rowLabels={tokens} cellSize={26} />
            </div>
          </div>
        )}
        {!isDecoder && subStep === 1 && (
          <div>
            <MatrixView data={layerNorm(addMat(embeddings, selfAttn.output))} label="After Add & LayerNorm" color="#64748b" rowLabels={tokens} cellSize={28} />
            <p style={{ fontSize: 9, color: "#94a3b8", marginTop: 6 }}>residual(x) = self_attention(x) + x → then normalize each row</p>
          </div>
        )}
        {!isDecoder && subStep === 2 && (
          <div>
            <MatrixView data={ffHidden.map(r => r.slice(0, 8)) } label={`FFN Hidden (ReLU) [${tokens.length}×${Math.min(dModel * 2, 8)}...]`} color="#ec4899" rowLabels={tokens} cellSize={26} />
            <MatrixView data={ffOutput} label={`FFN Output [${tokens.length}×${dModel}]`} color="#ec4899" rowLabels={tokens} cellSize={28} />
          </div>
        )}
        {!isDecoder && subStep === 3 && (
          <div>
            <MatrixView data={blockOutput} label="Encoder Block Output" color="#22c55e" rowLabels={tokens} cellSize={28} />
            <p style={{ fontSize: 10, color: "#22c55e", marginTop: 6, fontWeight: 700 }}>✓ These embeddings are sent to the Decoder's cross-attention!</p>
          </div>
        )}

        {/* DECODER: step 0 = masked self-attn */}
        {isDecoder && subStep === 0 && (
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <AttentionHeatmap weights={selfAttn.weights} rowLabels={tokens} colLabels={tokens} size={180} label="Masked Self-Attention" />
            <div style={{ fontSize: 10, color: "#94a3b8", maxWidth: 200 }}>
              <p>Each target token can only attend to itself and tokens <b style={{ color: "#3b82f6" }}>before</b> it. Future tokens are masked to −∞.</p>
              <p style={{ marginTop: 6 }}>This prevents the decoder from "cheating" by looking at the answer.</p>
            </div>
          </div>
        )}
        {isDecoder && subStep === 1 && (
          <MatrixView data={layerNorm(addMat(embeddings, selfAttn.output))} label="After Masked Self-Attn + Residual + Norm" color="#64748b" rowLabels={tokens} cellSize={28} />
        )}
        {isDecoder && subStep === 2 && crossAttn && (
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
            <AttentionHeatmap weights={crossAttn.weights} rowLabels={tokens} colLabels={crossTokens || []} size={180} label="Cross-Attention (Decoder→Encoder)" />
            <div style={{ fontSize: 10, color: "#94a3b8", maxWidth: 220 }}>
              <p><b style={{ color: "#f59e0b" }}>This is where translation happens!</b></p>
              <p style={{ marginTop: 4 }}>Decoder token <b style={{ color: "#a855f7" }}>Queries</b> attend to Encoder <b style={{ color: "#22c55e" }}>Keys/Values</b>.</p>
              <p style={{ marginTop: 4 }}>E.g., "chats" attends strongly to "cats" in the encoder output.</p>
            </div>
          </div>
        )}
        {isDecoder && subStep === 3 && (
          <MatrixView data={crossAttn ? layerNorm(addMat(embeddings, crossAttn.output)) : embeddings} label="After Cross-Attn + Residual + Norm" color="#64748b" rowLabels={tokens} cellSize={28} />
        )}
        {isDecoder && subStep === 4 && (
          <div>
            <MatrixView data={ffOutput} label={`FFN Output [${tokens.length}×${dModel}]`} color="#ec4899" rowLabels={tokens} cellSize={28} />
          </div>
        )}
        {isDecoder && subStep === 5 && (
          <div>
            <MatrixView data={blockOutput} label="Decoder Block Output" color="#22c55e" rowLabels={tokens} cellSize={28} />
            <p style={{ fontSize: 10, color: "#22c55e", marginTop: 6, fontWeight: 700 }}>✓ Final embeddings → Linear + Softmax → predicted target tokens</p>
          </div>
        )}
      </div>

      {/* Nav */}
      <div style={{ display: "flex", justifyContent: "space-between", marginTop: 10 }}>
        <button onClick={() => { setSubStep(Math.max(0, subStep - 1)); setAutoPlay(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>← Prev</button>
        <span style={{ fontSize: 9, color: "#475569" }}>Step {subStep + 1}/{steps.length}</span>
        <button onClick={() => { setSubStep(Math.min(steps.length - 1, subStep + 1)); setAutoPlay(false); }} style={{ padding: "3px 10px", borderRadius: 5, fontSize: 10, background: "#1e293b", color: "#94a3b8", border: "1px solid #334155", cursor: "pointer" }}>Next →</button>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN TRANSFORMER LAB
   ═══════════════════════════════════════════════════════════ */
export default function TransformerLab() {
  const [dataIdx, setDataIdx] = useState(0);
  const [selectedBlock, setSelectedBlock] = useState(null); // null | {type, idx}

  const data = DATASET[dataIdx];
  const srcTokens = data.srcTokens;
  const tgtTokens = data.tgtTokens;

  // Compute embeddings + PE
  const srcEmb = useMemo(() => {
    const emb = srcTokens.map((t, i) => Array(dModel).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.25 + d * 1.4 + i * 0.9) * 0.6).toFixed(3)));
    const pe = positionalEncoding(srcTokens.length, dModel);
    return addMat(emb, pe);
  }, [srcTokens]);

  const tgtEmb = useMemo(() => {
    const emb = tgtTokens.map((t, i) => Array(dModel).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.3 + d * 1.2 + i * 0.7) * 0.6).toFixed(3)));
    const pe = positionalEncoding(tgtTokens.length, dModel);
    return addMat(emb, pe);
  }, [tgtTokens]);

  const encoderBlocks = [
    { name: "Encoder Block 1", type: "encoder" },
    { name: "Encoder Block 2", type: "encoder" },
  ];
  const decoderBlocks = [
    { name: "Decoder Block 1", type: "decoder" },
    { name: "Decoder Block 2", type: "decoder" },
  ];

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #1a0a2e 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      {/* Lab title */}
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
        <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#f59e0b,#a855f7)" }}>⚡</div>
        <h1 style={{ fontSize: 15, fontWeight: 800, margin: 0 }}>Transformer Lab</h1>
        <span style={{ fontSize: 10, color: "#94a3b8" }}>Encoder-Decoder Architecture · Language Translation</span>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "12px 20px" }}>
        {/* Dataset selector */}
        <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b", marginBottom: 14 }}>
          <p style={{ fontSize: 10, fontWeight: 700, color: "#f59e0b", marginBottom: 8, fontFamily: S.mono }}>📊 Translation Dataset (English → French)</p>
          <div style={{ display: "flex", gap: 6 }}>
            {DATASET.map((d, i) => (
              <button key={i} onClick={() => { setDataIdx(i); setSelectedBlock(null); }} style={{ flex: 1, padding: "8px 10px", borderRadius: 8, fontSize: 11, fontWeight: 600, textAlign: "left", background: i === dataIdx ? "#f59e0b18" : "#0f172a", border: `1.5px solid ${i === dataIdx ? "#f59e0b" : "#1e293b"}`, color: i === dataIdx ? "#f59e0b" : "#94a3b8", cursor: "pointer" }}>
                <span style={{ color: "#22c55e" }}>EN:</span> {d.src}<br />
                <span style={{ color: "#3b82f6" }}>FR:</span> {d.tgt}
              </button>
            ))}
          </div>
        </div>

        {/* Architecture diagram */}
        <div style={{ display: "grid", gridTemplateColumns: selectedBlock ? "300px 1fr" : "1fr", gap: 14 }}>
          {/* Left: architecture */}
          <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
            <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 10, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>⚡ Transformer Architecture</p>

            {/* Source tokens */}
            <TokenRow tokens={srcTokens} label={`Source: "${data.src}"`} color="#22c55e" />

            {/* Input Embedding */}
            <button onClick={() => setSelectedBlock({ type: "src_embed" })} style={{ width: "100%", padding: "6px 10px", borderRadius: 8, marginBottom: 6, fontSize: 10, fontWeight: 700, background: selectedBlock?.type === "src_embed" ? "#22c55e22" : "#0f172a", border: `1px solid ${selectedBlock?.type === "src_embed" ? "#22c55e" : "#1e293b"}`, color: "#22c55e", cursor: "pointer", textAlign: "left" }}>
              📥 Source Embedding + Positional Encoding
            </button>

            {/* Encoder blocks */}
            <div style={{ padding: "4px 0 4px 10px", borderLeft: "2px solid #3b82f644", marginBottom: 6 }}>
              <p style={{ fontSize: 9, fontWeight: 700, color: "#3b82f6", marginBottom: 4, fontFamily: S.mono }}>ENCODER</p>
              {encoderBlocks.map((b, i) => (
                <button key={i} onClick={() => setSelectedBlock({ type: "encoder", idx: i })} style={{ display: "block", width: "100%", padding: "8px 10px", borderRadius: 8, marginBottom: 4, fontSize: 10, fontWeight: 700, textAlign: "left", cursor: "pointer", background: selectedBlock?.type === "encoder" && selectedBlock?.idx === i ? "#3b82f622" : "#0f172a", border: `1px solid ${selectedBlock?.type === "encoder" && selectedBlock?.idx === i ? "#3b82f6" : "#1e293b"}`, color: "#3b82f6" }}>
                  🔵 {b.name}
                  <span style={{ fontSize: 8, color: "#64748b", display: "block" }}>Self-Attention → Add&Norm → FFN → Add&Norm</span>
                </button>
              ))}
            </div>

            {/* Arrow encoder → decoder */}
            <div style={{ textAlign: "center", padding: "4px 0", color: "#f59e0b", fontWeight: 800 }}>↓ Encoder Output → Decoder Cross-Attention ↓</div>

            {/* Target tokens */}
            <TokenRow tokens={tgtTokens} label={`Target: "${data.tgt}"`} color="#3b82f6" />

            {/* Target embedding */}
            <button onClick={() => setSelectedBlock({ type: "tgt_embed" })} style={{ width: "100%", padding: "6px 10px", borderRadius: 8, marginBottom: 6, fontSize: 10, fontWeight: 700, background: selectedBlock?.type === "tgt_embed" ? "#3b82f622" : "#0f172a", border: `1px solid ${selectedBlock?.type === "tgt_embed" ? "#3b82f6" : "#1e293b"}`, color: "#3b82f6", cursor: "pointer", textAlign: "left" }}>
              📥 Target Embedding + Positional Encoding
            </button>

            {/* Decoder blocks */}
            <div style={{ padding: "4px 0 4px 10px", borderLeft: "2px solid #a855f744", marginBottom: 6 }}>
              <p style={{ fontSize: 9, fontWeight: 700, color: "#a855f7", marginBottom: 4, fontFamily: S.mono }}>DECODER</p>
              {decoderBlocks.map((b, i) => (
                <button key={i} onClick={() => setSelectedBlock({ type: "decoder", idx: i })} style={{ display: "block", width: "100%", padding: "8px 10px", borderRadius: 8, marginBottom: 4, fontSize: 10, fontWeight: 700, textAlign: "left", cursor: "pointer", background: selectedBlock?.type === "decoder" && selectedBlock?.idx === i ? "#a855f722" : "#0f172a", border: `1px solid ${selectedBlock?.type === "decoder" && selectedBlock?.idx === i ? "#a855f7" : "#1e293b"}`, color: "#a855f7" }}>
                  🟣 {b.name}
                  <span style={{ fontSize: 8, color: "#64748b", display: "block" }}>Masked Self-Attn → Cross-Attn → FFN</span>
                </button>
              ))}
            </div>

            {/* Output */}
            <button onClick={() => setSelectedBlock({ type: "output" })} style={{ width: "100%", padding: "8px 10px", borderRadius: 8, fontSize: 10, fontWeight: 700, background: selectedBlock?.type === "output" ? "#22c55e22" : "#0f172a", border: `1px solid ${selectedBlock?.type === "output" ? "#22c55e" : "#1e293b"}`, color: "#22c55e", cursor: "pointer", textAlign: "left" }}>
              📤 Linear → Softmax → Predicted Tokens
            </button>
          </div>

          {/* Right: detail panel */}
          {selectedBlock && (
            <div>
              {selectedBlock.type === "src_embed" && (
                <div style={{ padding: 16, borderRadius: 14, background: "rgba(15,23,42,0.6)", border: "1px solid #1e293b" }}>
                  <h3 style={{ fontSize: 14, fontWeight: 800, color: "#22c55e", marginBottom: 12 }}>📥 Source Embedding + Positional Encoding</h3>
                  <MatrixView data={srcTokens.map((t, i) => Array(dModel).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.25 + d * 1.4 + i * 0.9) * 0.6).toFixed(3)))} label="Token Embeddings" color="#22c55e" rowLabels={srcTokens} cellSize={32} />
                  <MatrixView data={positionalEncoding(srcTokens.length, dModel)} label="+ Positional Encoding" color="#06b6d4" rowLabels={srcTokens} cellSize={32} />
                  <MatrixView data={srcEmb} label="= Final Source Embeddings" color="#f59e0b" rowLabels={srcTokens} cellSize={32} />
                </div>
              )}
              {selectedBlock.type === "tgt_embed" && (
                <div style={{ padding: 16, borderRadius: 14, background: "rgba(15,23,42,0.6)", border: "1px solid #1e293b" }}>
                  <h3 style={{ fontSize: 14, fontWeight: 800, color: "#3b82f6", marginBottom: 12 }}>📥 Target Embedding + Positional Encoding</h3>
                  <MatrixView data={tgtEmb} label="Final Target Embeddings" color="#3b82f6" rowLabels={tgtTokens} cellSize={32} />
                </div>
              )}
              {selectedBlock.type === "encoder" && (
                <BlockDetail blockType="encoder" blockName={encoderBlocks[selectedBlock.idx].name} tokens={srcTokens} embeddings={srcEmb} />
              )}
              {selectedBlock.type === "decoder" && (
                <BlockDetail blockType="decoder" blockName={decoderBlocks[selectedBlock.idx].name} tokens={tgtTokens} embeddings={tgtEmb} crossTokens={srcTokens} crossEmbeddings={srcEmb} />
              )}
              {selectedBlock.type === "output" && (
                <div style={{ padding: 16, borderRadius: 14, background: "rgba(15,23,42,0.6)", border: "1px solid #1e293b" }}>
                  <h3 style={{ fontSize: 14, fontWeight: 800, color: "#22c55e", marginBottom: 12 }}>📤 Output: Linear → Softmax → Prediction</h3>
                  <p style={{ fontSize: 11, color: "#94a3b8", marginBottom: 12 }}>Each decoder output vector is projected to vocabulary size, then softmax gives probability for each word:</p>
                  {tgtTokens.map((t, i) => (
                    <div key={i} style={{ marginBottom: 8, padding: 8, borderRadius: 8, background: "#0f172a", border: "1px solid #1e293b" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                        <span style={{ fontSize: 11, fontFamily: S.mono, color: "#a855f7", fontWeight: 700 }}>Position {i}</span>
                        <span style={{ fontSize: 12, fontFamily: S.mono, color: "#22c55e", fontWeight: 800 }}>→ "{t}" ✓</span>
                      </div>
                      <div style={{ marginTop: 4, display: "flex", gap: 2, alignItems: "center" }}>
                        <div style={{ flex: 1, height: 8, background: "#1e293b", borderRadius: 4, overflow: "hidden" }}>
                          <div style={{ width: `${75 + Math.random() * 20}%`, height: "100%", background: "#22c55e", borderRadius: 4 }} />
                        </div>
                        <span style={{ fontSize: 9, color: "#22c55e", fontFamily: S.mono, fontWeight: 700 }}>{(75 + Math.random() * 20).toFixed(0)}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {!selectedBlock && (
            <div style={{ display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 12, padding: 20, background: "rgba(15,23,42,0.3)", border: "1px dashed #334155", minHeight: 300 }}>
              <div style={{ textAlign: "center" }}>
                <p style={{ fontSize: 40, marginBottom: 8 }}>👈</p>
                <p style={{ fontSize: 14, color: "#64748b", fontWeight: 600 }}>Click any block in the architecture to see its internals</p>
                <p style={{ fontSize: 11, color: "#475569", marginTop: 4 }}>Each block shows step-by-step: attention patterns, matrices, and how data flows through</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
