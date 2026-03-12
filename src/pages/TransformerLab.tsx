import { useState, useEffect, useRef, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════
   TRANSFORMER MATH ENGINE
   ═══════════════════════════════════════════════════════════ */
function matMul(a, b) { const R = []; for (let i = 0; i < a.length; i++) { R[i] = []; for (let j = 0; j < b[0].length; j++) { let s = 0; for (let k = 0; k < b.length; k++) s += a[i][k] * b[k][j]; R[i][j] = s; } } return R; }
function transpose(m) { return m[0].map((_, i) => m.map(r => r[i])); }
function softmaxRow(v) { const mx = Math.max(...v), e = v.map(x => Math.exp(x - mx)), s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); }
function randMat(r, c, scale = 0.3) { return Array(r).fill(0).map(() => Array(c).fill(0).map(() => (Math.random() - 0.5) * scale)); }
function addMat(a, b) { return a.map((row, i) => row.map((v, j) => v + (b[i]?.[j] || 0))); }
function layerNorm(m) { return m.map(row => { const mean = row.reduce((s, v) => s + v, 0) / row.length; const v = row.reduce((s, x) => s + (x - mean) ** 2, 0) / row.length; const std = Math.sqrt(v + 1e-5); return row.map(x => (x - mean) / std); }); }
function reluMat(m) { return m.map(r => r.map(v => Math.max(0, v))); }

function positionalEncoding(seqLen, dModel) {
  const pe = [];
  for (let pos = 0; pos < seqLen; pos++) {
    const row = [];
    for (let i = 0; i < dModel; i++) {
      if (i % 2 === 0) row.push(Math.sin(pos / Math.pow(10000, i / dModel)));
      else row.push(Math.cos(pos / Math.pow(10000, (i - 1) / dModel)));
    }
    pe.push(row);
  }
  return pe;
}

function scaledDotProductAttention(Q, K, V, mask = null) {
  const dK = K[0].length;
  const scores = matMul(Q, transpose(K));
  const scaled = scores.map(row => row.map(v => v / Math.sqrt(dK)));
  if (mask) { for (let i = 0; i < scaled.length; i++) for (let j = 0; j < scaled[0].length; j++) if (mask[i][j] === 0) scaled[i][j] = -1e9; }
  const attnWeights = scaled.map(softmaxRow);
  const output = matMul(attnWeights, V);
  return { scores: scaled, attnWeights, output };
}

function multiHeadAttention(input, Wqs, Wks, Wvs, Wo) {
  const headOutputs = [], headAttns = [];
  for (let h = 0; h < Wqs.length; h++) {
    const Q = matMul(input, Wqs[h]), K = matMul(input, Wks[h]), V = matMul(input, Wvs[h]);
    const { attnWeights, output } = scaledDotProductAttention(Q, K, V);
    headOutputs.push(output);
    headAttns.push(attnWeights);
  }
  // Concatenate heads
  const concat = input.map((_, i) => { const row = []; headOutputs.forEach(ho => row.push(...ho[i])); return row; });
  const projected = matMul(concat, Wo);
  return { output: projected, headAttns, headOutputs, concat };
}

function feedForward(input, W1, b1, W2, b2) {
  const h = matMul(input, W1).map((r, i) => r.map((v, j) => Math.max(0, v + (b1[j] || 0))));
  const out = matMul(h, W2).map((r, i) => r.map((v, j) => v + (b2[j] || 0)));
  return { hidden: h, output: out };
}

function transformerBlock(input, Wqs, Wks, Wvs, Wo, W1, b1, W2, b2) {
  const mha = multiHeadAttention(input, Wqs, Wks, Wvs, Wo);
  const addNorm1 = layerNorm(addMat(input, mha.output));
  const ff = feedForward(addNorm1, W1, b1, W2, b2);
  const addNorm2 = layerNorm(addMat(addNorm1, ff.output));
  return { mha, addNorm1, ff, output: addNorm2 };
}

/* ═══════════════════════════════════════════════════════════
   INPUT SENTENCES & TOKENIZATION
   ═══════════════════════════════════════════════════════════ */
const SENTENCES = [
  { id: "cat", text: "The cat sat on the mat", desc: "Simple subject-verb-object" },
  { id: "attention", text: "Attention is all you need", desc: "The famous paper title" },
  { id: "bank", text: "I went to the bank", desc: "Ambiguous word (river/money)" },
  { id: "king", text: "The king wore a crown", desc: "Semantic relationships" },
  { id: "quick", text: "The quick brown fox jumps", desc: "Classic test sentence" },
  { id: "students", text: "Students learn deep learning", desc: "Educational context" },
  { id: "hello", text: "Hello world from transformers", desc: "Tech greeting" },
  { id: "custom", text: "Custom input here", desc: "Edit this!" },
];

function tokenize(text) {
  return text.toLowerCase().split(/\s+/).filter(t => t.length > 0);
}

function buildVocab(tokens) {
  const unique = [...new Set(tokens)];
  const map = {}; unique.forEach((t, i) => map[t] = i);
  return { vocab: unique, toId: map, size: unique.length };
}

function embedTokens(tokenIds, dModel, embeddingMatrix) {
  return tokenIds.map(id => embeddingMatrix[id] || Array(dModel).fill(0));
}

const S = { mono: "'IBM Plex Mono', monospace", sans: "'IBM Plex Sans', system-ui, sans-serif" };

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS
   ═══════════════════════════════════════════════════════════ */
function AttentionHeatmap({ weights, tokens, size = 280, label, headIdx }) {
  const ref = useRef(null);
  const n = tokens.length;
  useEffect(() => {
    const c = ref.current; if (!c || !weights?.length) return;
    const cellW = size / (n + 2), cellH = size / (n + 2);
    c.width = size; c.height = size; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, size);
    const off = cellW * 2;
    // Cells
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = weights[i]?.[j] || 0;
      const r = Math.round(59 + v * 196), g = Math.round(130 - v * 80), b = Math.round(246 - v * 200);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(off + j * cellW, off + i * cellH, cellW - 1, cellH - 1);
      if (n <= 8) { ctx.fillStyle = v > 0.3 ? "#fff" : "#94a3b8"; ctx.font = `bold ${Math.min(10, cellW * 0.4)}px 'IBM Plex Mono'`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(v.toFixed(2), off + j * cellW + cellW / 2, off + i * cellH + cellH / 2); }
    }
    // Labels
    ctx.fillStyle = "#94a3b8"; ctx.font = `bold ${Math.min(10, cellW * 0.5)}px 'IBM Plex Mono'`;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let i = 0; i < n; i++) ctx.fillText(tokens[i].slice(0, 6), off - 4, off + i * cellH + cellH / 2);
    ctx.textAlign = "center"; ctx.textBaseline = "bottom";
    for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(off + j * cellW + cellW / 2, off - 4); ctx.rotate(-0.5); ctx.fillText(tokens[j].slice(0, 6), 0, 0); ctx.restore(); }
  }, [weights, tokens, size, n]);
  return (
    <div>
      {label && <p style={{ fontSize: 10, fontWeight: 700, color: "#94a3b8", marginBottom: 4, fontFamily: S.mono }}>{label}</p>}
      <canvas ref={ref} style={{ width: size, height: size, borderRadius: 8, border: "1px solid #334155", background: "#0f172a" }} />
    </div>
  );
}

function MatrixDisplay({ data, label, color = "#3b82f6", cellSize = 32, maxR = 8, maxC = 6 }) {
  if (!data?.length) return null;
  const rows = Math.min(data.length, maxR), cols = Math.min(data[0]?.length || 0, maxC);
  return (
    <div>
      {label && <p style={{ fontSize: 9, fontWeight: 700, color, marginBottom: 3, fontFamily: S.mono }}>{label} [{data.length}×{data[0]?.length}]</p>}
      <div style={{ display: "inline-grid", gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.slice(0, rows).map((row, i) => row.slice(0, cols).map((v, j) => (
          <div key={`${i}-${j}`} style={{ width: cellSize, height: 22, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 600, fontFamily: S.mono, borderRadius: 2, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 150 + 30).toString(16).padStart(2, '0')}`, color: "#fff" }}>
            {v.toFixed(2)}
          </div>
        )))}
      </div>
      {(data.length > maxR || (data[0]?.length || 0) > maxC) && <p style={{ fontSize: 7, color: "#475569" }}>showing {rows}×{cols} of {data.length}×{data[0]?.length}</p>}
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
          <div key={i} style={{ minWidth: 34, height: 20, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, fontWeight: 600, fontFamily: S.mono, borderRadius: 3, background: `${color}${Math.round(Math.min(Math.abs(v), 1) * 150 + 40).toString(16).padStart(2, '0')}`, color: "#fff", padding: "0 2px" }}>{v.toFixed(2)}</div>
        ))}
        {values.length > maxShow && <span style={{ fontSize: 7, color: "#475569", alignSelf: "center" }}>+{values.length - maxShow}</span>}
      </div>
    </div>
  );
}

function TensorShape({ shape, color = "#60a5fa" }) {
  return <span style={{ fontSize: 10, fontFamily: S.mono, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: `${color}15`, border: `1px solid ${color}33`, color }}>[{shape.join("×")}]</span>;
}

function PEPlot({ pe, size = 280 }) {
  const ref = useRef(null);
  useEffect(() => {
    const c = ref.current; if (!c || !pe?.length) return;
    c.width = size; c.height = 120; const ctx = c.getContext("2d");
    ctx.clearRect(0, 0, size, 120);
    const rows = pe.length, cols = pe[0].length;
    const cw = size / cols, ch = 120 / rows;
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const v = (pe[i][j] + 1) / 2;
      ctx.fillStyle = `hsl(${v * 260}, 70%, ${30 + v * 40}%)`;
      ctx.fillRect(j * cw, i * ch, cw + 0.5, ch + 0.5);
    }
    ctx.fillStyle = "#64748b"; ctx.font = "8px 'IBM Plex Mono'"; ctx.fillText("position →", 4, 10); ctx.fillText("dimension →", size - 70, 10);
  }, [pe, size]);
  return <canvas ref={ref} style={{ width: size, height: 120, borderRadius: 6, border: "1px solid #334155" }} />;
}

/* ═══════════════════════════════════════════════════════════
   STEP-BY-STEP SELF-ATTENTION MODULE
   ═══════════════════════════════════════════════════════════ */
function AttentionStepModule({ tokens, embeddings, dModel, numHeads, Wqs, Wks, Wvs, Wo, pe }) {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const [selectedToken, setSelectedToken] = useState(0);

  const seqLen = tokens.length;
  const dHead = Math.floor(dModel / numHeads);
  const embPlusPE = addMat(embeddings, pe);

  // Single-head attention for visualization (head 0)
  const Q0 = matMul(embPlusPE, Wqs[0]), K0 = matMul(embPlusPE, Wks[0]), V0 = matMul(embPlusPE, Wvs[0]);
  const { scores: rawScores, attnWeights, output: attnOut } = scaledDotProductAttention(Q0, K0, V0);

  // Full multi-head
  const mha = multiHeadAttention(embPlusPE, Wqs, Wks, Wvs, Wo);

  const STEPS = [
    { t: "📝 Input Tokens", d: `"${tokens.join(' ')}" → ${seqLen} tokens, each embedded as a ${dModel}-dim vector.`, c: "#22c55e" },
    { t: "🌊 Positional Encoding", d: `Sine/cosine encoding added to embeddings so the model knows token ORDER. Shape: [${seqLen}×${dModel}].`, c: "#06b6d4" },
    { t: "🔑 Q, K, V Projections", d: `Each token is projected into Query, Key, and Value vectors via learned weight matrices. Q=XW_Q, K=XW_K, V=XW_V.`, c: "#f59e0b" },
    { t: "📊 Attention Scores", d: `Scores = Q · K^T / √d_k. Each token's query "asks a question" and each key "offers an answer". High score = strong relevance.`, c: "#a855f7" },
    { t: "🎯 Softmax → Attention Weights", d: "Scores are softmax-normalized per row so each token's attention sums to 1.0. This creates the attention pattern.", c: "#3b82f6" },
    { t: "✖️ Weighted Values", d: "Output = Attention × V. Each token's output is a weighted combination of all value vectors, weighted by attention.", c: "#ec4899" },
    { t: "🧠 Multi-Head Attention", d: `${numHeads} parallel attention heads, each with d_head=${dHead}. Concatenated and projected through W_O.`, c: "#f59e0b" },
    { t: "📤 Final Output", d: "Multi-head attention output represents each token with context from ALL other tokens. This is the power of self-attention.", c: "#22c55e" },
  ];
  const totalSteps = STEPS.length;

  useEffect(() => {
    if (auto && step < totalSteps - 1) { const t = setTimeout(() => setStep(p => p + 1), 3500); return () => clearTimeout(t); }
    else setAuto(false);
  }, [auto, step, totalSteps]);

  return (
    <div>
      {/* Controls */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <div style={{ display: "flex", gap: 2 }}>{STEPS.map((s, i) => <button key={i} onClick={() => { setStep(i); setAuto(false); }} style={{ width: 28, height: 6, borderRadius: 3, background: i <= step ? s.c : "#1e293b", border: "none", cursor: "pointer" }} />)}</div>
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
        {/* Step 0: Tokens */}
        {step === 0 && (
          <div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
              {tokens.map((t, i) => (
                <div key={i} style={{ padding: "6px 12px", borderRadius: 8, background: "#22c55e22", border: "1px solid #22c55e44", textAlign: "center" }}>
                  <p style={{ fontSize: 13, fontWeight: 700, color: "#22c55e" }}>{t}</p>
                  <p style={{ fontSize: 8, color: "#475569", fontFamily: S.mono }}>id={i}</p>
                </div>
              ))}
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8", marginBottom: 8 }}>Each token gets embedded into a {dModel}-dimensional vector:</p>
            {tokens.slice(0, 4).map((t, i) => <VectorDisplay key={i} values={embeddings[i]} label={`"${t}" → embedding`} color="#22c55e" maxShow={dModel} />)}
            <p style={{ fontSize: 9, color: "#475569", marginTop: 6 }}>Embedding matrix shape: <TensorShape shape={[seqLen, dModel]} color="#22c55e" /></p>
          </div>
        )}

        {/* Step 1: Positional encoding */}
        {step === 1 && (
          <div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 12 }}>
              <div>
                <PEPlot pe={pe} size={240} />
                <p style={{ fontSize: 9, color: "#06b6d4", marginTop: 4 }}>Positional encoding heatmap</p>
              </div>
              <div>
                {tokens.slice(0, 3).map((t, i) => <VectorDisplay key={i} values={pe[i]} label={`PE[pos=${i}]`} color="#06b6d4" maxShow={dModel} />)}
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <MatrixDisplay data={embeddings} label="Embeddings" color="#22c55e" cellSize={28} maxC={dModel} />
              <span style={{ fontSize: 18, color: "#f59e0b", fontWeight: 800 }}>+</span>
              <MatrixDisplay data={pe} label="Pos Encoding" color="#06b6d4" cellSize={28} maxC={dModel} />
              <span style={{ fontSize: 18, color: "#22c55e", fontWeight: 800 }}>=</span>
              <MatrixDisplay data={embPlusPE} label="Input to Attention" color="#f59e0b" cellSize={28} maxC={dModel} />
            </div>
          </div>
        )}

        {/* Step 2: Q, K, V */}
        {step === 2 && (
          <div>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 12 }}>
              <MatrixDisplay data={Q0} label="Q = X·W_Q" color="#ef4444" cellSize={30} maxC={dHead} />
              <MatrixDisplay data={K0} label="K = X·W_K" color="#22c55e" cellSize={30} maxC={dHead} />
              <MatrixDisplay data={V0} label="V = X·W_V" color="#3b82f6" cellSize={30} maxC={dHead} />
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8" }}>
              <b style={{ color: "#ef4444" }}>Query</b> = "what am I looking for?" ·
              <b style={{ color: "#22c55e" }}> Key</b> = "what do I contain?" ·
              <b style={{ color: "#3b82f6" }}> Value</b> = "what info do I provide?"
            </p>
            <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Each: <TensorShape shape={[seqLen, dHead]} /></p>
          </div>
        )}

        {/* Step 3: Raw scores */}
        {step === 3 && (
          <div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <AttentionHeatmap weights={rawScores} tokens={tokens} size={220} label="Raw Scores (Q·K^T / √d_k)" />
              <div>
                <p style={{ fontSize: 10, color: "#94a3b8", marginBottom: 6 }}>Click a token to see its attention:</p>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 8 }}>
                  {tokens.map((t, i) => (
                    <button key={i} onClick={() => setSelectedToken(i)} style={{ padding: "3px 8px", borderRadius: 5, fontSize: 10, fontWeight: 700, background: selectedToken === i ? "#a855f7" : "#1e293b", color: selectedToken === i ? "#fff" : "#94a3b8", border: `1px solid ${selectedToken === i ? "#a855f7" : "#334155"}`, cursor: "pointer" }}>{t}</button>
                  ))}
                </div>
                <p style={{ fontSize: 10, color: "#a855f7", fontFamily: S.mono, marginBottom: 4 }}>"{tokens[selectedToken]}" attends to:</p>
                {tokens.map((t, j) => (
                  <div key={j} style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
                    <span style={{ fontSize: 9, color: "#64748b", width: 50, textAlign: "right", fontFamily: S.mono }}>{t}</span>
                    <div style={{ flex: 1, height: 14, background: "#1e293b", borderRadius: 3, overflow: "hidden" }}>
                      <div style={{ height: "100%", width: `${(rawScores[selectedToken]?.[j] || 0) * 30 + 10}%`, background: "#a855f7", borderRadius: 3 }} />
                    </div>
                    <span style={{ fontSize: 8, fontFamily: S.mono, color: "#94a3b8", width: 36 }}>{(rawScores[selectedToken]?.[j] || 0).toFixed(2)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Softmax attention weights */}
        {step === 4 && (
          <div>
            <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
              <AttentionHeatmap weights={attnWeights} tokens={tokens} size={240} label="Attention Weights (after softmax)" />
              <div>
                <p style={{ fontSize: 10, color: "#94a3b8", marginBottom: 8 }}>Each row sums to 1.0 — a probability distribution over all tokens.</p>
                {tokens.map((t, i) => (
                  <div key={i} style={{ marginBottom: 6 }}>
                    <p style={{ fontSize: 9, color: "#3b82f6", fontFamily: S.mono, fontWeight: 700 }}>"{t}" →</p>
                    <div style={{ display: "flex", gap: 2 }}>
                      {attnWeights[i]?.map((w, j) => (
                        <div key={j} style={{ flex: 1, height: 20, borderRadius: 3, background: `rgba(59,130,246,${w * 0.9 + 0.05})`, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 7, color: w > 0.15 ? "#fff" : "#64748b", fontFamily: S.mono, fontWeight: 700 }}>
                          {w.toFixed(2)}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Step 5: Weighted values */}
        {step === 5 && (
          <div>
            <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap", marginBottom: 12 }}>
              <MatrixDisplay data={attnWeights} label="Attention" color="#3b82f6" cellSize={28} />
              <span style={{ fontSize: 18, color: "#f59e0b", fontWeight: 800 }}>×</span>
              <MatrixDisplay data={V0} label="Values" color="#22c55e" cellSize={28} maxC={dHead} />
              <span style={{ fontSize: 18, color: "#ec4899", fontWeight: 800 }}>=</span>
              <MatrixDisplay data={attnOut} label="Output" color="#ec4899" cellSize={28} maxC={dHead} />
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8" }}>Each token's output is now a <b style={{ color: "#ec4899" }}>context-aware representation</b> — a weighted mix of all tokens' values, based on attention scores.</p>
            <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>"{tokens[0]}" output = {attnWeights[0]?.map((w, j) => `${w.toFixed(2)}×V("${tokens[j]}")`).join(" + ")}</p>
          </div>
        )}

        {/* Step 6: Multi-head */}
        {step === 6 && (
          <div>
            <p style={{ fontSize: 10, color: "#94a3b8", marginBottom: 10 }}>{numHeads} heads run in parallel, each learning different relationships:</p>
            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 12 }}>
              {mha.headAttns.map((ha, hi) => (
                <AttentionHeatmap key={hi} weights={ha} tokens={tokens} size={Math.min(160, 280 / numHeads)} label={`Head ${hi + 1}`} headIdx={hi} />
              ))}
            </div>
            <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              <div style={{ padding: "4px 8px", borderRadius: 6, background: "#f59e0b22", border: "1px solid #f59e0b44" }}>
                <span style={{ fontSize: 9, color: "#f59e0b", fontFamily: S.mono }}>Concat [{seqLen}×{dHead * numHeads}]</span>
              </div>
              <span style={{ fontSize: 14, color: "#64748b" }}>→ W_O →</span>
              <div style={{ padding: "4px 8px", borderRadius: 6, background: "#22c55e22", border: "1px solid #22c55e44" }}>
                <span style={{ fontSize: 9, color: "#22c55e", fontFamily: S.mono }}>Output [{seqLen}×{dModel}]</span>
              </div>
            </div>
          </div>
        )}

        {/* Step 7: Final */}
        {step === 7 && (
          <div>
            <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
              {tokens.map((t, i) => (
                <div key={i} style={{ textAlign: "center" }}>
                  <div style={{ padding: "4px 8px", borderRadius: 6, background: "#22c55e22", border: "1px solid #22c55e44", marginBottom: 4 }}>
                    <span style={{ fontSize: 10, fontWeight: 700, color: "#22c55e" }}>"{t}"</span>
                  </div>
                  <VectorDisplay values={mha.output[i]} label="" color="#22c55e" maxShow={dModel} />
                </div>
              ))}
            </div>
            <p style={{ fontSize: 10, color: "#94a3b8" }}>Each token now has a <b style={{ color: "#22c55e" }}>context-enriched</b> representation that encodes information from ALL other tokens via attention. This is what makes transformers so powerful.</p>
            <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>Output shape: <TensorShape shape={[seqLen, dModel]} color="#22c55e" /> · Params: {numHeads * 3 * dModel * dHead + dModel * dModel} (QKV weights + output projection)</p>
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
function FullModal({ tokens, embeddings, dModel, numHeads, Wqs, Wks, Wvs, Wo, pe, onClose }) {
  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 50, background: "rgba(2,6,23,0.95)", backdropFilter: "blur(16px)", overflow: "auto" }}>
      <div style={{ maxWidth: 1050, margin: "0 auto", padding: "20px 24px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <div style={{ width: 44, height: 44, borderRadius: 12, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 22, background: "#f59e0b22" }}>⚡</div>
            <div><h2 style={{ fontSize: 20, fontWeight: 800, color: "#fff", margin: 0 }}>Self-Attention — Step by Step</h2><p style={{ fontSize: 12, color: "#64748b", margin: 0 }}>"{tokens.join(' ')}" · {numHeads} heads · d={dModel}</p></div>
          </div>
          <button onClick={onClose} style={{ width: 36, height: 36, borderRadius: 8, fontSize: 18, color: "#64748b", background: "#1e293b", border: "1px solid #334155", cursor: "pointer" }}>×</button>
        </div>
        <div style={{ background: "rgba(15,23,42,0.6)", borderRadius: 14, border: "1px solid #1e293b", padding: 20 }}>
          <AttentionStepModule tokens={tokens} embeddings={embeddings} dModel={dModel} numHeads={numHeads} Wqs={Wqs} Wks={Wks} Wvs={Wvs} Wo={Wo} pe={pe} />
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN TRANSFORMER LAB
   ═══════════════════════════════════════════════════════════ */
export default function TransformerLab() {
  const [sentId, setSentId] = useState("attention");
  const [customText, setCustomText] = useState("Custom input here");
  const [dModel, setDModel] = useState(8);
  const [numHeads, setNumHeads] = useState(2);
  const [modal, setModal] = useState(false);
  const [selHead, setSelHead] = useState(0);

  const sentence = sentId === "custom" ? customText : SENTENCES.find(s => s.id === sentId)?.text || "";
  const tokens = useMemo(() => tokenize(sentence), [sentence]);
  const seqLen = tokens.length;
  const { vocab, toId } = useMemo(() => buildVocab(tokens), [tokens]);
  const dHead = Math.floor(dModel / numHeads);

  // Build weights
  const embMatrix = useMemo(() => randMat(vocab.size, dModel, 0.5), [vocab.size, dModel]);
  const embeddings = useMemo(() => tokens.map(t => embMatrix[toId[t]] || Array(dModel).fill(0)), [tokens, embMatrix, toId, dModel]);
  const pe = useMemo(() => positionalEncoding(seqLen, dModel), [seqLen, dModel]);
  const embPlusPE = useMemo(() => addMat(embeddings, pe), [embeddings, pe]);

  const Wqs = useMemo(() => Array(numHeads).fill(0).map(() => randMat(dModel, dHead)), [dModel, dHead, numHeads]);
  const Wks = useMemo(() => Array(numHeads).fill(0).map(() => randMat(dModel, dHead)), [dModel, dHead, numHeads]);
  const Wvs = useMemo(() => Array(numHeads).fill(0).map(() => randMat(dModel, dHead)), [dModel, dHead, numHeads]);
  const Wo = useMemo(() => randMat(dHead * numHeads, dModel), [dHead, numHeads, dModel]);

  // FFN weights
  const dFF = dModel * 2;
  const W1 = useMemo(() => randMat(dModel, dFF), [dModel, dFF]);
  const b1 = useMemo(() => Array(dFF).fill(0), [dFF]);
  const W2 = useMemo(() => randMat(dFF, dModel), [dFF, dModel]);
  const b2 = useMemo(() => Array(dModel).fill(0), [dModel]);

  // Full transformer block
  const block = useMemo(() => transformerBlock(embPlusPE, Wqs, Wks, Wvs, Wo, W1, b1, W2, b2), [embPlusPE, Wqs, Wks, Wvs, Wo, W1, b1, W2, b2]);

  const totalParams = numHeads * 3 * dModel * dHead + dHead * numHeads * dModel + dModel * dFF + dFF + dFF * dModel + dModel;

  return (
    <div style={{ fontFamily: S.sans, background: "linear-gradient(145deg, #020617 0%, #1a0a2e 50%, #020617 100%)", minHeight: "100vh", color: "#e2e8f0" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      <div style={{ position: "sticky", top: 0, zIndex: 40, backdropFilter: "blur(16px)", background: "rgba(2,6,23,0.88)", borderBottom: "1px solid #1e293b" }}>
        <div style={{ maxWidth: 1200, margin: "0 auto", padding: "8px 20px", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ width: 30, height: 30, borderRadius: 8, display: "flex", alignItems: "center", justifyContent: "center", fontSize: 16, background: "linear-gradient(135deg,#f59e0b,#a855f7)" }}>⚡</div>
          <h1 style={{ fontSize: 15, fontWeight: 800, color: "#fff", margin: 0 }}>Transformer Lab</h1>
          <span style={{ fontSize: 10, color: "#475569" }}>Interactive Self-Attention Simulator</span>
        </div>
      </div>

      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "16px 20px" }}>
        <div style={{ display: "grid", gridTemplateColumns: "240px 1fr", gap: 14 }}>
          {/* Sidebar */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>📝 Input Sentence</h3>
              <select value={sentId} onChange={e => setSentId(e.target.value)} style={{ width: "100%", padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }}>
                {SENTENCES.map(s => <option key={s.id} value={s.id}>{s.text}</option>)}
              </select>
              {sentId === "custom" && <input value={customText} onChange={e => setCustomText(e.target.value)} style={{ width: "100%", marginTop: 4, padding: "5px 8px", borderRadius: 6, fontSize: 11, background: "#0f172a", border: "1px solid #334155", color: "#e2e8f0" }} />}
              <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginTop: 8 }}>
                {tokens.map((t, i) => <span key={i} style={{ padding: "2px 6px", borderRadius: 4, fontSize: 10, background: "#22c55e22", color: "#22c55e", fontWeight: 600, border: "1px solid #22c55e33" }}>{t}</span>)}
              </div>
              <p style={{ fontSize: 9, color: "#475569", marginTop: 4 }}>{seqLen} tokens · vocab: {vocab.size}</p>
            </div>

            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>⚙️ Configuration</h3>
              <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 6 }}>d_model: <b style={{ color: "#fff" }}>{dModel}</b>
                <input type="range" min={4} max={16} step={2} value={dModel} onChange={e => setDModel(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
              </label>
              <label style={{ fontSize: 10, color: "#64748b", display: "block", marginBottom: 6 }}>Attention Heads: <b style={{ color: "#fff" }}>{numHeads}</b>
                <input type="range" min={1} max={4} value={numHeads} onChange={e => setNumHeads(+e.target.value)} style={{ width: "100%", marginTop: 2 }} />
              </label>
              <div style={{ padding: 8, borderRadius: 6, background: "#0f172a", border: "1px solid #1e293b", fontSize: 9, color: "#64748b" }}>
                <p>d_model = {dModel} · heads = {numHeads} · d_head = {dHead}</p>
                <p>d_ff = {dFF} · seq_len = {seqLen}</p>
                <p>Params: <b style={{ color: "#fff" }}>{totalParams.toLocaleString()}</b></p>
              </div>
            </div>

            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.7)", border: "1px solid #1e293b" }}>
              <h3 style={{ fontSize: 11, fontWeight: 700, color: "#fff", marginBottom: 6 }}>🏗 Architecture</h3>
              <div style={{ fontSize: 9, color: "#94a3b8" }}>
                <p>1. Token Embedding <TensorShape shape={[seqLen, dModel]} color="#22c55e" /></p>
                <p style={{ marginTop: 3 }}>2. + Positional Encoding</p>
                <p style={{ marginTop: 3 }}>3. Multi-Head Attention ({numHeads}h)</p>
                <p style={{ marginTop: 3 }}>4. Add & Layer Norm</p>
                <p style={{ marginTop: 3 }}>5. Feed-Forward ({dModel}→{dFF}→{dModel})</p>
                <p style={{ marginTop: 3 }}>6. Add & Layer Norm</p>
                <p style={{ marginTop: 3 }}>→ Output <TensorShape shape={[seqLen, dModel]} color="#22c55e" /></p>
              </div>
            </div>

            <button onClick={() => setModal(true)} style={{ width: "100%", padding: "10px 0", borderRadius: 10, fontSize: 12, fontWeight: 700, background: "#f59e0b18", color: "#f59e0b", border: "1px solid #f59e0b33", cursor: "pointer" }}>⛶ Step-by-Step Self-Attention</button>
          </div>

          {/* Main */}
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {/* Pipeline */}
            <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 9, fontWeight: 700, color: "#475569", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>⚡ Transformer Block Pipeline</p>
              <div style={{ display: "flex", gap: 4, alignItems: "center", flexWrap: "wrap" }}>
                {[
                  { l: "Tokens", c: "#22c55e" }, { l: "+PE", c: "#06b6d4" }, { l: `MHA (${numHeads}h)`, c: "#f59e0b" },
                  { l: "Add&Norm", c: "#64748b" }, { l: `FFN (${dFF})`, c: "#ec4899" }, { l: "Add&Norm", c: "#64748b" }, { l: "Output", c: "#22c55e" }
                ].map((s, i) => (
                  <div key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                    {i > 0 && <span style={{ fontSize: 10, color: "#334155" }}>→</span>}
                    <div style={{ padding: "3px 8px", borderRadius: 5, background: `${s.c}18`, border: `1px solid ${s.c}33` }}>
                      <span style={{ fontSize: 9, color: s.c, fontFamily: S.mono, fontWeight: 700 }}>{s.l}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Attention heatmaps */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>🎯 Attention Heatmaps</p>
                <div style={{ display: "flex", gap: 3 }}>
                  {Array.from({ length: numHeads }).map((_, i) => (
                    <button key={i} onClick={() => setSelHead(i)} style={{ padding: "2px 8px", borderRadius: 4, fontSize: 9, fontWeight: 700, background: selHead === i ? "#f59e0b" : "#1e293b", color: selHead === i ? "#000" : "#64748b", border: `1px solid ${selHead === i ? "#f59e0b" : "#334155"}`, cursor: "pointer" }}>Head {i + 1}</button>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", gap: 16, flexWrap: "wrap" }}>
                <AttentionHeatmap weights={block.mha.headAttns[selHead]} tokens={tokens} size={260} label={`Head ${selHead + 1} Attention`} />
                <div style={{ flex: 1, minWidth: 200 }}>
                  <p style={{ fontSize: 10, color: "#94a3b8", marginBottom: 8 }}>Each cell shows how much token (row) attends to token (column). Brighter = stronger attention.</p>
                  {tokens.map((t, i) => {
                    const maxJ = block.mha.headAttns[selHead]?.[i]?.indexOf(Math.max(...(block.mha.headAttns[selHead]?.[i] || [0])));
                    return (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 3 }}>
                        <span style={{ fontSize: 10, fontWeight: 700, color: "#f59e0b", width: 60, textAlign: "right" }}>{t}</span>
                        <span style={{ fontSize: 10, color: "#475569" }}>→</span>
                        <span style={{ fontSize: 10, fontWeight: 700, color: "#3b82f6" }}>{tokens[maxJ] || "?"}</span>
                        <span style={{ fontSize: 8, color: "#475569" }}>({(block.mha.headAttns[selHead]?.[i]?.[maxJ] * 100 || 0).toFixed(0)}%)</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Positional encoding + embeddings */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
              <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #06b6d433" }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: "#06b6d4", marginBottom: 6, fontFamily: S.mono }}>🌊 Positional Encoding</p>
                <PEPlot pe={pe} size={Math.min(400, 280)} />
                <p style={{ fontSize: 8, color: "#475569", marginTop: 4 }}>Each row = position, each col = dimension. Sine/cosine waves give unique position signatures.</p>
              </div>
              <div style={{ borderRadius: 12, padding: 12, background: "rgba(15,23,42,0.5)", border: "1px solid #22c55e33" }}>
                <p style={{ fontSize: 10, fontWeight: 700, color: "#22c55e", marginBottom: 6, fontFamily: S.mono }}>📊 Token Embeddings</p>
                <MatrixDisplay data={embeddings} label={`Embeddings [${seqLen}×${dModel}]`} color="#22c55e" cellSize={26} maxC={dModel} maxR={seqLen} />
                <div style={{ marginTop: 6 }}>
                  {tokens.slice(0, 4).map((t, i) => <p key={i} style={{ fontSize: 8, color: "#475569" }}>"{t}" → [{embeddings[i]?.slice(0, 3).map(v => v.toFixed(2)).join(", ")}...]</p>)}
                </div>
              </div>
            </div>

            {/* Transformer block output */}
            <div style={{ borderRadius: 12, padding: 14, background: "rgba(15,23,42,0.5)", border: "1px solid #1e293b" }}>
              <p style={{ fontSize: 10, fontWeight: 700, color: "#64748b", marginBottom: 8, letterSpacing: 1, textTransform: "uppercase", fontFamily: S.mono }}>📤 Transformer Block Output</p>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                {tokens.map((t, i) => (
                  <div key={i} style={{ padding: 8, borderRadius: 8, background: "#0f172a", border: "1px solid #1e293b", textAlign: "center" }}>
                    <span style={{ fontSize: 10, fontWeight: 700, color: "#22c55e" }}>"{t}"</span>
                    <VectorDisplay values={block.output[i]} label="" color="#22c55e" maxShow={dModel} />
                  </div>
                ))}
              </div>
              <p style={{ fontSize: 9, color: "#475569", marginTop: 6 }}>Output: <TensorShape shape={[seqLen, dModel]} color="#22c55e" /> — each token now carries context from ALL other tokens via self-attention.</p>
            </div>
          </div>
        </div>
      </div>

      {modal && <FullModal tokens={tokens} embeddings={embeddings} dModel={dModel} numHeads={numHeads} Wqs={Wqs} Wks={Wks} Wvs={Wvs} Wo={Wo} pe={pe} onClose={() => setModal(false)} />}
    </div>
  );
}
