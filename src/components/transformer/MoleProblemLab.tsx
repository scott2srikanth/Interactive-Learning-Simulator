import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';

/* ═══ DATA ═══ */
const PRESETS = [
  { sentence: 'American shrew mole', tokens: ['American','shrew','mole'], focus: 2, meaning: 'Small burrowing mammal', emoji: '🐾', color: '#22c55e', meaningColor: '#166534' },
  { sentence: 'One mole of carbon dioxide', tokens: ['One','mole','of','carbon','dioxide'], focus: 1, meaning: '6.022×10²³ particles (Avogadro\'s number)', emoji: '⚗️', color: '#3b82f6', meaningColor: '#1e3a5f' },
  { sentence: 'Take a biopsy of the mole', tokens: ['Take','a','biopsy','of','the','mole'], focus: 5, meaning: 'Skin growth / lesion', emoji: '🏥', color: '#f59e0b', meaningColor: '#78350f' },
];
const DK = 8; // visible dims
const F = "'JetBrains Mono', monospace";

function rgba(hex: string, a: number) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}
function softmax(arr: number[]) { const m = Math.max(...arr); const e = arr.map(v => Math.exp(v - m)); const s = e.reduce((a,b) => a+b, 0); return e.map(v => v / s); }
function dot(a: number[], b: number[]) { return a.reduce((s, v, i) => s + v * (b[i]||0), 0); }

/* ═══ GENERATE MOCK EMBEDDINGS/QKV ═══ */
function genData(tokens: string[], focus: number) {
  const embs = tokens.map((t, i) => Array(DK).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.3 + d * 1.5 + i * 0.7) * 0.6).toFixed(3)));
  const Wq = Array(DK).fill(0).map((_, r) => Array(DK).fill(0).map((_, c) => +(Math.sin(r * 2.1 + c * 1.3) * 0.4).toFixed(3)));
  const Wk = Array(DK).fill(0).map((_, r) => Array(DK).fill(0).map((_, c) => +(Math.cos(r * 1.7 + c * 2.3) * 0.4).toFixed(3)));
  const Wv = Array(DK).fill(0).map((_, r) => Array(DK).fill(0).map((_, c) => +(Math.sin(r * 0.9 + c * 3.1) * 0.4).toFixed(3)));
  const matmul = (v: number[], M: number[][]) => M[0].map((_, j) => v.reduce((s, vi, i) => s + vi * (M[i]?.[j]||0), 0));
  const Q = embs.map(e => matmul(e, Wq).map(v => +v.toFixed(3)));
  const K = embs.map(e => matmul(e, Wk).map(v => +v.toFixed(3)));
  const V = embs.map(e => matmul(e, Wv).map(v => +v.toFixed(3)));
  const scores = Q[focus].map((_, j) => dot(Q[focus], K[j]) / Math.sqrt(DK));
  const attn = softmax(scores);
  const output = Array(DK).fill(0).map((_, d) => tokens.reduce((s, _, i) => s + attn[i] * V[i][d], 0));
  return { embs, Q, K, V, scores, attn, output };
}

/* ═══ GLASSMORPHISM PANEL ═══ */
function Panel({ children, title, className = '', style = {} }: any) {
  return (
    <div style={{ background: 'rgba(15,23,42,0.75)', backdropFilter: 'blur(12px)', border: '1px solid rgba(59,130,246,0.15)', borderRadius: 12, padding: 14, ...style }} className={className}>
      {title && <p style={{ fontSize: 11, fontWeight: 800, color: '#94a3b8', marginBottom: 8, fontFamily: F, letterSpacing: 1, textTransform: 'uppercase' }}>{title}</p>}
      {children}
    </div>
  );
}

/* ═══ VECTOR BAR DISPLAY ═══ */
function VecBar({ values, label, color, height = 30 }: any) {
  return (
    <div style={{ marginBottom: 4 }}>
      {label && <p style={{ fontSize: 8, fontFamily: F, color: '#64748b', marginBottom: 1 }}>{label}</p>}
      <div style={{ display: 'flex', gap: 1, height, alignItems: 'flex-end' }}>
        {values.map((v: number, i: number) => (
          <div key={i} style={{ width: 10, height: `${Math.abs(v) / 0.8 * 100}%`, minHeight: 2, background: v >= 0 ? color : '#ef4444', borderRadius: 1, opacity: 0.4 + Math.abs(v) * 0.8, transition: 'height 0.4s' }} />
        ))}
      </div>
    </div>
  );
}

/* ═══ MAIN COMPONENT ═══ */
export default function MoleProblemLab() {
  const [preset, setPreset] = useState(0);
  const [step, setStep] = useState(0); // 0-5
  const [multiHead, setMultiHead] = useState(false);
  const [sharpness, setSharpness] = useState(1.0);
  const [hoveredToken, setHoveredToken] = useState(-1);
  const [showNoise, setShowNoise] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);
  const animRef = useRef(0);

  const ex = PRESETS[preset];
  const data = useMemo(() => genData(ex.tokens, ex.focus), [preset]);
  const adjustedScores = data.scores.map(s => s * sharpness);
  const adjustedAttn = softmax(adjustedScores);
  const n = ex.tokens.length;

  const STEPS = [
    { label: 'Embeddings', desc: 'Each token is mapped to a 512-dim vector. "mole" has the SAME vector in all 3 sentences.' },
    { label: 'Q, K, V Projections', desc: 'Each embedding is projected into Query (what am I looking for?), Key (what do I contain?), Value (what info do I share?).' },
    { label: 'Attention Scores', desc: `"${ex.tokens[ex.focus]}" Query dot-products with every Key. Higher score = more relevant.` },
    { label: 'Softmax', desc: 'Scores are normalized to probabilities (0-100%). This is how much each token contributes to "mole"\'s new meaning.' },
    { label: 'Weighted Output', desc: 'The Values are mixed according to attention weights, producing a new context-aware embedding for "mole".' },
    { label: 'Meaning Resolved', desc: `After attention, "mole" now encodes: ${ex.emoji} ${ex.meaning}` },
  ];

  // Canvas: 3D token cubes + attention arrows
  const CW = 520, CH = 280;
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = CW * dpr; c.height = CH * dpr;
    const ctx = c.getContext('2d')!;
    ctx.scale(dpr, dpr);

    const render = () => {
      tRef.current += 0.016;
      const t = tRef.current;
      ctx.clearRect(0, 0, CW, CH);

      // BG
      ctx.fillStyle = '#080818'; ctx.fillRect(0, 0, CW, CH);
      // Grid
      ctx.strokeStyle = '#1e293b22'; ctx.lineWidth = 0.5;
      for (let x = 0; x < CW; x += 30) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, CH); ctx.stroke(); }
      for (let y = 0; y < CH; y += 30) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CW, y); ctx.stroke(); }

      const gap = CW / (n + 1);
      const cubeY = 140;

      // ── Step 3+: Attention lines from "mole" to others ──
      if (step >= 2) {
        ex.tokens.forEach((_, j) => {
          const w = step >= 3 ? adjustedAttn[j] : Math.max(0, (adjustedScores[j] + 2) / 6);
          const x1 = gap * (ex.focus + 1), x2 = gap * (j + 1);
          const cpY = cubeY - 40 - w * 60;
          ctx.strokeStyle = rgba(ex.color, w * 0.8 + 0.05);
          ctx.lineWidth = w * 6 + 0.5;
          ctx.shadowColor = ex.color; ctx.shadowBlur = w * 10;
          ctx.beginPath();
          ctx.moveTo(x1, cubeY - 20);
          ctx.quadraticCurveTo((x1 + x2) / 2, cpY, x2, cubeY - 20);
          ctx.stroke();
          ctx.shadowBlur = 0;

          // Arrow dot
          if (w > 0.1) {
            ctx.fillStyle = rgba(ex.color, w + 0.2);
            ctx.beginPath(); ctx.arc(x2, cubeY - 22, 3 + w * 3, 0, Math.PI * 2); ctx.fill();
          }
        });
      }

      // ── Token cubes ──
      ex.tokens.forEach((tok, i) => {
        const x = gap * (i + 1);
        const y = cubeY;
        const isFocus = i === ex.focus;
        const isHov = hoveredToken === i;
        const bob = Math.sin(t * 1.5 + i * 0.8) * 4;
        const cubeW = 44, cubeH = 36, depth = 14;

        // 3D cube faces
        // Top
        ctx.fillStyle = rgba(isFocus ? ex.color : '#475569', isFocus ? 0.5 : 0.2);
        ctx.beginPath();
        ctx.moveTo(x - cubeW/2, y - cubeH/2 + bob);
        ctx.lineTo(x - cubeW/2 + depth, y - cubeH/2 - depth/2 + bob);
        ctx.lineTo(x + cubeW/2 + depth, y - cubeH/2 - depth/2 + bob);
        ctx.lineTo(x + cubeW/2, y - cubeH/2 + bob);
        ctx.closePath(); ctx.fill();
        // Right
        ctx.fillStyle = rgba(isFocus ? ex.color : '#475569', isFocus ? 0.35 : 0.12);
        ctx.beginPath();
        ctx.moveTo(x + cubeW/2, y - cubeH/2 + bob);
        ctx.lineTo(x + cubeW/2 + depth, y - cubeH/2 - depth/2 + bob);
        ctx.lineTo(x + cubeW/2 + depth, y + cubeH/2 - depth/2 + bob);
        ctx.lineTo(x + cubeW/2, y + cubeH/2 + bob);
        ctx.closePath(); ctx.fill();
        // Front
        ctx.fillStyle = rgba(isFocus ? ex.color : '#334155', isFocus ? 0.7 : 0.3);
        ctx.strokeStyle = rgba(isFocus ? ex.color : '#475569', isHov || isFocus ? 0.9 : 0.4);
        ctx.lineWidth = isFocus ? 2 : 1;
        if (isFocus) { ctx.shadowColor = ex.color; ctx.shadowBlur = 12; }
        ctx.beginPath(); ctx.roundRect(x - cubeW/2, y - cubeH/2 + bob, cubeW, cubeH, 4); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;

        // Token label
        ctx.fillStyle = isFocus ? '#fff' : '#94a3b8';
        ctx.font = `bold ${isFocus ? 13 : 11}px ${F}`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(tok, x, y + bob);

        // Step 1: embedding vectors as mini bars below cubes
        if (step >= 0) {
          data.embs[i].forEach((v, d) => {
            const bx = x - 18 + d * 5;
            const bh = Math.abs(v) * 25;
            ctx.fillStyle = rgba(isFocus ? ex.color : '#64748b', 0.4 + Math.abs(v) * 0.5);
            ctx.fillRect(bx, y + cubeH/2 + bob + 6, 4, bh);
          });
        }

        // Step 1: Q K V colored dots
        if (step >= 1) {
          const qp = { x: x - 14, y: y - cubeH/2 - 18 + bob };
          const kp = { x: x, y: y - cubeH/2 - 18 + bob };
          const vp = { x: x + 14, y: y - cubeH/2 - 18 + bob };
          [{ p: qp, c: '#ef4444', l: 'Q' }, { p: kp, c: '#22c55e', l: 'K' }, { p: vp, c: '#3b82f6', l: 'V' }].forEach(({ p, c, l }) => {
            ctx.fillStyle = rgba(c, 0.7);
            ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, Math.PI * 2); ctx.fill();
            ctx.fillStyle = '#fff'; ctx.font = `bold 7px ${F}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(l, p.x, p.y);
          });
        }

        // Step 3: softmax probability bars above cubes
        if (step >= 3) {
          const prob = adjustedAttn[i];
          const barH = prob * 60;
          const barX = x - 12;
          const barY = y - cubeH/2 - 35 + bob - barH;
          ctx.fillStyle = rgba(ex.color, 0.3 + prob * 0.6);
          ctx.beginPath(); ctx.roundRect(barX, barY, 24, barH, 3); ctx.fill();
          ctx.fillStyle = prob > 0.15 ? '#fff' : '#64748b';
          ctx.font = `bold 8px ${F}`; ctx.textAlign = 'center';
          ctx.fillText(`${(prob*100).toFixed(0)}%`, x, barY - 5);
        }

        // Hover: show embedding values
        if (isHov && step >= 0) {
          ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
          const tw = 110, th = 50;
          ctx.beginPath(); ctx.roundRect(x - tw/2, y + cubeH/2 + bob + 35, tw, th, 6); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#94a3b8'; ctx.font = `bold 8px ${F}`; ctx.textAlign = 'left';
          ctx.fillText(`E("${tok}")`, x - tw/2 + 6, y + cubeH/2 + bob + 48);
          ctx.fillStyle = '#e2e8f0'; ctx.font = `7px ${F}`;
          ctx.fillText(`[${data.embs[i].slice(0,4).map(v => v.toFixed(2)).join(', ')}...]`, x - tw/2 + 6, y + cubeH/2 + bob + 60);
          if (step >= 1) {
            ctx.fillStyle = '#ef4444'; ctx.fillText(`Q: [${data.Q[i].slice(0,3).map(v=>v.toFixed(2)).join(',')}...]`, x - tw/2 + 6, y + cubeH/2 + bob + 72);
          }
        }

        // Noise particles
        if (showNoise) {
          for (let p = 0; p < 3; p++) {
            const nx = x + Math.sin(t * 3 + i + p * 2) * 30;
            const ny = y + Math.cos(t * 2.5 + p * 3) * 20 + bob;
            ctx.fillStyle = rgba(isFocus ? ex.color : '#475569', 0.15);
            ctx.beginPath(); ctx.arc(nx, ny, 1.5, 0, Math.PI * 2); ctx.fill();
          }
        }
      });

      // Step 5: meaning morph animation
      if (step >= 5) {
        const morphX = CW / 2, morphY = CH - 40;
        const pulse = Math.sin(t * 3) * 0.15 + 0.85;
        ctx.fillStyle = rgba(ex.color, 0.15 * pulse);
        ctx.beginPath(); ctx.arc(morphX, morphY, 35, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = rgba(ex.color, 0.6 * pulse); ctx.lineWidth = 2;
        ctx.shadowColor = ex.color; ctx.shadowBlur = 15;
        ctx.stroke(); ctx.shadowBlur = 0;
        ctx.font = `${28}px sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(ex.emoji, morphX, morphY - 2);
        ctx.fillStyle = ex.color; ctx.font = `bold 10px ${F}`;
        ctx.fillText(ex.meaning.slice(0, 30), morphX, morphY + 28);
        // Arrow from "mole" to meaning
        const focusX = gap * (ex.focus + 1);
        ctx.setLineDash([4, 3]); ctx.strokeStyle = rgba(ex.color, 0.4); ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(focusX, cubeY + 40); ctx.lineTo(morphX, morphY - 40); ctx.stroke();
        ctx.setLineDash([]);
      }

      // Multi-head indicators
      if (multiHead && step >= 2) {
        const headColors = ['#ef4444','#22c55e','#3b82f6','#f59e0b'];
        headColors.forEach((hc, hi) => {
          const focusX = gap * (ex.focus + 1);
          ex.tokens.forEach((_, j) => {
            const tx = gap * (j + 1);
            const w = Math.abs(Math.sin(hi * 2.3 + j * 1.7 + t * 0.5)) * 0.6 + 0.1;
            ctx.strokeStyle = rgba(hc, w * 0.5); ctx.lineWidth = w * 2;
            const cpY = cubeY - 30 - hi * 12 - w * 20;
            ctx.beginPath(); ctx.moveTo(focusX, cubeY - 25); ctx.quadraticCurveTo((focusX + tx)/2, cpY, tx, cubeY - 25); ctx.stroke();
          });
          // Head label
          ctx.fillStyle = hc; ctx.font = `bold 7px ${F}`; ctx.textAlign = 'left';
          ctx.fillText(`H${hi+1}`, 8, 20 + hi * 12);
        });
      }

      animRef.current = requestAnimationFrame(render);
    };
    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [preset, step, multiHead, sharpness, hoveredToken, showNoise, adjustedAttn, adjustedScores, data, ex, n]);

  // Canvas mouse for hover
  const onCanvasMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return;
    const mx = (e.clientX - rect.left) * (CW / rect.width);
    const gap = CW / (n + 1);
    let found = -1;
    for (let i = 0; i < n; i++) { if (Math.abs(mx - gap * (i + 1)) < 28) { found = i; break; } }
    setHoveredToken(found);
  };

  return (
    <div style={{ fontFamily: F, color: '#e2e8f0' }}>
      {/* ── SENTENCE SELECTOR ── */}
      <Panel title="📝 Select a Sentence" style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {PRESETS.map((p, i) => (
            <button key={i} onClick={() => { setPreset(i); setStep(0); }}
              style={{ padding: '6px 12px', borderRadius: 8, fontSize: 11, fontWeight: 700, background: preset === i ? p.color : 'rgba(30,41,59,0.8)', color: preset === i ? '#fff' : '#94a3b8', border: `1.5px solid ${preset === i ? p.color : '#334155'}`, cursor: 'pointer', transition: 'all 0.2s' }}>
              {p.emoji} {p.sentence}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: 4, marginTop: 8, flexWrap: 'wrap' }}>
          {ex.tokens.map((tok, i) => (
            <span key={i} style={{ padding: '3px 10px', borderRadius: 6, fontSize: 12, fontWeight: 700, background: i === ex.focus ? rgba(ex.color, 0.2) : 'rgba(30,41,59,0.5)', color: i === ex.focus ? ex.color : '#94a3b8', border: `1px solid ${i === ex.focus ? ex.color+'66' : '#33415500'}` }}>
              {tok} {i === ex.focus && <span style={{ fontSize: 8 }}>← focus</span>}
            </span>
          ))}
        </div>
      </Panel>

      {/* ── MAIN: Canvas + Math side-by-side ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 220px', gap: 10, marginBottom: 10 }}>
        {/* Canvas */}
        <Panel title="🧊 Token Visualization" style={{ padding: 8 }}>
          <canvas ref={canvasRef} style={{ width: CW, height: CH, borderRadius: 10, display: 'block', maxWidth: '100%', cursor: 'crosshair' }}
            onMouseMove={onCanvasMove} onMouseLeave={() => setHoveredToken(-1)} />
        </Panel>

        {/* Math overlay */}
        <Panel title="📐 Math">
          <div style={{ fontSize: 10, lineHeight: 1.8 }}>
            {step >= 0 && <p style={{ color: step === 0 ? '#22c55e' : '#475569' }}>E = Embed(token) <span style={{ color: '#64748b', fontSize: 8 }}>[1×512]</span></p>}
            {step >= 1 && <><p style={{ color: step === 1 ? '#ef4444' : '#475569' }}>Q = E · W_Q <span style={{ color: '#64748b', fontSize: 8 }}>[1×64]</span></p>
              <p style={{ color: step === 1 ? '#22c55e' : '#475569' }}>K = E · W_K <span style={{ color: '#64748b', fontSize: 8 }}>[1×64]</span></p>
              <p style={{ color: step === 1 ? '#3b82f6' : '#475569' }}>V = E · W_V <span style={{ color: '#64748b', fontSize: 8 }}>[1×64]</span></p></>}
            {step >= 2 && <p style={{ color: step === 2 ? '#f59e0b' : '#475569' }}>Score = Q · Kᵀ / √d_k</p>}
            {step >= 3 && <p style={{ color: step === 3 ? '#a855f7' : '#475569' }}>α = softmax(Scores)</p>}
            {step >= 4 && <p style={{ color: step === 4 ? '#ec4899' : '#475569' }}>Out = Σ(αᵢ · Vᵢ)</p>}
            {step >= 5 && <p style={{ color: '#22c55e', fontWeight: 700 }}>→ {ex.emoji} {ex.meaning.slice(0, 25)}</p>}
          </div>
          {/* Tensor shapes toggle */}
          {step >= 1 && <div style={{ marginTop: 8, padding: 6, borderRadius: 6, background: '#0f172a', fontSize: 8, color: '#64748b' }}>
            W_Q: [512×64] · W_K: [512×64] · W_V: [512×64]<br/>
            Q·Kᵀ: [{n}×{n}] · α: [{n}] · Out: [1×64]
          </div>}
          {/* QKV vectors for focus token */}
          {step >= 1 && <div style={{ marginTop: 6 }}>
            <VecBar values={data.Q[ex.focus]} label={`Q("${ex.tokens[ex.focus]}")`} color="#ef4444" height={20} />
            <VecBar values={data.K[ex.focus]} label={`K("${ex.tokens[ex.focus]}")`} color="#22c55e" height={20} />
            <VecBar values={data.V[ex.focus]} label={`V("${ex.tokens[ex.focus]}")`} color="#3b82f6" height={20} />
          </div>}
        </Panel>
      </div>

      {/* ── STEP CONTROLS ── */}
      <Panel style={{ marginBottom: 10 }}>
        <div style={{ display: 'flex', gap: 4, marginBottom: 8, flexWrap: 'wrap' }}>
          {STEPS.map((s, i) => (
            <button key={i} onClick={() => setStep(i)}
              style={{ padding: '4px 10px', borderRadius: 6, fontSize: 9, fontWeight: 700, background: step === i ? ex.color : step > i ? rgba(ex.color, 0.15) : '#0f172a', color: step >= i ? '#fff' : '#475569', border: `1px solid ${step === i ? ex.color : step > i ? rgba(ex.color, 0.3) : '#1e293b'}`, cursor: 'pointer' }}>
              {i + 1}. {s.label}
            </button>
          ))}
        </div>
        <p style={{ fontSize: 11, color: '#c8d6e5', margin: 0 }}>{STEPS[step].desc}</p>
        <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
          <button onClick={() => setStep(Math.max(0, step - 1))} style={{ padding: '3px 10px', borderRadius: 5, fontSize: 10, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>← Prev</button>
          <button onClick={() => setStep(Math.min(5, step + 1))} style={{ padding: '3px 10px', borderRadius: 5, fontSize: 10, background: ex.color, color: '#fff', border: 'none', cursor: 'pointer' }}>Next →</button>
        </div>
      </Panel>

      {/* ── BOTTOM ROW: Heatmap + Controls + Meaning ── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
        {/* Attention Heatmap */}
        <Panel title="🔥 Attention Heatmap">
          <div style={{ display: 'inline-grid', gridTemplateColumns: `28px repeat(${n}, 32px)`, gap: 2 }}>
            <span />{ex.tokens.map((t, j) => <span key={j} style={{ fontSize: 7, fontFamily: F, color: '#60a5fa', textAlign: 'center', fontWeight: 700 }}>{t.slice(0,4)}</span>)}
            {ex.tokens.map((t, i) => (
              <React.Fragment key={i}>
                <span style={{ fontSize: 7, fontFamily: F, color: '#60a5fa', textAlign: 'right', fontWeight: 700 }}>{t.slice(0,4)}</span>
                {ex.tokens.map((_, j) => {
                  const isRow = i === ex.focus;
                  const v = isRow ? adjustedAttn[j] : (i === j ? 0.4 : 0.05 + Math.random() * 0.1);
                  return <div key={j} style={{ width: 32, height: 22, borderRadius: 3, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 7, fontWeight: 700, fontFamily: F, color: v > 0.2 ? '#fff' : '#475569', background: rgba(isRow ? ex.color : '#3b82f6', v * 0.85 + 0.05), border: isRow && j === ex.focus ? `1.5px solid ${ex.color}` : '1px solid transparent' }}>
                    {(v * 100).toFixed(0)}%
                  </div>;
                })}
              </React.Fragment>
            ))}
          </div>
          <p style={{ fontSize: 7, color: '#475569', marginTop: 4 }}>Row "{ex.tokens[ex.focus]}" highlighted — shows attention distribution</p>
        </Panel>

        {/* Controls */}
        <Panel title="🎛️ Controls">
          <div style={{ marginBottom: 10 }}>
            <label style={{ fontSize: 9, color: '#94a3b8', display: 'block', marginBottom: 2 }}>Attention Sharpness (temperature⁻¹)</label>
            <input type="range" min="0.2" max="3" step="0.1" value={sharpness} onChange={e => setSharpness(+e.target.value)}
              style={{ width: '100%', accentColor: ex.color }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: '#475569' }}><span>Soft</span><span>{sharpness.toFixed(1)}</span><span>Sharp</span></div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <button onClick={() => setMultiHead(!multiHead)}
              style={{ padding: '5px 10px', borderRadius: 6, fontSize: 10, fontWeight: 700, background: multiHead ? '#f59e0b22' : '#0f172a', color: multiHead ? '#f59e0b' : '#64748b', border: `1px solid ${multiHead ? '#f59e0b' : '#334155'}`, cursor: 'pointer' }}>
              👁️ Multi-Head: {multiHead ? 'ON' : 'OFF'}
            </button>
            <button onClick={() => setShowNoise(!showNoise)}
              style={{ padding: '5px 10px', borderRadius: 6, fontSize: 10, fontWeight: 700, background: showNoise ? '#a855f722' : '#0f172a', color: showNoise ? '#a855f7' : '#64748b', border: `1px solid ${showNoise ? '#a855f7' : '#334155'}`, cursor: 'pointer' }}>
              ✨ Noise: {showNoise ? 'ON' : 'OFF'}
            </button>
            <button onClick={() => setPreset(Math.floor(Math.random() * 3))}
              style={{ padding: '5px 10px', borderRadius: 6, fontSize: 10, fontWeight: 700, background: '#0f172a', color: '#64748b', border: '1px solid #334155', cursor: 'pointer' }}>
              🔀 Random Context
            </button>
          </div>
        </Panel>

        {/* Meaning Transformation */}
        <Panel title={`${ex.emoji} Meaning Resolved`}>
          {step >= 5 ? (
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 48, marginBottom: 8 }}>{ex.emoji}</div>
              <p style={{ fontSize: 14, fontWeight: 800, color: ex.color }}>{ex.meaning}</p>
              <p style={{ fontSize: 9, color: '#64748b', marginTop: 6 }}>"{ex.tokens[ex.focus]}" embedding now encodes this meaning</p>
              <div style={{ display: 'flex', gap: 6, marginTop: 10, justifyContent: 'center' }}>
                {PRESETS.filter((_, i) => i !== preset).map((p, i) => (
                  <div key={i} style={{ fontSize: 24, opacity: 0.2, filter: 'grayscale(1)' }}>{p.emoji}</div>
                ))}
              </div>
              <p style={{ fontSize: 8, color: '#334155', marginTop: 4 }}>other meanings faded out</p>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '20px 0' }}>
              <div style={{ display: 'flex', gap: 12, justifyContent: 'center', marginBottom: 8 }}>
                {PRESETS.map((p, i) => <span key={i} style={{ fontSize: 28, opacity: 0.4 }}>{p.emoji}</span>)}
              </div>
              <p style={{ fontSize: 10, color: '#475569' }}>Complete all steps to resolve meaning...</p>
              <p style={{ fontSize: 9, color: '#334155' }}>Step {step + 1}/6</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
