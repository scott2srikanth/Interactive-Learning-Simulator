import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const F = { display: "'Playfair Display', Georgia, serif", mono: "'JetBrains Mono', monospace", sans: "'DM Sans', system-ui, sans-serif" };

/* ═══ SHARED ═══ */
export function TokenPill({ word, idx, highlight, glow, delay = 0, sub }: any) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: delay * 0.08, type: 'spring', stiffness: 200 }} className="relative flex flex-col items-center">
      <motion.div animate={glow ? { boxShadow: ['0 0 0px rgba(59,130,246,0)', '0 0 20px rgba(59,130,246,0.6)', '0 0 0px rgba(59,130,246,0)'] } : {}} transition={{ repeat: Infinity, duration: 2 }}
        style={{ padding: '6px 14px', borderRadius: 10, fontFamily: F.mono, fontSize: 13, fontWeight: 700, background: highlight || 'rgba(59,130,246,0.15)', color: highlight ? '#fff' : '#60a5fa', border: `1.5px solid ${highlight || 'rgba(59,130,246,0.3)'}` }}>{word}</motion.div>
      {sub && <span style={{ fontSize: 9, color: '#64748b', marginTop: 2, fontFamily: F.mono }}>{sub}</span>}
    </motion.div>
  );
}

export function EmbeddingBar({ values, label, color = '#3b82f6', delay = 0, height = 40 }: any) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  return (
    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay }} style={{ marginBottom: 6 }}>
      {label && <p style={{ fontSize: 9, fontFamily: F.mono, color: '#64748b', marginBottom: 2 }}>{label}</p>}
      <div style={{ display: 'flex', gap: 1, height }}>{values.map((v: number, i: number) => (
        <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${Math.abs(v) / 1.5 * 100}%` }} transition={{ delay: delay + i * 0.03, type: 'spring', stiffness: 120 }}
          style={{ width: 14, background: v >= 0 ? color : '#ef4444', borderRadius: 2, alignSelf: 'flex-end', opacity: 0.4 + Math.abs(v) * 0.6 }} />
      ))}</div>
    </motion.div>
  );
}

export function FlowArrow({ label, delay = 0, color = '#475569' }: any) {
  return (
    <motion.div initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }} transition={{ delay }} className="flex flex-col items-center mx-2" style={{ minWidth: 30 }}>
      <div style={{ width: 40, height: 2, background: color, position: 'relative' }}><div style={{ position: 'absolute', right: -4, top: -4, width: 0, height: 0, borderLeft: `8px solid ${color}`, borderTop: '5px solid transparent', borderBottom: '5px solid transparent' }} /></div>
      {label && <span style={{ fontSize: 8, color, fontFamily: F.mono, marginTop: 2 }}>{label}</span>}
    </motion.div>
  );
}

export function MatrixGrid({ data, rowLabels, colLabel, color = '#3b82f6', cellSize = 32, delay = 0, highlightCell }: any) {
  if (!data?.length) return null;
  const rows = data.length, cols = data[0]?.length || 0;
  return (
    <div>{colLabel && <p style={{ fontSize: 9, fontFamily: F.mono, fontWeight: 700, color, marginBottom: 4 }}>{colLabel}</p>}
      <div style={{ display: 'inline-grid', gridTemplateColumns: rowLabels ? `40px repeat(${cols}, ${cellSize}px)` : `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.map((row: number[], i: number) => (
          <React.Fragment key={i}>{rowLabels && <span style={{ fontSize: 8, fontFamily: F.mono, color: '#64748b', alignSelf: 'center', textAlign: 'right', paddingRight: 4 }}>{rowLabels[i]}</span>}
          {row.map((v: number, j: number) => {
            const hl = highlightCell && highlightCell[0] === i && highlightCell[1] === j;
            return <motion.div key={j} initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: delay + (i * cols + j) * 0.015 }}
              style={{ width: cellSize, height: cellSize - 4, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: cellSize < 28 ? 7 : 9, fontWeight: 700, fontFamily: F.mono, borderRadius: 3, background: hl ? '#facc15' : `${color}${Math.round(Math.min(Math.abs(v), 1) * 140 + 30).toString(16).padStart(2, '0')}`, color: '#fff', border: hl ? '2px solid #facc15' : `1px solid ${color}33` }}>{v.toFixed(2)}</motion.div>;
          })}</React.Fragment>
        ))}
      </div>
    </div>
  );
}

export function AttentionArrows({ tokens, weights, targetIdx, width = 500, height = 80 }: any) {
  const n = tokens.length; const gap = width / (n + 1);
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {tokens.map((_: string, j: number) => { const w = weights[j] || 0; const x1 = gap * (j + 1), x2 = gap * (targetIdx + 1); const cpY = height * 0.3 - w * height * 0.4;
        return <motion.g key={j} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: j * 0.1 }}><motion.path d={`M${x1},${height - 10} Q${(x1 + x2) / 2},${cpY} ${x2},${height - 10}`} fill="none" stroke="#3b82f6" strokeWidth={w * 4 + 0.5} initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: w * 0.8 + 0.1 }} transition={{ delay: j * 0.12, duration: 0.6 }} /></motion.g>;
      })}
      {tokens.map((t: string, i: number) => <text key={i} x={gap * (i + 1)} y={height} textAnchor="middle" fill={i === targetIdx ? '#facc15' : '#94a3b8'} fontFamily={F.mono} fontSize={11} fontWeight={700}>{t}</text>)}
    </svg>
  );
}

export function DotGrid({ tokens, scores, delay = 0 }: any) {
  const n = tokens.length; const cs = 36;
  return (
    <div><p style={{ fontSize: 9, fontFamily: F.mono, color: '#64748b', marginBottom: 4 }}>Dot sizes = attention strength</p>
      <div style={{ display: 'inline-grid', gridTemplateColumns: `40px repeat(${n}, ${cs}px)`, gap: 2, alignItems: 'center' }}>
        <span />{tokens.map((t: string, j: number) => <span key={j} style={{ fontSize: 8, fontFamily: F.mono, color: '#60a5fa', textAlign: 'center', fontWeight: 700 }}>{t.slice(0, 5)}</span>)}
        {tokens.map((t: string, i: number) => (
          <React.Fragment key={i}><span style={{ fontSize: 8, fontFamily: F.mono, color: '#60a5fa', textAlign: 'right', fontWeight: 700 }}>{t.slice(0, 5)}</span>
          {tokens.map((_: string, j: number) => { const v = scores?.[i]?.[j] || 0; const size = Math.min(Math.max(Math.abs(v) * 10, 4), cs - 4);
            return <motion.div key={j} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: delay + (i * n + j) * 0.02 }} style={{ width: cs, height: cs, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ width: size, height: size, borderRadius: '50%', background: v > 0 ? `rgba(59,130,246,${Math.min(v * 0.4 + 0.15, 0.9)})` : `rgba(239,68,68,${Math.min(Math.abs(v) * 0.3 + 0.1, 0.6)})` }} />
            </motion.div>;
          })}</React.Fragment>
        ))}
      </div>
    </div>
  );
}

export function SoftmaxAnim({ input, output, tokens, colIdx, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep(p => Math.min(p + 1, 3)), 1200); return () => clearInterval(t); }, []);
  return (
    <div style={{ padding: 12, borderRadius: 10, background: 'rgba(168,85,247,0.08)', border: '1px solid rgba(168,85,247,0.2)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#a855f7', fontWeight: 700, marginBottom: 8 }}>Softmax for "{tokens[colIdx]}"</p>
      <div className="flex gap-6 flex-wrap">
        <div><p style={{ fontSize: 8, color: '#64748b' }}>Raw scores</p>{input.map((v: number, i: number) => (
          <motion.div key={i} initial={{ x: -10, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: delay + i * 0.05 }} className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 9, fontFamily: F.mono, color: '#94a3b8', width: 40 }}>{tokens[i]}</span><span style={{ fontSize: 10, fontFamily: F.mono, color: '#e2e8f0', fontWeight: 700 }}>{v.toFixed(2)}</span>
          </motion.div>))}</div>
        {step >= 1 && <div><p style={{ fontSize: 8, color: '#64748b' }}>exp(score)</p>{input.map((v: number, i: number) => (
          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }} className="flex items-center gap-2 mb-1">
            <span style={{ fontSize: 10, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700 }}>{Math.exp(v).toFixed(2)}</span>
          </motion.div>))}</div>}
        {step >= 2 && <div><p style={{ fontSize: 8, color: '#64748b' }}>÷ sum = softmax</p>{output.map((v: number, i: number) => (
          <motion.div key={i} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }} className="flex items-center gap-2 mb-1">
            <div style={{ width: 60, height: 14, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}><motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: i * 0.08, duration: 0.5 }} style={{ height: '100%', background: '#a855f7', borderRadius: 3 }} /></div>
            <span style={{ fontSize: 10, fontFamily: F.mono, color: '#a855f7', fontWeight: 700 }}>{(v * 100).toFixed(0)}%</span>
          </motion.div>))}</div>}
      </div>
    </div>
  );
}

export function ValueWeightedSum({ tokens, attnWeights, values, targetIdx, delay = 0 }: any) {
  return (
    <div style={{ padding: 12, borderRadius: 10, background: 'rgba(236,72,153,0.08)', border: '1px solid rgba(236,72,153,0.2)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#ec4899', fontWeight: 700, marginBottom: 8 }}>Output for "{tokens[targetIdx]}" = Σ (attention × value)</p>
      {tokens.map((t: string, i: number) => { const w = attnWeights[i]; return (
        <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay + i * 0.08 }} className="flex items-center gap-2 mb-2 flex-wrap">
          <span style={{ fontSize: 9, fontFamily: F.mono, color: '#a855f7', width: 30, fontWeight: 700 }}>{(w * 100).toFixed(0)}%</span><span style={{ fontSize: 8, color: '#64748b' }}>×</span>
          <span style={{ fontSize: 9, fontFamily: F.mono, color: '#ec4899' }}>V("{t}")</span><span style={{ fontSize: 8, color: '#64748b' }}>=</span>
          <div className="flex gap-0.5">{values[i]?.slice(0, 6).map((v: number, j: number) => (
            <div key={j} style={{ width: 20, height: 14, borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 6, fontFamily: F.mono, fontWeight: 700, color: '#fff', background: `rgba(236,72,153,${Math.abs(v * w) * 2 + 0.1})` }}>{(v * w).toFixed(1)}</div>
          ))}</div>
        </motion.div>);
      })}
    </div>
  );
}

export function QKVProjection({ embedding, query, keyVec, value, token, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep(p => Math.min(p + 1, 3)), 1500); return () => clearInterval(t); }, []);
  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.15)' }}>
      <p style={{ fontSize: 11, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700, marginBottom: 10 }}>Projecting "{token}" into Q, K, V</p>
      <div className="flex gap-4 items-center flex-wrap">
        <div><EmbeddingBar values={embedding} label={`E("${token}")`} color="#22c55e" height={36} /></div>
        {step >= 1 && <><FlowArrow label="× W_Q" color="#ef4444" delay={0.2} /><div><EmbeddingBar values={query} label="Query" color="#ef4444" height={30} delay={0.3} /></div></>}
        {step >= 2 && <><FlowArrow label="× W_K" color="#22c55e" delay={0.4} /><div><EmbeddingBar values={keyVec} label="Key" color="#22c55e" height={30} delay={0.5} /></div></>}
        {step >= 3 && <><FlowArrow label="× W_V" color="#3b82f6" delay={0.6} /><div><EmbeddingBar values={value} label="Value" color="#3b82f6" height={30} delay={0.7} /></div></>}
      </div>
      <div className="flex gap-3 mt-4">
        {[{ l: 'Query', c: '#ef4444', d: '"What am I looking for?"' }, { l: 'Key', c: '#22c55e', d: '"What do I contain?"' }, { l: 'Value', c: '#3b82f6', d: '"What info do I provide?"' }].map((item, i) => (
          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: step > i ? 1 : 0.2 }} style={{ padding: '4px 10px', borderRadius: 6, background: `${item.c}15`, border: `1px solid ${item.c}33`, flex: 1 }}>
            <span style={{ fontSize: 10, fontFamily: F.mono, color: item.c, fontWeight: 700 }}>{item.l}</span>
            <p style={{ fontSize: 8, color: '#94a3b8', marginTop: 2 }}>{item.d}</p>
          </motion.div>))}
      </div>
    </div>
  );
}

export function MultiHeadSplit({ tokens, numHeads, delay = 0 }: any) {
  const headColors = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#a855f7', '#ec4899', '#06b6d4', '#84cc16'];
  const headLabels = ['Syntactic', 'Semantic', 'Positional', 'Coreference', 'Hierarchical', 'Discourse', 'Temporal', 'Stylistic'];
  return (
    <div><p style={{ fontSize: 10, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700, marginBottom: 10 }}>{numHeads} attention heads — each learns different relationships:</p>
      <div className="flex gap-3 flex-wrap">{Array.from({ length: numHeads }).map((_, h) => (
        <motion.div key={h} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: delay + h * 0.15 }}
          style={{ padding: 10, borderRadius: 10, background: `${headColors[h]}08`, border: `1.5px solid ${headColors[h]}33`, minWidth: 100 }}>
          <p style={{ fontSize: 10, fontFamily: F.mono, color: headColors[h], fontWeight: 700, marginBottom: 4 }}>Head {h + 1}</p>
          <p style={{ fontSize: 8, color: '#94a3b8' }}>{headLabels[h] || `Pattern ${h + 1}`}</p>
          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(tokens.length, 5)}, 10px)`, gap: 1, marginTop: 6 }}>
            {Array.from({ length: Math.min(tokens.length, 5) ** 2 }).map((_, i) => (
              <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: delay + h * 0.15 + i * 0.008 }}
                style={{ width: 10, height: 10, borderRadius: 2, background: `${headColors[h]}${Math.round(Math.random() * 180 + 40).toString(16).padStart(2, '0')}` }} />
            ))}
          </div>
        </motion.div>
      ))}</div>
    </div>
  );
}

export function TransformerBlockAnim({ delay = 0 }: { delay?: number }) {
  const [activeStage, setActiveStage] = useState(-1);
  useEffect(() => { const t = setInterval(() => setActiveStage(p => p < 5 ? p + 1 : -1), 1500); return () => clearInterval(t); }, []);
  const stages = [
    { label: 'Input', icon: '📥', color: '#22c55e', desc: 'Token embeddings + positional encoding enter the block' },
    { label: 'Multi-Head Attention', icon: '👁️', color: '#f59e0b', desc: 'Each token attends to all others via Q·K softmax × V' },
    { label: 'Add & Normalize', icon: '➕', color: '#64748b', desc: 'Residual: output = attention(x) + x, then layer norm' },
    { label: 'Feed-Forward', icon: '🧮', color: '#ec4899', desc: 'Two linear layers with ReLU: d_model → 4×d_model → d_model' },
    { label: 'Add & Normalize', icon: '➕', color: '#64748b', desc: 'Second residual: output = FFN(x) + x, then layer norm' },
    { label: 'Output', icon: '📤', color: '#22c55e', desc: 'Context-enriched embeddings ready for next block or prediction' },
  ];
  return (
    <div>
      <div className="flex items-center gap-1 flex-wrap mb-4">{stages.map((s, i) => (
        <React.Fragment key={i}>{i > 0 && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.1 }} style={{ color: '#334155', fontSize: 12 }}>→</motion.span>}
          <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: delay + i * 0.1 }} onClick={() => setActiveStage(i)}
            style={{ padding: '6px 12px', borderRadius: 8, fontSize: 10, fontWeight: 700, fontFamily: F.mono, background: i <= activeStage ? s.color : 'rgba(255,255,255,0.05)', color: i <= activeStage ? '#fff' : '#64748b', border: `1.5px solid ${i <= activeStage ? s.color : '#334155'}`, cursor: 'pointer' }}>{s.icon} {s.label}</motion.button>
        </React.Fragment>
      ))}</div>
      <AnimatePresence mode="wait">{activeStage >= 0 && (
        <motion.div key={activeStage} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
          style={{ padding: 12, borderRadius: 10, background: `${stages[activeStage].color}10`, border: `1px solid ${stages[activeStage].color}30` }}>
          <p style={{ fontSize: 12, fontFamily: F.mono, fontWeight: 700, color: stages[activeStage].color }}>{stages[activeStage].icon} {stages[activeStage].label}</p>
          <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{stages[activeStage].desc}</p>
        </motion.div>
      )}</AnimatePresence>
    </div>
  );
}

/* ═══ NEW: Tokenization Animation ═══ */
export function TokenizationAnim({ sentence, tokens, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep(p => Math.min(p + 1, 2)), 1500); return () => clearInterval(t); }, []);
  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
        {step === 0 && <p style={{ fontSize: 16, fontFamily: F.mono, color: '#22c55e', fontWeight: 700, textAlign: 'center', padding: 16 }}>"{sentence}"</p>}
        {step >= 1 && <div className="flex gap-1 justify-center flex-wrap" style={{ margin: '12px 0' }}>
          {tokens.map((t: string, i: number) => (
            <motion.div key={i} initial={{ opacity: 0, scale: 0 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.12, type: 'spring' }}
              style={{ padding: '6px 12px', borderRadius: 8, background: '#22c55e22', border: '1.5px solid #22c55e44', fontFamily: F.mono, fontSize: 13, fontWeight: 700, color: '#22c55e' }}>{t}
              {step >= 2 && <div style={{ fontSize: 8, color: '#64748b', textAlign: 'center', marginTop: 2 }}>id={i}</div>}
            </motion.div>
          ))}
        </div>}
      </motion.div>
      <div className="flex gap-2 justify-center mt-2">
        {['Raw text', 'Split into tokens', 'Assign IDs'].map((l, i) => (
          <span key={i} style={{ fontSize: 9, padding: '2px 8px', borderRadius: 4, fontFamily: F.mono, fontWeight: 700, background: i <= step ? '#22c55e22' : 'transparent', color: i <= step ? '#22c55e' : '#475569', border: `1px solid ${i <= step ? '#22c55e33' : '#33415500'}` }}>{l}</span>
        ))}
      </div>
    </div>
  );
}

/* ═══ NEW: Mask Grid Animation ═══ */
export function MaskGridAnim({ tokens, delay = 0 }: any) {
  const n = tokens.length;
  return (
    <div style={{ padding: 12, borderRadius: 10, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.15)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700, marginBottom: 8 }}>Causal Mask — Who can see whom?</p>
      <div style={{ display: 'inline-grid', gridTemplateColumns: `36px repeat(${n}, 44px)`, gap: 2 }}>
        <span />{tokens.map((t: string, j: number) => <span key={j} style={{ fontSize: 8, fontFamily: F.mono, color: '#f59e0b', textAlign: 'center', fontWeight: 700 }}>{t.slice(0, 4)}</span>)}
        {tokens.map((t: string, i: number) => (
          <React.Fragment key={i}>
            <span style={{ fontSize: 8, fontFamily: F.mono, color: '#f59e0b', textAlign: 'right', fontWeight: 700 }}>{t.slice(0, 4)}</span>
            {tokens.map((_: string, j: number) => (
              <motion.div key={j} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: delay + (i * n + j) * 0.04 }}
                style={{ width: 44, height: 28, borderRadius: 5, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 12, fontWeight: 800, background: j <= i ? '#22c55e22' : '#ef444422', color: j <= i ? '#22c55e' : '#ef4444', border: `1px solid ${j <= i ? '#22c55e33' : '#ef444433'}` }}>
                {j <= i ? '✓' : '✗'}
              </motion.div>
            ))}
          </React.Fragment>
        ))}
      </div>
      <p style={{ fontSize: 9, color: '#94a3b8', marginTop: 6 }}>✓ = can attend, ✗ = masked to -∞ (becomes 0% after softmax)</p>
    </div>
  );
}

/* ═══ NEW: Cross-Attention Flow ═══ */
export function CrossAttentionFlow({ encTokens, decTokens, connections, delay = 0 }: any) {
  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'rgba(236,72,153,0.06)', border: '1px solid rgba(236,72,153,0.15)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#ec4899', fontWeight: 700, marginBottom: 10 }}>Cross-Attention: Decoder queries → Encoder keys</p>
      <div className="flex gap-12 justify-center">
        <div><p style={{ fontSize: 9, color: '#22c55e', fontWeight: 700, marginBottom: 4 }}>Encoder (English)</p>
          {encTokens.map((t: string, i: number) => <div key={i} style={{ padding: '4px 10px', marginBottom: 3, borderRadius: 6, background: '#22c55e18', border: '1px solid #22c55e33', fontFamily: F.mono, fontSize: 11, color: '#22c55e', fontWeight: 600 }}>{t}</div>)}
        </div>
        <div><p style={{ fontSize: 9, color: '#a855f7', fontWeight: 700, marginBottom: 4 }}>Decoder (Hindi)</p>
          {decTokens.map((t: string, i: number) => {
            const conn = connections[i];
            return <motion.div key={i} initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay + i * 0.2 }}
              style={{ padding: '4px 10px', marginBottom: 3, borderRadius: 6, background: '#a855f718', border: '1px solid #a855f733', fontFamily: F.mono, fontSize: 11, color: '#a855f7', fontWeight: 600 }}>
              {t} <span style={{ fontSize: 8, color: '#ec4899' }}>← {conn?.from} ({conn?.pct}%)</span>
            </motion.div>;
          })}
        </div>
      </div>
    </div>
  );
}

/* ═══ 3D MODEL MEMORY VISUALIZATION — Canvas rendered ═══ */
export function MemoryCube3D({ layers, delay = 0 }: any) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotX, setRotX] = useState(-20);
  const [rotY, setRotY] = useState(25);
  const [zoom, setZoom] = useState(1);
  const [hovered, setHovered] = useState(-1);
  const [dragging, setDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [inferStep, setInferStep] = useState(-1);
  const [autoInfer, setAutoInfer] = useState(false);

  const W = 600, H = 480;
  const n = layers.length;

  // 3D projection
  const project = (x: number, y: number, z: number) => {
    const radY = (rotY * Math.PI) / 180, radX = (rotX * Math.PI) / 180;
    let x1 = x * Math.cos(radY) - z * Math.sin(radY);
    let z1 = x * Math.sin(radY) + z * Math.cos(radY);
    let y1 = y * Math.cos(radX) - z1 * Math.sin(radX);
    let z2 = y * Math.sin(radX) + z1 * Math.cos(radX);
    const scale = (500 * zoom) / (500 + z2);
    return { x: W / 2 + x1 * scale, y: H / 2 + y1 * scale, scale, z: z2 };
  };

  // Draw
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    c.width = W * 2; c.height = H * 2;
    const ctx = c.getContext('2d')!;
    ctx.scale(2, 2);
    ctx.clearRect(0, 0, W, H);

    // Background grid
    ctx.strokeStyle = '#1e293b33';
    ctx.lineWidth = 0.5;
    for (let i = -200; i <= 200; i += 40) {
      const p1 = project(i, 160, -200), p2 = project(i, 160, 200);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      const p3 = project(-200, 160, i), p4 = project(200, 160, i);
      ctx.beginPath(); ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y); ctx.stroke();
    }

    // Sort layers by z-depth for painter's algorithm
    const layerData = layers.map((layer: any, i: number) => {
      const yPos = 140 - i * 26;
      const w = (layer.w || 160) * 0.7;
      const h = 18;
      const d = 50;
      // 8 corners of a box
      const corners = [
        project(-w/2, yPos, -d/2), project(w/2, yPos, -d/2),
        project(w/2, yPos, d/2), project(-w/2, yPos, d/2),
        project(-w/2, yPos - h, -d/2), project(w/2, yPos - h, -d/2),
        project(w/2, yPos - h, d/2), project(-w/2, yPos - h, d/2),
      ];
      const avgZ = corners.reduce((s, c) => s + c.z, 0) / 8;
      return { layer, i, yPos, w, h, d, corners, avgZ };
    });

    layerData.sort((a: any, b: any) => b.avgZ - a.avgZ);

    layerData.forEach(({ layer, i: idx, corners }: any) => {
      const isHov = hovered === idx;
      const isInfer = inferStep >= 0 && idx <= inferStep;
      const color = layer.color || '#3b82f6';
      const alpha = isHov ? 'cc' : isInfer ? 'aa' : '66';

      // Top face
      ctx.fillStyle = color + alpha;
      ctx.beginPath();
      ctx.moveTo(corners[4].x, corners[4].y); ctx.lineTo(corners[5].x, corners[5].y);
      ctx.lineTo(corners[6].x, corners[6].y); ctx.lineTo(corners[7].x, corners[7].y);
      ctx.closePath(); ctx.fill();
      ctx.strokeStyle = isHov ? '#fff' : color + '88'; ctx.lineWidth = isHov ? 2 : 0.8; ctx.stroke();

      // Front face
      ctx.fillStyle = color + (isHov ? 'bb' : isInfer ? '88' : '44');
      ctx.beginPath();
      ctx.moveTo(corners[0].x, corners[0].y); ctx.lineTo(corners[1].x, corners[1].y);
      ctx.lineTo(corners[5].x, corners[5].y); ctx.lineTo(corners[4].x, corners[4].y);
      ctx.closePath(); ctx.fill(); ctx.stroke();

      // Right face
      ctx.fillStyle = color + (isHov ? '99' : isInfer ? '66' : '33');
      ctx.beginPath();
      ctx.moveTo(corners[1].x, corners[1].y); ctx.lineTo(corners[2].x, corners[2].y);
      ctx.lineTo(corners[6].x, corners[6].y); ctx.lineTo(corners[5].x, corners[5].y);
      ctx.closePath(); ctx.fill(); ctx.stroke();

      // Label on top face
      const cx = (corners[4].x + corners[5].x + corners[6].x + corners[7].x) / 4;
      const cy = (corners[4].y + corners[5].y + corners[6].y + corners[7].y) / 4;
      ctx.fillStyle = isHov ? '#fff' : '#e2e8f0cc';
      ctx.font = `bold ${Math.max(8, 10 * corners[4].scale)}px JetBrains Mono, monospace`;
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      const labelText = layer.label.length > 22 ? layer.label.slice(0, 20) + '..' : layer.label;
      ctx.fillText(labelText, cx, cy);

      // Inference glow
      if (isInfer && inferStep === idx) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(corners[4].x, corners[4].y); ctx.lineTo(corners[5].x, corners[5].y);
        ctx.lineTo(corners[6].x, corners[6].y); ctx.lineTo(corners[7].x, corners[7].y);
        ctx.closePath(); ctx.stroke();
        ctx.shadowBlur = 0;
      }

      // Wireframe connection to next layer (inference flow)
      if (isInfer && idx < n - 1 && inferStep > idx) {
        const nextIdx = idx + 1;
        const nextY = 140 - nextIdx * 26;
        const p1 = project(0, nextY + 18, 0);
        const p2 = project(0, nextY + 18 + 8, 0);
        ctx.strokeStyle = '#facc1588';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(cx, cy + 6); ctx.lineTo(p1.x, p1.y); ctx.stroke();
        ctx.setLineDash([]);
      }
    });

  }, [layers, rotX, rotY, zoom, hovered, inferStep, n]);

  // Mouse handlers
  const onMouseDown = (e: any) => { setDragging(true); setLastMouse({ x: e.clientX, y: e.clientY }); };
  const onMouseMove = (e: any) => {
    if (!dragging) {
      // Hover detection
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const scaleX = W / rect.width, scaleY = H / rect.height;
      let found = -1;
      for (let i = 0; i < n; i++) {
        const yPos = 140 - i * 26;
        const p = project(0, yPos - 9, 0);
        if (Math.abs(mx * scaleX - p.x) < 80 && Math.abs(my * scaleY - p.y) < 14) { found = i; break; }
      }
      setHovered(found);
      return;
    }
    const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
    setRotY(prev => prev + dx * 0.5);
    setRotX(prev => Math.max(-60, Math.min(30, prev + dy * 0.5)));
    setLastMouse({ x: e.clientX, y: e.clientY });
  };
  const onMouseUp = () => setDragging(false);
  const onWheel = (e: any) => { e.preventDefault(); setZoom(prev => Math.max(0.4, Math.min(2.5, prev - e.deltaY * 0.001))); };

  // Auto inference
  useEffect(() => {
    if (autoInfer && inferStep < n - 1) {
      const t = setTimeout(() => setInferStep(p => p + 1), 600);
      return () => clearTimeout(t);
    } else if (inferStep >= n - 1) setAutoInfer(false);
  }, [autoInfer, inferStep, n]);

  return (
    <div style={{ position: 'relative' }}>
      <canvas ref={canvasRef} style={{ width: W, height: H, borderRadius: 12, border: '1px solid #1e293b', cursor: dragging ? 'grabbing' : 'grab', display: 'block', margin: '0 auto', maxWidth: '100%' }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp} onWheel={onWheel} />

      {/* Controls */}
      <div style={{ display: 'flex', gap: 6, justifyContent: 'center', marginTop: 8 }}>
        <button onClick={() => { setInferStep(-1); setTimeout(() => { setInferStep(0); setAutoInfer(true); }, 100); }}
          style={{ padding: '5px 14px', borderRadius: 6, fontSize: 10, fontWeight: 700, background: autoInfer ? '#dc2626' : '#16a34a', color: '#fff', border: 'none', cursor: 'pointer' }}>
          {autoInfer ? '⏸ Pause' : '▶ Run Inference'}
        </button>
        <button onClick={() => { setInferStep(-1); setAutoInfer(false); }} style={{ padding: '5px 10px', borderRadius: 6, fontSize: 10, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>↺ Reset</button>
        <button onClick={() => setZoom(1)} style={{ padding: '5px 10px', borderRadius: 6, fontSize: 10, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>⊙ Reset View</button>
      </div>

      <p style={{ textAlign: 'center', fontSize: 8, color: '#475569', marginTop: 4, fontFamily: F.mono }}>🖱️ Drag to rotate · Scroll to zoom · Hover for details · ▶ to animate inference</p>

      {/* Hover info */}
      {hovered >= 0 && layers[hovered] && (
        <div style={{ position: 'absolute', top: 12, right: 12, padding: '10px 14px', borderRadius: 10, background: 'rgba(0,0,0,0.9)', border: `1.5px solid ${layers[hovered].color}`, maxWidth: 260, zIndex: 10 }}>
          <p style={{ fontSize: 12, fontFamily: F.mono, color: layers[hovered].color, fontWeight: 700, marginBottom: 4 }}>{layers[hovered].label}</p>
          <p style={{ fontSize: 10, color: '#c8d6e5', marginBottom: 4 }}>{layers[hovered].desc}</p>
          <div style={{ display: 'flex', gap: 8 }}>
            <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: F.mono }}>Shape: {layers[hovered].shape}</span>
            <span style={{ fontSize: 9, color: '#22c55e', fontFamily: F.mono }}>{layers[hovered].params}</span>
          </div>
        </div>
      )}

      {/* Inference step label */}
      {inferStep >= 0 && inferStep < n && (
        <div style={{ textAlign: 'center', marginTop: 6, padding: '4px 12px', borderRadius: 6, background: `${layers[inferStep].color}22`, border: `1px solid ${layers[inferStep].color}44`, display: 'inline-block', marginLeft: '50%', transform: 'translateX(-50%)' }}>
          <span style={{ fontSize: 10, fontFamily: F.mono, color: layers[inferStep].color, fontWeight: 700 }}>
            ⚡ Data flowing through: {layers[inferStep].label}
          </span>
        </div>
      )}
    </div>
  );
}

/* ═══ NEW: Inference Pipeline Animation ═══ */
export function InferencePipeline({ srcTokens, tgtTokens, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);
  const steps = [
    { label: 'Input: English tokens', tokens: srcTokens, color: '#22c55e', desc: 'Source sentence is tokenized and embedded' },
    { label: 'Encoder processes all at once', tokens: srcTokens, color: '#3b82f6', desc: 'Self-attention + FFN applied N× times. Output: context-rich source embeddings.' },
    ...tgtTokens.map((t: string, i: number) => ({
      label: `Decoder predicts token ${i + 1}: "${t}"`, tokens: tgtTokens.slice(0, i + 1), color: '#a855f7',
      desc: `Masked self-attention on previous Hindi tokens → Cross-attention to encoder → FFN → Softmax → "${t}"`
    })),
    { label: 'Translation complete!', tokens: tgtTokens, color: '#22c55e', desc: `"${srcTokens.join(' ')}" → "${tgtTokens.join(' ')}"` },
  ];
  useEffect(() => { if (auto && step < steps.length - 1) { const t = setTimeout(() => setStep(p => p + 1), 1800); return () => clearTimeout(t); } else setAuto(false); }, [auto, step, steps.length]);

  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'rgba(34,197,94,0.06)', border: '1px solid rgba(34,197,94,0.15)' }}>
      <div className="flex gap-2 mb-3">
        <button onClick={() => setAuto(!auto)} style={{ padding: '4px 12px', borderRadius: 6, fontSize: 10, fontWeight: 700, background: auto ? '#dc2626' : '#16a34a', color: '#fff', border: 'none', cursor: 'pointer' }}>{auto ? '⏸' : '▶ Play Inference'}</button>
        <button onClick={() => setStep(0)} style={{ padding: '4px 8px', borderRadius: 6, fontSize: 10, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>↺</button>
        <span style={{ fontSize: 9, color: '#475569', alignSelf: 'center' }}>Step {step + 1}/{steps.length}</span>
      </div>
      <div className="flex gap-1 mb-3">{steps.map((_, i) => <div key={i} onClick={() => { setStep(i); setAuto(false); }} style={{ flex: 1, height: 4, borderRadius: 2, background: i <= step ? steps[i].color : '#1e293b', cursor: 'pointer' }} />)}</div>
      <AnimatePresence mode="wait">
        <motion.div key={step} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
          <p style={{ fontSize: 12, fontFamily: F.mono, fontWeight: 700, color: steps[step].color, marginBottom: 6 }}>{steps[step].label}</p>
          <div className="flex gap-2 flex-wrap mb-2">{steps[step].tokens.map((t: string, i: number) => (
            <motion.div key={i} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: i * 0.1 }}
              style={{ padding: '5px 10px', borderRadius: 7, background: `${steps[step].color}22`, border: `1px solid ${steps[step].color}44`, fontFamily: F.mono, fontSize: 12, fontWeight: 700, color: steps[step].color }}>{t}</motion.div>
          ))}</div>
          <p style={{ fontSize: 10, color: '#94a3b8' }}>{steps[step].desc}</p>
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

// Re-export ContextMeaningDemo
export function ContextMeaningDemo({ delay = 0 }: { delay?: number }) {
  const examples = [
    { sentence: ['American', 'shrew', 'mole'], focus: 2, meaning: 'small burrowing mammal', color: '#22c55e' },
    { sentence: ['One', 'mole', 'of', 'carbon'], focus: 1, meaning: 'unit of measurement (6.02×10²³)', color: '#3b82f6' },
    { sentence: ['Take', 'biopsy', 'of', 'mole'], focus: 3, meaning: 'skin growth/lesion', color: '#f59e0b' },
  ];
  const [active, setActive] = useState(0);
  useEffect(() => { const t = setInterval(() => setActive(p => (p + 1) % 3), 3000); return () => clearInterval(t); }, []);
  const ex = examples[active];
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay }}
      style={{ padding: 16, borderRadius: 12, background: 'rgba(59,130,246,0.06)', border: '1px solid rgba(59,130,246,0.15)' }}>
      <div className="flex gap-2 mb-4">{examples.map((e, i) => (
        <button key={i} onClick={() => setActive(i)} style={{ padding: '3px 8px', borderRadius: 6, fontSize: 9, fontWeight: 700, fontFamily: F.mono, background: i === active ? e.color : 'transparent', color: i === active ? '#fff' : '#64748b', border: `1px solid ${i === active ? e.color : '#334155'}`, cursor: 'pointer' }}>{e.sentence.join(' ')}</button>
      ))}</div>
      <AnimatePresence mode="wait">
        <motion.div key={active} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
          <div className="flex gap-3 items-end mb-3 flex-wrap">{ex.sentence.map((w, i) => (
            <TokenPill key={i} word={w} idx={i} highlight={i === ex.focus ? ex.color : undefined} glow={i === ex.focus} sub={i === ex.focus ? '← same word' : undefined} />
          ))}</div>
          <motion.div initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }}
            style={{ padding: '6px 14px', borderRadius: 8, background: `${ex.color}18`, border: `1px solid ${ex.color}33`, display: 'inline-block' }}>
            <span style={{ fontSize: 11, fontFamily: F.mono, color: ex.color, fontWeight: 700 }}>"{ex.sentence[ex.focus]}" → {ex.meaning}</span>
          </motion.div>
          <p style={{ fontSize: 10, color: '#94a3b8', marginTop: 8 }}>The embedding for "<b style={{ color: ex.color }}>{ex.sentence[ex.focus]}</b>" starts the same in all 3 sentences. Attention updates it based on context.</p>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}

export function PipelineStep({ label, icon, color, active, onClick, delay = 0 }: any) {
  return (
    <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }} onClick={onClick}
      style={{ padding: '6px 12px', borderRadius: 8, fontSize: 10, fontWeight: 700, fontFamily: F.mono, background: active ? color : 'rgba(255,255,255,0.05)', color: active ? '#fff' : '#64748b', border: `1.5px solid ${active ? color : '#334155'}`, cursor: 'pointer', transition: 'all 0.2s', transform: active ? 'scale(1.05)' : 'scale(1)' }}>{icon} {label}</motion.button>
  );
}
