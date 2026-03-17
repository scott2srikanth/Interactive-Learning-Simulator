import React, { useState, useEffect, useRef, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

/* ═══════════════════════════════════════════════════════════
   SHARED PRIMITIVES
   ═══════════════════════════════════════════════════════════ */
const F = { display: "'Playfair Display', Georgia, serif", mono: "'JetBrains Mono', monospace", sans: "'DM Sans', system-ui, sans-serif" };

// Animated token pill
export function TokenPill({ word, idx, highlight, glow, delay = 0, sub }: { word: string; idx: number; highlight?: string; glow?: boolean; delay?: number; sub?: string }) {
  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: delay * 0.08, type: 'spring', stiffness: 200 }}
      className="relative flex flex-col items-center"
    >
      <motion.div animate={glow ? { boxShadow: ['0 0 0px rgba(59,130,246,0)', '0 0 20px rgba(59,130,246,0.6)', '0 0 0px rgba(59,130,246,0)'] } : {}}
        transition={{ repeat: Infinity, duration: 2 }}
        style={{ padding: '6px 14px', borderRadius: 10, fontFamily: F.mono, fontSize: 13, fontWeight: 700,
          background: highlight || 'rgba(59,130,246,0.15)', color: highlight ? '#fff' : '#60a5fa',
          border: `1.5px solid ${highlight || 'rgba(59,130,246,0.3)'}` }}>
        {word}
      </motion.div>
      {sub && <span style={{ fontSize: 9, color: '#64748b', marginTop: 2, fontFamily: F.mono }}>{sub}</span>}
    </motion.div>
  );
}

// Embedding vector bar — animated fill
export function EmbeddingBar({ values, label, color = '#3b82f6', delay = 0, height = 40 }: any) {
  if (!values || !Array.isArray(values) || values.length === 0) return null;
  return (
    <motion.div initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay }}
      style={{ marginBottom: 6 }}>
      {label && <p style={{ fontSize: 9, fontFamily: F.mono, color: '#64748b', marginBottom: 2 }}>{label}</p>}
      <div style={{ display: 'flex', gap: 1, height }}>
        {values.map((v: number, i: number) => (
          <motion.div key={i} initial={{ height: 0 }} animate={{ height: `${Math.abs(v) / 1.5 * 100}%` }}
            transition={{ delay: delay + i * 0.03, type: 'spring', stiffness: 120 }}
            style={{ width: 14, background: v >= 0 ? color : '#ef4444', borderRadius: 2, alignSelf: 'flex-end', opacity: 0.4 + Math.abs(v) * 0.6 }}
          />
        ))}
      </div>
    </motion.div>
  );
}

// Animated arrow between elements
export function FlowArrow({ label, delay = 0, color = '#475569' }: { label?: string; delay?: number; color?: string }) {
  return (
    <motion.div initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }} transition={{ delay }}
      className="flex flex-col items-center mx-2" style={{ minWidth: 30 }}>
      <div style={{ width: 40, height: 2, background: color, position: 'relative' }}>
        <div style={{ position: 'absolute', right: -4, top: -4, width: 0, height: 0, borderLeft: `8px solid ${color}`, borderTop: '5px solid transparent', borderBottom: '5px solid transparent' }} />
      </div>
      {label && <span style={{ fontSize: 8, color, fontFamily: F.mono, marginTop: 2 }}>{label}</span>}
    </motion.div>
  );
}

// Matrix cell grid with animated reveal
export function MatrixGrid({ data, rowLabels, colLabel, color = '#3b82f6', cellSize = 32, delay = 0, highlightCell }: any) {
  if (!data?.length) return null;
  const rows = data.length, cols = data[0]?.length || 0;
  return (
    <div>
      {colLabel && <p style={{ fontSize: 9, fontFamily: F.mono, fontWeight: 700, color, marginBottom: 4 }}>{colLabel}</p>}
      <div style={{ display: 'inline-grid', gridTemplateColumns: rowLabels ? `40px repeat(${cols}, ${cellSize}px)` : `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.map((row: number[], i: number) => (
          <React.Fragment key={i}>
            {rowLabels && <span style={{ fontSize: 8, fontFamily: F.mono, color: '#64748b', alignSelf: 'center', textAlign: 'right', paddingRight: 4 }}>{rowLabels[i]}</span>}
            {row.map((v: number, j: number) => {
              const hl = highlightCell && highlightCell[0] === i && highlightCell[1] === j;
              return (
                <motion.div key={j} initial={{ opacity: 0, scale: 0.5 }} animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: delay + (i * cols + j) * 0.015 }}
                  style={{ width: cellSize, height: cellSize - 4, display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: cellSize < 28 ? 7 : 9, fontWeight: 700, fontFamily: F.mono, borderRadius: 3,
                    background: hl ? '#facc15' : `${color}${Math.round(Math.min(Math.abs(v), 1) * 140 + 30).toString(16).padStart(2, '0')}`,
                    color: '#fff', border: hl ? '2px solid #facc15' : `1px solid ${color}33`,
                    transition: 'background 0.3s' }}>
                  {v.toFixed(2)}
                </motion.div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// Attention arrows SVG — lines from source tokens to target with varying opacity
export function AttentionArrows({ tokens, weights, targetIdx, width = 500, height = 80 }: any) {
  const n = tokens.length;
  const gap = width / (n + 1);
  return (
    <svg width={width} height={height} style={{ overflow: 'visible' }}>
      {tokens.map((_: string, j: number) => {
        const w = weights[j] || 0;
        const x1 = gap * (j + 1), x2 = gap * (targetIdx + 1);
        const cpY = height * 0.3 - w * height * 0.4;
        return (
          <motion.g key={j} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: j * 0.1 }}>
            <motion.path d={`M${x1},${height - 10} Q${(x1 + x2) / 2},${cpY} ${x2},${height - 10}`}
              fill="none" stroke="#3b82f6" strokeWidth={w * 4 + 0.5}
              initial={{ pathLength: 0, opacity: 0 }} animate={{ pathLength: 1, opacity: w * 0.8 + 0.1 }}
              transition={{ delay: j * 0.12, duration: 0.6 }} />
          </motion.g>
        );
      })}
      {tokens.map((t: string, i: number) => (
        <text key={i} x={gap * (i + 1)} y={height} textAnchor="middle" fill={i === targetIdx ? '#facc15' : '#94a3b8'}
          fontFamily={F.mono} fontSize={11} fontWeight={700}>{t}</text>
      ))}
    </svg>
  );
}

// Dot product grid — circle sizes represent dot product magnitudes
export function DotGrid({ tokens, scores, delay = 0 }: any) {
  const n = tokens.length;
  const cs = 36;
  return (
    <div>
      <p style={{ fontSize: 9, fontFamily: F.mono, color: '#64748b', marginBottom: 4 }}>Attention Scores (dot size = magnitude)</p>
      <div style={{ display: 'inline-grid', gridTemplateColumns: `40px repeat(${n}, ${cs}px)`, gap: 2, alignItems: 'center' }}>
        <span />
        {tokens.map((t: string, j: number) => <span key={j} style={{ fontSize: 8, fontFamily: F.mono, color: '#60a5fa', textAlign: 'center', fontWeight: 700 }}>{t.slice(0, 5)}</span>)}
        {tokens.map((t: string, i: number) => (
          <React.Fragment key={i}>
            <span style={{ fontSize: 8, fontFamily: F.mono, color: '#60a5fa', textAlign: 'right', fontWeight: 700 }}>{t.slice(0, 5)}</span>
            {tokens.map((_: string, j: number) => {
              const v = scores?.[i]?.[j] || 0;
              const size = Math.min(Math.max(Math.abs(v) * 10, 4), cs - 4);
              return (
                <motion.div key={j} initial={{ scale: 0 }} animate={{ scale: 1 }} transition={{ delay: delay + (i * n + j) * 0.02 }}
                  style={{ width: cs, height: cs, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <div style={{ width: size, height: size, borderRadius: '50%', background: v > 0 ? `rgba(59,130,246,${Math.min(v * 0.4 + 0.15, 0.9)})` : `rgba(239,68,68,${Math.min(Math.abs(v) * 0.3 + 0.1, 0.6)})` }} />
                </motion.div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

// Softmax column animation
export function SoftmaxAnim({ input, output, tokens, colIdx, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep(p => Math.min(p + 1, 3)), 1200); return () => clearInterval(t); }, []);
  return (
    <div style={{ padding: 12, borderRadius: 10, background: 'rgba(168,85,247,0.08)', border: '1px solid rgba(168,85,247,0.2)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#a855f7', fontWeight: 700, marginBottom: 8 }}>
        Softmax for column "{tokens[colIdx]}"
      </p>
      <div className="flex gap-6 flex-wrap">
        <div>
          <p style={{ fontSize: 8, color: '#64748b' }}>Raw scores</p>
          {input.map((v: number, i: number) => (
            <motion.div key={i} initial={{ x: -10, opacity: 0 }} animate={{ x: 0, opacity: 1 }} transition={{ delay: delay + i * 0.05 }}
              className="flex items-center gap-2 mb-1">
              <span style={{ fontSize: 9, fontFamily: F.mono, color: '#94a3b8', width: 40 }}>{tokens[i]}</span>
              <span style={{ fontSize: 10, fontFamily: F.mono, color: '#e2e8f0', fontWeight: 700 }}>{v.toFixed(2)}</span>
            </motion.div>
          ))}
        </div>
        {step >= 1 && <div>
          <p style={{ fontSize: 8, color: '#64748b' }}>exp(score)</p>
          {input.map((v: number, i: number) => (
            <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.05 }}
              className="flex items-center gap-2 mb-1">
              <span style={{ fontSize: 10, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700 }}>{Math.exp(v).toFixed(2)}</span>
            </motion.div>
          ))}
        </div>}
        {step >= 2 && <div>
          <p style={{ fontSize: 8, color: '#64748b' }}>÷ sum = softmax</p>
          {output.map((v: number, i: number) => (
            <motion.div key={i} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.05 }}
              className="flex items-center gap-2 mb-1">
              <div style={{ width: 60, height: 14, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                <motion.div initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ delay: i * 0.08, duration: 0.5 }}
                  style={{ height: '100%', background: '#a855f7', borderRadius: 3 }} />
              </div>
              <span style={{ fontSize: 10, fontFamily: F.mono, color: '#a855f7', fontWeight: 700 }}>{(v * 100).toFixed(0)}%</span>
            </motion.div>
          ))}
        </div>}
      </div>
    </div>
  );
}

// Value weighted sum animation
export function ValueWeightedSum({ tokens, attnWeights, values, targetIdx, delay = 0 }: any) {
  return (
    <div style={{ padding: 12, borderRadius: 10, background: 'rgba(236,72,153,0.08)', border: '1px solid rgba(236,72,153,0.2)' }}>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#ec4899', fontWeight: 700, marginBottom: 8 }}>
        Output for "{tokens[targetIdx]}" = Σ (attention × value)
      </p>
      {tokens.map((t: string, i: number) => {
        const w = attnWeights[i];
        return (
          <motion.div key={i} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: delay + i * 0.08 }}
            className="flex items-center gap-2 mb-2 flex-wrap">
            <span style={{ fontSize: 9, fontFamily: F.mono, color: '#a855f7', width: 30, fontWeight: 700 }}>{(w * 100).toFixed(0)}%</span>
            <span style={{ fontSize: 8, color: '#64748b' }}>×</span>
            <span style={{ fontSize: 9, fontFamily: F.mono, color: '#ec4899' }}>V("{t}")</span>
            <span style={{ fontSize: 8, color: '#64748b' }}>=</span>
            <div className="flex gap-0.5">
              {values[i]?.slice(0, 6).map((v: number, j: number) => (
                <div key={j} style={{ width: 20, height: 14, borderRadius: 2, display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 6, fontFamily: F.mono, fontWeight: 700, color: '#fff',
                  background: `rgba(236,72,153,${Math.abs(v * w) * 2 + 0.1})` }}>
                  {(v * w).toFixed(1)}
                </div>
              ))}
            </div>
          </motion.div>
        );
      })}
    </div>
  );
}

// Animated pipeline step
export function PipelineStep({ label, icon, color, active, onClick, delay = 0 }: any) {
  return (
    <motion.button initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }}
      onClick={onClick}
      style={{ padding: '6px 12px', borderRadius: 8, fontSize: 10, fontWeight: 700, fontFamily: F.mono,
        background: active ? color : 'rgba(255,255,255,0.05)', color: active ? '#fff' : '#64748b',
        border: `1.5px solid ${active ? color : '#334155'}`, cursor: 'pointer', transition: 'all 0.2s',
        transform: active ? 'scale(1.05)' : 'scale(1)' }}>
      {icon} {label}
    </motion.button>
  );
}

// Contextual meaning animation — same word, different arrows
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
      <div className="flex gap-2 mb-4">
        {examples.map((e, i) => (
          <button key={i} onClick={() => setActive(i)}
            style={{ padding: '3px 8px', borderRadius: 6, fontSize: 9, fontWeight: 700, fontFamily: F.mono,
              background: i === active ? e.color : 'transparent', color: i === active ? '#fff' : '#64748b',
              border: `1px solid ${i === active ? e.color : '#334155'}`, cursor: 'pointer' }}>
            {e.sentence.join(' ')}
          </button>
        ))}
      </div>
      <AnimatePresence mode="wait">
        <motion.div key={active} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}>
          <div className="flex gap-3 items-end mb-3 flex-wrap">
            {ex.sentence.map((w, i) => (
              <TokenPill key={i} word={w} idx={i}
                highlight={i === ex.focus ? ex.color : undefined}
                glow={i === ex.focus}
                sub={i === ex.focus ? '← same word' : undefined} />
            ))}
          </div>
          <motion.div initial={{ opacity: 0, scaleX: 0 }} animate={{ opacity: 1, scaleX: 1 }}
            style={{ padding: '6px 14px', borderRadius: 8, background: `${ex.color}18`, border: `1px solid ${ex.color}33`, display: 'inline-block' }}>
            <span style={{ fontSize: 11, fontFamily: F.mono, color: ex.color, fontWeight: 700 }}>
              "{ex.sentence[ex.focus]}" → {ex.meaning}
            </span>
          </motion.div>
          <p style={{ fontSize: 10, color: '#94a3b8', marginTop: 8 }}>
            The embedding for "<b style={{ color: ex.color }}>{ex.sentence[ex.focus]}</b>" starts the same in all 3 sentences. 
            Attention updates it based on context so the model knows which meaning is intended.
          </p>
        </motion.div>
      </AnimatePresence>
    </motion.div>
  );
}

// QKV projection animation — shows embedding → multiply by W → get Q/K/V
export function QKVProjection({ embedding, query, keyVec, value, token, delay = 0 }: any) {
  const [step, setStep] = useState(0);
  useEffect(() => { const t = setInterval(() => setStep(p => Math.min(p + 1, 3)), 1500); return () => clearInterval(t); }, []);
  return (
    <div style={{ padding: 14, borderRadius: 12, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.15)' }}>
      <p style={{ fontSize: 11, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700, marginBottom: 10 }}>
        Projecting "{token}" into Q, K, V
      </p>
      <div className="flex gap-4 items-center flex-wrap">
        <div>
          <EmbeddingBar values={embedding} label={`E("${token}")`} color="#22c55e" height={36} />
        </div>
        {step >= 1 && <>
          <FlowArrow label="× W_Q" color="#ef4444" delay={0.2} />
          <div><EmbeddingBar values={query} label="Query" color="#ef4444" height={30} delay={0.3} /></div>
        </>}
        {step >= 2 && <>
          <FlowArrow label="× W_K" color="#22c55e" delay={0.4} />
          <div><EmbeddingBar values={keyVec} label="Key" color="#22c55e" height={30} delay={0.5} /></div>
        </>}
        {step >= 3 && <>
          <FlowArrow label="× W_V" color="#3b82f6" delay={0.6} />
          <div><EmbeddingBar values={value} label="Value" color="#3b82f6" height={30} delay={0.7} /></div>
        </>}
      </div>
      <div className="flex gap-3 mt-4">
        {[{ l: 'Query', c: '#ef4444', d: '"What am I looking for?"' },
          { l: 'Key', c: '#22c55e', d: '"What do I contain?"' },
          { l: 'Value', c: '#3b82f6', d: '"What info do I provide?"' }].map((item, i) => (
          <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: step > i ? 1 : 0.2 }}
            style={{ padding: '4px 10px', borderRadius: 6, background: `${item.c}15`, border: `1px solid ${item.c}33`, flex: 1 }}>
            <span style={{ fontSize: 10, fontFamily: F.mono, color: item.c, fontWeight: 700 }}>{item.l}</span>
            <p style={{ fontSize: 8, color: '#94a3b8', marginTop: 2 }}>{item.d}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// Multi-head split animation
export function MultiHeadSplit({ tokens, numHeads, delay = 0 }: any) {
  const headColors = ['#ef4444', '#22c55e', '#3b82f6', '#f59e0b', '#a855f7', '#ec4899'];
  const headLabels = ['Syntactic', 'Semantic', 'Positional', 'Coreference', 'Hierarchical', 'Discourse'];
  return (
    <div>
      <p style={{ fontSize: 10, fontFamily: F.mono, color: '#f59e0b', fontWeight: 700, marginBottom: 10 }}>
        {numHeads} attention heads — each learns different relationships:
      </p>
      <div className="flex gap-3 flex-wrap">
        {Array.from({ length: numHeads }).map((_, h) => (
          <motion.div key={h} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: delay + h * 0.15 }}
            style={{ padding: 10, borderRadius: 10, background: `${headColors[h]}08`, border: `1.5px solid ${headColors[h]}33`, minWidth: 100 }}>
            <p style={{ fontSize: 10, fontFamily: F.mono, color: headColors[h], fontWeight: 700, marginBottom: 4 }}>
              Head {h + 1}
            </p>
            <p style={{ fontSize: 8, color: '#94a3b8' }}>{headLabels[h] || `Pattern ${h + 1}`}</p>
            {/* Mini attention grid */}
            <div style={{ display: 'grid', gridTemplateColumns: `repeat(${Math.min(tokens.length, 5)}, 10px)`, gap: 1, marginTop: 6 }}>
              {Array.from({ length: Math.min(tokens.length, 5) ** 2 }).map((_, i) => (
                <motion.div key={i} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: delay + h * 0.15 + i * 0.008 }}
                  style={{ width: 10, height: 10, borderRadius: 2,
                    background: `${headColors[h]}${Math.round(Math.random() * 180 + 40).toString(16).padStart(2, '0')}` }} />
              ))}
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}

// Full transformer block pipeline animation
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
      <div className="flex items-center gap-1 flex-wrap mb-4">
        {stages.map((s, i) => (
          <React.Fragment key={i}>
            {i > 0 && <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 + i * 0.1 }}
              style={{ color: '#334155', fontSize: 12 }}>→</motion.span>}
            <PipelineStep label={s.label} icon={s.icon} color={s.color} active={i === activeStage || i <= activeStage}
              onClick={() => setActiveStage(i)} delay={delay + i * 0.1} />
          </React.Fragment>
        ))}
      </div>
      <AnimatePresence mode="wait">
        {activeStage >= 0 && (
          <motion.div key={activeStage} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}
            style={{ padding: 12, borderRadius: 10, background: `${stages[activeStage].color}10`, border: `1px solid ${stages[activeStage].color}30` }}>
            <p style={{ fontSize: 12, fontFamily: F.mono, fontWeight: 700, color: stages[activeStage].color }}>
              {stages[activeStage].icon} {stages[activeStage].label}
            </p>
            <p style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{stages[activeStage].desc}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
