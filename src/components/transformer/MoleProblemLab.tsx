import React, { useState, useMemo } from 'react';

/* ═══ DATA ═══ */
const PRESETS = [
  { sentence: 'American shrew mole', tokens: ['American','shrew','mole'], focus: 2, meaning: 'Small burrowing mammal', emoji: '🐾', color: '#22c55e' },
  { sentence: 'One mole of carbon dioxide', tokens: ['One','mole','of','carbon','dioxide'], focus: 1, meaning: '6.022×10²³ (Avogadro)', emoji: '⚗️', color: '#3b82f6' },
  { sentence: 'Take a biopsy of the mole', tokens: ['Take','a','biopsy','of','the','mole'], focus: 5, meaning: 'Skin growth / lesion', emoji: '🏥', color: '#f59e0b' },
];
const DK = 6;
const FM = "'JetBrains Mono', monospace";
function rgba(h: string, a: number) { const r=parseInt(h.slice(1,3),16),g=parseInt(h.slice(3,5),16),b=parseInt(h.slice(5,7),16); return `rgba(${r},${g},${b},${a})`; }
function softmax(arr: number[]) { const m=Math.max(...arr); const e=arr.map(v=>Math.exp(v-m)); const s=e.reduce((a,b)=>a+b,0); return e.map(v=>v/(s||1)); }
function dot(a: number[], b: number[]) { return a.reduce((s,v,i)=>s+v*(b[i]||0),0); }

function genData(tokens: string[], focus: number) {
  const n = tokens.length;
  const embs = tokens.map((t,i) => Array(DK).fill(0).map((_,d) => +(Math.sin(t.charCodeAt(0)*0.3+d*1.5+i*0.7)*0.6).toFixed(3)));
  const mm = (v: number[], M: number[][]) => M[0].map((_,j) => v.reduce((s,vi,k) => s+vi*(M[k]?.[j]||0),0));
  const Wq = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.sin(r*2.1+c*1.3)*0.4));
  const Wk = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.cos(r*1.7+c*2.3)*0.4));
  const Wv = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.sin(r*0.9+c*3.1)*0.4));
  const Q = embs.map(e => mm(e, Wq).map(v => +v.toFixed(3)));
  const K = embs.map(e => mm(e, Wk).map(v => +v.toFixed(3)));
  const V = embs.map(e => mm(e, Wv).map(v => +v.toFixed(3)));
  // FIXED: iterate over TOKENS (0..n-1), not Q dimensions
  const scores = Array.from({length: n}, (_, j) => +(dot(Q[focus], K[j]) / Math.sqrt(DK)).toFixed(3));
  const attn = softmax(scores);
  const output = Array(DK).fill(0).map((_,d) => +tokens.reduce((s,_2,i) => s+attn[i]*V[i][d], 0).toFixed(3));
  return { embs, Q, K, V, scores, attn, output };
}

/* ═══ SMALL COMPONENTS ═══ */
function P({ children, title, style={} }: any) {
  return <div style={{ background:'rgba(15,23,42,0.8)', border:'1px solid rgba(59,130,246,0.12)', borderRadius:10, padding:12, ...style }}>
    {title && <div style={{ fontSize:9, fontWeight:800, color:'#64748b', marginBottom:6, fontFamily:FM, textTransform:'uppercase', letterSpacing:0.5 }}>{title}</div>}
    {children}
  </div>;
}

function VecBar({ vals, label, color }: { vals: number[], label: string, color: string }) {
  return <div style={{ marginBottom:4 }}>
    <span style={{ fontSize:7, fontFamily:FM, color:'#475569' }}>{label}</span>
    <div style={{ display:'flex', gap:1, height:18, alignItems:'flex-end' }}>
      {vals.map((v,i) => <div key={i} style={{ width:7, height:`${Math.min(Math.abs(v)/0.6*100,100)}%`, minHeight:2, background:v>=0?color:'#ef4444', borderRadius:1, opacity:0.4+Math.abs(v)*0.8 }} />)}
    </div>
  </div>;
}

/* ═══ MAIN ═══ */
export default function MoleProblemLab() {
  const [preset, setPreset] = useState(0);
  const [step, setStep] = useState(0);
  const [multiHead, setMultiHead] = useState(false);
  const [sharpness, setSharpness] = useState(1.0);
  const [hovTok, setHovTok] = useState(-1);

  const ex = PRESETS[preset];
  const data = useMemo(() => genData(ex.tokens, ex.focus), [preset]);
  const n = ex.tokens.length;
  const { adjustedAttn } = useMemo(() => {
    const s = data.scores.map(v => v * sharpness);
    return { adjustedScores: s, adjustedAttn: softmax(s) };
  }, [data.scores, sharpness]);

  const STEPS = [
    { label:'1. Embeddings', desc:'Each token is mapped to a 512-dim vector. The word "mole" gets the SAME initial vector in all 3 sentences — it doesn\'t know its context yet.' },
    { label:'2. Q, K, V', desc:'Each embedding is projected into 3 vectors: Query ("What am I looking for?"), Key ("What do I contain?"), Value ("What info do I share?").' },
    { label:'3. Attention Scores', desc:`We compute dot(Q_mole, K_each) — measuring how relevant each token is to "${ex.tokens[ex.focus]}". Higher = more relevant.` },
    { label:'4. Softmax', desc:'Scores are normalized to probabilities (sum = 100%). This is the ATTENTION PATTERN — how much weight each token gets.' },
    { label:'5. Weighted Output', desc:'Each Value vector is multiplied by its attention weight and summed. This produces a NEW embedding for "mole" that encodes context.' },
    { label:'6. Meaning ✓', desc:`After attention: "${ex.tokens[ex.focus]}" is now understood as ${ex.emoji} ${ex.meaning}. The embedding has been transformed by context!` },
  ];

  return (
    <div style={{ fontFamily:FM, color:'#e2e8f0' }}>
      {/* ── SENTENCE SELECTOR ── */}
      <P title="📝 Select a sentence" style={{ marginBottom:8 }}>
        <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
          {PRESETS.map((p,i) => (
            <button key={i} onClick={() => { setPreset(i); setStep(0); }}
              style={{ padding:'5px 12px', borderRadius:7, fontSize:11, fontWeight:700, background:preset===i?p.color:'rgba(30,41,59,0.8)', color:preset===i?'#fff':'#94a3b8', border:`1.5px solid ${preset===i?p.color:'#334155'}`, cursor:'pointer', transition:'all 0.2s' }}>
              {p.emoji} {p.sentence}
            </button>
          ))}
        </div>
      </P>

      {/* ── TOKENS + ATTENTION ARROWS (pure DOM) ── */}
      <P title="🧊 Token Visualization" style={{ marginBottom:8 }}>
        {/* Softmax bars (step 3+) */}
        {step >= 3 && <div style={{ display:'flex', justifyContent:'center', gap:8, marginBottom:4, height:50, alignItems:'flex-end' }}>
          {ex.tokens.map((tok,i) => {
            const w = adjustedAttn[i];
            return <div key={i} style={{ textAlign:'center', width:60 }}>
              <div style={{ fontSize:8, color:w>0.15?ex.color:'#475569', fontWeight:700, marginBottom:2 }}>{(w*100).toFixed(0)}%</div>
              <div style={{ height:w*40, background:rgba(ex.color, 0.3+w*0.5), borderRadius:4, transition:'height 0.5s', margin:'0 auto', width:30 }} />
            </div>;
          })}
        </div>}

        {/* Attention lines (SVG) */}
        {step >= 2 && <svg width="100%" height={60} style={{ display:'block', overflow:'visible' }}>
          {ex.tokens.map((_,j) => {
            const w = step >= 3 ? adjustedAttn[j] : Math.max(0.05, (data.scores[j]+1.5)/4);
            const x1 = `${(ex.focus+0.5)/n*100}%`, x2 = `${(j+0.5)/n*100}%`;
            return <g key={j}>
              <line x1={x1} y1={55} x2={x2} y2={55} stroke={rgba(ex.color, w*0.6+0.05)} strokeWidth={w*6+0.5} strokeLinecap="round" />
              <circle cx={x2} cy={55} r={2+w*4} fill={rgba(ex.color, w*0.8)} />
              {multiHead && ['#ef4444','#22c55e','#3b82f6','#f59e0b'].map((hc,hi) => {
                const hw = Math.abs(Math.sin(hi*2.3+j*1.7))*0.5+0.1;
                return <line key={hi} x1={x1} y1={48-hi*4} x2={x2} y2={48-hi*4} stroke={rgba(hc, hw*0.4)} strokeWidth={hw*2} strokeLinecap="round" />;
              })}
            </g>;
          })}
          {multiHead && <g>{['#ef4444','#22c55e','#3b82f6','#f59e0b'].map((c,i) => <text key={i} x={4} y={50-i*4} fill={c} fontSize={7} fontFamily={FM}>H{i+1}</text>)}</g>}
        </svg>}

        {/* Token cubes */}
        <div style={{ display:'flex', justifyContent:'center', gap:8, flexWrap:'wrap', margin:'8px 0' }}>
          {ex.tokens.map((tok,i) => {
            const isFoc = i===ex.focus;
            const isHov = hovTok===i;
            return <div key={i} style={{ textAlign:'center', cursor:'pointer', position:'relative' }}
              onMouseEnter={() => setHovTok(i)} onMouseLeave={() => setHovTok(-1)}>
              {/* QKV dots */}
              {step >= 1 && <div style={{ display:'flex', gap:3, justifyContent:'center', marginBottom:4 }}>
                {[{c:'#ef4444',l:'Q'},{c:'#22c55e',l:'K'},{c:'#3b82f6',l:'V'}].map(({c,l}) =>
                  <div key={l} style={{ width:14, height:14, borderRadius:7, background:rgba(c,0.7), display:'flex', alignItems:'center', justifyContent:'center', fontSize:6, color:'#fff', fontWeight:800 }}>{l}</div>
                )}
              </div>}
              {/* Cube */}
              <div style={{ padding:'8px 14px', borderRadius:8, fontWeight:700, fontSize:isFoc?14:12,
                background:isFoc?rgba(ex.color,0.2):'rgba(30,41,59,0.6)', color:isFoc?'#fff':'#94a3b8',
                border:`2px solid ${isFoc?ex.color:isHov?'#475569':'transparent'}`,
                boxShadow:isFoc?`0 0 20px ${rgba(ex.color,0.4)}`:'none',
                transition:'all 0.3s', transform:isFoc?'scale(1.08)':'scale(1)' }}>
                {tok}
              </div>
              {/* Embedding bars */}
              {step >= 0 && <div style={{ display:'flex', gap:1, justifyContent:'center', marginTop:4, height:20, alignItems:'flex-end' }}>
                {data.embs[i].map((v,d) => <div key={d} style={{ width:5, height:`${Math.abs(v)/0.6*100}%`, minHeight:1, background:rgba(isFoc?ex.color:'#475569',0.4+Math.abs(v)*0.5), borderRadius:1 }} />)}
              </div>}
              {/* Hover popup */}
              {isHov && <div style={{ position:'absolute', top:'100%', left:'50%', transform:'translateX(-50%)', marginTop:8, padding:'6px 10px', borderRadius:6, background:'rgba(0,0,0,0.9)', border:'1px solid #334155', zIndex:20, whiteSpace:'nowrap', fontSize:8, textAlign:'left' }}>
                <div style={{ color:'#94a3b8', fontWeight:700 }}>E("{tok}")</div>
                <div style={{ color:'#e2e8f0' }}>[{data.embs[i].map(v=>v.toFixed(2)).join(', ')}]</div>
                {step>=1 && <><div style={{ color:'#ef4444', marginTop:2 }}>Q: [{data.Q[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div>
                  <div style={{ color:'#22c55e' }}>K: [{data.K[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div>
                  <div style={{ color:'#3b82f6' }}>V: [{data.V[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div></>}
                {step>=3 && <div style={{ color:ex.color, marginTop:2, fontWeight:700 }}>Attention: {(adjustedAttn[i]*100).toFixed(1)}%</div>}
              </div>}
            </div>;
          })}
        </div>

        {/* Output vector (step 4+) */}
        {step >= 4 && <div style={{ textAlign:'center', marginTop:8 }}>
          <div style={{ fontSize:9, color:'#ec4899', fontWeight:700, marginBottom:4 }}>Output = Σ(α × V)</div>
          <div style={{ display:'flex', gap:2, justifyContent:'center' }}>
            {data.output.map((v,d) => <div key={d} style={{ width:24, height:20, borderRadius:3, display:'flex', alignItems:'center', justifyContent:'center', fontSize:7, fontWeight:700, color:'#fff', background:rgba('#ec4899', Math.abs(v)*2+0.15) }}>{v.toFixed(2)}</div>)}
          </div>
        </div>}

        {/* Meaning (step 5) */}
        {step >= 5 && <div style={{ textAlign:'center', marginTop:12, padding:12, borderRadius:10, background:rgba(ex.color,0.08), border:`1px solid ${rgba(ex.color,0.25)}` }}>
          <span style={{ fontSize:36 }}>{ex.emoji}</span>
          <p style={{ fontSize:13, fontWeight:800, color:ex.color, margin:'4px 0' }}>{ex.meaning}</p>
          <p style={{ fontSize:8, color:'#475569' }}>"{ex.tokens[ex.focus]}" embedding transformed by attention</p>
          <div style={{ display:'flex', gap:10, justifyContent:'center', marginTop:8 }}>
            {PRESETS.filter((_,i)=>i!==preset).map((p,i) => <span key={i} style={{ fontSize:20, opacity:0.15, filter:'grayscale(1)' }}>{p.emoji}</span>)}
          </div>
        </div>}
      </P>

      {/* ── STEP CONTROLS + MATH (side by side) ── */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 200px', gap:8, marginBottom:8 }}>
        <P>
          <div style={{ display:'flex', gap:3, marginBottom:8, flexWrap:'wrap' }}>
            {STEPS.map((s,i) => (
              <button key={i} onClick={() => setStep(i)}
                style={{ padding:'4px 8px', borderRadius:5, fontSize:8, fontWeight:700, background:step===i?ex.color:step>i?rgba(ex.color,0.12):'#0f172a', color:step>=i?'#fff':'#475569', border:`1px solid ${step===i?ex.color:step>i?rgba(ex.color,0.25):'#1e293b'}`, cursor:'pointer', transition:'all 0.2s' }}>
                {s.label}
              </button>
            ))}
          </div>
          <p style={{ fontSize:11, color:'#c8d6e5', margin:0, lineHeight:1.6 }}>{STEPS[step].desc}</p>
          <div style={{ display:'flex', gap:6, marginTop:8 }}>
            <button onClick={() => setStep(Math.max(0,step-1))} disabled={step===0} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:'#1e293b', color:step===0?'#334155':'#94a3b8', border:'1px solid #334155', cursor:step===0?'default':'pointer' }}>← Prev</button>
            <button onClick={() => setStep(Math.min(5,step+1))} disabled={step===5} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:step===5?'#334155':ex.color, color:'#fff', border:'none', cursor:step===5?'default':'pointer' }}>Next →</button>
            <button onClick={() => { let s=0; const iv=setInterval(()=>{s++;if(s>5)clearInterval(iv);else setStep(s);},1200); }} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:'#0f172a', color:'#94a3b8', border:'1px solid #334155', cursor:'pointer' }}>▶ Auto Play</button>
          </div>
        </P>

        {/* Math panel */}
        <P title="📐 Math">
          <div style={{ fontSize:9, lineHeight:2.2 }}>
            <p style={{ color:step===0?'#22c55e':'#334155', margin:0 }}>E = Embed(tok) <span style={{fontSize:7}}>[1×512]</span></p>
            {step>=1 && <><p style={{ color:step===1?'#ef4444':'#334155', margin:0 }}>Q = E·W_Q <span style={{fontSize:7}}>[1×64]</span></p>
              <p style={{ color:step===1?'#22c55e':'#334155', margin:0 }}>K = E·W_K</p>
              <p style={{ color:step===1?'#3b82f6':'#334155', margin:0 }}>V = E·W_V</p></>}
            {step>=2 && <p style={{ color:step===2?'#f59e0b':'#334155', margin:0 }}>Score=Q·Kᵀ/√d_k</p>}
            {step>=3 && <p style={{ color:step===3?'#a855f7':'#334155', margin:0 }}>α=softmax(Scores)</p>}
            {step>=4 && <p style={{ color:step===4?'#ec4899':'#334155', margin:0 }}>Out=Σ(αᵢ·Vᵢ)</p>}
            {step>=5 && <p style={{ color:'#22c55e', margin:0, fontWeight:700 }}>→{ex.emoji}{ex.meaning.slice(0,18)}</p>}
          </div>
          {step>=1 && <div style={{ marginTop:6 }}>
            <VecBar vals={data.Q[ex.focus]} label={`Q("${ex.tokens[ex.focus]}")`} color="#ef4444" />
            <VecBar vals={data.K[ex.focus]} label="K" color="#22c55e" />
            <VecBar vals={data.V[ex.focus]} label="V" color="#3b82f6" />
          </div>}
        </P>
      </div>

      {/* ── BOTTOM: Heatmap + Controls ── */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
        {/* Heatmap */}
        <P title="🔥 Attention Heatmap">
          <div style={{ display:'inline-grid', gridTemplateColumns:`28px repeat(${n},${Math.min(40, 220/n)}px)`, gap:2 }}>
            <span />{ex.tokens.map((t,j) => <span key={j} style={{ fontSize:7, fontFamily:FM, color:'#60a5fa', textAlign:'center', fontWeight:700 }}>{t.slice(0,5)}</span>)}
            {ex.tokens.map((t,i) => (
              <React.Fragment key={i}>
                <span style={{ fontSize:7, fontFamily:FM, color:'#60a5fa', textAlign:'right', fontWeight:700 }}>{t.slice(0,5)}</span>
                {ex.tokens.map((_,j) => {
                  const isRow = i===ex.focus;
                  const v = isRow ? adjustedAttn[j] : (i===j?0.35:0.06);
                  return <div key={j} style={{ height:20, borderRadius:3, display:'flex', alignItems:'center', justifyContent:'center', fontSize:7, fontWeight:700, fontFamily:FM, color:v>0.12?'#fff':'#475569', background:rgba(isRow?ex.color:'#3b82f6',v*0.8+0.05), border:isRow&&j===ex.focus?`1.5px solid ${ex.color}`:'1px solid transparent', transition:'background 0.3s' }}>
                    {step>=3 ? `${(v*100).toFixed(0)}%` : '·'}
                  </div>;
                })}
              </React.Fragment>
            ))}
          </div>
          <p style={{ fontSize:7, color:'#334155', marginTop:4 }}>Row "{ex.tokens[ex.focus]}" = attention from focus token</p>
        </P>

        {/* Controls */}
        <P title="🎛️ Controls">
          <label style={{ fontSize:8, color:'#64748b', display:'block', marginBottom:2 }}>Attention Sharpness (1/temperature)</label>
          <input type="range" min="0.2" max="3" step="0.1" value={sharpness} onChange={e => setSharpness(+e.target.value)} style={{ width:'100%', accentColor:ex.color }} />
          <div style={{ display:'flex', justifyContent:'space-between', fontSize:7, color:'#475569', marginBottom:10 }}><span>Soft (uniform)</span><span>{sharpness.toFixed(1)}</span><span>Sharp (peaked)</span></div>
          <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
            <button onClick={() => setMultiHead(!multiHead)}
              style={{ padding:'4px 10px', borderRadius:5, fontSize:9, fontWeight:700, background:multiHead?'#f59e0b22':'#0f172a', color:multiHead?'#f59e0b':'#64748b', border:`1px solid ${multiHead?'#f59e0b':'#334155'}`, cursor:'pointer' }}>
              👁️ Multi-Head: {multiHead?'ON':'OFF'}
            </button>
            <button onClick={() => { setPreset(p => (p+1)%3); setStep(0); }}
              style={{ padding:'4px 10px', borderRadius:5, fontSize:9, fontWeight:700, background:'#0f172a', color:'#64748b', border:'1px solid #334155', cursor:'pointer' }}>
              🔀 Next Context
            </button>
          </div>
        </P>
      </div>
    </div>
  );
}
