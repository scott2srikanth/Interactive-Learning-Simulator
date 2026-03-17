import React, { useState, useMemo } from 'react';
import { useThemeStore } from '../../store/themeStore';

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
  const scores = Array.from({length: n}, (_, j) => +(dot(Q[focus], K[j]) / Math.sqrt(DK)).toFixed(3));
  const attn = softmax(scores);
  const output = Array(DK).fill(0).map((_,d) => +tokens.reduce((s,_2,i) => s+attn[i]*V[i][d], 0).toFixed(3));
  return { embs, Q, K, V, scores, attn, output };
}

// Theme-aware colors
function useC() {
  const dark = useThemeStore(s => s.dark);
  return {
    panelBg: dark ? 'rgba(15,23,42,0.8)' : 'rgba(255,255,255,0.95)',
    panelBorder: dark ? 'rgba(59,130,246,0.12)' : 'rgba(0,0,0,0.08)',
    text: dark ? '#e2e8f0' : '#1e293b',
    textMuted: dark ? '#94a3b8' : '#64748b',
    textDim: dark ? '#64748b' : '#94a3b8',
    textFaint: dark ? '#475569' : '#cbd5e1',
    textLabel: dark ? '#64748b' : '#475569',
    cubeBg: dark ? 'rgba(30,41,59,0.6)' : 'rgba(241,245,249,0.9)',
    cubeBorder: dark ? '#475569' : '#cbd5e1',
    cubeText: dark ? '#94a3b8' : '#475569',
    btnBg: dark ? '#0f172a' : '#f1f5f9',
    btnBorder: dark ? '#334155' : '#e2e8f0',
    btnText: dark ? '#64748b' : '#475569',
    btnActiveBg: dark ? 'rgba(30,41,59,0.8)' : '#fff',
    heatBg: dark ? 'rgba(59,130,246,' : 'rgba(59,130,246,',
    stepInactiveBg: dark ? '#0f172a' : '#f8fafc',
    stepInactiveBorder: dark ? '#1e293b' : '#e2e8f0',
    stepInactiveText: dark ? '#475569' : '#94a3b8',
    popupBg: dark ? 'rgba(0,0,0,0.9)' : 'rgba(255,255,255,0.97)',
    popupBorder: dark ? '#334155' : '#e2e8f0',
    popupText: dark ? '#e2e8f0' : '#1e293b',
    barIdle: dark ? '#475569' : '#cbd5e1',
    dark,
  };
}

function VecBar({ vals, label, color, c }: any) {
  return <div style={{ marginBottom:3 }}>
    <span style={{ fontSize:7, fontFamily:FM, color:c.textDim }}>{label}</span>
    <div style={{ display:'flex', gap:1, height:18, alignItems:'flex-end' }}>
      {vals.map((v: number, i: number) => <div key={i} style={{ width:7, height:`${Math.min(Math.abs(v)/0.6*100,100)}%`, minHeight:2, background:v>=0?color:'#ef4444', borderRadius:1, opacity:0.4+Math.abs(v)*0.8 }} />)}
    </div>
  </div>;
}

export default function MoleProblemLab() {
  const [preset, setPreset] = useState(0);
  const [step, setStep] = useState(0);
  const [multiHead, setMultiHead] = useState(false);
  const [sharpness, setSharpness] = useState(1.0);
  const [hovTok, setHovTok] = useState(-1);
  const c = useC();

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
    { label:'3. Scores', desc:`We compute dot(Q_mole, K_each) — measuring how relevant each token is to "${ex.tokens[ex.focus]}".` },
    { label:'4. Softmax', desc:'Scores are normalized to probabilities (sum = 100%). This is the attention pattern.' },
    { label:'5. Output', desc:'Each Value vector is multiplied by its attention weight and summed → new context-aware embedding.' },
    { label:'6. Meaning ✓', desc:`After attention: "${ex.tokens[ex.focus]}" → ${ex.emoji} ${ex.meaning}` },
  ];

  const panelStyle = { background:c.panelBg, border:`1px solid ${c.panelBorder}`, borderRadius:10, padding:12 };

  return (
    <div style={{ fontFamily:FM, color:c.text }}>
      {/* Sentence selector */}
      <div style={{ ...panelStyle, marginBottom:8 }}>
        <div style={{ fontSize:9, fontWeight:800, color:c.textMuted, marginBottom:6, textTransform:'uppercase', letterSpacing:0.5 }}>📝 Select a sentence</div>
        <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
          {PRESETS.map((p,i) => (
            <button key={i} onClick={() => { setPreset(i); setStep(0); }}
              style={{ padding:'5px 12px', borderRadius:7, fontSize:11, fontWeight:700, background:preset===i?p.color:c.btnActiveBg, color:preset===i?'#fff':c.btnText, border:`1.5px solid ${preset===i?p.color:c.btnBorder}`, cursor:'pointer', transition:'all 0.2s' }}>
              {p.emoji} {p.sentence}
            </button>
          ))}
        </div>
      </div>

      {/* Token visualization */}
      <div style={{ ...panelStyle, marginBottom:8 }}>
        <div style={{ fontSize:9, fontWeight:800, color:c.textMuted, marginBottom:6, textTransform:'uppercase', letterSpacing:0.5 }}>🧊 Token Visualization</div>

        {/* Softmax bars */}
        {step >= 3 && <div style={{ display:'flex', justifyContent:'center', gap:8, marginBottom:4, height:50, alignItems:'flex-end' }}>
          {ex.tokens.map((tok,i) => {
            const w = adjustedAttn[i];
            return <div key={i} style={{ textAlign:'center', width:60 }}>
              <div style={{ fontSize:8, color:w>0.15?ex.color:c.textDim, fontWeight:700, marginBottom:2 }}>{(w*100).toFixed(0)}%</div>
              <div style={{ height:w*40, background:rgba(ex.color, c.dark ? 0.3+w*0.5 : 0.15+w*0.4), borderRadius:4, transition:'height 0.5s', margin:'0 auto', width:30 }} />
            </div>;
          })}
        </div>}

        {/* Attention arrows SVG */}
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
          {multiHead && <g>{['#ef4444','#22c55e','#3b82f6','#f59e0b'].map((col,i) => <text key={i} x={4} y={50-i*4} fill={col} fontSize={7} fontFamily={FM}>H{i+1}</text>)}</g>}
        </svg>}

        {/* Token cubes */}
        <div style={{ display:'flex', justifyContent:'center', gap:10, flexWrap:'wrap', margin:'8px 0' }}>
          {ex.tokens.map((tok,i) => {
            const isFoc = i===ex.focus;
            const isHov = hovTok===i;
            return <div key={i} style={{ textAlign:'center', cursor:'pointer', position:'relative' }}
              onMouseEnter={() => setHovTok(i)} onMouseLeave={() => setHovTok(-1)}>
              {step >= 1 && <div style={{ display:'flex', gap:3, justifyContent:'center', marginBottom:4 }}>
                {[{col:'#ef4444',l:'Q'},{col:'#22c55e',l:'K'},{col:'#3b82f6',l:'V'}].map(({col,l}) =>
                  <div key={l} style={{ width:14, height:14, borderRadius:7, background:rgba(col,0.7), display:'flex', alignItems:'center', justifyContent:'center', fontSize:6, color:'#fff', fontWeight:800 }}>{l}</div>
                )}
              </div>}
              <div style={{ padding:'8px 16px', borderRadius:8, fontWeight:700, fontSize:isFoc?15:12,
                background:isFoc?rgba(ex.color, c.dark?0.2:0.1) : c.cubeBg, color:isFoc?(c.dark?'#fff':'#0f172a'):c.cubeText,
                border:`2px solid ${isFoc?ex.color:isHov?c.cubeBorder:'transparent'}`,
                boxShadow:isFoc?`0 0 20px ${rgba(ex.color,0.3)}`:'none',
                transition:'all 0.3s', transform:isFoc?'scale(1.08)':'scale(1)' }}>
                {tok}
              </div>
              {step >= 0 && <div style={{ display:'flex', gap:1, justifyContent:'center', marginTop:4, height:20, alignItems:'flex-end' }}>
                {data.embs[i].map((v,d) => <div key={d} style={{ width:5, height:`${Math.abs(v)/0.6*100}%`, minHeight:1, background:rgba(isFoc?ex.color:c.barIdle, 0.4+Math.abs(v)*0.5), borderRadius:1 }} />)}
              </div>}
              {isHov && <div style={{ position:'absolute', top:'100%', left:'50%', transform:'translateX(-50%)', marginTop:8, padding:'6px 10px', borderRadius:6, background:c.popupBg, border:`1px solid ${c.popupBorder}`, zIndex:20, whiteSpace:'nowrap', fontSize:8, textAlign:'left', boxShadow:'0 4px 12px rgba(0,0,0,0.15)' }}>
                <div style={{ color:c.textMuted, fontWeight:700 }}>E("{tok}")</div>
                <div style={{ color:c.popupText }}>[{data.embs[i].map(v=>v.toFixed(2)).join(', ')}]</div>
                {step>=1 && <><div style={{ color:'#ef4444', marginTop:2 }}>Q: [{data.Q[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div>
                  <div style={{ color:'#22c55e' }}>K: [{data.K[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div>
                  <div style={{ color:'#3b82f6' }}>V: [{data.V[i].slice(0,3).map(v=>v.toFixed(2)).join(', ')}...]</div></>}
                {step>=3 && <div style={{ color:ex.color, marginTop:2, fontWeight:700 }}>Attn: {(adjustedAttn[i]*100).toFixed(1)}%</div>}
              </div>}
            </div>;
          })}
        </div>

        {step >= 4 && <div style={{ textAlign:'center', marginTop:8 }}>
          <div style={{ fontSize:9, color:'#ec4899', fontWeight:700, marginBottom:4 }}>Output = Σ(α × V)</div>
          <div style={{ display:'flex', gap:2, justifyContent:'center' }}>
            {data.output.map((v,d) => <div key={d} style={{ width:24, height:20, borderRadius:3, display:'flex', alignItems:'center', justifyContent:'center', fontSize:7, fontWeight:700, color:'#fff', background:rgba('#ec4899', Math.abs(v)*2+0.15) }}>{v.toFixed(2)}</div>)}
          </div>
        </div>}

        {step >= 5 && <div style={{ textAlign:'center', marginTop:12, padding:12, borderRadius:10, background:rgba(ex.color, c.dark?0.08:0.06), border:`1px solid ${rgba(ex.color,0.25)}` }}>
          <span style={{ fontSize:36 }}>{ex.emoji}</span>
          <p style={{ fontSize:13, fontWeight:800, color:ex.color, margin:'4px 0' }}>{ex.meaning}</p>
          <p style={{ fontSize:8, color:c.textDim }}>"{ex.tokens[ex.focus]}" embedding transformed by attention</p>
          <div style={{ display:'flex', gap:10, justifyContent:'center', marginTop:8 }}>
            {PRESETS.filter((_,i2)=>i2!==preset).map((p,i2) => <span key={i2} style={{ fontSize:20, opacity:0.15, filter:'grayscale(1)' }}>{p.emoji}</span>)}
          </div>
        </div>}
      </div>

      {/* Steps + Math */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 200px', gap:8, marginBottom:8 }}>
        <div style={panelStyle}>
          <div style={{ display:'flex', gap:3, marginBottom:8, flexWrap:'wrap' }}>
            {STEPS.map((s,i) => (
              <button key={i} onClick={() => setStep(i)}
                style={{ padding:'4px 8px', borderRadius:5, fontSize:8, fontWeight:700,
                  background:step===i?ex.color:step>i?rgba(ex.color, c.dark?0.12:0.08):c.stepInactiveBg,
                  color:step===i?'#fff':step>i?(c.dark?'#fff':ex.color):c.stepInactiveText,
                  border:`1px solid ${step===i?ex.color:step>i?rgba(ex.color,0.25):c.stepInactiveBorder}`, cursor:'pointer', transition:'all 0.2s' }}>
                {s.label}
              </button>
            ))}
          </div>
          <p style={{ fontSize:11, color:c.text, margin:0, lineHeight:1.6 }}>{STEPS[step].desc}</p>
          <div style={{ display:'flex', gap:6, marginTop:8 }}>
            <button onClick={() => setStep(Math.max(0,step-1))} disabled={step===0} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:c.btnBg, color:step===0?c.textFaint:c.textMuted, border:`1px solid ${c.btnBorder}`, cursor:step===0?'default':'pointer' }}>← Prev</button>
            <button onClick={() => setStep(Math.min(5,step+1))} disabled={step===5} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:step===5?c.btnBg:ex.color, color:step===5?c.textFaint:'#fff', border:'none', cursor:step===5?'default':'pointer', borderRadius:5 }}>Next →</button>
            <button onClick={() => { let s2=0; const iv=setInterval(()=>{s2++;if(s2>5)clearInterval(iv);else setStep(s2);},1200); }} style={{ padding:'4px 10px', borderRadius:5, fontSize:10, background:c.btnBg, color:c.textMuted, border:`1px solid ${c.btnBorder}`, cursor:'pointer' }}>▶ Auto</button>
          </div>
        </div>

        <div style={panelStyle}>
          <div style={{ fontSize:9, fontWeight:800, color:c.textMuted, marginBottom:6, textTransform:'uppercase' }}>📐 Math</div>
          <div style={{ fontSize:9, lineHeight:2.2 }}>
            <p style={{ color:step===0?'#22c55e':c.textFaint, margin:0 }}>E = Embed(tok) <span style={{fontSize:7}}>[1×512]</span></p>
            {step>=1 && <><p style={{ color:step===1?'#ef4444':c.textFaint, margin:0 }}>Q = E·W_Q <span style={{fontSize:7}}>[1×64]</span></p>
              <p style={{ color:step===1?'#22c55e':c.textFaint, margin:0 }}>K = E·W_K</p>
              <p style={{ color:step===1?'#3b82f6':c.textFaint, margin:0 }}>V = E·W_V</p></>}
            {step>=2 && <p style={{ color:step===2?'#f59e0b':c.textFaint, margin:0 }}>Score=Q·Kᵀ/√d_k</p>}
            {step>=3 && <p style={{ color:step===3?'#a855f7':c.textFaint, margin:0 }}>α=softmax(Scores)</p>}
            {step>=4 && <p style={{ color:step===4?'#ec4899':c.textFaint, margin:0 }}>Out=Σ(αᵢ·Vᵢ)</p>}
            {step>=5 && <p style={{ color:'#22c55e', margin:0, fontWeight:700 }}>→{ex.emoji}{ex.meaning.slice(0,18)}</p>}
          </div>
          {step>=1 && <div style={{ marginTop:6 }}>
            <VecBar vals={data.Q[ex.focus]} label={`Q("${ex.tokens[ex.focus]}")`} color="#ef4444" c={c} />
            <VecBar vals={data.K[ex.focus]} label="K" color="#22c55e" c={c} />
            <VecBar vals={data.V[ex.focus]} label="V" color="#3b82f6" c={c} />
          </div>}
        </div>
      </div>

      {/* Heatmap + Controls */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:8 }}>
        <div style={panelStyle}>
          <div style={{ fontSize:9, fontWeight:800, color:c.textMuted, marginBottom:6, textTransform:'uppercase' }}>🔥 Attention Heatmap</div>
          <div style={{ display:'inline-grid', gridTemplateColumns:`28px repeat(${n},${Math.min(40, 220/n)}px)`, gap:2 }}>
            <span />{ex.tokens.map((t,j) => <span key={j} style={{ fontSize:7, fontFamily:FM, color:ex.color, textAlign:'center', fontWeight:700 }}>{t.slice(0,5)}</span>)}
            {ex.tokens.map((t,i) => (
              <React.Fragment key={i}>
                <span style={{ fontSize:7, fontFamily:FM, color:ex.color, textAlign:'right', fontWeight:700 }}>{t.slice(0,5)}</span>
                {ex.tokens.map((_,j) => {
                  const isRow = i===ex.focus;
                  const v = isRow ? adjustedAttn[j] : (i===j?0.35:0.06);
                  return <div key={j} style={{ height:22, borderRadius:3, display:'flex', alignItems:'center', justifyContent:'center', fontSize:7, fontWeight:700, fontFamily:FM,
                    color:v>0.12?'#fff':c.textDim,
                    background:rgba(isRow?ex.color:'#3b82f6', c.dark ? v*0.8+0.05 : v*0.6+0.03),
                    border:isRow&&j===ex.focus?`1.5px solid ${ex.color}`:'1px solid transparent', transition:'background 0.3s' }}>
                    {step>=3 ? `${(v*100).toFixed(0)}%` : '·'}
                  </div>;
                })}
              </React.Fragment>
            ))}
          </div>
          <p style={{ fontSize:7, color:c.textFaint, marginTop:4 }}>Row "{ex.tokens[ex.focus]}" = attention from focus</p>
        </div>

        <div style={panelStyle}>
          <div style={{ fontSize:9, fontWeight:800, color:c.textMuted, marginBottom:6, textTransform:'uppercase' }}>🎛️ Controls</div>
          <label style={{ fontSize:8, color:c.textLabel, display:'block', marginBottom:2 }}>Attention Sharpness (1/temperature)</label>
          <input type="range" min="0.2" max="3" step="0.1" value={sharpness} onChange={e => setSharpness(+e.target.value)} style={{ width:'100%', accentColor:ex.color }} />
          <div style={{ display:'flex', justifyContent:'space-between', fontSize:7, color:c.textDim, marginBottom:10 }}><span>Soft</span><span>{sharpness.toFixed(1)}</span><span>Sharp</span></div>
          <div style={{ display:'flex', gap:4, flexWrap:'wrap' }}>
            <button onClick={() => setMultiHead(!multiHead)}
              style={{ padding:'4px 10px', borderRadius:5, fontSize:9, fontWeight:700, background:multiHead?rgba('#f59e0b',0.15):c.btnBg, color:multiHead?'#f59e0b':c.btnText, border:`1px solid ${multiHead?'#f59e0b':c.btnBorder}`, cursor:'pointer' }}>
              👁️ Multi-Head: {multiHead?'ON':'OFF'}
            </button>
            <button onClick={() => { setPreset(p => (p+1)%3); setStep(0); }}
              style={{ padding:'4px 10px', borderRadius:5, fontSize:9, fontWeight:700, background:c.btnBg, color:c.btnText, border:`1px solid ${c.btnBorder}`, cursor:'pointer' }}>
              🔀 Next Context
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
