import React, { useState, useEffect, useRef, useMemo } from 'react';

const PRESETS = [
  { sentence: 'American shrew mole', tokens: ['American','shrew','mole'], focus: 2, meaning: 'Small burrowing mammal', emoji: '🐾', color: '#22c55e' },
  { sentence: 'One mole of carbon dioxide', tokens: ['One','mole','of','carbon','dioxide'], focus: 1, meaning: '6.022×10²³ (Avogadro)', emoji: '⚗️', color: '#3b82f6' },
  { sentence: 'Take a biopsy of the mole', tokens: ['Take','a','biopsy','of','the','mole'], focus: 5, meaning: 'Skin growth / lesion', emoji: '🏥', color: '#f59e0b' },
];
const DK = 8;
const FM = "'JetBrains Mono', monospace";

function rgba(hex: string, a: number) {
  const r = parseInt(hex.slice(1,3),16), g = parseInt(hex.slice(3,5),16), b = parseInt(hex.slice(5,7),16);
  return `rgba(${r},${g},${b},${a})`;
}
function softmax(arr: number[]) { const m = Math.max(...arr); const e = arr.map(v => Math.exp(v-m)); const s = e.reduce((a,b)=>a+b,0); return e.map(v => v/s); }

// roundRect polyfill
function drawRoundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function genData(tokens: string[], focus: number) {
  const embs = tokens.map((t,i) => Array(DK).fill(0).map((_,d) => +(Math.sin(t.charCodeAt(0)*0.3+d*1.5+i*0.7)*0.6).toFixed(3)));
  const mm = (v: number[], M: number[][]) => M[0].map((_,j) => v.reduce((s,vi,i2) => s+vi*(M[i2]?.[j]||0), 0));
  const Wq = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.sin(r*2.1+c*1.3)*0.4));
  const Wk = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.cos(r*1.7+c*2.3)*0.4));
  const Wv = Array(DK).fill(0).map((_,r) => Array(DK).fill(0).map((_,c) => Math.sin(r*0.9+c*3.1)*0.4));
  const Q = embs.map(e => mm(e, Wq));
  const K = embs.map(e => mm(e, Wk));
  const V = embs.map(e => mm(e, Wv));
  const dot = (a: number[], b: number[]) => a.reduce((s,v,i) => s+v*(b[i]||0), 0);
  const scores = Q[focus].map((_,j) => dot(Q[focus], K[j]) / Math.sqrt(DK));
  const attn = softmax(scores);
  const output = Array(DK).fill(0).map((_,d) => tokens.reduce((s,_2,i) => s+attn[i]*V[i][d], 0));
  return { embs, Q, K, V, scores, attn, output };
}

function Panel({ children, title, style={} }: any) {
  return <div style={{ background:'rgba(15,23,42,0.75)', backdropFilter:'blur(12px)', border:'1px solid rgba(59,130,246,0.15)', borderRadius:12, padding:14, ...style }}>
    {title && <p style={{ fontSize:10, fontWeight:800, color:'#94a3b8', marginBottom:8, fontFamily:FM, letterSpacing:1, textTransform:'uppercase' }}>{title}</p>}
    {children}
  </div>;
}

function VecBar({ values, label, color, height=24 }: any) {
  if (!values?.length) return null;
  return <div style={{ marginBottom:3 }}>
    {label && <p style={{ fontSize:7, fontFamily:FM, color:'#64748b', marginBottom:1 }}>{label}</p>}
    <div style={{ display:'flex', gap:1, height, alignItems:'flex-end' }}>
      {values.map((v: number, i: number) => <div key={i} style={{ width:8, height:`${Math.min(Math.abs(v)/0.8*100, 100)}%`, minHeight:2, background:v>=0?color:'#ef4444', borderRadius:1, opacity:0.4+Math.abs(v)*0.8 }} />)}
    </div>
  </div>;
}

export default function MoleProblemLab() {
  const [preset, setPreset] = useState(0);
  const [step, setStep] = useState(0);
  const [multiHead, setMultiHead] = useState(false);
  const [sharpness, setSharpness] = useState(1.0);
  const [hoveredToken, setHoveredToken] = useState(-1);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const tRef = useRef(0);
  const animRef = useRef(0);

  const ex = PRESETS[preset];
  const data = useMemo(() => genData(ex.tokens, ex.focus), [preset]);
  const n = ex.tokens.length;

  // Compute adjusted attention with memoization
  const { adjustedAttn, adjustedScores } = useMemo(() => {
    const s = data.scores.map(v => v * sharpness);
    return { adjustedScores: s, adjustedAttn: softmax(s) };
  }, [data.scores, sharpness]);

  const STEPS = [
    { label:'Embeddings', desc:'Each token → 512-dim vector. "mole" has the SAME vector in all 3 sentences.' },
    { label:'Q, K, V', desc:'Each embedding projected into Query, Key, Value vectors via learned weight matrices.' },
    { label:'Scores', desc:`"${ex.tokens[ex.focus]}" Query dot-products with every Key → raw compatibility scores.` },
    { label:'Softmax', desc:'Scores normalized to probabilities (sum=100%). Shows how much each token matters.' },
    { label:'Output', desc:'Values mixed by attention weights → new context-aware embedding for "mole".' },
    { label:'Meaning', desc:`After attention: "${ex.tokens[ex.focus]}" → ${ex.emoji} ${ex.meaning}` },
  ];

  const CW = 520, CH = 260;

  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = CW * dpr; c.height = CH * dpr;
    const ctx = c.getContext('2d')!;
    ctx.scale(dpr, dpr);

    // Copy values into render scope to avoid stale closures
    const _step = step, _preset = preset, _multi = multiHead, _hov = hoveredToken;
    const _ex = PRESETS[_preset];
    const _data = data;
    const _attn = adjustedAttn;
    const _scores = adjustedScores;
    const _n = _ex.tokens.length;
    const _focus = _ex.focus;
    const _color = _ex.color;

    const render = () => {
      tRef.current += 0.016;
      const t = tRef.current;
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = '#080818'; ctx.fillRect(0, 0, CW, CH);

      // Grid
      ctx.strokeStyle = '#1e293b22'; ctx.lineWidth = 0.5;
      for (let x = 0; x < CW; x += 30) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,CH); ctx.stroke(); }
      for (let y = 0; y < CH; y += 30) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(CW,y); ctx.stroke(); }

      const gap = CW / (_n + 1);
      const cubeY = 120;

      // Step 2+: Attention lines
      if (_step >= 2) {
        for (let j = 0; j < _n; j++) {
          const w = _step >= 3 ? _attn[j] : Math.max(0, (_scores[j]+2)/6);
          const x1 = gap*(_focus+1), x2 = gap*(j+1);
          const cpY = cubeY - 30 - w * 50;
          ctx.strokeStyle = rgba(_color, w*0.8+0.05);
          ctx.lineWidth = w*6+0.5;
          ctx.shadowColor = _color; ctx.shadowBlur = w*8;
          ctx.beginPath(); ctx.moveTo(x1, cubeY-18); ctx.quadraticCurveTo((x1+x2)/2, cpY, x2, cubeY-18); ctx.stroke();
          ctx.shadowBlur = 0;
          if (w > 0.1) {
            ctx.fillStyle = rgba(_color, w+0.2);
            ctx.beginPath(); ctx.arc(x2, cubeY-20, 2+w*3, 0, Math.PI*2); ctx.fill();
          }
        }
      }

      // Multi-head overlays
      if (_multi && _step >= 2) {
        const hc = ['#ef4444','#22c55e','#3b82f6','#f59e0b'];
        hc.forEach((c, hi) => {
          for (let j = 0; j < _n; j++) {
            const w = Math.abs(Math.sin(hi*2.3+j*1.7+t*0.5))*0.5+0.1;
            const x1 = gap*(_focus+1), x2 = gap*(j+1);
            ctx.strokeStyle = rgba(c, w*0.4); ctx.lineWidth = w*2;
            ctx.beginPath(); ctx.moveTo(x1, cubeY-22); ctx.quadraticCurveTo((x1+x2)/2, cubeY-40-hi*10-w*15, x2, cubeY-22); ctx.stroke();
          }
        });
      }

      // Token cubes
      for (let i = 0; i < _n; i++) {
        const x = gap*(i+1);
        const isFoc = i === _focus;
        const isHov = _hov === i;
        const bob = Math.sin(t*1.5+i*0.8)*3;
        const cw2 = 40, ch2 = 32, dp = 12;

        // Top face
        ctx.fillStyle = rgba(isFoc ? _color : '#475569', isFoc ? 0.4 : 0.15);
        ctx.beginPath();
        ctx.moveTo(x-cw2/2, cubeY-ch2/2+bob); ctx.lineTo(x-cw2/2+dp, cubeY-ch2/2-dp/2+bob);
        ctx.lineTo(x+cw2/2+dp, cubeY-ch2/2-dp/2+bob); ctx.lineTo(x+cw2/2, cubeY-ch2/2+bob);
        ctx.closePath(); ctx.fill();
        // Right face
        ctx.fillStyle = rgba(isFoc ? _color : '#475569', isFoc ? 0.3 : 0.1);
        ctx.beginPath();
        ctx.moveTo(x+cw2/2, cubeY-ch2/2+bob); ctx.lineTo(x+cw2/2+dp, cubeY-ch2/2-dp/2+bob);
        ctx.lineTo(x+cw2/2+dp, cubeY+ch2/2-dp/2+bob); ctx.lineTo(x+cw2/2, cubeY+ch2/2+bob);
        ctx.closePath(); ctx.fill();
        // Front face
        ctx.fillStyle = rgba(isFoc ? _color : '#334155', isFoc ? 0.65 : 0.25);
        ctx.strokeStyle = rgba(isFoc ? _color : '#475569', isHov||isFoc ? 0.8 : 0.3);
        ctx.lineWidth = isFoc ? 2 : 1;
        if (isFoc) { ctx.shadowColor = _color; ctx.shadowBlur = 10; }
        drawRoundRect(ctx, x-cw2/2, cubeY-ch2/2+bob, cw2, ch2, 4);
        ctx.fill(); ctx.stroke(); ctx.shadowBlur = 0;

        // Label
        ctx.fillStyle = isFoc ? '#fff' : '#94a3b8';
        ctx.font = `bold ${isFoc?12:10}px ${FM}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(_ex.tokens[i], x, cubeY+bob);

        // Embedding bars
        if (_step >= 0) {
          _data.embs[i].forEach((v, d) => {
            ctx.fillStyle = rgba(isFoc ? _color : '#64748b', 0.3+Math.abs(v)*0.5);
            ctx.fillRect(x-16+d*4.5, cubeY+ch2/2+bob+5, 3.5, Math.abs(v)*22);
          });
        }

        // QKV dots
        if (_step >= 1) {
          [{c:'#ef4444',l:'Q',dx:-12},{c:'#22c55e',l:'K',dx:0},{c:'#3b82f6',l:'V',dx:12}].forEach(({c,l,dx}) => {
            const px = x+dx, py = cubeY-ch2/2-14+bob;
            ctx.fillStyle = rgba(c, 0.7);
            ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI*2); ctx.fill();
            ctx.fillStyle = '#fff'; ctx.font = `bold 6px ${FM}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(l, px, py);
          });
        }

        // Softmax bars
        if (_step >= 3) {
          const prob = _attn[i];
          const barH = prob*50;
          ctx.fillStyle = rgba(_color, 0.3+prob*0.5);
          drawRoundRect(ctx, x-10, cubeY-ch2/2-30+bob-barH, 20, barH, 3);
          ctx.fill();
          ctx.fillStyle = prob>0.12 ? '#fff' : '#475569';
          ctx.font = `bold 8px ${FM}`; ctx.textAlign = 'center';
          ctx.fillText(`${(prob*100).toFixed(0)}%`, x, cubeY-ch2/2-34+bob-barH);
        }

        // Hover popup
        if (isHov) {
          const tw = 100, th = 40, px = x-tw/2, py = cubeY+ch2/2+bob+32;
          ctx.fillStyle = 'rgba(0,0,0,0.85)'; ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
          drawRoundRect(ctx, px, py, tw, th, 5); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#94a3b8'; ctx.font = `bold 7px ${FM}`; ctx.textAlign = 'left';
          ctx.fillText(`E("${_ex.tokens[i]}")`, px+5, py+12);
          ctx.fillStyle = '#e2e8f0'; ctx.font = `6px ${FM}`;
          ctx.fillText(`[${_data.embs[i].slice(0,4).map(v=>v.toFixed(2)).join(', ')}...]`, px+5, py+24);
          if (_step>=1) { ctx.fillStyle='#ef4444'; ctx.fillText(`Q:[${_data.Q[i].slice(0,3).map(v=>v.toFixed(2)).join(',')}...]`, px+5, py+34); }
        }
      }

      // Step 5: meaning morph
      if (_step >= 5) {
        const mx2 = CW/2, my2 = CH-35;
        const pulse = Math.sin(t*3)*0.15+0.85;
        ctx.fillStyle = rgba(_color, 0.12*pulse);
        ctx.beginPath(); ctx.arc(mx2, my2, 30, 0, Math.PI*2); ctx.fill();
        ctx.strokeStyle = rgba(_color, 0.5*pulse); ctx.lineWidth = 2;
        ctx.shadowColor = _color; ctx.shadowBlur = 12; ctx.stroke(); ctx.shadowBlur = 0;
        ctx.font = '24px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(_ex.emoji, mx2, my2-2);
        ctx.fillStyle = _color; ctx.font = `bold 9px ${FM}`;
        ctx.fillText(_ex.meaning.slice(0,28), mx2, my2+22);
      }

      animRef.current = requestAnimationFrame(render);
    };
    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [preset, step, multiHead, hoveredToken, data, adjustedAttn, adjustedScores]);

  const onCanvasMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return;
    const mx = (e.clientX - rect.left) * (CW / rect.width);
    const gap = CW / (n+1);
    let found = -1;
    for (let i = 0; i < n; i++) { if (Math.abs(mx - gap*(i+1)) < 24) { found = i; break; } }
    setHoveredToken(found);
  };

  return (
    <div style={{ fontFamily:FM, color:'#e2e8f0' }}>
      {/* Sentence selector */}
      <Panel title="📝 Select Sentence" style={{ marginBottom:8 }}>
        <div style={{ display:'flex', gap:5, flexWrap:'wrap' }}>
          {PRESETS.map((p,i) => (
            <button key={i} onClick={() => { setPreset(i); setStep(0); }}
              style={{ padding:'5px 10px', borderRadius:7, fontSize:10, fontWeight:700, background:preset===i?p.color:'rgba(30,41,59,0.8)', color:preset===i?'#fff':'#94a3b8', border:`1.5px solid ${preset===i?p.color:'#334155'}`, cursor:'pointer' }}>
              {p.emoji} {p.sentence}
            </button>
          ))}
        </div>
        <div style={{ display:'flex', gap:3, marginTop:6, flexWrap:'wrap' }}>
          {ex.tokens.map((tok,i) => (
            <span key={i} style={{ padding:'2px 8px', borderRadius:5, fontSize:11, fontWeight:700, background:i===ex.focus?rgba(ex.color,0.2):'rgba(30,41,59,0.5)', color:i===ex.focus?ex.color:'#94a3b8', border:`1px solid ${i===ex.focus?ex.color+'66':'transparent'}` }}>
              {tok}{i===ex.focus && <span style={{ fontSize:7 }}> ← focus</span>}
            </span>
          ))}
        </div>
      </Panel>

      {/* Canvas + Math */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 200px', gap:8, marginBottom:8 }}>
        <Panel title="🧊 Tokens" style={{ padding:6 }}>
          <canvas ref={canvasRef} style={{ width:CW, height:CH, borderRadius:8, display:'block', maxWidth:'100%', cursor:'crosshair' }}
            onMouseMove={onCanvasMove} onMouseLeave={() => setHoveredToken(-1)} />
        </Panel>
        <Panel title="📐 Math">
          <div style={{ fontSize:9, lineHeight:2 }}>
            {step>=0 && <p style={{ color:step===0?'#22c55e':'#475569' }}>E = Embed(tok) <span style={{ fontSize:7, color:'#334155' }}>[1×512]</span></p>}
            {step>=1 && <><p style={{ color:step===1?'#ef4444':'#475569' }}>Q = E·W_Q <span style={{ fontSize:7, color:'#334155' }}>[1×64]</span></p>
              <p style={{ color:step===1?'#22c55e':'#475569' }}>K = E·W_K</p>
              <p style={{ color:step===1?'#3b82f6':'#475569' }}>V = E·W_V</p></>}
            {step>=2 && <p style={{ color:step===2?'#f59e0b':'#475569' }}>Score = Q·Kᵀ/√d_k</p>}
            {step>=3 && <p style={{ color:step===3?'#a855f7':'#475569' }}>α = softmax(Scores)</p>}
            {step>=4 && <p style={{ color:step===4?'#ec4899':'#475569' }}>Out = Σ(αᵢ·Vᵢ)</p>}
            {step>=5 && <p style={{ color:'#22c55e', fontWeight:700 }}>→ {ex.emoji} {ex.meaning.slice(0,20)}</p>}
          </div>
          {step>=1 && <div style={{ marginTop:6 }}>
            <VecBar values={data.Q[ex.focus]} label={`Q("${ex.tokens[ex.focus]}")`} color="#ef4444" />
            <VecBar values={data.K[ex.focus]} label="K" color="#22c55e" />
            <VecBar values={data.V[ex.focus]} label="V" color="#3b82f6" />
          </div>}
          {step>=1 && <div style={{ marginTop:6, padding:4, borderRadius:4, background:'#0f172a', fontSize:7, color:'#475569' }}>
            W_Q:[512×64] W_K:[512×64] W_V:[512×64]
          </div>}
        </Panel>
      </div>

      {/* Steps */}
      <Panel style={{ marginBottom:8 }}>
        <div style={{ display:'flex', gap:3, marginBottom:6, flexWrap:'wrap' }}>
          {STEPS.map((s,i) => (
            <button key={i} onClick={() => setStep(i)}
              style={{ padding:'3px 8px', borderRadius:5, fontSize:8, fontWeight:700, background:step===i?ex.color:step>i?rgba(ex.color,0.15):'#0f172a', color:step>=i?'#fff':'#475569', border:`1px solid ${step===i?ex.color:step>i?rgba(ex.color,0.3):'#1e293b'}`, cursor:'pointer' }}>
              {i+1}. {s.label}
            </button>
          ))}
        </div>
        <p style={{ fontSize:10, color:'#c8d6e5', margin:0 }}>{STEPS[step].desc}</p>
        <div style={{ display:'flex', gap:6, marginTop:6 }}>
          <button onClick={() => setStep(Math.max(0,step-1))} style={{ padding:'3px 8px', borderRadius:4, fontSize:9, background:'#1e293b', color:'#94a3b8', border:'1px solid #334155', cursor:'pointer' }}>← Prev</button>
          <button onClick={() => setStep(Math.min(5,step+1))} style={{ padding:'3px 8px', borderRadius:4, fontSize:9, background:ex.color, color:'#fff', border:'none', cursor:'pointer' }}>Next →</button>
        </div>
      </Panel>

      {/* Bottom: Heatmap + Controls + Meaning */}
      <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr 1fr', gap:8 }}>
        {/* Heatmap */}
        <Panel title="🔥 Heatmap">
          <div style={{ display:'inline-grid', gridTemplateColumns:`24px repeat(${n},28px)`, gap:1 }}>
            <span />{ex.tokens.map((t,j) => <span key={j} style={{ fontSize:6, fontFamily:FM, color:'#60a5fa', textAlign:'center', fontWeight:700 }}>{t.slice(0,4)}</span>)}
            {ex.tokens.map((t,i) => (
              <React.Fragment key={i}>
                <span style={{ fontSize:6, fontFamily:FM, color:'#60a5fa', textAlign:'right', fontWeight:700 }}>{t.slice(0,4)}</span>
                {ex.tokens.map((_,j) => {
                  const isRow = i===ex.focus;
                  const v = isRow ? adjustedAttn[j] : (i===j ? 0.35 : 0.06);
                  return <div key={j} style={{ width:28, height:18, borderRadius:2, display:'flex', alignItems:'center', justifyContent:'center', fontSize:6, fontWeight:700, fontFamily:FM, color:v>0.15?'#fff':'#475569', background:rgba(isRow?ex.color:'#3b82f6', v*0.8+0.05), border:isRow&&j===ex.focus?`1px solid ${ex.color}`:'none' }}>
                    {(v*100).toFixed(0)}%
                  </div>;
                })}
              </React.Fragment>
            ))}
          </div>
        </Panel>

        {/* Controls */}
        <Panel title="🎛️ Controls">
          <label style={{ fontSize:8, color:'#94a3b8', display:'block', marginBottom:2 }}>Sharpness</label>
          <input type="range" min="0.2" max="3" step="0.1" value={sharpness} onChange={e => setSharpness(+e.target.value)}
            style={{ width:'100%', accentColor:ex.color }} />
          <div style={{ display:'flex', justifyContent:'space-between', fontSize:7, color:'#475569', marginBottom:8 }}><span>Soft</span><span>{sharpness.toFixed(1)}</span><span>Sharp</span></div>
          <button onClick={() => setMultiHead(!multiHead)}
            style={{ display:'block', width:'100%', padding:'4px 8px', borderRadius:5, fontSize:9, fontWeight:700, marginBottom:4, background:multiHead?'#f59e0b22':'#0f172a', color:multiHead?'#f59e0b':'#64748b', border:`1px solid ${multiHead?'#f59e0b':'#334155'}`, cursor:'pointer' }}>
            👁️ Multi-Head: {multiHead ? 'ON' : 'OFF'}
          </button>
          <button onClick={() => setPreset(Math.floor(Math.random()*3))}
            style={{ display:'block', width:'100%', padding:'4px 8px', borderRadius:5, fontSize:9, fontWeight:700, background:'#0f172a', color:'#64748b', border:'1px solid #334155', cursor:'pointer' }}>
            🔀 Random
          </button>
        </Panel>

        {/* Meaning */}
        <Panel title={`${ex.emoji} Meaning`}>
          {step>=5 ? (
            <div style={{ textAlign:'center' }}>
              <div style={{ fontSize:40, marginBottom:6 }}>{ex.emoji}</div>
              <p style={{ fontSize:12, fontWeight:800, color:ex.color }}>{ex.meaning}</p>
              <p style={{ fontSize:8, color:'#475569', marginTop:4 }}>"{ex.tokens[ex.focus]}" now encodes this</p>
              <div style={{ display:'flex', gap:8, marginTop:8, justifyContent:'center' }}>
                {PRESETS.filter((_,i) => i!==preset).map((p,i) => <span key={i} style={{ fontSize:20, opacity:0.2, filter:'grayscale(1)' }}>{p.emoji}</span>)}
              </div>
            </div>
          ) : (
            <div style={{ textAlign:'center', padding:'12px 0' }}>
              <div style={{ display:'flex', gap:8, justifyContent:'center', marginBottom:6 }}>
                {PRESETS.map((p,i) => <span key={i} style={{ fontSize:24, opacity:0.35 }}>{p.emoji}</span>)}
              </div>
              <p style={{ fontSize:9, color:'#475569' }}>Complete all 6 steps</p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}
