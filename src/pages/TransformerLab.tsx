import { useState, useEffect, useRef, useMemo } from "react";

/* ═══════════════════════════════════════════════════════════
   CONFIG — following the original paper
   ═══════════════════════════════════════════════════════════ */
const D_MODEL = 512;
const D_K = 64; // d_model / num_heads
const NUM_HEADS = 8;
const D_FF = 2048;
const SHOW_DIMS = 8; // we show 8 of 512 dims for visual clarity

/* ═══════════════════════════════════════════════════════════
   DATASET — English → Hindi
   ═══════════════════════════════════════════════════════════ */
const DATASET = [
  { en: "I love India", hi: "मुझे भारत पसंद है", enT: ["I", "love", "India"], hiT: ["मुझे", "भारत", "पसंद", "है"] },
  { en: "The sun is bright", hi: "सूरज चमकीला है", enT: ["The", "sun", "is", "bright"], hiT: ["सूरज", "चमकीला", "है"] },
  { en: "She reads books", hi: "वह किताबें पढ़ती है", enT: ["She", "reads", "books"], hiT: ["वह", "किताबें", "पढ़ती", "है"] },
];

/* ═══════════════════════════════════════════════════════════
   MATH HELPERS (simulated at SHOW_DIMS for visuals)
   ═══════════════════════════════════════════════════════════ */
function srow(v) { if(!v?.length) return []; const mx=Math.max(...v); const e=v.map(x=>Math.exp(x-mx)); const s=e.reduce((a,b)=>a+b,0); return e.map(x=>x/(s||1)); }
function mm(a,b) { if(!a?.length||!b?.length) return []; return a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*(b[k]?.[j]||0),0))); }
function tp(m) { return m[0].map((_,i)=>m.map(r=>r[i])); }
function rM(r,c,s=0.15) { return Array(r).fill(0).map(()=>Array(c).fill(0).map(()=>+((Math.random()-0.5)*s*2).toFixed(3))); }
function addM(a,b) { return a.map((r,i)=>r.map((v,j)=>+(v+(b[i]?.[j]||0)).toFixed(3))); }
function lnorm(m) { return m.map(r=>{const mu=r.reduce((s,v)=>s+v,0)/r.length; const va=r.reduce((s,v)=>s+(v-mu)**2,0)/r.length; return r.map(v=>+((v-mu)/Math.sqrt(va+1e-5)).toFixed(3));}); }
function embed(tokens) { return tokens.map((t,i)=>Array(SHOW_DIMS).fill(0).map((_,d)=>+(Math.sin(t.charCodeAt(0)*0.25+d*1.4+i*0.9)*0.5).toFixed(3))); }
function posEnc(len) { return Array.from({length:len}).map((_,p)=>Array(SHOW_DIMS).fill(0).map((_,i)=>+(i%2===0?Math.sin(p/Math.pow(10000,i/D_MODEL)):Math.cos(p/Math.pow(10000,(i-1)/D_MODEL))).toFixed(3))); }
function attn(Q,K,V) { const dk=K[0]?.length||1; const sc=mm(Q,tp(K)).map(r=>r.map(v=>+(v/Math.sqrt(dk)).toFixed(3))); const w=sc.map(srow); return {scores:sc,weights:w,output:mm(w,V)}; }

const S = { mono: "'JetBrains Mono', monospace", sans: "'DM Sans', system-ui, sans-serif" };

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS
   ═══════════════════════════════════════════════════════════ */
function TokenChip({t, color, sub}) {
  return <div style={{padding:"5px 10px",borderRadius:8,background:`${color}18`,border:`1.5px solid ${color}44`,fontSize:13,fontWeight:700,fontFamily:S.mono,color,textAlign:"center"}}>{t}{sub&&<div style={{fontSize:7,color:"#64748b",marginTop:1}}>{sub}</div>}</div>;
}

function MiniMatrix({data, label, color, rowL, show=SHOW_DIMS, cs=26}) {
  if(!data?.length) return null;
  const rows=data.length, cols=Math.min(data[0]?.length||0,show);
  return <div style={{marginBottom:8}}>
    {label&&<p style={{fontSize:9,fontFamily:S.mono,fontWeight:700,color,marginBottom:3}}>{label} [{rows}×{D_MODEL}] <span style={{color:"#475569"}}>(showing {cols} dims)</span></p>}
    <div style={{display:"inline-grid",gridTemplateColumns:rowL?`32px repeat(${cols},${cs}px)`:`repeat(${cols},${cs}px)`,gap:1}}>
      {data.map((row,i)=><>{rowL&&<span style={{fontSize:7,fontFamily:S.mono,color:"#64748b",alignSelf:"center",textAlign:"right",paddingRight:2}}>{rowL[i]?.slice(0,4)}</span>}{row.slice(0,cols).map((v,j)=><div key={`${i}-${j}`} style={{width:cs,height:cs-4,display:"flex",alignItems:"center",justifyContent:"center",fontSize:7,fontWeight:700,fontFamily:S.mono,borderRadius:3,background:`${color}${Math.round(Math.min(Math.abs(v),1)*150+30).toString(16).padStart(2,"0")}`,color:"#fff"}}>{v.toFixed(2)}</div>)}</>)}
    </div>
  </div>;
}

function HeatmapCanvas({weights, rL, cL, size=180, label}) {
  const ref=useRef(null); const n=rL?.length||0, m=cL?.length||0;
  useEffect(()=>{
    const c=ref.current; if(!c||!n||!m) return;
    c.width=size; c.height=size; const ctx=c.getContext("2d");
    ctx.clearRect(0,0,size,size);
    const off=36,cw=(size-off)/m,ch=(size-off)/n;
    for(let i=0;i<n;i++) for(let j=0;j<m;j++){ const v=weights[i]?.[j]||0; ctx.fillStyle=`rgba(168,85,247,${v*0.85+0.05})`; ctx.fillRect(off+j*cw,off+i*ch,cw-1,ch-1); if(n<=6){ctx.fillStyle=v>0.15?"#fff":"#64748b";ctx.font=`bold ${Math.min(9,cw*0.28)}px monospace`;ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText((v*100).toFixed(0)+"%",off+j*cw+cw/2,off+i*ch+ch/2);}}
    ctx.fillStyle="#94a3b8";ctx.font="bold 7px monospace";ctx.textAlign="right";ctx.textBaseline="middle";
    for(let i=0;i<n;i++) ctx.fillText(rL[i].slice(0,5),off-2,off+i*ch+ch/2);
    ctx.textAlign="center";ctx.textBaseline="bottom";
    for(let j=0;j<m;j++){ctx.save();ctx.translate(off+j*cw+cw/2,off-2);ctx.rotate(-0.4);ctx.fillText(cL[j].slice(0,5),0,0);ctx.restore();}
  },[weights,n,m,size,rL,cL]);
  return <div>{label&&<p style={{fontSize:9,fontFamily:S.mono,fontWeight:700,color:"#a855f7",marginBottom:3}}>{label}</p>}<canvas ref={ref} style={{width:size,height:size,borderRadius:8,border:"1px solid #334155"}} /></div>;
}

function StoryCard({icon, title, children, color}) {
  return <div style={{padding:14,borderRadius:12,background:`${color}08`,border:`1px solid ${color}25`,marginBottom:10}}>
    <div style={{display:"flex",alignItems:"center",gap:8,marginBottom:8}}>
      <span style={{fontSize:20}}>{icon}</span>
      <h4 style={{fontSize:13,fontWeight:800,color,margin:0}}>{title}</h4>
    </div>
    <div style={{fontSize:11,color:"#94a3b8",lineHeight:1.6}}>{children}</div>
  </div>;
}

function DimBadge() {
  return <span style={{fontSize:8,padding:"1px 5px",borderRadius:4,background:"#f59e0b22",color:"#f59e0b",fontFamily:S.mono,fontWeight:700}}>d_model={D_MODEL}</span>;
}

/* ═══════════════════════════════════════════════════════════
   STORYTELLING DETAIL PANELS
   ═══════════════════════════════════════════════════════════ */
function InputEmbeddingStory({tokens, lang, color}) {
  const embs = embed(tokens);
  const pe = posEnc(tokens.length);
  const final = addM(embs, pe);
  return <div>
    <StoryCard icon="📖" title="Chapter 1: Tokenization" color={color}>
      <p>Our sentence is split into <b style={{color:"#fff"}}>tokens</b> — the smallest units the model understands.</p>
      <div style={{display:"flex",gap:4,margin:"8px 0",flexWrap:"wrap"}}>{tokens.map((t,i)=><TokenChip key={i} t={t} color={color} sub={`id=${i}`} />)}</div>
      <p>Each token gets a unique integer ID from a vocabulary of ~50,000 words.</p>
    </StoryCard>
    <StoryCard icon="🔢" title="Chapter 2: Token Embedding" color="#22c55e">
      <p>Each token ID is looked up in an <b style={{color:"#fff"}}>embedding matrix</b> of size [{tokens.length} × {D_MODEL}]. This gives each token a {D_MODEL}-dimensional vector — a point in a vast semantic space.</p>
      <MiniMatrix data={embs} label="Token Embeddings" color="#22c55e" rowL={tokens} />
      <p>Similar words like "sun" and "star" would have nearby vectors. <DimBadge /></p>
    </StoryCard>
    <StoryCard icon="🌊" title="Chapter 3: Positional Encoding" color="#06b6d4">
      <p>Transformers process all tokens at once — they have <b style={{color:"#fff"}}>no inherent sense of order</b>. We add sine/cosine waves to encode each token's position.</p>
      <MiniMatrix data={pe} label="Positional Encoding" color="#06b6d4" rowL={tokens.map((_,i)=>`pos${i}`)} />
      <p style={{fontFamily:S.mono,fontSize:9,color:"#06b6d4"}}>PE(pos,2i) = sin(pos/10000^(2i/{D_MODEL})) · PE(pos,2i+1) = cos(...)</p>
    </StoryCard>
    <StoryCard icon="✨" title="Chapter 4: Final Embedding" color="#f59e0b">
      <p>We <b style={{color:"#fff"}}>add</b> token embeddings + positional encodings = the final input to the transformer.</p>
      <MiniMatrix data={final} label="Final Input Embedding" color="#f59e0b" rowL={tokens} />
      <p>Each token now knows <b style={{color:"#fff"}}>what it means</b> AND <b style={{color:"#fff"}}>where it is</b> in the sentence. <DimBadge /></p>
    </StoryCard>
  </div>;
}

function MultiHeadAttnStory({tokens, embs, isMasked, crossTokens, crossEmbs, title}) {
  const Wq=useMemo(()=>rM(SHOW_DIMS,SHOW_DIMS),[]);
  const Wk=useMemo(()=>rM(SHOW_DIMS,SHOW_DIMS),[]);
  const Wv=useMemo(()=>rM(SHOW_DIMS,SHOW_DIMS),[]);
  const srcE = crossEmbs || embs;
  const srcT = crossTokens || tokens;
  const Q=mm(embs,Wq), K=mm(srcE,Wk), V=mm(srcE,Wv);
  const {scores, weights, output} = attn(Q,K,V);
  const [step,setStep]=useState(0);
  const [auto,setAuto]=useState(false);
  useEffect(()=>{if(auto&&step<5){const t=setTimeout(()=>setStep(p=>p+1),3000);return()=>clearTimeout(t);}else setAuto(false);},[auto,step]);

  return <div>
    <div style={{display:"flex",gap:4,marginBottom:8}}>
      <button onClick={()=>setAuto(!auto)} style={{padding:"3px 10px",borderRadius:5,fontSize:10,fontWeight:700,background:auto?"#dc2626":"#16a34a",color:"#fff",border:"none",cursor:"pointer"}}>{auto?"⏸":"▶ Play Story"}</button>
      <button onClick={()=>setStep(0)} style={{padding:"3px 8px",borderRadius:5,fontSize:10,background:"#1e293b",color:"#94a3b8",border:"1px solid #334155",cursor:"pointer"}}>↺</button>
      <span style={{fontSize:9,color:"#475569",alignSelf:"center"}}>Step {step+1}/6</span>
    </div>
    <div style={{display:"flex",gap:2,marginBottom:10}}>{[0,1,2,3,4,5].map(i=><button key={i} onClick={()=>{setStep(i);setAuto(false);}} style={{flex:1,height:5,borderRadius:3,background:i<=step?["#ef4444","#22c55e","#3b82f6","#a855f7","#ec4899","#22c55e"][i]:"#1e293b",border:"none",cursor:"pointer"}} />)}</div>

    {step>=0 && <StoryCard icon="🔴" title="Step 1: Queries — 'What am I looking for?'" color="#ef4444">
      <p>Each token's embedding is multiplied by <b style={{color:"#fff"}}>W_Q</b> [{D_MODEL}×{D_K}] to create a <b style={{color:"#ef4444"}}>Query vector</b>.</p>
      <p>Think of it as each token asking: <i>"What kind of information do I need?"</i></p>
      <MiniMatrix data={Q} label={`Queries Q [${tokens.length}×${D_K}]`} color="#ef4444" rowL={tokens} />
      {isMasked && <p style={{color:"#f59e0b"}}>⚠️ Masked: each token can only query tokens before it (no peeking at the future!)</p>}
    </StoryCard>}

    {step>=1 && <StoryCard icon="🟢" title="Step 2: Keys — 'What do I contain?'" color="#22c55e">
      <p>Each {crossTokens?"encoder":"source"} token's embedding × <b style={{color:"#fff"}}>W_K</b> [{D_MODEL}×{D_K}] = <b style={{color:"#22c55e"}}>Key vector</b>.</p>
      <p>Keys advertise: <i>"Here's what information I have to offer."</i></p>
      <MiniMatrix data={K} label={`Keys K [${srcT.length}×${D_K}]`} color="#22c55e" rowL={srcT} />
      {crossTokens && <p style={{color:"#f59e0b"}}>📌 Keys come from the <b>Encoder output</b>, not from the decoder itself!</p>}
    </StoryCard>}

    {step>=2 && <StoryCard icon="🔵" title="Step 3: Values — 'What info do I provide?'" color="#3b82f6">
      <p>Each {crossTokens?"encoder":"source"} token's embedding × <b style={{color:"#fff"}}>W_V</b> [{D_MODEL}×{D_K}] = <b style={{color:"#3b82f6"}}>Value vector</b>.</p>
      <p>Values carry the <b style={{color:"#fff"}}>actual content</b> that will be mixed into the output.</p>
      <MiniMatrix data={V} label={`Values V [${srcT.length}×${D_K}]`} color="#3b82f6" rowL={srcT} />
      <p>Query asks the question, Key matches it, <b style={{color:"#3b82f6"}}>Value provides the answer</b>.</p>
    </StoryCard>}

    {step>=3 && <StoryCard icon="📊" title="Step 4: Attention Scores — Q · Kᵀ / √d_k" color="#a855f7">
      <p>Dot product of each Query with every Key, divided by √{D_K} = <b style={{color:"#fff"}}>{Math.sqrt(D_K).toFixed(0)}</b>. High score = strong match.</p>
      <p>Then <b style={{color:"#a855f7"}}>softmax</b> makes each row sum to 100% — a probability distribution over which tokens to attend to.</p>
      <HeatmapCanvas weights={weights} rL={tokens} cL={srcT} size={180} label="Attention Weights (after softmax)" />
      <p style={{fontFamily:S.mono,fontSize:9,color:"#a855f7"}}>Attention(Q,K,V) = softmax(Q·Kᵀ / √d_k) · V</p>
    </StoryCard>}

    {step>=4 && <StoryCard icon="✖️" title="Step 5: Weighted Values — Attention × V" color="#ec4899">
      <p>Each token's output = <b style={{color:"#fff"}}>attention weights × Value vectors</b>. This is the magic — each token now absorbs relevant information from all other tokens!</p>
      <MiniMatrix data={output} label={`Attention Output [${tokens.length}×${D_K}]`} color="#ec4899" rowL={tokens} />
      <p>For example, if "पसंद" attends 45% to "love" and 30% to "India", its output is 0.45×V("love") + 0.30×V("India") + ...</p>
    </StoryCard>}

    {step>=5 && <StoryCard icon="🧠" title="Step 6: Multi-Head — 8 heads in parallel" color="#22c55e">
      <p>We run <b style={{color:"#fff"}}>{NUM_HEADS} attention heads</b> simultaneously, each with its own W_Q, W_K, W_V [{D_MODEL}×{D_K}].</p>
      <div style={{display:"flex",gap:4,flexWrap:"wrap",margin:"8px 0"}}>
        {Array.from({length:NUM_HEADS}).map((_,h)=><div key={h} style={{padding:"4px 8px",borderRadius:6,background:["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7","#ec4899","#06b6d4","#84cc16"][h]+"22",border:`1px solid ${["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7","#ec4899","#06b6d4","#84cc16"][h]}44`,fontSize:9,fontFamily:S.mono,fontWeight:700,color:["#ef4444","#22c55e","#3b82f6","#f59e0b","#a855f7","#ec4899","#06b6d4","#84cc16"][h]}}>Head {h+1}<br/><span style={{fontSize:7,color:"#64748b"}}>{D_K} dims</span></div>)}
      </div>
      <p>Concat all 8 heads: [{tokens.length}×{D_K*NUM_HEADS}] → multiply by <b style={{color:"#fff"}}>W_O</b> [{D_MODEL}×{D_MODEL}] → back to [{tokens.length}×{D_MODEL}].</p>
      <p style={{fontFamily:S.mono,fontSize:9,color:"#22c55e"}}>MultiHead = Concat(head₁...head₈) × W_O</p>
      <p><b style={{color:"#fff"}}>Total parameters in this layer:</b> 4 × {D_MODEL} × {D_K} × {NUM_HEADS} = <b style={{color:"#f59e0b"}}>{(4*D_MODEL*D_K*NUM_HEADS).toLocaleString()}</b></p>
    </StoryCard>}
  </div>;
}

function AddNormStory({tokens, inputM, outputM, label}) {
  const res = addM(inputM, outputM);
  const normed = lnorm(res);
  return <StoryCard icon="➕" title={label || "Add & Layer Normalize"} color="#eab308">
    <p><b style={{color:"#fff"}}>Residual connection:</b> output = sublayer(x) + x. This lets gradients flow directly through the network (like a highway bypass).</p>
    <p><b style={{color:"#fff"}}>Layer Norm:</b> normalizes each token's vector to have mean=0, variance=1. Stabilizes training.</p>
    <MiniMatrix data={normed} label="After Add & LayerNorm" color="#eab308" rowL={tokens} />
    <p style={{fontFamily:S.mono,fontSize:9,color:"#eab308"}}>LayerNorm(x + Sublayer(x))</p>
  </StoryCard>;
}

function FFNStory({tokens, inputM}) {
  const W1=useMemo(()=>rM(SHOW_DIMS,SHOW_DIMS*2),[]);
  const W2=useMemo(()=>rM(SHOW_DIMS*2,SHOW_DIMS),[]);
  const hidden = mm(inputM,W1).map(r=>r.map(v=>Math.max(0,v)));
  const out = mm(hidden,W2);
  return <StoryCard icon="🧮" title="Feed-Forward Network" color="#60a5fa">
    <p>A 2-layer neural network applied to <b style={{color:"#fff"}}>each token independently</b> (same weights, different tokens).</p>
    <p style={{fontFamily:S.mono,fontSize:10,color:"#60a5fa"}}>FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂</p>
    <p>Layer 1: [{D_MODEL}→{D_FF}] with ReLU. Layer 2: [{D_FF}→{D_MODEL}].</p>
    <MiniMatrix data={hidden.map(r=>r.slice(0,8))} label={`Hidden [${tokens.length}×${D_FF}] (showing 8)`} color="#60a5fa" rowL={tokens} cs={24} />
    <MiniMatrix data={out} label={`FFN Output [${tokens.length}×${D_MODEL}]`} color="#60a5fa" rowL={tokens} />
    <p><b style={{color:"#fff"}}>Parameters:</b> {D_MODEL}×{D_FF} + {D_FF}×{D_MODEL} = <b style={{color:"#f59e0b"}}>{(D_MODEL*D_FF*2).toLocaleString()}</b></p>
  </StoryCard>;
}

function LinearSoftmaxStory({tokens}) {
  return <div>
    <StoryCard icon="📐" title="Linear Projection" color="#c4b5fd">
      <p>Each decoder output vector [{D_MODEL}] is multiplied by a <b style={{color:"#fff"}}>weight matrix [{D_MODEL}×vocab_size]</b> to produce a score for every word in the vocabulary.</p>
      <p>This creates <b style={{color:"#c4b5fd"}}>logits</b> — raw scores before normalization.</p>
    </StoryCard>
    <StoryCard icon="📊" title="Softmax → Output Probabilities" color="#86efac">
      <p>Softmax converts logits into a <b style={{color:"#fff"}}>probability distribution</b> over the entire vocabulary. The highest probability word is the prediction.</p>
      {tokens.map((t,i)=><div key={i} style={{marginBottom:6,padding:6,borderRadius:6,background:"#0f172a",border:"1px solid #1e293b"}}>
        <div style={{display:"flex",justifyContent:"space-between"}}><span style={{fontSize:11,fontFamily:S.mono,color:"#a855f7"}}>Position {i}</span><span style={{fontSize:12,fontFamily:S.mono,color:"#22c55e",fontWeight:800}}>→ "{t}" ✓</span></div>
        <div style={{marginTop:3,height:6,background:"#1e293b",borderRadius:3,overflow:"hidden"}}><div style={{width:`${82+i*4}%`,height:"100%",background:"#22c55e",borderRadius:3}} /></div>
        <span style={{fontSize:8,fontFamily:S.mono,color:"#22c55e"}}>{82+i*4}% confidence</span>
      </div>)}
    </StoryCard>
  </div>;
}

/* ═══════════════════════════════════════════════════════════
   SVG ARCHITECTURE DIAGRAM
   ═══════════════════════════════════════════════════════════ */
function ArchBlock({x,y,w,h,label,color,sel,onClick,sub}) {
  return <g onClick={onClick} style={{cursor:"pointer"}}>
    <rect x={x} y={y} width={w} height={h} rx={5} fill={sel?color:`${color}55`} stroke={sel?"#fff":color} strokeWidth={sel?2.5:1.2} />
    <text x={x+w/2} y={y+h/2-(sub?3:0)} textAnchor="middle" dominantBaseline="central" fill={sel?"#000":"#e2e8f0"} fontSize={9} fontWeight={700} fontFamily="DM Sans">{label}</text>
    {sub&&<text x={x+w/2} y={y+h/2+8} textAnchor="middle" fill={sel?"#000":"#94a3b8"} fontSize={7} fontFamily="monospace">{sub}</text>}
  </g>;
}

function ArchDiagram({sel,onSel}) {
  const bw=125,bh=28;
  const ex=25,dx=200;
  // Y positions bottom-up
  const Y = {
    in_emb: 400, pe_enc: 370, enc_mha: 330, enc_an1: 300, enc_ff: 270, enc_an2: 240,
    out_emb: 400, pe_dec: 370, dec_mmha: 330, dec_an1: 300, dec_mha: 270, dec_an2: 240, dec_ff: 210, dec_an3: 180, linear: 140, softmax: 110
  };
  const arr=(x1,y1,x2,y2)=><line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#475569" strokeWidth={1.3} markerEnd="url(#ah)" />;

  return <svg viewBox="0 0 380 460" style={{width:"100%",maxWidth:380,height:"auto"}}>
    <defs><marker id="ah" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto"><polygon points="0 0,7 2.5,0 5" fill="#475569"/></marker></defs>

    {/* Labels */}
    <text x={ex+bw/2} y={435} textAnchor="middle" fill="#94a3b8" fontSize={9} fontWeight={700}>Inputs (English)</text>
    <text x={dx+bw/2} y={435} textAnchor="middle" fill="#94a3b8" fontSize={9} fontWeight={700}>Outputs (Hindi, shifted)</text>
    <text x={dx+bw/2} y={95} textAnchor="middle" fill="#22c55e" fontSize={10} fontWeight={800}>Output Probabilities</text>

    {/* Nx boxes */}
    <rect x={ex-6} y={Y.enc_an2-6} width={bw+12} height={Y.enc_mha-Y.enc_an2+bh+12} rx={6} fill="none" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4,3" />
    <text x={ex-12} y={(Y.enc_an2+Y.enc_mha+bh)/2} fill="#94a3b8" fontSize={10} fontWeight={800} textAnchor="end">N×</text>
    <rect x={dx-6} y={Y.dec_an3-6} width={bw+12} height={Y.dec_mmha-Y.dec_an3+bh+12} rx={6} fill="none" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4,3" />
    <text x={dx+bw+16} y={(Y.dec_an3+Y.dec_mmha+bh)/2} fill="#94a3b8" fontSize={10} fontWeight={800}>N×</text>

    {/* PE circles */}
    <circle cx={ex+bw/2-10} cy={Y.pe_enc+bh/2} r={7} fill="none" stroke="#94a3b8" /><text x={ex+bw/2-10} y={Y.pe_enc+bh/2+1} textAnchor="middle" dominantBaseline="central" fill="#94a3b8" fontSize={7}>~</text>
    <circle cx={ex+bw/2+10} cy={Y.pe_enc+bh/2} r={7} fill="none" stroke="#94a3b8" /><text x={ex+bw/2+10} y={Y.pe_enc+bh/2+1} textAnchor="middle" dominantBaseline="central" fill="#94a3b8" fontSize={9}>⊕</text>
    <circle cx={dx+bw/2+10} cy={Y.pe_dec+bh/2} r={7} fill="none" stroke="#94a3b8" /><text x={dx+bw/2+10} y={Y.pe_dec+bh/2+1} textAnchor="middle" dominantBaseline="central" fill="#94a3b8" fontSize={7}>~</text>
    <circle cx={dx+bw/2+30} cy={Y.pe_dec+bh/2} r={7} fill="none" stroke="#94a3b8" /><text x={dx+bw/2+30} y={Y.pe_dec+bh/2+1} textAnchor="middle" dominantBaseline="central" fill="#94a3b8" fontSize={9}>⊕</text>

    {/* Arrows */}
    {arr(ex+bw/2,420,ex+bw/2,Y.in_emb+bh)}
    {arr(ex+bw/2,Y.in_emb,ex+bw/2,Y.pe_enc+bh+8)}
    {arr(ex+bw/2,Y.pe_enc,ex+bw/2,Y.enc_mha+bh)}
    {arr(ex+bw/2,Y.enc_mha,ex+bw/2,Y.enc_an1+bh)}
    {arr(ex+bw/2,Y.enc_an1,ex+bw/2,Y.enc_ff+bh)}
    {arr(ex+bw/2,Y.enc_ff,ex+bw/2,Y.enc_an2+bh)}
    {arr(dx+bw/2,420,dx+bw/2,Y.out_emb+bh)}
    {arr(dx+bw/2,Y.out_emb,dx+bw/2,Y.pe_dec+bh+8)}
    {arr(dx+bw/2,Y.pe_dec,dx+bw/2,Y.dec_mmha+bh)}
    {arr(dx+bw/2,Y.dec_mmha,dx+bw/2,Y.dec_an1+bh)}
    {arr(dx+bw/2,Y.dec_an1,dx+bw/2,Y.dec_mha+bh)}
    {arr(dx+bw/2,Y.dec_mha,dx+bw/2,Y.dec_an2+bh)}
    {arr(dx+bw/2,Y.dec_an2,dx+bw/2,Y.dec_ff+bh)}
    {arr(dx+bw/2,Y.dec_ff,dx+bw/2,Y.dec_an3+bh)}
    {arr(dx+bw/2,Y.dec_an3,dx+bw/2,Y.linear+bh)}
    {arr(dx+bw/2,Y.linear,dx+bw/2,Y.softmax+bh)}

    {/* Cross-attention arrow */}
    <path d={`M${ex+bw},${Y.enc_an2+bh/2} L${dx},${Y.dec_mha+bh/2}`} fill="none" stroke="#f59e0b" strokeWidth={2} markerEnd="url(#ah)" />
    <text x={(ex+bw+dx)/2} y={Y.dec_mha+bh/2-8} textAnchor="middle" fill="#f59e0b" fontSize={7} fontWeight={700}>K, V from Encoder</text>

    {/* Blocks */}
    <ArchBlock x={ex} y={Y.in_emb} w={bw} h={bh} label="Input Embedding" color="#fca5a5" sel={sel==="in_emb"} onClick={()=>onSel("in_emb")} />
    <ArchBlock x={ex} y={Y.enc_mha} w={bw} h={bh} label="Multi-Head Attention" color="#fdba74" sel={sel==="enc_mha"} onClick={()=>onSel("enc_mha")} />
    <ArchBlock x={ex} y={Y.enc_an1} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="enc_an1"} onClick={()=>onSel("enc_an1")} />
    <ArchBlock x={ex} y={Y.enc_ff} w={bw} h={bh} label="Feed Forward" color="#93c5fd" sel={sel==="enc_ff"} onClick={()=>onSel("enc_ff")} />
    <ArchBlock x={ex} y={Y.enc_an2} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="enc_an2"} onClick={()=>onSel("enc_an2")} />
    <ArchBlock x={dx} y={Y.out_emb} w={bw} h={bh} label="Output Embedding" color="#fca5a5" sel={sel==="out_emb"} onClick={()=>onSel("out_emb")} />
    <ArchBlock x={dx} y={Y.dec_mmha} w={bw} h={bh} label="Masked Multi-Head" sub="Attention" color="#fdba74" sel={sel==="dec_mmha"} onClick={()=>onSel("dec_mmha")} />
    <ArchBlock x={dx} y={Y.dec_an1} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="dec_an1"} onClick={()=>onSel("dec_an1")} />
    <ArchBlock x={dx} y={Y.dec_mha} w={bw} h={bh} label="Multi-Head Attention" sub="(Cross)" color="#fdba74" sel={sel==="dec_mha"} onClick={()=>onSel("dec_mha")} />
    <ArchBlock x={dx} y={Y.dec_an2} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="dec_an2"} onClick={()=>onSel("dec_an2")} />
    <ArchBlock x={dx} y={Y.dec_ff} w={bw} h={bh} label="Feed Forward" color="#93c5fd" sel={sel==="dec_ff"} onClick={()=>onSel("dec_ff")} />
    <ArchBlock x={dx} y={Y.dec_an3} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="dec_an3"} onClick={()=>onSel("dec_an3")} />
    <ArchBlock x={dx} y={Y.linear} w={bw} h={bh} label="Linear" color="#c4b5fd" sel={sel==="linear"} onClick={()=>onSel("linear")} />
    <ArchBlock x={dx} y={Y.softmax} w={bw} h={bh} label="Softmax" color="#86efac" sel={sel==="softmax"} onClick={()=>onSel("softmax")} />
  </svg>;
}

/* ═══════════════════════════════════════════════════════════
   MAIN LAB
   ═══════════════════════════════════════════════════════════ */
export default function TransformerLab() {
  const [dIdx,setDIdx]=useState(0);
  const [sel,setSel]=useState(null);
  const d=DATASET[dIdx];
  const sE=useMemo(()=>addM(embed(d.enT),posEnc(d.enT.length)),[d]);
  const tE=useMemo(()=>addM(embed(d.hiT),posEnc(d.hiT.length)),[d]);

  const detail = () => {
    if(!sel) return <div style={{display:"flex",alignItems:"center",justifyContent:"center",minHeight:400}}><div style={{textAlign:"center"}}><p style={{fontSize:40,marginBottom:8}}>👈</p><p style={{fontSize:14,color:"#64748b",fontWeight:600}}>Click any block in the architecture</p><p style={{fontSize:11,color:"#475569"}}>Each block tells its story step-by-step</p></div></div>;
    if(sel==="in_emb") return <InputEmbeddingStory tokens={d.enT} lang="English" color="#ef4444" />;
    if(sel==="out_emb") return <InputEmbeddingStory tokens={d.hiT} lang="Hindi" color="#3b82f6" />;
    if(sel==="enc_mha") return <MultiHeadAttnStory tokens={d.enT} embs={sE} title="Encoder Self-Attention" />;
    if(sel==="enc_an1"||sel==="enc_an2") return <AddNormStory tokens={d.enT} inputM={sE} outputM={sE} />;
    if(sel==="enc_ff") return <FFNStory tokens={d.enT} inputM={sE} />;
    if(sel==="dec_mmha") return <MultiHeadAttnStory tokens={d.hiT} embs={tE} isMasked={true} title="Decoder Masked Self-Attention" />;
    if(sel==="dec_an1"||sel==="dec_an2"||sel==="dec_an3") return <AddNormStory tokens={d.hiT} inputM={tE} outputM={tE} />;
    if(sel==="dec_mha") return <MultiHeadAttnStory tokens={d.hiT} embs={tE} crossTokens={d.enT} crossEmbs={sE} title="Cross-Attention (Decoder→Encoder)" />;
    if(sel==="dec_ff") return <FFNStory tokens={d.hiT} inputM={tE} />;
    if(sel==="linear"||sel==="softmax") return <LinearSoftmaxStory tokens={d.hiT} />;
    return null;
  };

  return (
    <div style={{fontFamily:S.sans,background:"linear-gradient(145deg,#020617 0%,#1a0a2e 50%,#020617 100%)",minHeight:"100vh",color:"#e2e8f0"}}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
      <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>

      <div style={{maxWidth:1200,margin:"0 auto",padding:"8px 20px",display:"flex",alignItems:"center",gap:10}}>
        <div style={{width:30,height:30,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16,background:"linear-gradient(135deg,#f59e0b,#a855f7)"}}>⚡</div>
        <h1 style={{fontSize:15,fontWeight:800,margin:0}}>Transformer Lab</h1>
        <span style={{fontSize:10,color:"#94a3b8"}}>English → Hindi · d_model={D_MODEL} · {NUM_HEADS} heads · d_ff={D_FF}</span>
      </div>

      <div style={{maxWidth:1200,margin:"0 auto",padding:"8px 20px"}}>
        {/* Dataset */}
        <div style={{borderRadius:10,padding:10,background:"rgba(15,23,42,0.7)",border:"1px solid #1e293b",marginBottom:10}}>
          <p style={{fontSize:9,fontFamily:S.mono,fontWeight:700,color:"#f59e0b",marginBottom:6}}>📊 Translation Dataset (English → Hindi)</p>
          <div style={{display:"flex",gap:6}}>{DATASET.map((dd,i)=>(
            <button key={i} onClick={()=>{setDIdx(i);setSel(null);}} style={{flex:1,padding:"8px 8px",borderRadius:6,fontSize:11,fontWeight:600,textAlign:"left",background:i===dIdx?"#f59e0b15":"#0f172a",border:`1.5px solid ${i===dIdx?"#f59e0b":"#1e293b"}`,color:i===dIdx?"#f59e0b":"#94a3b8",cursor:"pointer"}}>
              <span style={{color:"#22c55e"}}>EN:</span> {dd.en}<br/>
              <span style={{color:"#f59e0b"}}>HI:</span> {dd.hi}
            </button>
          ))}</div>
        </div>

        <div style={{display:"grid",gridTemplateColumns:"400px 1fr",gap:12}}>
          <div style={{borderRadius:12,padding:10,background:"rgba(15,23,42,0.5)",border:"1px solid #1e293b",overflowY:"auto",maxHeight:"80vh"}}>
            <p style={{fontSize:9,fontFamily:S.mono,fontWeight:700,color:"#64748b",marginBottom:6,textTransform:"uppercase",letterSpacing:1}}>⚡ Architecture — Click any block</p>
            <ArchDiagram sel={sel} onSel={setSel} />
          </div>
          <div style={{overflowY:"auto",maxHeight:"80vh",paddingRight:6}}>{detail()}</div>
        </div>
      </div>
    </div>
  );
}
