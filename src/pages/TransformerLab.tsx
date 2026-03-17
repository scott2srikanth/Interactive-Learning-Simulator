import { useState, useEffect, useRef, useMemo } from "react";

/* ═══ CONFIG ═══ */
const D=512, DK=64, H=8, DFF=2048, SHOW=8;
const DATASET=[
  {en:"I love India",hi:"मुझे भारत पसंद है",eT:["I","love","India"],hT:["मुझे","भारत","पसंद","है"]},
  {en:"The sun is bright",hi:"सूरज चमकीला है",eT:["The","sun","is","bright"],hT:["सूरज","चमकीला","है"]},
  {en:"She reads books",hi:"वह किताबें पढ़ती है",eT:["She","reads","books"],hT:["वह","किताबें","पढ़ती","है"]},
];
const F={mono:"'JetBrains Mono',monospace",sans:"'DM Sans',system-ui,sans-serif"};

/* ═══ MATH ═══ */
function srow(v){if(!v?.length)return[];const m=Math.max(...v),e=v.map(x=>Math.exp(x-m)),s=e.reduce((a,b)=>a+b,0);return e.map(x=>x/(s||1));}
function mm(a,b){if(!a?.length||!b?.length)return[];return a.map(r=>b[0].map((_,j)=>r.reduce((s,v,k)=>s+v*(b[k]?.[j]||0),0)));}
function tp(m){return m[0].map((_,i)=>m.map(r=>r[i]));}
function rM(r,c,s=.15){return Array(r).fill(0).map(()=>Array(c).fill(0).map(()=>+((Math.random()-.5)*s*2).toFixed(3)));}
function addM(a,b){return a.map((r,i)=>r.map((v,j)=>+(v+(b[i]?.[j]||0)).toFixed(3)));}
function lnorm(m){return m.map(r=>{const mu=r.reduce((s,v)=>s+v,0)/r.length;const va=r.reduce((s,v)=>s+(v-mu)**2,0)/r.length;return r.map(v=>+((v-mu)/Math.sqrt(va+1e-5)).toFixed(3));});}
function emb(tokens){return tokens.map((t,i)=>Array(SHOW).fill(0).map((_,d)=>+(Math.sin(t.charCodeAt(0)*.25+d*1.4+i*.9)*.5).toFixed(3)));}
function pe(len){return Array.from({length:len}).map((_,p)=>Array(SHOW).fill(0).map((_,i)=>+(i%2===0?Math.sin(p/Math.pow(10000,i/D)):Math.cos(p/Math.pow(10000,(i-1)/D))).toFixed(3)));}
function attn(Q,K,V,mask){const dk=K[0]?.length||1;let sc=mm(Q,tp(K)).map(r=>r.map(v=>+(v/Math.sqrt(dk)).toFixed(3)));if(mask)sc=sc.map((r,i)=>r.map((v,j)=>j>i?-999:v));const w=sc.map(srow);return{scores:sc,weights:w,output:mm(w,V)};}

/* ═══ UI PRIMITIVES ═══ */
function Chip({t,c,sub}){return<div style={{padding:"4px 10px",borderRadius:7,background:`${c}18`,border:`1.5px solid ${c}44`,fontSize:12,fontWeight:700,fontFamily:F.mono,color:c,textAlign:"center"}}>{t}{sub&&<div style={{fontSize:7,color:"#64748b",marginTop:1}}>{sub}</div>}</div>;}

function IOBadge({label,shape,color}){return<div style={{display:"inline-flex",alignItems:"center",gap:4,padding:"3px 8px",borderRadius:6,background:`${color}12`,border:`1px solid ${color}33`,marginRight:6,marginBottom:4}}><span style={{fontSize:8,fontWeight:700,color,fontFamily:F.mono}}>{label}</span><span style={{fontSize:8,color:"#64748b",fontFamily:F.mono}}>{shape}</span></div>;}

function MiniMat({data,label,color,rL,cs=26}){if(!data?.length)return null;const rows=data.length,cols=Math.min(data[0]?.length||0,SHOW);return<div style={{marginBottom:8}}>{label&&<p style={{fontSize:9,fontFamily:F.mono,fontWeight:700,color,marginBottom:3}}>{label}</p>}<div style={{display:"inline-grid",gridTemplateColumns:rL?`32px repeat(${cols},${cs}px)`:`repeat(${cols},${cs}px)`,gap:1}}>{data.map((row,i)=><>{rL&&<span style={{fontSize:7,fontFamily:F.mono,color:"#64748b",alignSelf:"center",textAlign:"right",paddingRight:2}}>{rL[i]?.slice(0,4)}</span>}{row.slice(0,cols).map((v,j)=><div key={`${i}-${j}`} style={{width:cs,height:cs-4,display:"flex",alignItems:"center",justifyContent:"center",fontSize:7,fontWeight:700,fontFamily:F.mono,borderRadius:3,background:`${color}${Math.round(Math.min(Math.abs(v),1)*150+30).toString(16).padStart(2,"0")}`,color:"#fff"}}>{v.toFixed(2)}</div>)}</>)}</div></div>;}

function Heatmap({w,rL,cL,sz=170,label}){const ref=useRef(null);const n=rL?.length||0,m=cL?.length||0;useEffect(()=>{const c=ref.current;if(!c||!n||!m)return;c.width=sz;c.height=sz;const x=c.getContext("2d");x.clearRect(0,0,sz,sz);const o=34,cw=(sz-o)/m,ch=(sz-o)/n;for(let i=0;i<n;i++)for(let j=0;j<m;j++){const v=w[i]?.[j]||0;x.fillStyle=`rgba(168,85,247,${v*.85+.05})`;x.fillRect(o+j*cw,o+i*ch,cw-1,ch-1);if(n<=6){x.fillStyle=v>.15?"#fff":"#64748b";x.font=`bold ${Math.min(9,cw*.28)}px monospace`;x.textAlign="center";x.textBaseline="middle";x.fillText((v*100).toFixed(0)+"%",o+j*cw+cw/2,o+i*ch+ch/2);}}x.fillStyle="#94a3b8";x.font="bold 7px monospace";x.textAlign="right";x.textBaseline="middle";for(let i=0;i<n;i++)x.fillText(rL[i].slice(0,4),o-2,o+i*ch+ch/2);x.textAlign="center";x.textBaseline="bottom";for(let j=0;j<m;j++){x.save();x.translate(o+j*cw+cw/2,o-2);x.rotate(-.4);x.fillText(cL[j].slice(0,4),0,0);x.restore();}},[w,n,m,sz,rL,cL]);return<div>{label&&<p style={{fontSize:9,fontFamily:F.mono,fontWeight:700,color:"#a855f7",marginBottom:3}}>{label}</p>}<canvas ref={ref} style={{width:sz,height:sz,borderRadius:8,border:"1px solid #334155"}} /></div>;}

function Story({icon,title,children,color,input,output}){return<div style={{padding:14,borderRadius:12,background:`${color}08`,border:`1px solid ${color}25`,marginBottom:10}}><div style={{display:"flex",alignItems:"center",gap:8,marginBottom:6}}><span style={{fontSize:18}}>{icon}</span><h4 style={{fontSize:13,fontWeight:800,color,margin:0}}>{title}</h4></div>{input&&<div style={{marginBottom:8}}><IOBadge label="INPUT" shape={input} color="#22c55e" /></div>}<div style={{fontSize:11,color:"#94a3b8",lineHeight:1.7}}>{children}</div>{output&&<div style={{marginTop:8}}><IOBadge label="OUTPUT" shape={output} color="#f59e0b" /></div>}</div>;}

/* ═══ BLOCK DETAIL PANELS — each visually DISTINCT ═══ */

function InputEmbedStory({tokens,color}){
  const e=emb(tokens),p=pe(tokens.length),f=addM(e,p);
  return<div>
    <Story icon="📝" title="Step 1: Tokenize" color={color} input={`"${tokens.join(' ')}"`} output={`[${tokens.length} tokens]`}>
      <p>The sentence is split into individual tokens — the smallest units the model can process.</p>
      <div style={{display:"flex",gap:4,margin:"8px 0",flexWrap:"wrap"}}>{tokens.map((t,i)=><Chip key={i} t={t} c={color} sub={`id=${i}`} />)}</div>
    </Story>
    <Story icon="🔢" title="Step 2: Token → Vector" color="#22c55e" input={`Token IDs [${tokens.length}]`} output={`Embeddings [${tokens.length} × ${D}]`}>
      <p>Each token is looked up in a giant table of {D}-dimensional vectors. Think of it as giving each word a unique "fingerprint" in {D}-dimensional space.</p>
      <MiniMat data={e} label={`Token Embeddings (showing ${SHOW} of ${D} dims)`} color="#22c55e" rL={tokens} />
    </Story>
    <Story icon="🌊" title="Step 3: Add Position Info" color="#06b6d4" input={`Embeddings [${tokens.length}×${D}]`} output={`Position-aware Embeddings [${tokens.length}×${D}]`}>
      <p>Transformers see all words at once — they don't know word order! We add unique sine/cosine waves to tell the model "this word is 1st, this is 2nd..."</p>
      <MiniMat data={p} label="Positional Encoding" color="#06b6d4" rL={tokens.map((_,i)=>`pos${i}`)} />
      <p style={{marginTop:6}}>Final = Token Embedding + Position Encoding</p>
      <MiniMat data={f} label="✅ Final Input to Transformer" color="#f59e0b" rL={tokens} />
    </Story>
  </div>;
}

/* ─── MHA: Self-Attention (Encoder) — ALL tokens see ALL tokens ─── */
function MHASelfStory({tokens,embs}){
  const Wq=useMemo(()=>rM(SHOW,SHOW),[]),Wk=useMemo(()=>rM(SHOW,SHOW),[]),Wv=useMemo(()=>rM(SHOW,SHOW),[]);
  const Q=mm(embs,Wq),K=mm(embs,Wk),V=mm(embs,Wv);
  const{weights,output}=attn(Q,K,V,false);
  return<div>
    <div style={{padding:10,borderRadius:10,background:"#3b82f612",border:"1px solid #3b82f633",marginBottom:10}}>
      <p style={{fontSize:12,fontWeight:800,color:"#3b82f6"}}>🔵 Self-Attention (Encoder)</p>
      <p style={{fontSize:10,color:"#94a3b8"}}>Every source token can see <b style={{color:"#fff"}}>every other source token</b> — no restrictions. This is how the encoder understands the full input sentence.</p>
      <IOBadge label="INPUT" shape={`Source Embeddings [${tokens.length}×${D}]`} color="#22c55e" />
      <IOBadge label="OUTPUT" shape={`Context-enriched Source [${tokens.length}×${D}]`} color="#f59e0b" />
    </div>
    <Story icon="🔴" title="1. Queries (Q) — 'What am I looking for?'" color="#ef4444" input={`Embeddings × W_Q [${D}×${DK}]`} output={`Q [${tokens.length}×${DK}]`}>
      <p>Each token creates a Query vector by multiplying its embedding with W_Q. The query encodes: <i>"What kind of context do I need?"</i></p>
      <MiniMat data={Q} label="Q (Queries)" color="#ef4444" rL={tokens} />
    </Story>
    <Story icon="🟢" title="2. Keys (K) — 'What do I contain?'" color="#22c55e" input={`Embeddings × W_K [${D}×${DK}]`} output={`K [${tokens.length}×${DK}]`}>
      <p>Each token also creates a Key vector. Keys advertise: <i>"Here's what information I have."</i></p>
      <MiniMat data={K} label="K (Keys)" color="#22c55e" rL={tokens} />
    </Story>
    <Story icon="🔵" title="3. Values (V) — 'What info do I provide?'" color="#3b82f6" input={`Embeddings × W_V [${D}×${DK}]`} output={`V [${tokens.length}×${DK}]`}>
      <p>Values carry the <b style={{color:"#fff"}}>actual content</b> that gets mixed. When a token "pays attention" to another, it absorbs that token's Value.</p>
      <MiniMat data={V} label="V (Values)" color="#3b82f6" rL={tokens} />
    </Story>
    <Story icon="📊" title="4. Attention = softmax(Q·Kᵀ/√64) × V" color="#a855f7" input={`Q [${tokens.length}×${DK}], K [${tokens.length}×${DK}], V [${tokens.length}×${DK}]`} output={`Attended Output [${tokens.length}×${DK}]`}>
      <p>Dot product Q·Kᵀ measures compatibility. Divide by √{DK}=8 for stability. Softmax makes each row sum to 100%. <b style={{color:"#fff"}}>Notice: no masking — every cell can have a value!</b></p>
      <div style={{display:"flex",gap:12,flexWrap:"wrap"}}><Heatmap w={weights} rL={tokens} cL={tokens} label="Attention Weights (FULL — no mask)" /><MiniMat data={output} label="Output = Attention × V" color="#ec4899" rL={tokens} /></div>
    </Story>
  </div>;
}

/* ─── Masked MHA (Decoder) — tokens can ONLY see past ─── */
function MHAMaskedStory({tokens,embs}){
  const Wq=useMemo(()=>rM(SHOW,SHOW),[]),Wk=useMemo(()=>rM(SHOW,SHOW),[]),Wv=useMemo(()=>rM(SHOW,SHOW),[]);
  const Q=mm(embs,Wq),K=mm(embs,Wk),V=mm(embs,Wv);
  const{scores,weights,output}=attn(Q,K,V,true);
  return<div>
    <div style={{padding:10,borderRadius:10,background:"#f59e0b12",border:"1px solid #f59e0b33",marginBottom:10}}>
      <p style={{fontSize:12,fontWeight:800,color:"#f59e0b"}}>🟡 Masked Self-Attention (Decoder)</p>
      <p style={{fontSize:10,color:"#94a3b8"}}>Each target token can <b style={{color:"#fff"}}>ONLY see itself and tokens BEFORE it</b> — never the future! This prevents cheating during training.</p>
      <IOBadge label="INPUT" shape={`Target Embeddings [${tokens.length}×${D}]`} color="#22c55e" />
      <IOBadge label="OUTPUT" shape={`Masked Context [${tokens.length}×${D}]`} color="#f59e0b" />
    </div>
    <Story icon="🎭" title="The Mask — Why?" color="#f59e0b">
      <p>Imagine translating "I love India" → "मुझे भारत पसंद है". When predicting "पसंद", the model should only know "मुझे" and "भारत" — NOT "है" (which comes after). The mask enforces this:</p>
      <div style={{display:"inline-grid",gridTemplateColumns:`40px repeat(${tokens.length},50px)`,gap:2,margin:"8px 0"}}>
        <span />{tokens.map((t,j)=><span key={j} style={{fontSize:8,fontFamily:F.mono,color:"#f59e0b",textAlign:"center",fontWeight:700}}>{t.slice(0,4)}</span>)}
        {tokens.map((t,i)=><>{<span style={{fontSize:8,fontFamily:F.mono,color:"#f59e0b",textAlign:"right",fontWeight:700}}>{t.slice(0,4)}</span>}{tokens.map((_,j)=><div key={j} style={{width:50,height:24,borderRadius:4,display:"flex",alignItems:"center",justifyContent:"center",fontSize:10,fontWeight:800,background:j<=i?"#22c55e33":"#ef444433",color:j<=i?"#22c55e":"#ef4444",border:`1px solid ${j<=i?"#22c55e44":"#ef444444"}`}}>{j<=i?"✓":"✗"}</div>)}</>)}
      </div>
      <p>✓ = can see, ✗ = masked (set to -∞ before softmax → becomes 0%)</p>
    </Story>
    <Story icon="📊" title="Masked Attention Heatmap" color="#a855f7" input="Q, K, V from target tokens" output={`Masked Output [${tokens.length}×${DK}]`}>
      <p><b style={{color:"#fff"}}>Compare with encoder's self-attention above</b> — notice the <b style={{color:"#f59e0b"}}>triangular pattern</b>! The upper-right is all zeros because those are future tokens.</p>
      <div style={{display:"flex",gap:12,flexWrap:"wrap"}}><Heatmap w={weights} rL={tokens} cL={tokens} label="⚠️ Masked — upper triangle is 0%" /><MiniMat data={output} label="Masked Output" color="#f59e0b" rL={tokens} /></div>
    </Story>
  </div>;
}

/* ─── Cross MHA (Decoder attends to Encoder) ─── */
function MHACrossStory({decTokens,decEmbs,encTokens,encEmbs}){
  const Wq=useMemo(()=>rM(SHOW,SHOW),[]),Wk=useMemo(()=>rM(SHOW,SHOW),[]),Wv=useMemo(()=>rM(SHOW,SHOW),[]);
  const Q=mm(decEmbs,Wq),K=mm(encEmbs,Wk),V=mm(encEmbs,Wv);
  const{weights,output}=attn(Q,K,V,false);
  return<div>
    <div style={{padding:10,borderRadius:10,background:"#ec489912",border:"1px solid #ec489933",marginBottom:10}}>
      <p style={{fontSize:12,fontWeight:800,color:"#ec4899"}}>🩷 Cross-Attention (Decoder → Encoder)</p>
      <p style={{fontSize:10,color:"#94a3b8"}}><b style={{color:"#fff"}}>THIS IS WHERE TRANSLATION HAPPENS!</b> Decoder tokens ask questions (Q) about the encoder's understanding (K, V).</p>
      <IOBadge label="Q from" shape={`Decoder [${decTokens.length}×${DK}]`} color="#a855f7" />
      <IOBadge label="K,V from" shape={`Encoder [${encTokens.length}×${DK}]`} color="#22c55e" />
      <IOBadge label="OUTPUT" shape={`Cross-attended [${decTokens.length}×${DK}]`} color="#f59e0b" />
    </div>
    <Story icon="🔑" title="Key Difference from Self-Attention" color="#ec4899">
      <p><b style={{color:"#ef4444"}}>Queries</b> come from the <b style={{color:"#a855f7"}}>DECODER</b> (Hindi tokens).</p>
      <p><b style={{color:"#22c55e"}}>Keys + Values</b> come from the <b style={{color:"#22c55e"}}>ENCODER</b> (English tokens).</p>
      <p>So "पसंद" (Hindi for "like") sends a Query asking: "Which English word am I translating?" The encoder's Keys for "love" match strongly → its Value flows into the decoder!</p>
    </Story>
    <Story icon="📊" title="Cross-Attention Heatmap" color="#a855f7" input={`Q[${decTokens.length}×${DK}] from decoder, K,V[${encTokens.length}×${DK}] from encoder`} output={`Cross output [${decTokens.length}×${DK}]`}>
      <p><b style={{color:"#fff"}}>Notice: rows are Hindi, columns are English!</b> This is NOT a square matrix — it's [{decTokens.length}×{encTokens.length}].</p>
      <div style={{display:"flex",gap:12,flexWrap:"wrap"}}><Heatmap w={weights} rL={decTokens} cL={encTokens} label="Decoder→Encoder Attention" /><div>
        <p style={{fontSize:9,fontFamily:F.mono,color:"#ec4899",marginBottom:4}}>Strongest connections:</p>
        {decTokens.map((dt,i)=>{const mx=Math.max(...weights[i]);const j=weights[i].indexOf(mx);return<p key={i} style={{fontSize:10,color:"#94a3b8"}}><b style={{color:"#a855f7"}}>{dt}</b> → <b style={{color:"#22c55e"}}>{encTokens[j]}</b> <span style={{color:"#f59e0b"}}>{(mx*100).toFixed(0)}%</span></p>;})}
      </div></div>
    </Story>
  </div>;
}

/* ─── Add & Norm ─── */
function AddNormStory({tokens,inputM,subM,label}){
  const res=addM(inputM,subM||inputM),normed=lnorm(res);
  return<Story icon="➕" title={label||"Add & Layer Normalize"} color="#eab308" input={`x [${tokens.length}×${D}] + sublayer(x) [${tokens.length}×${D}]`} output={`Normalized [${tokens.length}×${D}]`}>
    <p><b style={{color:"#fff"}}>Residual:</b> output = sublayer(x) + x. Like a highway bypass — the original signal passes through even if the sublayer does nothing useful. Prevents vanishing gradients.</p>
    <p><b style={{color:"#fff"}}>LayerNorm:</b> For each token, normalize its {D} values to mean=0, std=1. Stabilizes training dramatically.</p>
    <MiniMat data={normed} label="After Add & LayerNorm" color="#eab308" rL={tokens} />
  </Story>;
}

/* ─── FFN ─── */
function FFNStory({tokens,inputM}){
  const W1=useMemo(()=>rM(SHOW,SHOW*2),[]),W2=useMemo(()=>rM(SHOW*2,SHOW),[]);
  const hidden=mm(inputM,W1).map(r=>r.map(v=>Math.max(0,v)));
  const out=mm(hidden,W2);
  return<Story icon="🧮" title="Feed-Forward Network" color="#60a5fa" input={`Each token vector [${D}]`} output={`Transformed token vector [${D}]`}>
    <p>A simple 2-layer neural network applied to <b style={{color:"#fff"}}>each token independently</b> (no interaction between tokens here — that already happened in attention).</p>
    <p style={{fontFamily:F.mono,fontSize:10,color:"#60a5fa"}}>FFN(x) = ReLU(x·W₁ + b₁)·W₂ + b₂</p>
    <p>Layer 1: [{D} → {DFF}] with ReLU (expand). Layer 2: [{DFF} → {D}] (compress back).</p>
    <MiniMat data={hidden.map(r=>r.slice(0,8))} label={`Hidden [${tokens.length}×${DFF}] (ReLU applied)`} color="#60a5fa" rL={tokens} cs={24} />
    <MiniMat data={out} label={`Output [${tokens.length}×${D}]`} color="#60a5fa" rL={tokens} />
    <p><b style={{color:"#fff"}}>Parameters:</b> {D}×{DFF} + {DFF}×{D} = <b style={{color:"#f59e0b"}}>{(D*DFF*2).toLocaleString()}</b></p>
  </Story>;
}

/* ─── Linear + Softmax — DISTINCT panels ─── */
function LinearStory({tokens}){
  const vocabSample=["मुझे","भारत","पसंद","है","वह","सूरज","चमकीला","किताबें","पढ़ती","...50K more"];
  const logits=tokens.map(()=>vocabSample.map(()=>+(Math.random()*8-4).toFixed(1)));
  return<Story icon="📐" title="Linear Projection" color="#c4b5fd" input={`Decoder output [${tokens.length}×${D}]`} output={`Logits [${tokens.length}×vocab_size(50,257)]`}>
    <p>Each decoder output vector ({D} dims) is multiplied by a weight matrix [{D}×50,257] to produce a <b style={{color:"#fff"}}>score for every word in the Hindi vocabulary</b>.</p>
    <p>These raw scores are called <b style={{color:"#c4b5fd"}}>logits</b> — they can be any number (positive or negative).</p>
    <div style={{overflowX:"auto"}}><table style={{fontSize:8,fontFamily:F.mono,borderCollapse:"collapse"}}>
      <tr><td style={{padding:3,color:"#64748b"}}></td>{vocabSample.map((w,j)=><td key={j} style={{padding:"2px 4px",color:"#c4b5fd",fontWeight:700,textAlign:"center"}}>{w}</td>)}</tr>
      {tokens.map((t,i)=><tr key={i}><td style={{padding:3,color:"#a855f7",fontWeight:700}}>{t}</td>{logits[i].map((v,j)=><td key={j} style={{padding:"2px 4px",textAlign:"center",color:v>2?"#22c55e":v<-2?"#ef4444":"#94a3b8",fontWeight:v>2?800:400}}>{v}</td>)}</tr>)}
    </table></div>
    <p style={{marginTop:6}}>High positive score = model thinks this word is likely. Negative = unlikely.</p>
  </Story>;
}

function SoftmaxStory({tokens}){
  return<Story icon="📊" title="Softmax → Next Word Prediction" color="#86efac" input={`Logits [${tokens.length}×50,257]`} output={`Probability distribution per position`}>
    <p>Softmax converts raw logits into <b style={{color:"#fff"}}>probabilities that sum to 100%</b>. The highest probability word at each position is the prediction.</p>
    {tokens.map((t,i)=>{
      const conf=82+i*4+Math.floor(Math.random()*5);
      const others=[{w:"[random]",p:100-conf-8},{w:"[other]",p:8}];
      return<div key={i} style={{marginBottom:10,padding:10,borderRadius:8,background:"#0f172a",border:"1px solid #1e293b"}}>
        <p style={{fontSize:10,fontFamily:F.mono,color:"#a855f7",marginBottom:4}}>Position {i}: Which Hindi word comes here?</p>
        <div style={{display:"flex",alignItems:"center",gap:6,marginBottom:6}}>
          <div style={{flex:1,height:20,background:"#1e293b",borderRadius:4,overflow:"hidden",position:"relative"}}>
            <div style={{width:`${conf}%`,height:"100%",background:"#22c55e",borderRadius:4}} />
            <span style={{position:"absolute",left:8,top:2,fontSize:9,color:"#fff",fontWeight:700}}>{t} — {conf}%</span>
          </div>
        </div>
        <div style={{display:"flex",gap:4}}>{others.map((o,j)=>
          <div key={j} style={{flex:1,height:12,background:"#1e293b",borderRadius:3,overflow:"hidden",position:"relative"}}>
            <div style={{width:`${o.p}%`,height:"100%",background:"#475569",borderRadius:3}} />
            <span style={{position:"absolute",left:4,top:0,fontSize:7,color:"#94a3b8"}}>{o.w} {o.p}%</span>
          </div>
        )}</div>
        <p style={{fontSize:10,color:"#22c55e",fontWeight:800,marginTop:4}}>✅ Predicted: "{t}"</p>
      </div>;
    })}
    <p><b style={{color:"#22c55e"}}>The full translated sentence:</b> {tokens.join(" ")}</p>
  </Story>;
}

/* ═══ SVG ARCHITECTURE ═══ */
function ABlock({x,y,w,h,label,color,sel,onClick,sub}){return<g onClick={onClick} style={{cursor:"pointer"}}><rect x={x} y={y} width={w} height={h} rx={5} fill={sel?color:`${color}55`} stroke={sel?"#fff":color} strokeWidth={sel?2.5:1.2} /><text x={x+w/2} y={y+h/2-(sub?3:0)} textAnchor="middle" dominantBaseline="central" fill={sel?"#000":"#e2e8f0"} fontSize={8.5} fontWeight={700} fontFamily="DM Sans">{label}</text>{sub&&<text x={x+w/2} y={y+h/2+7} textAnchor="middle" fill={sel?"#000":"#94a3b8"} fontSize={6.5} fontFamily="monospace">{sub}</text>}</g>;}

function Arch({sel,onSel}){
  const bw=120,bh=26,ex=25,dx=195;
  const Y={ie:395,ep:365,em:330,ea1:300,ef:270,ea2:240,oe:395,op:365,dm:330,da1:300,cx:270,da2:240,df:210,da3:180,li:145,sm:115};
  const ar=(x1,y1,x2,y2)=><line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#475569" strokeWidth={1.2} markerEnd="url(#ah)" />;
  return<svg viewBox="0 0 370 450" style={{width:"100%",maxWidth:370}}>
    <defs><marker id="ah" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto"><polygon points="0 0,6 2.5,0 5" fill="#475569"/></marker></defs>
    <text x={ex+bw/2} y={425} textAnchor="middle" fill="#94a3b8" fontSize={8} fontWeight={700}>Inputs (English)</text>
    <text x={dx+bw/2} y={425} textAnchor="middle" fill="#94a3b8" fontSize={8} fontWeight={700}>Outputs (Hindi)</text>
    <text x={dx+bw/2} y={100} textAnchor="middle" fill="#22c55e" fontSize={9} fontWeight={800}>Output Probabilities</text>
    <rect x={ex-5} y={Y.ea2-5} width={bw+10} height={Y.em-Y.ea2+bh+10} rx={6} fill="none" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4,3" />
    <text x={ex-10} y={(Y.ea2+Y.em+bh)/2} fill="#94a3b8" fontSize={10} fontWeight={800} textAnchor="end">N×</text>
    <rect x={dx-5} y={Y.da3-5} width={bw+10} height={Y.dm-Y.da3+bh+10} rx={6} fill="none" stroke="#64748b" strokeWidth={1.2} strokeDasharray="4,3" />
    <text x={dx+bw+14} y={(Y.da3+Y.dm+bh)/2} fill="#94a3b8" fontSize={10} fontWeight={800}>N×</text>
    {ar(ex+bw/2,415,ex+bw/2,Y.ie+bh)}{ar(ex+bw/2,Y.ie,ex+bw/2,Y.ep+bh)}{ar(ex+bw/2,Y.ep,ex+bw/2,Y.em+bh)}{ar(ex+bw/2,Y.em,ex+bw/2,Y.ea1+bh)}{ar(ex+bw/2,Y.ea1,ex+bw/2,Y.ef+bh)}{ar(ex+bw/2,Y.ef,ex+bw/2,Y.ea2+bh)}
    {ar(dx+bw/2,415,dx+bw/2,Y.oe+bh)}{ar(dx+bw/2,Y.oe,dx+bw/2,Y.op+bh)}{ar(dx+bw/2,Y.op,dx+bw/2,Y.dm+bh)}{ar(dx+bw/2,Y.dm,dx+bw/2,Y.da1+bh)}{ar(dx+bw/2,Y.da1,dx+bw/2,Y.cx+bh)}{ar(dx+bw/2,Y.cx,dx+bw/2,Y.da2+bh)}{ar(dx+bw/2,Y.da2,dx+bw/2,Y.df+bh)}{ar(dx+bw/2,Y.df,dx+bw/2,Y.da3+bh)}{ar(dx+bw/2,Y.da3,dx+bw/2,Y.li+bh)}{ar(dx+bw/2,Y.li,dx+bw/2,Y.sm+bh)}
    <path d={`M${ex+bw},${Y.ea2+bh/2} L${dx},${Y.cx+bh/2}`} fill="none" stroke="#ec4899" strokeWidth={2} markerEnd="url(#ah)" />
    <text x={(ex+bw+dx)/2} y={Y.cx+bh/2-6} textAnchor="middle" fill="#ec4899" fontSize={6.5} fontWeight={700}>K, V from Encoder</text>
    <ABlock x={ex} y={Y.ie} w={bw} h={bh} label="Input Embedding" color="#fca5a5" sel={sel==="ie"} onClick={()=>onSel("ie")} />
    <ABlock x={ex} y={Y.ep} w={bw} h={bh} label="+ Pos Encoding" color="#67e8f9" sel={sel==="ep"} onClick={()=>onSel("ie")} />
    <ABlock x={ex} y={Y.em} w={bw} h={bh} label="Multi-Head Attention" sub="(Self)" color="#fdba74" sel={sel==="em"} onClick={()=>onSel("em")} />
    <ABlock x={ex} y={Y.ea1} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="ea1"} onClick={()=>onSel("ea1")} />
    <ABlock x={ex} y={Y.ef} w={bw} h={bh} label="Feed Forward" color="#93c5fd" sel={sel==="ef"} onClick={()=>onSel("ef")} />
    <ABlock x={ex} y={Y.ea2} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="ea2"} onClick={()=>onSel("ea2")} />
    <ABlock x={dx} y={Y.oe} w={bw} h={bh} label="Output Embedding" color="#fca5a5" sel={sel==="oe"} onClick={()=>onSel("oe")} />
    <ABlock x={dx} y={Y.op} w={bw} h={bh} label="+ Pos Encoding" color="#67e8f9" sel={sel==="op"} onClick={()=>onSel("oe")} />
    <ABlock x={dx} y={Y.dm} w={bw} h={bh} label="Masked Multi-Head" sub="Attention (Self)" color="#fcd34d" sel={sel==="dm"} onClick={()=>onSel("dm")} />
    <ABlock x={dx} y={Y.da1} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="da1"} onClick={()=>onSel("da1")} />
    <ABlock x={dx} y={Y.cx} w={bw} h={bh} label="Multi-Head Attention" sub="(Cross: Dec→Enc)" color="#f9a8d4" sel={sel==="cx"} onClick={()=>onSel("cx")} />
    <ABlock x={dx} y={Y.da2} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="da2"} onClick={()=>onSel("da2")} />
    <ABlock x={dx} y={Y.df} w={bw} h={bh} label="Feed Forward" color="#93c5fd" sel={sel==="df"} onClick={()=>onSel("df")} />
    <ABlock x={dx} y={Y.da3} w={bw} h={bh} label="Add & Norm" color="#fde68a" sel={sel==="da3"} onClick={()=>onSel("da3")} />
    <ABlock x={dx} y={Y.li} w={bw} h={bh} label="Linear" color="#c4b5fd" sel={sel==="li"} onClick={()=>onSel("li")} />
    <ABlock x={dx} y={Y.sm} w={bw} h={bh} label="Softmax" color="#86efac" sel={sel==="sm"} onClick={()=>onSel("sm")} />
  </svg>;
}

/* ═══ MAIN ═══ */
export default function TransformerLab(){
  const[dI,setDI]=useState(0),[sel,setSel]=useState(null);
  const d=DATASET[dI];
  const sE=useMemo(()=>addM(emb(d.eT),pe(d.eT.length)),[d]);
  const tE=useMemo(()=>addM(emb(d.hT),pe(d.hT.length)),[d]);

  const detail=()=>{
    if(!sel)return<div style={{display:"flex",alignItems:"center",justifyContent:"center",minHeight:400}}><div style={{textAlign:"center"}}><p style={{fontSize:36,marginBottom:8}}>👈</p><p style={{fontSize:14,color:"#64748b",fontWeight:600}}>Click any block to see what happens inside</p><p style={{fontSize:11,color:"#475569"}}>Each block shows INPUT → PROCESSING → OUTPUT</p></div></div>;
    if(sel==="ie")return<InputEmbedStory tokens={d.eT} color="#ef4444" />;
    if(sel==="oe")return<InputEmbedStory tokens={d.hT} color="#3b82f6" />;
    if(sel==="em")return<MHASelfStory tokens={d.eT} embs={sE} />;
    if(sel==="dm")return<MHAMaskedStory tokens={d.hT} embs={tE} />;
    if(sel==="cx")return<MHACrossStory decTokens={d.hT} decEmbs={tE} encTokens={d.eT} encEmbs={sE} />;
    if(sel==="ea1"||sel==="ea2")return<AddNormStory tokens={d.eT} inputM={sE} />;
    if(sel==="da1"||sel==="da2"||sel==="da3")return<AddNormStory tokens={d.hT} inputM={tE} />;
    if(sel==="ef")return<FFNStory tokens={d.eT} inputM={sE} />;
    if(sel==="df")return<FFNStory tokens={d.hT} inputM={tE} />;
    if(sel==="li")return<LinearStory tokens={d.hT} />;
    if(sel==="sm")return<SoftmaxStory tokens={d.hT} />;
    return null;
  };

  return<div style={{fontFamily:F.sans,background:"linear-gradient(145deg,#020617 0%,#1a0a2e 50%,#020617 100%)",minHeight:"100vh",color:"#e2e8f0"}}>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap" rel="stylesheet" />
    <style>{`*::-webkit-scrollbar{width:5px} *::-webkit-scrollbar-thumb{background:#334155;border-radius:3px}`}</style>
    <div style={{maxWidth:1200,margin:"0 auto",padding:"8px 20px",display:"flex",alignItems:"center",gap:10}}>
      <div style={{width:30,height:30,borderRadius:8,display:"flex",alignItems:"center",justifyContent:"center",fontSize:16,background:"linear-gradient(135deg,#f59e0b,#a855f7)"}}>⚡</div>
      <h1 style={{fontSize:15,fontWeight:800,margin:0}}>Transformer Lab</h1>
      <span style={{fontSize:9,color:"#94a3b8",fontFamily:F.mono}}>EN→HI · d={D} · heads={H} · d_ff={DFF}</span>
    </div>
    <div style={{maxWidth:1200,margin:"0 auto",padding:"6px 20px"}}>
      <div style={{borderRadius:10,padding:8,background:"rgba(15,23,42,0.7)",border:"1px solid #1e293b",marginBottom:10}}>
        <div style={{display:"flex",gap:6}}>{DATASET.map((dd,i)=><button key={i} onClick={()=>{setDI(i);setSel(null);}} style={{flex:1,padding:"6px 8px",borderRadius:6,fontSize:10,fontWeight:600,textAlign:"left",background:i===dI?"#f59e0b15":"#0f172a",border:`1.5px solid ${i===dI?"#f59e0b":"#1e293b"}`,color:i===dI?"#f59e0b":"#94a3b8",cursor:"pointer"}}><span style={{color:"#22c55e"}}>EN:</span> {dd.en} → <span style={{color:"#f59e0b"}}>HI:</span> {dd.hi}</button>)}</div>
      </div>
      <div style={{display:"grid",gridTemplateColumns:"390px 1fr",gap:12}}>
        <div style={{borderRadius:12,padding:8,background:"rgba(15,23,42,0.5)",border:"1px solid #1e293b",overflowY:"auto",maxHeight:"82vh"}}><Arch sel={sel} onSel={setSel} /></div>
        <div style={{overflowY:"auto",maxHeight:"82vh",paddingRight:6}}>{detail()}</div>
      </div>
    </div>
  </div>;
}
