import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { ArrowLeft } from 'lucide-react';

/* ═══════════════════════════════════════════════════════════
   SHARED MATH UTILS
   ═══════════════════════════════════════════════════════════ */
const sigmoid = (x: number) => 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
const reluFn = (x: number) => Math.max(0, x);
const tanhFn = (x: number) => Math.tanh(x);
const softmaxArr = (a: number[]) => { const mx = Math.max(...a), e = a.map(v => Math.exp(v - mx)), s = e.reduce((p, c) => p + c, 0); return e.map(v => v / s); };
function activate(x: number, fn: string) { return fn === 'sigmoid' ? sigmoid(x) : fn === 'relu' ? reluFn(x) : fn === 'tanh' ? tanhFn(x) : x; }
function activateDeriv(x: number, fn: string) { if (fn === 'sigmoid') { const s = sigmoid(x); return s * (1 - s); } if (fn === 'relu') return x > 0 ? 1 : 0; if (fn === 'tanh') return 1 - Math.tanh(x) ** 2; return 1; }
function randMat(r: number, c: number) { const s = Math.sqrt(2 / c); return Array(r).fill(0).map(() => Array(c).fill(0).map(() => (Math.random() - 0.5) * s * 2)); }
function randVec(n: number) { return Array(n).fill(0).map(() => (Math.random() - 0.5) * 0.1); }

/* ═══════════════════════════════════════════════════════════
   MINI CANVAS / GRID COMPONENTS
   ═══════════════════════════════════════════════════════════ */
const S = { mono: "'monospace'", bg: '#0f172a', border: '#1e293b' };

function SmallGrid({ data, cellSize = 32, label, highlight, colorFn }: any) {
  if (!data?.length) return null;
  const rows = data.length, cols = data[0]?.length || 0;
  const defaultColor = (v: number) => { const n = Math.max(0, Math.min(1, v)); return `rgb(${Math.round(n * 255)},${Math.round(n * 255)},${Math.round(n * 255)})`; };
  const cf = colorFn || defaultColor;
  return (
    <div className="inline-block">
      {label && <p className="text-xs font-bold text-gray-400 mb-1" style={{ fontFamily: 'monospace' }}>{label}</p>}
      <div style={{ display: 'inline-grid', gridTemplateColumns: `repeat(${cols}, ${cellSize}px)`, gap: 1 }}>
        {data.flat().map((v: number, i: number) => {
          const r = Math.floor(i / cols), c = i % cols;
          const hl = highlight && r >= highlight.r && r < highlight.r + highlight.h && c >= highlight.c && c < highlight.c + highlight.w;
          return (
            <div key={i} style={{ width: cellSize, height: cellSize, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: cellSize < 28 ? 7 : 9, fontWeight: 600, fontFamily: 'monospace', borderRadius: 3, background: hl ? '#facc1555' : cf(v), color: '#fff', border: hl ? '2px solid #facc15' : '1px solid #334155' }}>
              {rows * cols <= 64 ? v.toFixed(1) : ''}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ActivationCanvas({ fn, w = 200, h = 120 }: { fn: string; w?: number; h?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    c.width = w; c.height = h; const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let px = 0; px < w; px++) {
      const x = (px / w) * 8 - 4;
      const y = activate(x, fn);
      const py = h - ((y + 1.5) / 3) * h;
      px === 0 ? ctx.moveTo(px, Math.max(0, Math.min(h, py))) : ctx.lineTo(px, Math.max(0, Math.min(h, py)));
    }
    ctx.stroke();
    ctx.fillStyle = '#e2e8f0'; ctx.font = 'bold 11px monospace'; ctx.textAlign = 'center';
    ctx.fillText(fn === 'relu' ? 'ReLU' : fn === 'sigmoid' ? 'Sigmoid' : fn === 'tanh' ? 'Tanh' : fn, w / 2, 14);
  }, [fn, w, h]);
  return <canvas ref={ref} style={{ width: w, height: h, borderRadius: 8, border: '1px solid #334155' }} />;
}

/* ═══════════════════════════════════════════════════════════
   CNN LESSON LABS
   ═══════════════════════════════════════════════════════════ */
function ConvolutionLab() {
  const [kernel, setKernel] = useState([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]);
  const [stride, setStride] = useState(1);
  const [pos, setPos] = useState({ r: 0, c: 0 });
  const [playing, setPlaying] = useState(false);
  const input = [[0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0], [0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]];
  const k = kernel.length;
  const oH = Math.floor((input.length - k) / stride) + 1, oW = Math.floor((input[0].length - k) / stride) + 1;
  const output: number[][] = [];
  for (let i = 0; i < oH; i++) { const row: number[] = []; for (let j = 0; j < oW; j++) { let s = 0; for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) s += input[i * stride + a][j * stride + b] * kernel[a][b]; row.push(s); } output.push(row); }

  useEffect(() => { if (!playing) return; const t = setTimeout(() => { let { r, c } = pos; c++; if (c >= oW) { c = 0; r++; } if (r >= oH) { setPlaying(false); return; } setPos({ r, c }); }, 400); return () => clearTimeout(t); }, [playing, pos, oH, oW]);

  const iR = pos.r * stride, iC = pos.c * stride;
  let sum = 0; const prods: { iv: number; kv: number; p: number }[] = [];
  for (let a = 0; a < k; a++) for (let b = 0; b < k; b++) { const iv = input[iR + a]?.[iC + b] ?? 0, kv = kernel[a][b], p = iv * kv; prods.push({ iv, kv, p }); sum += p; }

  const PRESET_KERNELS: Record<string, number[][]> = {
    edge: [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], vert: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
    horiz: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], sharpen: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    blur: [[1, 1, 1], [1, 1, 1], [1, 1, 1]].map(r => r.map(v => +(v / 9).toFixed(2))),
  };

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap">
        {Object.entries(PRESET_KERNELS).map(([name, k]) => (
          <button key={name} onClick={() => { setKernel(k); setPos({ r: 0, c: 0 }); }} className="px-3 py-1 rounded text-xs font-semibold bg-blue-900 text-blue-300 border border-blue-700 hover:bg-blue-800">{name}</button>
        ))}
        <select value={stride} onChange={e => { setStride(+e.target.value); setPos({ r: 0, c: 0 }); }} className="px-2 py-1 rounded text-xs bg-gray-800 text-white border border-gray-600">
          <option value={1}>Stride 1</option><option value={2}>Stride 2</option>
        </select>
        <button onClick={() => { setPos({ r: 0, c: 0 }); setPlaying(!playing); }} className={`px-3 py-1 rounded text-xs font-bold text-white ${playing ? 'bg-red-600' : 'bg-green-600'}`}>{playing ? '⏸ Pause' : '▶ Animate'}</button>
        <button onClick={() => setPos({ r: 0, c: 0 })} className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">↺</button>
      </div>
      <div className="flex gap-6 items-start flex-wrap">
        <SmallGrid data={input} cellSize={36} label="Input 6×6" highlight={{ r: iR, c: iC, h: k, w: k }} colorFn={(v: number) => v > 0.5 ? '#3b82f6' : '#1e293b'} />
        <div className="text-center">
          <SmallGrid data={kernel} cellSize={36} label="Kernel 3×3" colorFn={(v: number) => v > 0 ? `rgba(250,204,21,${Math.abs(v) / 8 + 0.15})` : `rgba(239,68,68,${Math.abs(v) / 8 + 0.15})`} />
        </div>
        <SmallGrid data={output} cellSize={36} label={`Output ${oH}×${oW}`} colorFn={(v: number) => { const n = (v + 10) / 20; return `rgb(${Math.round(n * 200 + 30)},${Math.round(n * 150 + 30)},${Math.round((1 - n) * 200 + 30)})`; }} />
      </div>
      <div className="p-3 bg-gray-900 rounded-lg text-xs font-mono">
        <p className="text-green-400 mb-1">Position ({pos.r},{pos.c}):</p>
        <div className="flex gap-1 flex-wrap">{prods.map((p, i) => <span key={i} className="text-gray-300"><span className="text-blue-400">{p.iv.toFixed(1)}</span>×<span className="text-yellow-400">{p.kv.toFixed(1)}</span>=<b className={p.p >= 0 ? 'text-green-400' : 'text-red-400'}>{p.p.toFixed(1)}</b>{i < prods.length - 1 ? ' + ' : ''}</span>)}</div>
        <p className="text-yellow-300 mt-1 font-bold">Σ = {sum.toFixed(2)}</p>
      </div>
    </div>
  );
}

function PoolingLab() {
  const input = [[4, 2, 7, 1, 3, 5], [6, 8, 1, 3, 2, 4], [3, 1, 9, 5, 7, 2], [5, 7, 2, 8, 1, 6], [2, 4, 6, 3, 9, 1], [8, 3, 5, 7, 4, 2]];
  const [poolType, setPoolType] = useState('max');
  const [poolSize, setPoolSize] = useState(2);
  const [pos, setPos] = useState({ r: 0, c: 0 });
  const [playing, setPlaying] = useState(false);
  const oH = Math.floor(input.length / poolSize), oW = Math.floor(input[0].length / poolSize);
  const output: number[][] = [];
  for (let i = 0; i < oH; i++) { const row: number[] = []; for (let j = 0; j < oW; j++) { const vals: number[] = []; for (let a = 0; a < poolSize; a++) for (let b = 0; b < poolSize; b++) vals.push(input[i * poolSize + a]?.[j * poolSize + b] ?? 0); row.push(poolType === 'max' ? Math.max(...vals) : vals.reduce((s, v) => s + v, 0) / vals.length); } output.push(row); }

  useEffect(() => { if (!playing) return; const t = setTimeout(() => { let { r, c } = pos; c++; if (c >= oW) { c = 0; r++; } if (r >= oH) { setPlaying(false); return; } setPos({ r, c }); }, 500); return () => clearTimeout(t); }, [playing, pos, oH, oW]);

  const iR = pos.r * poolSize, iC = pos.c * poolSize;
  const windowVals: number[] = [];
  for (let a = 0; a < poolSize; a++) for (let b = 0; b < poolSize; b++) windowVals.push(input[iR + a]?.[iC + b] ?? 0);
  const result = poolType === 'max' ? Math.max(...windowVals) : windowVals.reduce((s, v) => s + v, 0) / windowVals.length;

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        <button onClick={() => setPoolType('max')} className={`px-3 py-1 rounded text-xs font-bold ${poolType === 'max' ? 'bg-orange-600 text-white' : 'bg-gray-800 text-gray-400'}`}>Max Pool</button>
        <button onClick={() => setPoolType('avg')} className={`px-3 py-1 rounded text-xs font-bold ${poolType === 'avg' ? 'bg-cyan-600 text-white' : 'bg-gray-800 text-gray-400'}`}>Avg Pool</button>
        <button onClick={() => { setPos({ r: 0, c: 0 }); setPlaying(!playing); }} className={`px-3 py-1 rounded text-xs font-bold text-white ${playing ? 'bg-red-600' : 'bg-green-600'}`}>{playing ? '⏸' : '▶ Animate'}</button>
      </div>
      <div className="flex gap-6 items-start flex-wrap">
        <SmallGrid data={input} cellSize={36} label="Input 6×6" highlight={{ r: iR, c: iC, h: poolSize, w: poolSize }} colorFn={(v: number) => `rgba(59,130,246,${v / 10 + 0.1})`} />
        <SmallGrid data={output} cellSize={44} label={`${poolType === 'max' ? 'Max' : 'Avg'} Pool ${oH}×${oW}`} colorFn={(v: number) => `rgba(34,197,94,${v / 10 + 0.15})`} />
      </div>
      <div className="p-3 bg-gray-900 rounded-lg text-xs font-mono">
        <span className="text-orange-400">{poolType}([{windowVals.join(', ')}]) = </span>
        <b className="text-green-400 text-sm">{result.toFixed(2)}</b>
      </div>
    </div>
  );
}

function ActivationLab() {
  const [fn, setFn] = useState('relu');
  const [input, setInput] = useState([-2, -1, -0.5, 0, 0.5, 1, 2, 3]);
  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {['relu', 'sigmoid', 'tanh'].map(f => <button key={f} onClick={() => setFn(f)} className={`px-3 py-1 rounded text-xs font-bold ${fn === f ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}>{f}</button>)}
      </div>
      <ActivationCanvas fn={fn} w={300} h={140} />
      <div className="grid grid-cols-4 gap-2">
        {input.map((v, i) => (
          <div key={i} className="p-2 bg-gray-900 rounded text-center text-xs font-mono">
            <span className="text-blue-400">{v.toFixed(1)}</span>
            <span className="text-gray-500"> → </span>
            <b className="text-green-400">{activate(v, fn).toFixed(3)}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   ANN BACKPROPAGATION LAB — full gradient visualization
   ═══════════════════════════════════════════════════════════ */
function BackpropLab() {
  const [lr, setLr] = useState(0.5);
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);

  // Simple 2-1-1 network: 2 inputs, 1 hidden, 1 output
  const [w1, setW1] = useState([0.5, -0.3]); // hidden weights
  const [b1, setB1] = useState(0.1);
  const [w2, setW2] = useState([0.8]); // output weight
  const [b2, setB2] = useState(-0.2);
  const x = [1.0, 0.5]; // input
  const target = 1.0;
  const actFn = 'sigmoid';

  // Forward pass
  const z1 = x[0] * w1[0] + x[1] * w1[1] + b1;
  const a1 = activate(z1, actFn);
  const z2 = a1 * w2[0] + b2;
  const a2 = activate(z2, actFn);
  const loss = 0.5 * (target - a2) ** 2;

  // Backward pass
  const dL_da2 = -(target - a2);
  const da2_dz2 = activateDeriv(z2, actFn);
  const dL_dz2 = dL_da2 * da2_dz2;
  const dL_dw2 = dL_dz2 * a1;
  const dL_db2 = dL_dz2;
  const dL_da1 = dL_dz2 * w2[0];
  const da1_dz1 = activateDeriv(z1, actFn);
  const dL_dz1 = dL_da1 * da1_dz1;
  const dL_dw1_0 = dL_dz1 * x[0];
  const dL_dw1_1 = dL_dz1 * x[1];
  const dL_db1 = dL_dz1;

  const doUpdate = () => {
    setW2([w2[0] - lr * dL_dw2]);
    setB2(b2 - lr * dL_db2);
    setW1([w1[0] - lr * dL_dw1_0, w1[1] - lr * dL_dw1_1]);
    setB1(b1 - lr * dL_db1);
    setStep(0);
  };

  const STEPS = [
    { t: '1. Forward: Hidden Layer', d: `z₁ = x₁×w₁ + x₂×w₂ + b = ${x[0]}×${w1[0].toFixed(3)} + ${x[1]}×${w1[1].toFixed(3)} + ${b1.toFixed(3)} = ${z1.toFixed(4)}`, d2: `a₁ = σ(z₁) = σ(${z1.toFixed(4)}) = ${a1.toFixed(4)}`, c: '#3b82f6' },
    { t: '2. Forward: Output Layer', d: `z₂ = a₁×w₃ + b₂ = ${a1.toFixed(4)}×${w2[0].toFixed(3)} + ${b2.toFixed(3)} = ${z2.toFixed(4)}`, d2: `ŷ = σ(z₂) = ${a2.toFixed(4)}  |  Target = ${target}`, c: '#a855f7' },
    { t: '3. Loss Computation', d: `L = ½(target - ŷ)² = ½(${target} - ${a2.toFixed(4)})² = ${loss.toFixed(6)}`, d2: '', c: '#ef4444' },
    { t: '4. Backprop: Output Gradients', d: `∂L/∂ŷ = -(target - ŷ) = ${dL_da2.toFixed(4)}`, d2: `∂L/∂z₂ = ∂L/∂ŷ × σ'(z₂) = ${dL_da2.toFixed(4)} × ${da2_dz2.toFixed(4)} = ${dL_dz2.toFixed(4)}`, c: '#f59e0b' },
    { t: '5. Backprop: Weight Gradients', d: `∂L/∂w₃ = ∂L/∂z₂ × a₁ = ${dL_dz2.toFixed(4)} × ${a1.toFixed(4)} = ${dL_dw2.toFixed(6)}`, d2: `∂L/∂w₁ = ${dL_dw1_0.toFixed(6)}, ∂L/∂w₂ = ${dL_dw1_1.toFixed(6)}`, c: '#f59e0b' },
    { t: '6. Weight Update (Gradient Descent)', d: `w₃_new = ${w2[0].toFixed(3)} - ${lr}×${dL_dw2.toFixed(4)} = ${(w2[0] - lr * dL_dw2).toFixed(4)}`, d2: `w₁_new = ${(w1[0] - lr * dL_dw1_0).toFixed(4)}, w₂_new = ${(w1[1] - lr * dL_dw1_1).toFixed(4)}`, c: '#22c55e' },
  ];

  useEffect(() => { if (auto && step < STEPS.length - 1) { const t = setTimeout(() => setStep(p => p + 1), 2000); return () => clearTimeout(t); } else setAuto(false); }, [auto, step, STEPS.length]);

  return (
    <div className="space-y-4">
      <div className="flex gap-2 flex-wrap items-center">
        <label className="text-xs text-gray-400">LR:
          <select value={lr} onChange={e => setLr(+e.target.value)} className="ml-1 px-2 py-1 rounded text-xs bg-gray-800 text-white border border-gray-600">
            {[0.1, 0.3, 0.5, 1.0, 2.0].map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <button onClick={() => setAuto(!auto)} className={`px-3 py-1 rounded text-xs font-bold text-white ${auto ? 'bg-red-600' : 'bg-green-600'}`}>{auto ? '⏸' : '▶ Auto Step'}</button>
        <button onClick={() => setStep(0)} className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-300">↺</button>
        <button onClick={doUpdate} className="px-3 py-1 rounded text-xs font-bold bg-blue-600 text-white">Apply Update</button>
      </div>

      {/* Network diagram */}
      <div className="p-4 bg-gray-900 rounded-lg">
        <div className="flex items-center justify-center gap-8 flex-wrap">
          <div className="text-center">
            <p className="text-xs text-green-400 font-bold mb-2">Input</p>
            {x.map((v, i) => <div key={i} className="w-12 h-12 rounded-full bg-green-900 border-2 border-green-500 flex items-center justify-center text-xs font-mono text-green-300 mb-1">{v}</div>)}
          </div>
          <div className="text-center text-xs font-mono space-y-1">
            <p className="text-yellow-400">w₁={w1[0].toFixed(3)}</p>
            <p className="text-yellow-400">w₂={w1[1].toFixed(3)}</p>
            <p className="text-gray-500">b₁={b1.toFixed(3)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-blue-400 font-bold mb-2">Hidden</p>
            <div className="w-14 h-14 rounded-full bg-blue-900 border-2 border-blue-500 flex items-center justify-center text-xs font-mono text-blue-300">
              {a1.toFixed(3)}
            </div>
          </div>
          <div className="text-center text-xs font-mono space-y-1">
            <p className="text-yellow-400">w₃={w2[0].toFixed(3)}</p>
            <p className="text-gray-500">b₂={b2.toFixed(3)}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-purple-400 font-bold mb-2">Output</p>
            <div className="w-14 h-14 rounded-full bg-purple-900 border-2 border-purple-500 flex items-center justify-center text-xs font-mono text-purple-300">
              {a2.toFixed(3)}
            </div>
            <p className="text-xs text-red-400 mt-1">Loss: {loss.toFixed(4)}</p>
          </div>
        </div>
      </div>

      {/* Step progress */}
      <div className="flex gap-1">{STEPS.map((_, i) => <button key={i} onClick={() => { setStep(i); setAuto(false); }} className="flex-1 h-2 rounded-full" style={{ background: i <= step ? STEPS[i].c : '#1e293b' }} />)}</div>

      {/* Current step */}
      <div className="p-4 rounded-lg" style={{ background: `${STEPS[step].c}11`, border: `1px solid ${STEPS[step].c}44` }}>
        <h4 className="font-bold text-white text-sm mb-2">{STEPS[step].t}</h4>
        <p className="text-xs font-mono text-gray-300">{STEPS[step].d}</p>
        {STEPS[step].d2 && <p className="text-xs font-mono text-gray-400 mt-1">{STEPS[step].d2}</p>}
      </div>

      {/* Gradient table */}
      {step >= 4 && (
        <div className="p-3 bg-gray-900 rounded-lg">
          <p className="text-xs font-bold text-orange-400 mb-2">Gradient Table:</p>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2 bg-gray-800 rounded"><span className="text-gray-400">∂L/∂w₁ = </span><span className="text-orange-300">{dL_dw1_0.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-800 rounded"><span className="text-gray-400">∂L/∂w₂ = </span><span className="text-orange-300">{dL_dw1_1.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-800 rounded"><span className="text-gray-400">∂L/∂w₃ = </span><span className="text-orange-300">{dL_dw2.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-800 rounded"><span className="text-gray-400">∂L/∂b₁ = </span><span className="text-orange-300">{dL_db1.toFixed(6)}</span></div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   RNN LESSON LABS
   ═══════════════════════════════════════════════════════════ */
function RNNCellLab() {
  const [step, setStep] = useState(0);
  const seq = [0.8, 0.3, -0.5, 0.9, -0.2];
  const hiddenSize = 2;
  const Wxh = [[0.4, 0.2], [-0.3, 0.5]];
  const Whh = [[0.1, -0.2], [0.3, 0.1]];
  const bh = [0, 0];
  const steps: { x: number; hPrev: number[]; z: number[]; hNew: number[] }[] = [];
  let h = [0, 0];
  for (let t = 0; t < seq.length; t++) {
    const x = seq[t];
    const xP = Wxh.map(row => row[0] * x);
    const hP = Whh.map((row, i) => row.reduce((s, w, j) => s + w * h[j], 0));
    const z = xP.map((v, i) => v + hP[i] + bh[i]);
    const hNew = z.map(Math.tanh);
    steps.push({ x, hPrev: [...h], z, hNew });
    h = hNew;
  }
  const cur = steps[step] || steps[0];

  return (
    <div className="space-y-4">
      <div className="flex gap-2">{seq.map((v, i) => <button key={i} onClick={() => setStep(i)} className={`px-3 py-2 rounded text-xs font-bold ${i === step ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}>t={i} (x={v})</button>)}</div>
      <div className="p-4 bg-gray-900 rounded-lg space-y-2 text-xs font-mono">
        <p className="text-green-400">x_{step} = {cur.x.toFixed(2)}</p>
        <p className="text-purple-400">h_{step > 0 ? step - 1 : 'init'} = [{cur.hPrev.map(v => v.toFixed(3)).join(', ')}]</p>
        <p className="text-yellow-400">z = W_xh·x + W_hh·h + b = [{cur.z.map(v => v.toFixed(3)).join(', ')}]</p>
        <p className="text-cyan-400 font-bold">h_{step} = tanh(z) = [{cur.hNew.map(v => v.toFixed(3)).join(', ')}]</p>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   VAE LESSON LABS
   ═══════════════════════════════════════════════════════════ */
function ReparamLab() {
  const [mu, setMu] = useState([0.5, -0.3]);
  const [logVar, setLogVar] = useState([-0.5, 0.2]);
  const [samples, setSamples] = useState<number[][]>([]);

  const sample = () => {
    const std = logVar.map(v => Math.exp(0.5 * v));
    const eps = std.map(() => Math.sqrt(-2 * Math.log(Math.random())) * Math.cos(2 * Math.PI * Math.random()));
    const z = mu.map((m, i) => m + std[i] * eps[i]);
    setSamples(prev => [...prev.slice(-20), z]);
    return { std, eps, z };
  };
  const [lastSample, setLastSample] = useState<any>(null);

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-wrap">
        <label className="text-xs text-gray-400">μ₁: <input type="range" min={-2} max={2} step={0.1} value={mu[0]} onChange={e => setMu([+e.target.value, mu[1]])} className="w-24" /> {mu[0].toFixed(1)}</label>
        <label className="text-xs text-gray-400">μ₂: <input type="range" min={-2} max={2} step={0.1} value={mu[1]} onChange={e => setMu([mu[0], +e.target.value])} className="w-24" /> {mu[1].toFixed(1)}</label>
      </div>
      <button onClick={() => setLastSample(sample())} className="px-4 py-2 rounded text-sm font-bold bg-purple-600 text-white">🎲 Sample z = μ + σ × ε</button>
      {lastSample && (
        <div className="p-3 bg-gray-900 rounded-lg text-xs font-mono space-y-1">
          <p className="text-purple-400">μ = [{mu.map(v => v.toFixed(2)).join(', ')}]</p>
          <p className="text-pink-400">σ = [{lastSample.std.map((v: number) => v.toFixed(3)).join(', ')}]</p>
          <p className="text-gray-400">ε = [{lastSample.eps.map((v: number) => v.toFixed(3)).join(', ')}]</p>
          <p className="text-green-400 font-bold">z = [{lastSample.z.map((v: number) => v.toFixed(3)).join(', ')}]</p>
        </div>
      )}
      {samples.length > 0 && (
        <div className="p-3 bg-gray-900 rounded-lg">
          <p className="text-xs text-gray-400 mb-2">{samples.length} samples plotted:</p>
          <div className="relative" style={{ width: 200, height: 200, background: '#0f172a', borderRadius: 8, border: '1px solid #334155' }}>
            {samples.map((z, i) => (
              <div key={i} className="absolute w-2 h-2 rounded-full bg-blue-500" style={{ left: `${(z[0] + 3) / 6 * 100}%`, top: `${(z[1] + 3) / 6 * 100}%`, opacity: 0.6 + i / samples.length * 0.4 }} />
            ))}
            <div className="absolute w-3 h-3 rounded-full bg-red-500 border border-white" style={{ left: `${(mu[0] + 3) / 6 * 100}%`, top: `${(mu[1] + 3) / 6 * 100}%` }} title="μ" />
          </div>
        </div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   TRANSFORMER LESSON LABS
   ═══════════════════════════════════════════════════════════ */
function AttentionLab() {
  const tokens = ['The', 'cat', 'sat', 'on', 'mat'];
  const n = tokens.length;
  // Simulate Q·K^T scores
  const scores: number[][] = tokens.map((_, i) => tokens.map((_, j) => Math.sin(i * 3.7 + j * 2.1) * 0.5 + (i === j ? 1 : 0)));
  const dk = 4;
  const scaled = scores.map(row => row.map(v => v / Math.sqrt(dk)));
  const attnWeights = scaled.map(row => { const mx = Math.max(...row); const e = row.map(v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); });
  const [selTok, setSelTok] = useState(1);

  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const size = 220; c.width = size; c.height = size;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, size, size);
    const off = 40, cs = (size - off) / n;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const v = attnWeights[i][j];
      ctx.fillStyle = `rgba(59,130,246,${v * 0.9 + 0.05})`;
      ctx.fillRect(off + j * cs, off + i * cs, cs - 1, cs - 1);
      ctx.fillStyle = v > 0.25 ? '#fff' : '#64748b'; ctx.font = 'bold 9px monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(v.toFixed(2), off + j * cs + cs / 2, off + i * cs + cs / 2);
    }
    ctx.fillStyle = '#94a3b8'; ctx.font = 'bold 9px sans-serif'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) ctx.fillText(tokens[i], off - 3, off + i * cs + cs / 2);
    ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
    for (let j = 0; j < n; j++) { ctx.save(); ctx.translate(off + j * cs + cs / 2, off - 3); ctx.rotate(-0.4); ctx.fillText(tokens[j], 0, 0); ctx.restore(); }
  }, [attnWeights, n]);

  return (
    <div className="space-y-4">
      <div className="flex gap-6 items-start flex-wrap">
        <canvas ref={ref} style={{ width: 220, height: 220, borderRadius: 8, border: '1px solid #334155' }} />
        <div>
          <p className="text-xs text-gray-400 mb-2">Click a token to see its attention:</p>
          <div className="flex gap-2 mb-3">{tokens.map((t, i) => <button key={i} onClick={() => setSelTok(i)} className={`px-2 py-1 rounded text-xs font-bold ${i === selTok ? 'bg-blue-600 text-white' : 'bg-gray-800 text-gray-400'}`}>{t}</button>)}</div>
          <p className="text-xs text-blue-400 font-mono mb-2">"{tokens[selTok]}" attends to:</p>
          {tokens.map((t, j) => (
            <div key={j} className="flex items-center gap-2 mb-1">
              <span className="text-xs text-gray-500 w-10 text-right font-mono">{t}</span>
              <div className="flex-1 h-4 bg-gray-800 rounded overflow-hidden" style={{ maxWidth: 150 }}>
                <div className="h-full bg-blue-500 rounded" style={{ width: `${attnWeights[selTok][j] * 100}%` }} />
              </div>
              <span className="text-xs text-gray-400 font-mono w-10">{(attnWeights[selTok][j] * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LESSON LAB MAPPING
   ═══════════════════════════════════════════════════════════ */
const LESSON_LABS: Record<string, { title: string; component: React.FC }> = {
  'cnn-1': { title: 'Image as Matrix', component: () => <div className="space-y-4"><p className="text-sm text-gray-300">Try changing pixel values to see how images are represented as numbers:</p><SmallGrid data={[[0,0,0,0,0,0,0,0],[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[0,1,0,1,0,1,0,0],[0,1,0,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,0]]} cellSize={32} label="Smiley face as 8×8 matrix" colorFn={(v: number) => v > 0.5 ? '#facc15' : '#1e293b'} /></div> },
  'cnn-2': { title: 'Convolution Explorer', component: ConvolutionLab },
  'cnn-3': { title: 'Filter Gallery', component: ConvolutionLab },
  'cnn-4': { title: 'Activation Explorer', component: ActivationLab },
  'cnn-5': { title: 'Pooling Explorer', component: PoolingLab },
  'cnn-6': { title: 'CNN Pipeline', component: ConvolutionLab },
  'ann-1': { title: 'Neuron Playground', component: ActivationLab },
  'ann-2': { title: 'Forward Pass Visualizer', component: BackpropLab },
  'ann-3': { title: 'Activation Explorer', component: ActivationLab },
  'ann-4': { title: 'Loss & Gradient Descent', component: BackpropLab },
  'ann-5': { title: 'Backpropagation Step-by-Step', component: BackpropLab },
  'ann-6': { title: 'Decision Boundary Lab', component: BackpropLab },
  'rnn-1': { title: 'Sequence Viewer', component: RNNCellLab },
  'rnn-2': { title: 'RNN Cell Step-by-Step', component: RNNCellLab },
  'rnn-3': { title: 'Gradient Decay Demo', component: RNNCellLab },
  'rnn-4': { title: 'LSTM Gate Explorer', component: RNNCellLab },
  'rnn-5': { title: 'Cell State Viewer', component: RNNCellLab },
  'rnn-6': { title: 'RNN Variants', component: RNNCellLab },
  'vae-1': { title: 'Autoencoder Demo', component: ReparamLab },
  'vae-2': { title: 'Latent Space Explorer', component: ReparamLab },
  'vae-3': { title: 'Distribution Sampler', component: ReparamLab },
  'vae-4': { title: 'Reparameterization Trick', component: ReparamLab },
  'vae-5': { title: 'Loss Calculator', component: ReparamLab },
  'vae-6': { title: 'Interpolation Lab', component: ReparamLab },
  'tf-1': { title: 'Attention vs RNN', component: AttentionLab },
  'tf-2': { title: 'Embedding Viewer', component: AttentionLab },
  'tf-3': { title: 'Positional Encoding', component: AttentionLab },
  'tf-4': { title: 'Self-Attention Lab', component: AttentionLab },
  'tf-5': { title: 'Multi-Head Attention', component: AttentionLab },
  'tf-6': { title: 'Transformer Block', component: AttentionLab },
};

/* ═══════════════════════════════════════════════════════════
   MAIN LESSON LAB PAGE
   ═══════════════════════════════════════════════════════════ */
export const LessonLab: React.FC = () => {
  const { topicId, lessonId } = useParams();
  const navigate = useNavigate();
  const lab = LESSON_LABS[lessonId || ''];

  if (!lab) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md">
          <div className="p-8 text-center">
            <h2 className="text-2xl font-bold text-white mb-4">Lab Not Found</h2>
            <p className="text-slate-300 mb-6">This lesson lab doesn't exist yet.</p>
            <div className="flex gap-3 justify-center">
              <Button onClick={() => navigate(`/topics/${topicId}/lessons`)}>← Back to Lessons</Button>
              <Button variant="secondary" onClick={() => navigate(`/topics/${topicId}/lab`)}>Full Lab</Button>
            </div>
          </div>
        </Card>
      </div>
    );
  }

  const LabComponent = lab.component;

  return (
    <div className="min-h-screen" style={{ background: 'linear-gradient(145deg, #020617, #0c1222, #020617)' }}>
      <div className="sticky top-0 z-40 backdrop-blur-xl" style={{ background: 'rgba(2,6,23,0.88)', borderBottom: '1px solid #1e293b' }}>
        <div className="max-w-4xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Button variant="ghost" onClick={() => navigate(`/topics/${topicId}/lessons`)}>
              <ArrowLeft className="w-4 h-4 mr-1" />
              Lessons
            </Button>
            <div>
              <h1 className="text-lg font-bold text-white">🧪 {lab.title}</h1>
              <p className="text-xs text-gray-500">Lesson Lab · {lessonId}</p>
            </div>
          </div>
          <Link to={`/topics/${topicId}/lab`}>
            <Button variant="secondary" size="sm">Full {topicId?.toUpperCase()} Lab →</Button>
          </Link>
        </div>
      </div>

      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="rounded-2xl p-6" style={{ background: 'rgba(15,23,42,0.6)', border: '1px solid #1e293b' }}>
          <LabComponent />
        </div>
        <div className="mt-6 flex justify-between">
          <Button variant="ghost" onClick={() => navigate(`/topics/${topicId}/lessons`)}>← Back to Lesson</Button>
          <Link to={`/topics/${topicId}/lab`}><Button>Open Full Lab →</Button></Link>
        </div>
      </div>
    </div>
  );
};
