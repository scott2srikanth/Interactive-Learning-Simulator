import React, { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { ArrowLeft } from 'lucide-react';
import MoleProblemLab from '../components/transformer/MoleProblemLab';

function MoleProblemLabWrapper() { return <MoleProblemLab />; }

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
        <select value={stride} onChange={e => { setStride(+e.target.value); setPos({ r: 0, c: 0 }); }} className="px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600">
          <option value={1}>Stride 1</option><option value={2}>Stride 2</option>
        </select>
        <button onClick={() => { setPos({ r: 0, c: 0 }); setPlaying(!playing); }} className={`px-3 py-1 rounded text-xs font-bold text-white ${playing ? 'bg-red-600' : 'bg-green-600'}`}>{playing ? '⏸ Pause' : '▶ Animate'}</button>
        <button onClick={() => setPos({ r: 0, c: 0 })} className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-700 dark:text-gray-300">↺</button>
      </div>
      <div className="flex gap-6 items-start flex-wrap">
        <SmallGrid data={input} cellSize={36} label="Input 6×6" highlight={{ r: iR, c: iC, h: k, w: k }} colorFn={(v: number) => v > 0.5 ? '#3b82f6' : '#1e293b'} />
        <div className="text-center">
          <SmallGrid data={kernel} cellSize={36} label="Kernel 3×3" colorFn={(v: number) => v > 0 ? `rgba(250,204,21,${Math.abs(v) / 8 + 0.15})` : `rgba(239,68,68,${Math.abs(v) / 8 + 0.15})`} />
        </div>
        <SmallGrid data={output} cellSize={36} label={`Output ${oH}×${oW}`} colorFn={(v: number) => { const n = (v + 10) / 20; return `rgb(${Math.round(n * 200 + 30)},${Math.round(n * 150 + 30)},${Math.round((1 - n) * 200 + 30)})`; }} />
      </div>
      <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg text-xs font-mono">
        <p className="text-green-400 mb-1">Position ({pos.r},{pos.c}):</p>
        <div className="flex gap-1 flex-wrap">{prods.map((p, i) => <span key={i} className="text-gray-700 dark:text-gray-300"><span className="text-blue-400">{p.iv.toFixed(1)}</span>×<span className="text-yellow-400">{p.kv.toFixed(1)}</span>=<b className={p.p >= 0 ? 'text-green-400' : 'text-red-400'}>{p.p.toFixed(1)}</b>{i < prods.length - 1 ? ' + ' : ''}</span>)}</div>
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
        <button onClick={() => setPoolType('max')} className={`px-3 py-1 rounded text-xs font-bold ${poolType === 'max' ? 'bg-orange-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>Max Pool</button>
        <button onClick={() => setPoolType('avg')} className={`px-3 py-1 rounded text-xs font-bold ${poolType === 'avg' ? 'bg-cyan-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>Avg Pool</button>
        <button onClick={() => { setPos({ r: 0, c: 0 }); setPlaying(!playing); }} className={`px-3 py-1 rounded text-xs font-bold text-white ${playing ? 'bg-red-600' : 'bg-green-600'}`}>{playing ? '⏸' : '▶ Animate'}</button>
      </div>
      <div className="flex gap-6 items-start flex-wrap">
        <SmallGrid data={input} cellSize={36} label="Input 6×6" highlight={{ r: iR, c: iC, h: poolSize, w: poolSize }} colorFn={(v: number) => `rgba(59,130,246,${v / 10 + 0.1})`} />
        <SmallGrid data={output} cellSize={44} label={`${poolType === 'max' ? 'Max' : 'Avg'} Pool ${oH}×${oW}`} colorFn={(v: number) => `rgba(34,197,94,${v / 10 + 0.15})`} />
      </div>
      <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg text-xs font-mono">
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
        {['relu', 'sigmoid', 'tanh'].map(f => <button key={f} onClick={() => setFn(f)} className={`px-3 py-1 rounded text-xs font-bold ${fn === f ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{f}</button>)}
      </div>
      <ActivationCanvas fn={fn} w={300} h={140} />
      <div className="grid grid-cols-4 gap-2">
        {input.map((v, i) => (
          <div key={i} className="p-2 bg-gray-100 dark:bg-gray-900 rounded text-center text-xs font-mono">
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
        <label className="text-xs text-gray-600 dark:text-gray-400">LR:
          <select value={lr} onChange={e => setLr(+e.target.value)} className="ml-1 px-2 py-1 rounded text-xs bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600">
            {[0.1, 0.3, 0.5, 1.0, 2.0].map(v => <option key={v} value={v}>{v}</option>)}
          </select>
        </label>
        <button onClick={() => setAuto(!auto)} className={`px-3 py-1 rounded text-xs font-bold text-white ${auto ? 'bg-red-600' : 'bg-green-600'}`}>{auto ? '⏸' : '▶ Auto Step'}</button>
        <button onClick={() => setStep(0)} className="px-2 py-1 rounded text-xs bg-gray-700 text-gray-700 dark:text-gray-300">↺</button>
        <button onClick={doUpdate} className="px-3 py-1 rounded text-xs font-bold bg-blue-600 text-gray-900 dark:text-white">Apply Update</button>
      </div>

      {/* Network diagram */}
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
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
        <p className="text-xs font-mono text-gray-700 dark:text-gray-300">{STEPS[step].d}</p>
        {STEPS[step].d2 && <p className="text-xs font-mono text-gray-400 mt-1">{STEPS[step].d2}</p>}
      </div>

      {/* Gradient table */}
      {step >= 4 && (
        <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg">
          <p className="text-xs font-bold text-orange-400 mb-2">Gradient Table:</p>
          <div className="grid grid-cols-2 gap-2 text-xs font-mono">
            <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded"><span className="text-gray-600 dark:text-gray-400">∂L/∂w₁ = </span><span className="text-orange-300">{dL_dw1_0.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded"><span className="text-gray-600 dark:text-gray-400">∂L/∂w₂ = </span><span className="text-orange-300">{dL_dw1_1.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded"><span className="text-gray-600 dark:text-gray-400">∂L/∂w₃ = </span><span className="text-orange-300">{dL_dw2.toFixed(6)}</span></div>
            <div className="p-2 bg-gray-100 dark:bg-gray-800 rounded"><span className="text-gray-600 dark:text-gray-400">∂L/∂b₁ = </span><span className="text-orange-300">{dL_db1.toFixed(6)}</span></div>
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
      <div className="flex gap-2">{seq.map((v, i) => <button key={i} onClick={() => setStep(i)} className={`px-3 py-2 rounded text-xs font-bold ${i === step ? 'bg-blue-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>t={i} (x={v})</button>)}</div>
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg space-y-2 text-xs font-mono">
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
        <label className="text-xs text-gray-600 dark:text-gray-400">μ₁: <input type="range" min={-2} max={2} step={0.1} value={mu[0]} onChange={e => setMu([+e.target.value, mu[1]])} className="w-24" /> {mu[0].toFixed(1)}</label>
        <label className="text-xs text-gray-600 dark:text-gray-400">μ₂: <input type="range" min={-2} max={2} step={0.1} value={mu[1]} onChange={e => setMu([mu[0], +e.target.value])} className="w-24" /> {mu[1].toFixed(1)}</label>
      </div>
      <button onClick={() => setLastSample(sample())} className="px-4 py-2 rounded text-sm font-bold bg-purple-600 text-gray-900 dark:text-white">🎲 Sample z = μ + σ × ε</button>
      {lastSample && (
        <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg text-xs font-mono space-y-1">
          <p className="text-purple-400">μ = [{mu.map(v => v.toFixed(2)).join(', ')}]</p>
          <p className="text-pink-400">σ = [{lastSample.std.map((v: number) => v.toFixed(3)).join(', ')}]</p>
          <p className="text-gray-600 dark:text-gray-400">ε = [{lastSample.eps.map((v: number) => v.toFixed(3)).join(', ')}]</p>
          <p className="text-green-400 font-bold">z = [{lastSample.z.map((v: number) => v.toFixed(3)).join(', ')}]</p>
        </div>
      )}
      {samples.length > 0 && (
        <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">{samples.length} samples plotted:</p>
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
   TRANSFORMER LESSON LABS — 6 distinct labs
   ═══════════════════════════════════════════════════════════ */

// tf-1: RNN vs Transformer comparison
function TFComparisonLab() {
  const tokens = ['Attention', 'is', 'all', 'you', 'need'];
  const [step, setStep] = useState(0);
  const [mode, setMode] = useState<'rnn'|'transformer'>('rnn');
  useEffect(() => { const t = setInterval(() => setStep(p => (p + 1) % tokens.length), 800); return () => clearInterval(t); }, []);
  return (
    <div className="space-y-4">
      <div className="flex gap-2 mb-4">{['rnn','transformer'].map(m => <button key={m} onClick={() => setMode(m as any)} className={`px-4 py-2 rounded text-xs font-bold ${mode === m ? 'bg-yellow-600 text-white' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{m === 'rnn' ? '🔄 RNN (Sequential)' : '⚡ Transformer (Parallel)'}</button>)}</div>
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
        <p className="text-xs text-gray-600 dark:text-gray-400 mb-3">{mode === 'rnn' ? 'RNN processes tokens one-by-one, left to right:' : 'Transformer processes ALL tokens simultaneously:'}</p>
        <div className="flex gap-3 items-center flex-wrap">
          {tokens.map((t, i) => {
            const active = mode === 'rnn' ? i <= step : true;
            const current = mode === 'rnn' ? i === step : true;
            return (
              <div key={i} className="text-center">
                <div className={`px-3 py-2 rounded-lg text-sm font-bold transition-all duration-300 ${active ? (current ? 'bg-yellow-500 text-black scale-110' : 'bg-blue-600 text-white') : 'bg-gray-800 text-gray-600'}`} style={{ transform: current && mode === 'rnn' ? 'scale(1.1)' : 'scale(1)' }}>{t}</div>
                <p className="text-xs mt-1" style={{ color: active ? '#22c55e' : '#475569' }}>{active ? '✓' : '⏳'}</p>
              </div>
            );
          })}
        </div>
        {mode === 'rnn' && <div className="mt-3 flex items-center gap-2"><span className="text-xs text-orange-400">Hidden state h →</span><div className="h-2 bg-orange-500 rounded" style={{ width: `${((step + 1) / tokens.length) * 100}%`, maxWidth: 200, transition: 'width 0.5s' }} /><span className="text-xs text-gray-500">carries ALL info</span></div>}
        {mode === 'transformer' && <div className="mt-3"><p className="text-xs text-green-400">Every token sees every other token directly via attention — no bottleneck!</p><div className="flex gap-1 mt-2">{tokens.map((_, i) => <div key={i} className="flex gap-0.5">{tokens.map((_, j) => <div key={j} className="w-3 h-3 rounded-sm" style={{ background: `rgba(59,130,246,${0.2 + Math.random() * 0.6})` }} />)}</div>)}</div></div>}
      </div>
      <div className="p-3 rounded-lg" style={{ background: mode === 'rnn' ? '#7f1d1d22' : '#052e1622', border: `1px solid ${mode === 'rnn' ? '#dc262633' : '#16a34a33'}` }}>
        <p className="text-xs text-gray-700 dark:text-gray-300">{mode === 'rnn' ? '❌ Sequential → slow, can\'t parallelize. Token 5 must wait for tokens 1-4. Long-range dependencies lost through bottleneck.' : '✅ Parallel → fast on GPUs. Every token directly attends to every other. No information bottleneck. O(n²) attention but highly parallelizable.'}</p>
      </div>
    </div>
  );
}

// tf-2: Embedding Viewer
function EmbeddingLab() {
  const [inputText, setInputText] = useState('king queen man woman');
  const tokens = inputText.toLowerCase().split(/\s+/).filter(t => t);
  const dModel = 6;
  // Deterministic embeddings based on char codes
  const embed = (t: string) => Array(dModel).fill(0).map((_, d) => Math.sin(t.charCodeAt(d % t.length) * 0.37 + d * 1.7) * 0.8);
  const embeddings = tokens.map(embed);
  // Cosine similarity
  const cosine = (a: number[], b: number[]) => { const dot = a.reduce((s, v, i) => s + v * b[i], 0); const ma = Math.sqrt(a.reduce((s, v) => s + v * v, 0)); const mb = Math.sqrt(b.reduce((s, v) => s + v * v, 0)); return dot / (ma * mb + 1e-8); };
  const simMatrix = tokens.map((_, i) => tokens.map((_, j) => cosine(embeddings[i], embeddings[j])));

  return (
    <div className="space-y-4">
      <div><label className="text-xs text-gray-600 dark:text-gray-400">Enter tokens (space-separated):</label><input value={inputText} onChange={e => setInputText(e.target.value)} className="w-full mt-1 px-3 py-2 rounded-lg text-sm bg-gray-200 dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600" /></div>
      <p className="text-xs text-gray-600 dark:text-gray-400">Each token → {dModel}-dim embedding vector:</p>
      <div className="space-y-2">{tokens.slice(0, 6).map((t, i) => (
        <div key={i} className="flex items-center gap-3">
          <span className="text-xs font-bold text-green-400 w-16 text-right font-mono">"{t}"</span>
          <span className="text-gray-600">→</span>
          <div className="flex gap-1">{embeddings[i].map((v, j) => <div key={j} className="w-10 h-6 rounded text-center text-xs font-mono font-bold flex items-center justify-center" style={{ background: `rgba(59,130,246,${Math.abs(v) * 0.7 + 0.1})`, color: '#fff' }}>{v.toFixed(2)}</div>)}</div>
        </div>
      ))}</div>
      {tokens.length >= 2 && (
        <div>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-4 mb-2">Cosine Similarity Matrix (similar tokens → higher value):</p>
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `60px repeat(${tokens.length}, 48px)` }}>
            <div />
            {tokens.map((t, j) => <div key={j} className="text-center text-xs font-bold text-blue-400 truncate">{t.slice(0, 5)}</div>)}
            {tokens.map((t, i) => (<React.Fragment key={i}><div className="text-xs font-bold text-blue-400 text-right pr-1 self-center">{t.slice(0, 5)}</div>{simMatrix[i].map((v, j) => <div key={j} className="h-8 rounded flex items-center justify-center text-xs font-mono font-bold" style={{ background: `rgba(${v > 0.7 ? '34,197,94' : v > 0.3 ? '250,204,21' : '239,68,68'},${Math.abs(v) * 0.6 + 0.15})`, color: '#fff' }}>{v.toFixed(2)}</div>)}</React.Fragment>))}
          </div>
        </div>
      )}
    </div>
  );
}

// tf-3: Positional Encoding visualizer
function PELab() {
  const [dModel, setDModel] = useState(8);
  const [seqLen, setSeqLen] = useState(10);
  const ref = useRef<HTMLCanvasElement>(null);
  const pe: number[][] = [];
  for (let pos = 0; pos < seqLen; pos++) { const row: number[] = []; for (let i = 0; i < dModel; i++) { row.push(i % 2 === 0 ? Math.sin(pos / Math.pow(10000, i / dModel)) : Math.cos(pos / Math.pow(10000, (i - 1) / dModel))); } pe.push(row); }

  useEffect(() => {
    const c = ref.current; if (!c) return;
    const W = 320, H = 180; c.width = W; c.height = H;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#0f172a'; ctx.fillRect(0, 0, W, H);
    const cw = (W - 40) / dModel, ch = (H - 30) / seqLen;
    for (let i = 0; i < seqLen; i++) for (let j = 0; j < dModel; j++) {
      const v = (pe[i][j] + 1) / 2;
      ctx.fillStyle = `hsl(${v * 240}, 80%, ${25 + v * 45}%)`;
      ctx.fillRect(35 + j * cw, 20 + i * ch, cw - 0.5, ch - 0.5);
    }
    ctx.fillStyle = '#94a3b8'; ctx.font = '8px monospace'; ctx.textAlign = 'right';
    for (let i = 0; i < seqLen; i++) ctx.fillText(`pos ${i}`, 32, 20 + i * ch + ch / 2 + 3);
    ctx.textAlign = 'center';
    for (let j = 0; j < dModel; j++) ctx.fillText(`d${j}`, 35 + j * cw + cw / 2, 14);
    ctx.fillText('dim →', W / 2, H - 4); ctx.save(); ctx.translate(8, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('position →', 0, 0); ctx.restore();
  }, [dModel, seqLen, pe]);

  return (
    <div className="space-y-4">
      <div className="flex gap-4 flex-wrap">
        <label className="text-xs text-gray-600 dark:text-gray-400">Dimensions: <b className="text-gray-900 dark:text-white">{dModel}</b><input type="range" min={4} max={16} step={2} value={dModel} onChange={e => setDModel(+e.target.value)} className="ml-2 w-24" /></label>
        <label className="text-xs text-gray-600 dark:text-gray-400">Seq Length: <b className="text-gray-900 dark:text-white">{seqLen}</b><input type="range" min={4} max={20} value={seqLen} onChange={e => setSeqLen(+e.target.value)} className="ml-2 w-24" /></label>
      </div>
      <canvas ref={ref} style={{ width: 320, height: 180, borderRadius: 8, border: '1px solid #334155' }} />
      <p className="text-xs text-gray-600 dark:text-gray-400">Each row is a unique positional signature. Sine waves have different frequencies per dimension — the model can learn to compute relative positions from these patterns.</p>
      <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg text-xs font-mono">
        <p className="text-cyan-400">PE(pos, 2i) = sin(pos / 10000^(2i/{dModel}))</p>
        <p className="text-orange-400">PE(pos, 2i+1) = cos(pos / 10000^(2i/{dModel}))</p>
      </div>
      <p className="text-xs text-gray-600 dark:text-gray-400">Sample — Position 0: [{pe[0]?.map(v => v.toFixed(2)).join(', ')}]</p>
      <p className="text-xs text-gray-600 dark:text-gray-400">Position {seqLen - 1}: [{pe[seqLen - 1]?.map(v => v.toFixed(2)).join(', ')}]</p>
    </div>
  );
}

// tf-4: Self-Attention Q,K,V Lab (the existing one, enhanced)
function SelfAttentionLab() {
  const [sentence, setSentence] = useState('The cat sat on mat');
  const tokens = sentence.toLowerCase().split(/\s+/).filter(t => t);
  const n = tokens.length;
  const dk = 4;
  const dModel = 6;

  // Generate deterministic embeddings, Q, K, V
  const embeddings = tokens.map((t, i) => Array(dModel).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.3 + d * 1.5 + i * 0.7) * 0.7).toFixed(2)));
  const queries = embeddings.map(e => e.slice(0, dk).map((v, d) => +(v * 0.8 + Math.cos(d) * 0.2).toFixed(2)));
  const keys = embeddings.map(e => e.slice(0, dk).map((v, d) => +(v * 0.6 - Math.sin(d) * 0.3).toFixed(2)));
  const values = embeddings.map(e => e.slice(0, dk).map((v, d) => +(v * 0.5 + Math.sin(d * 2) * 0.4).toFixed(2)));

  const scores: number[][] = queries.map(q => keys.map(k => q.reduce((s, v, i) => s + v * (k[i] || 0), 0)));
  const scaled = scores.map(row => row.map(v => v / Math.sqrt(dk)));
  const softmaxRow = (row: number[]) => { if (!row.length) return []; const mx = Math.max(...row); const e = row.map(v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / (s || 1)); };
  const attnWeights = scaled.map(softmaxRow);

  const [selTok, setSelTok] = useState(0);
  const safeSelTok = Math.min(selTok, n - 1);
  const [step, setStep] = useState(0); // 0=embed, 1=QKV, 2=scores, 3=softmax

  return (
    <div className="space-y-4">
      <input value={sentence} onChange={e => { setSentence(e.target.value); setSelTok(0); }} className="w-full px-3 py-2 rounded-lg text-sm bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600" placeholder="Enter a sentence..." />

      {/* Step selector */}
      <div className="flex gap-1 flex-wrap">{['1. Embeddings', '2. Q, K, V', '3. Scores Q·Kᵀ', '4. Softmax'].map((label, i) => (
        <button key={i} onClick={() => setStep(i)} className={`px-3 py-1.5 rounded text-xs font-bold ${step === i ? 'bg-blue-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{label}</button>
      ))}</div>

      {n === 0 && <p className="text-sm text-gray-500">Type a sentence above to begin.</p>}

      {n > 0 && step === 0 && (
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-2">
          <p className="text-xs font-bold text-green-500 mb-2">Token Embeddings [{n} × {dModel}]</p>
          {tokens.map((t, i) => (
            <div key={i} className="flex items-center gap-2">
              <span className="text-xs font-mono w-12 text-right text-green-600 font-bold">{t}</span>
              <div className="flex gap-1">{embeddings[i].map((v, d) => <div key={d} className="w-10 h-5 rounded text-center flex items-center justify-center" style={{ fontSize: 7, fontFamily: 'monospace', fontWeight: 700, color: '#fff', background: `rgba(34,197,94,${Math.abs(v) * 0.8 + 0.1})` }}>{v}</div>)}</div>
            </div>
          ))}
        </div>
      )}

      {n > 0 && step === 1 && (
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-3">
          <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">Each embedding is projected into <b className="text-gray-900 dark:text-white">Query</b>, <b className="text-gray-900 dark:text-white">Key</b>, <b className="text-gray-900 dark:text-white">Value</b> via learned weight matrices:</p>
          {tokens.map((t, i) => (
            <div key={i} className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-mono w-10 text-right text-gray-500 font-bold">{t}</span>
              <span className="text-xs text-gray-400">→</span>
              <div className="flex gap-0.5">{queries[i].map((v, d) => <div key={d} className="w-8 h-5 rounded text-center flex items-center justify-center" style={{ fontSize: 6, fontFamily: 'monospace', fontWeight: 700, color: '#fff', background: `rgba(239,68,68,${Math.abs(v) + 0.15})` }}>{v}</div>)}</div>
              <div className="flex gap-0.5">{keys[i].map((v, d) => <div key={d} className="w-8 h-5 rounded text-center flex items-center justify-center" style={{ fontSize: 6, fontFamily: 'monospace', fontWeight: 700, color: '#fff', background: `rgba(34,197,94,${Math.abs(v) + 0.15})` }}>{v}</div>)}</div>
              <div className="flex gap-0.5">{values[i].map((v, d) => <div key={d} className="w-8 h-5 rounded text-center flex items-center justify-center" style={{ fontSize: 6, fontFamily: 'monospace', fontWeight: 700, color: '#fff', background: `rgba(59,130,246,${Math.abs(v) + 0.15})` }}>{v}</div>)}</div>
            </div>
          ))}
          <div className="flex gap-4 mt-2">{[{l:'Q (Query)',c:'#ef4444'},{l:'K (Key)',c:'#22c55e'},{l:'V (Value)',c:'#3b82f6'}].map(x => <span key={x.l} className="text-xs font-bold" style={{color:x.c}}>{x.l}</span>)}</div>
        </div>
      )}

      {n > 0 && step === 2 && (
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg">
          <p className="text-xs font-bold text-purple-500 mb-2">Scores = Q · Kᵀ / √{dk} [{n}×{n}]</p>
          <div className="overflow-x-auto">
            <div className="inline-grid gap-1" style={{ gridTemplateColumns: `50px repeat(${n}, 44px)` }}>
              <div />{tokens.map((t, j) => <div key={j} className="text-center text-xs font-bold text-blue-500 truncate">{t}</div>)}
              {tokens.map((t, i) => (<React.Fragment key={i}><div className="text-xs font-bold text-blue-500 text-right pr-1 self-center">{t}</div>{scaled[i]?.map((v, j) => <div key={j} className="h-8 rounded flex items-center justify-center text-xs font-mono font-bold" style={{ background: `rgba(168,85,247,${Math.max(0,(v+1)/3)*0.7+0.05})`, color: '#fff' }}>{v.toFixed(2)}</div>)}</React.Fragment>))}
            </div>
          </div>
        </div>
      )}

      {n > 0 && step === 3 && (
        <div className="p-4 bg-gray-50 dark:bg-gray-900 rounded-lg space-y-3">
          <p className="text-xs font-bold text-purple-500 mb-2">Attention Weights (softmax per row)</p>
          <div className="flex gap-2 mb-2 flex-wrap">{tokens.map((t, i) => <button key={i} onClick={() => setSelTok(i)} className={`px-2 py-1 rounded text-xs font-bold ${i === safeSelTok ? 'bg-purple-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{t}</button>)}</div>
          <p className="text-xs text-gray-600 dark:text-gray-400">"{tokens[safeSelTok]}" attends to:</p>
          {tokens.map((t, j) => (
            <div key={j} className="flex items-center gap-2 mb-1">
              <span className="text-xs w-12 text-right font-mono text-gray-500">{t}</span>
              <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-800 rounded overflow-hidden" style={{ maxWidth: 180 }}>
                <div className="h-full bg-purple-500 rounded" style={{ width: `${(attnWeights[safeSelTok]?.[j] || 0) * 100}%` }} />
              </div>
              <span className="text-xs w-10 font-mono text-gray-600 dark:text-gray-400">{((attnWeights[safeSelTok]?.[j] || 0) * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// tf-5: Multi-Head Attention
function MultiHeadLab() {
  const tokens = ['I', 'love', 'deep', 'learning'];
  const n = tokens.length;
  const numHeads = 4;
  const headNames = ['Syntax', 'Semantics', 'Position', 'Context'];
  // Generate different patterns for each head
  const heads = Array.from({ length: numHeads }).map((_, h) =>
    tokens.map((_, i) => { const row = tokens.map((_, j) => Math.exp(Math.sin(i * (h + 1) * 2.3 + j * (h + 1) * 1.7) + (i === j ? 0.5 : 0))); const s = row.reduce((a, b) => a + b, 0); return row.map(v => v / s); })
  );
  const [selHead, setSelHead] = useState(0);

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-600 dark:text-gray-400">Each head learns a <b className="text-gray-900 dark:text-white">different type of relationship</b> between tokens:</p>
      <div className="flex gap-2 flex-wrap">{headNames.map((name, h) => <button key={h} onClick={() => setSelHead(h)} className={`px-3 py-1.5 rounded text-xs font-bold ${selHead === h ? 'bg-yellow-600 text-black' : 'bg-gray-200 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>Head {h + 1}: {name}</button>)}</div>
      <div className="flex gap-6 items-start flex-wrap">
        {/* Selected head heatmap */}
        <div>
          <p className="text-xs font-bold text-yellow-400 mb-2">Head {selHead + 1}: {headNames[selHead]}</p>
          <div className="inline-grid gap-1" style={{ gridTemplateColumns: `50px repeat(${n}, 44px)` }}>
            <div />{tokens.map((t, j) => <div key={j} className="text-center text-xs font-bold text-blue-400">{t}</div>)}
            {tokens.map((t, i) => (<React.Fragment key={i}><div className="text-xs font-bold text-blue-400 text-right pr-1 self-center">{t}</div>{heads[selHead][i].map((v, j) => <div key={j} className="h-8 rounded flex items-center justify-center text-xs font-mono font-bold" style={{ background: `rgba(59,130,246,${v * 0.85 + 0.05})`, color: v > 0.2 ? '#fff' : '#64748b' }}>{v.toFixed(2)}</div>)}</React.Fragment>))}
          </div>
        </div>
        {/* All heads side-by-side mini */}
        <div>
          <p className="text-xs font-bold text-gray-600 dark:text-gray-400 mb-2">All {numHeads} heads compared:</p>
          <div className="flex gap-3">
            {heads.map((head, h) => (
              <div key={h} className="text-center cursor-pointer" onClick={() => setSelHead(h)} style={{ opacity: h === selHead ? 1 : 0.5 }}>
                <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `repeat(${n}, 16px)` }}>
                  {head.flat().map((v, i) => <div key={i} className="w-4 h-4 rounded-sm" style={{ background: `rgba(59,130,246,${v * 0.9 + 0.05})` }} />)}
                </div>
                <p className="text-xs mt-1" style={{ color: h === selHead ? '#facc15' : '#475569' }}>{headNames[h]}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg text-xs font-mono text-gray-600 dark:text-gray-400">
        MultiHead = Concat(Head₁, Head₂, Head₃, Head₄) × W_O → <span className="text-green-400">[{n}×{n * 2}] → [{n}×{n * 2}]</span>
      </div>
    </div>
  );
}

// tf-6: Full Transformer Block
function TransformerBlockLab() {
  const tokens = ['The', 'cat', 'sat'];
  const dModel = 4;
  const [curStage, setCurStage] = useState(0);
  const stages = ['Input+PE', 'Multi-Head Attn', 'Add & Norm', 'Feed-Forward', 'Add & Norm', 'Output'];
  // Simulate values at each stage
  const vals = tokens.map((_, i) => Array(dModel).fill(0).map((_, d) => Math.sin(i * 2.3 + d * 1.7) * 0.5));
  const stageVals = stages.map((_, s) => tokens.map((_, i) => Array(dModel).fill(0).map((_, d) => Math.sin(i * 2.3 + d * 1.7 + s * 0.8) * (0.5 + s * 0.1))));
  const stageColors = ['#22c55e', '#f59e0b', '#64748b', '#ec4899', '#64748b', '#22c55e'];

  return (
    <div className="space-y-4">
      <p className="text-xs text-gray-600 dark:text-gray-400">Click each stage to see the tensor values flowing through:</p>
      <div className="flex gap-1 items-center flex-wrap">
        {stages.map((s, i) => (
          <React.Fragment key={i}>
            {i > 0 && <span className="text-gray-600 text-xs">→</span>}
            <button onClick={() => setCurStage(i)} className="px-2 py-1.5 rounded text-xs font-bold transition-all" style={{ background: curStage === i ? stageColors[i] : '#1e293b', color: curStage === i ? (i === 2 || i === 4 ? '#fff' : '#000') : '#64748b', border: `1px solid ${curStage === i ? stageColors[i] : '#334155'}` }}>{s}</button>
          </React.Fragment>
        ))}
      </div>
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
        <p className="text-xs font-bold mb-3" style={{ color: stageColors[curStage] }}>Stage: {stages[curStage]} — tensor [{tokens.length}×{dModel}]</p>
        {tokens.map((t, i) => (
          <div key={i} className="flex items-center gap-3 mb-2">
            <span className="text-xs font-bold text-green-400 w-10 text-right">"{t}"</span>
            <div className="flex gap-1">
              {stageVals[curStage][i].map((v, d) => (
                <div key={d} className="w-12 h-7 rounded flex items-center justify-center text-xs font-mono font-bold" style={{ background: `${stageColors[curStage]}${Math.round(Math.abs(v) * 150 + 40).toString(16).padStart(2, '0')}`, color: '#fff' }}>{v.toFixed(2)}</div>
              ))}
            </div>
          </div>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-gray-900 text-xs text-gray-600 dark:text-gray-400">
        {curStage === 0 && 'Embeddings + Positional Encoding → position-aware token representations.'}
        {curStage === 1 && 'Multi-Head Self-Attention: every token attends to every other token via Q·Kᵀ/√dₖ softmax.'}
        {curStage === 2 && 'Residual connection (x + attention(x)) followed by Layer Normalization.'}
        {curStage === 3 && 'Feed-Forward Network: two linear layers with ReLU. Applied per-position independently.'}
        {curStage === 4 && 'Second residual connection (x + FFN(x)) followed by Layer Normalization.'}
        {curStage === 5 && 'Final output: each token now carries context from ALL other tokens. Ready for next block or output.'}
      </div>
    </div>
  );
}

// tf-1: Context changes meaning — interactive "mole" demo
function ContextMeaningLab() {
  const examples = [
    { sentence: 'American shrew mole', words: ['American', 'shrew', 'mole'], focus: 2, meaning: 'Small burrowing mammal', emoji: '🐾', color: '#22c55e' },
    { sentence: 'One mole of carbon dioxide', words: ['One', 'mole', 'of', 'carbon', 'dioxide'], focus: 1, meaning: '6.022 × 10²³ particles (Avogadro)', emoji: '⚗️', color: '#3b82f6' },
    { sentence: 'Take a biopsy of the mole', words: ['Take', 'a', 'biopsy', 'of', 'the', 'mole'], focus: 5, meaning: 'Skin growth / lesion', emoji: '🏥', color: '#f59e0b' },
  ];
  const [sel, setSel] = useState(0);
  const [showResult, setShowResult] = useState(false);
  useEffect(() => { setShowResult(false); const t = setTimeout(() => setShowResult(true), 600); return () => clearTimeout(t); }, [sel]);

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-700 dark:text-gray-300">The word <b>"mole"</b> has the <b>same initial embedding</b> in all 3 sentences. Click each to see how attention updates its meaning:</p>
      <div className="flex gap-2 flex-wrap">{examples.map((ex, i) => (
        <button key={i} onClick={() => setSel(i)} className={`px-3 py-2 rounded-lg text-xs font-bold transition-all ${sel === i ? 'text-white scale-105' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`} style={sel === i ? { background: ex.color } : {}}>
          {ex.emoji} {ex.sentence}
        </button>
      ))}</div>
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg">
        <div className="flex gap-2 flex-wrap mb-4">{examples[sel].words.map((w, i) => (
          <div key={i} className={`px-3 py-2 rounded-lg text-sm font-bold transition-all ${i === examples[sel].focus ? 'text-white scale-110' : 'bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300'}`}
            style={i === examples[sel].focus ? { background: examples[sel].color, boxShadow: `0 0 15px ${examples[sel].color}66` } : {}}>
            {w}
          </div>
        ))}</div>
        <div className="flex items-center gap-3">
          <div className="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded">
            <div className="h-1 rounded transition-all duration-700" style={{ width: showResult ? '100%' : '0%', background: examples[sel].color }} />
          </div>
        </div>
        {showResult && (
          <div className="mt-3 p-3 rounded-lg" style={{ background: `${examples[sel].color}15`, border: `1px solid ${examples[sel].color}33` }}>
            <p className="text-xs text-gray-600 dark:text-gray-400">After attention processes context:</p>
            <p className="text-sm font-bold" style={{ color: examples[sel].color }}>
              "{examples[sel].words[examples[sel].focus]}" → {examples[sel].meaning}
            </p>
          </div>
        )}
      </div>
      <div className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
        <p className="text-xs text-gray-700 dark:text-gray-300"><b>Key insight:</b> The initial embedding of "mole" is identical in all 3 sentences. Attention reads the surrounding tokens and <b>adds context information</b> to update the embedding so the model knows which meaning is intended.</p>
      </div>
    </div>
  );
}

// tf-5: Attention Pattern — dot products, scaling, softmax
function AttentionPatternLab() {
  const [sentence, setSentence] = useState('A fluffy blue creature roamed');
  const tokens = sentence.toLowerCase().split(/\s+/).filter(t => t);
  const n = tokens.length;
  const dk = 4;
  // Generate Q·K scores — adjectives should attend strongly to nouns
  const scores: number[][] = tokens.map((ti, i) => tokens.map((tj, j) => {
    const isAdj = ['fluffy', 'blue'].includes(ti);
    const isNoun = ['creature'].includes(tj);
    if (isAdj && isNoun) return 2.0 + Math.sin(i + j) * 0.3;
    if (i === j) return 1.0;
    return -0.5 + Math.sin(i * 2.7 + j * 1.3) * 0.4;
  }));
  const scaled = scores.map(row => row.map(v => v / Math.sqrt(dk)));
  const softmaxRow = (row: number[]) => { const mx = Math.max(...row); const e = row.map(v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0); return e.map(v => v / s); };
  const attn = scaled.map(softmaxRow);
  const [selRow, setSelRow] = useState(0);
  const [step, setStep] = useState<'raw'|'scaled'|'softmax'>('raw');

  const curData = step === 'raw' ? scores : step === 'scaled' ? scaled : attn;

  return (
    <div className="space-y-4">
      <div><label className="text-xs text-gray-600 dark:text-gray-400">Sentence:</label><input value={sentence} onChange={e => setSentence(e.target.value)} className="w-full mt-1 px-3 py-2 rounded-lg text-sm bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white border border-gray-300 dark:border-gray-600" /></div>
      <div className="flex gap-2">{(['raw', 'scaled', 'softmax'] as const).map(s => (
        <button key={s} onClick={() => setStep(s)} className={`px-3 py-1.5 rounded text-xs font-bold ${step === s ? 'bg-purple-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>
          {s === 'raw' ? '1. Q·Kᵀ (raw)' : s === 'scaled' ? '2. ÷ √d_k (scaled)' : '3. Softmax (%)'}
        </button>
      ))}</div>
      {/* Grid */}
      <div className="overflow-x-auto">
        <div className="inline-grid gap-1" style={{ gridTemplateColumns: `50px repeat(${n}, 48px)` }}>
          <div />
          {tokens.map((t, j) => <div key={j} className="text-center text-xs font-bold text-blue-500 truncate">{t}</div>)}
          {tokens.map((t, i) => (<React.Fragment key={i}>
            <div className="text-xs font-bold text-blue-500 text-right pr-1 self-center cursor-pointer" onClick={() => setSelRow(i)} style={{ textDecoration: selRow === i ? 'underline' : 'none' }}>{t}</div>
            {curData[i]?.map((v, j) => (
              <div key={j} className="h-9 rounded flex items-center justify-center text-xs font-mono font-bold transition-all" style={{
                background: step === 'softmax' ? `rgba(168,85,247,${v * 0.85 + 0.05})` : `rgba(59,130,246,${Math.max(0, (v + 2) / 5) * 0.7 + 0.05})`,
                color: (step === 'softmax' ? v > 0.15 : (v + 2) / 5 > 0.3) ? '#fff' : '#94a3b8',
                border: selRow === i ? '2px solid #facc15' : '1px solid transparent'
              }}>{step === 'softmax' ? `${(v * 100).toFixed(0)}%` : v.toFixed(2)}</div>
            ))}
          </React.Fragment>))}
        </div>
      </div>
      {/* Selected row bar chart */}
      <div className="p-3 bg-gray-100 dark:bg-gray-900 rounded-lg">
        <p className="text-xs text-gray-600 dark:text-gray-400 mb-2">"{tokens[selRow]}" attends to:</p>
        {tokens.map((t, j) => (
          <div key={j} className="flex items-center gap-2 mb-1">
            <span className="text-xs w-16 text-right font-mono text-gray-500">{t}</span>
            <div className="flex-1 h-4 bg-gray-200 dark:bg-gray-800 rounded overflow-hidden" style={{ maxWidth: 200 }}>
              <div className="h-full rounded transition-all duration-500" style={{ width: `${(step === 'softmax' ? attn[selRow][j] : Math.max(0, (curData[selRow]?.[j] || 0) + 1) / 4) * 100}%`, background: '#a855f7' }} />
            </div>
            <span className="text-xs w-12 font-mono text-gray-600 dark:text-gray-400">{step === 'softmax' ? `${(attn[selRow][j] * 100).toFixed(0)}%` : curData[selRow]?.[j]?.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// tf-6: Value weighted sum — shows how output embedding is computed
function ValueOutputLab() {
  const tokens = ['A', 'fluffy', 'blue', 'creature', 'roamed'];
  const [targetIdx, setTargetIdx] = useState(3);
  // Simulated attention weights for target token
  const attnWeights: Record<number, number[]> = {
    0: [0.6, 0.1, 0.1, 0.1, 0.1],
    1: [0.05, 0.5, 0.3, 0.1, 0.05],
    2: [0.05, 0.3, 0.5, 0.1, 0.05],
    3: [0.07, 0.42, 0.38, 0.08, 0.05],
    4: [0.1, 0.05, 0.05, 0.3, 0.5],
  };
  const values = tokens.map((t, i) => Array(6).fill(0).map((_, d) => +(Math.sin(t.charCodeAt(0) * 0.2 + d * 1.3 + i) * 0.6).toFixed(2)));
  const weights = attnWeights[targetIdx] || attnWeights[3];
  // Compute weighted sum
  const output = Array(6).fill(0).map((_, d) => tokens.reduce((s, _, i) => s + weights[i] * values[i][d], 0));

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-700 dark:text-gray-300">Select a token to see how its output is computed as a <b>weighted sum</b> of all Value vectors:</p>
      <div className="flex gap-2 flex-wrap">{tokens.map((t, i) => (
        <button key={i} onClick={() => setTargetIdx(i)} className={`px-3 py-2 rounded-lg text-xs font-bold ${i === targetIdx ? 'bg-purple-600 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400'}`}>{t}</button>
      ))}</div>
      <div className="p-4 bg-gray-100 dark:bg-gray-900 rounded-lg space-y-3">
        <p className="text-xs font-bold text-purple-500 mb-2">Output("{tokens[targetIdx]}") = Σ attention_weight × Value</p>
        {tokens.map((t, i) => (
          <div key={i} className="flex items-center gap-2 flex-wrap">
            <span className="text-xs font-mono w-8 font-bold" style={{ color: weights[i] > 0.2 ? '#a855f7' : '#64748b' }}>{(weights[i] * 100).toFixed(0)}%</span>
            <span className="text-xs text-gray-500">×</span>
            <span className="text-xs font-mono text-pink-500">V("{t}")</span>
            <span className="text-xs text-gray-500">=</span>
            <div className="flex gap-0.5">{values[i].map((v, d) => (
              <div key={d} className="w-9 h-5 rounded text-center flex items-center justify-center" style={{
                fontSize: 7, fontFamily: 'monospace', fontWeight: 700, color: '#fff',
                background: `rgba(236,72,153,${Math.abs(v * weights[i]) * 3 + 0.1})`
              }}>{(v * weights[i]).toFixed(2)}</div>
            ))}</div>
          </div>
        ))}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-2 mt-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-green-500">SUM =</span>
            <div className="flex gap-0.5">{output.map((v, d) => (
              <div key={d} className="w-9 h-6 rounded text-center flex items-center justify-center" style={{
                fontSize: 8, fontFamily: 'monospace', fontWeight: 700, color: '#fff',
                background: `rgba(34,197,94,${Math.abs(v) * 2 + 0.2})`
              }}>{v.toFixed(2)}</div>
            ))}</div>
          </div>
          <p className="text-xs text-gray-500 mt-2">This is the <b>new context-aware embedding</b> for "{tokens[targetIdx]}" — it now encodes information from all tokens, weighted by relevance.</p>
        </div>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════
   LESSON LAB MAPPING
   ═══════════════════════════════════════════════════════════ */
const LESSON_LABS: Record<string, { title: string; component: React.FC }> = {
  'cnn-1': { title: 'Image as Matrix', component: () => <div className="space-y-4"><p className="text-sm text-gray-700 dark:text-gray-300">Try changing pixel values to see how images are represented as numbers:</p><SmallGrid data={[[0,0,0,0,0,0,0,0],[0,0,1,1,1,1,0,0],[0,1,0,0,0,0,1,0],[0,1,0,1,0,1,0,0],[0,1,0,0,0,0,1,0],[0,0,1,0,0,1,0,0],[0,0,0,1,1,0,0,0],[0,0,0,0,0,0,0,0]]} cellSize={32} label="Smiley face as 8×8 matrix" colorFn={(v: number) => v > 0.5 ? '#facc15' : '#1e293b'} /></div> },
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
  'tf-1': { title: 'The Mole Problem — Attention Mechanism Explorer', component: MoleProblemLabWrapper },
  'tf-2': { title: 'Tokenization Explorer', component: EmbeddingLab },
  'tf-3': { title: 'Token Embedding Explorer', component: EmbeddingLab },
  'tf-4': { title: 'Positional Encoding Visualizer', component: PELab },
  'tf-5': { title: 'Query, Key, Value Projections', component: SelfAttentionLab },
  'tf-6': { title: 'Attention Scores & Softmax', component: AttentionPatternLab },
  'tf-7': { title: 'Value Weighted Sum', component: ValueOutputLab },
  'tf-8': { title: 'Masked Attention Viewer', component: AttentionPatternLab },
  'tf-9': { title: 'Cross-Attention Explorer', component: AttentionPatternLab },
  'tf-10': { title: 'Multi-Head Attention Viewer', component: MultiHeadLab },
  'tf-11': { title: 'Feed-Forward Network', component: TransformerBlockLab },
  'tf-12': { title: 'Transformer Block Pipeline', component: TransformerBlockLab },
  'tf-13': { title: 'Model in Memory & Inference', component: TransformerBlockLab },
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
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <Navbar actions={<><NavLink to={`/topics/${topicId}/lessons`}>← Lessons</NavLink><NavLink to={`/topics/${topicId}/lab`} primary>Full Lab</NavLink></>} />

      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="mb-4">
          <h1 className="text-xl font-bold text-gray-900 dark:text-gray-900 dark:text-white">🧪 {lab.title}</h1>
          <p className="text-xs text-gray-500 dark:text-gray-600 dark:text-gray-400">Lesson Lab · {lessonId}</p>
        </div>
        <div className="rounded-2xl p-6 bg-white dark:bg-slate-800/60 border border-gray-200 dark:border-slate-700 shadow-lg">
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
