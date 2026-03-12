import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { useUserStore } from '../store/userStore';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';
import { Navbar, NavLink } from '../components/ui/Navbar';
import { CheckCircle, Circle, ArrowLeft } from 'lucide-react';
import { motion } from 'framer-motion';
import { TOPICS } from '../types/topics';

/* ═══════════════════════════════════════════════════════════
   VISUAL COMPONENTS FOR LESSONS
   ═══════════════════════════════════════════════════════════ */
// Math formula block
const MathBlock: React.FC<{formula: string; label?: string}> = ({formula, label}) => (
  <div className="my-4 p-4 bg-slate-100 dark:bg-slate-900 rounded-lg border border-slate-200 dark:border-slate-700 text-center">
    {label && <p className="text-xs text-slate-500 dark:text-slate-400 mb-2">{label}</p>}
    <p className="text-lg font-bold text-blue-700 dark:text-cyan-300 font-mono">{formula}</p>
  </div>
);

// Colored info box
const InfoBox: React.FC<{color: string; title: string; children: React.ReactNode}> = ({color, title, children}) => {
  const colors: Record<string, string> = { blue: 'bg-blue-50 dark:bg-blue-900/20 border-blue-300 dark:border-blue-700', green: 'bg-green-50 dark:bg-green-900/20 border-green-300 dark:border-green-700', yellow: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-300 dark:border-yellow-700', red: 'bg-red-50 dark:bg-red-900/20 border-red-300 dark:border-red-700', purple: 'bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700', orange: 'bg-orange-50 dark:bg-orange-900/20 border-orange-300 dark:border-orange-700' };
  return <div className={`my-4 p-4 rounded-lg border-l-4 ${colors[color] || colors.blue}`}><h4 className="font-bold text-gray-900 dark:text-white mb-2">{title}</h4><div className="text-sm text-gray-700 dark:text-gray-300">{children}</div></div>;
};

// Flow diagram
const FlowDiagram: React.FC<{steps: {label: string; color: string}[]}> = ({steps}) => (
  <div className="my-4 flex flex-wrap items-center gap-2 justify-center">
    {steps.map((s, i) => (
      <React.Fragment key={i}>
        {i > 0 && <span className="text-gray-400 text-lg">→</span>}
        <div className={`px-3 py-2 rounded-lg text-white text-sm font-semibold ${s.color}`}>{s.label}</div>
      </React.Fragment>
    ))}
  </div>
);

// Grid visual — small matrix of colored cells
const GridVisual: React.FC<{data: number[][]; cellSize?: number; label?: string; colorFn?: (v:number) => string}> = ({data, cellSize=28, label, colorFn}) => (
  <div className="my-3 inline-block">
    {label && <p className="text-xs font-semibold text-gray-600 mb-1">{label}</p>}
    <div style={{display:'inline-grid', gridTemplateColumns:`repeat(${data[0]?.length||1}, ${cellSize}px)`, gap:1}}>
      {data.flat().map((v,i) => (
        <div key={i} style={{width:cellSize, height:cellSize, display:'flex', alignItems:'center', justifyContent:'center', fontSize: cellSize < 24 ? 7 : 9, fontWeight:600, fontFamily:'monospace', borderRadius:3, background: colorFn ? colorFn(v) : `rgb(${Math.round(v*255)},${Math.round(v*255)},${Math.round(v*255)})`, color: v > 0.5 ? '#000' : '#fff', border:'1px solid #e2e8f0'}}>
          {v.toFixed(1)}
        </div>
      ))}
    </div>
  </div>
);

// Neuron diagram
const NeuronDiagram: React.FC<{inputs: string[]; output: string; activation: string}> = ({inputs, output, activation}) => (
  <div className="my-4 flex items-center justify-center gap-4 flex-wrap">
    <div className="flex flex-col gap-1">
      {inputs.map((inp,i) => <div key={i} className="px-2 py-1 bg-blue-100 rounded text-xs font-mono text-blue-800 border border-blue-300">{inp}</div>)}
    </div>
    <div className="flex flex-col items-center">
      <span className="text-gray-400 text-sm">→ Σ →</span>
    </div>
    <div className="w-16 h-16 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center text-white font-bold text-sm shadow-lg">{activation}</div>
    <span className="text-gray-400 text-sm">→</span>
    <div className="px-3 py-2 bg-green-100 rounded-lg text-sm font-semibold text-green-800 border border-green-300">{output}</div>
  </div>
);

// Activation function plot (canvas)
const ActivationPlot: React.FC<{fn: string; width?: number; height?: number}> = ({fn, width=200, height=120}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if(!c) return;
    c.width = width; c.height = height;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,width,height);
    ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0,height/2); ctx.lineTo(width,height/2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(width/2,0); ctx.lineTo(width/2,height); ctx.stroke();
    ctx.strokeStyle = '#3b82f6'; ctx.lineWidth = 3; ctx.beginPath();
    for(let px=0;px<width;px++){
      const x = (px/width)*8-4;
      let y: number;
      if(fn==='relu') y = Math.max(0,x);
      else if(fn==='sigmoid') y = 1/(1+Math.exp(-x));
      else if(fn==='tanh') y = Math.tanh(x);
      else y = x;
      const py = height - ((y+1.5)/3)*height;
      px===0 ? ctx.moveTo(px,Math.max(0,Math.min(height,py))) : ctx.lineTo(px,Math.max(0,Math.min(height,py)));
    }
    ctx.stroke();
    ctx.fillStyle = '#1e293b'; ctx.font = 'bold 12px monospace'; ctx.textAlign = 'center';
    ctx.fillText(fn==='relu'?'ReLU':fn==='sigmoid'?'Sigmoid':fn==='tanh'?'Tanh':'Linear', width/2, 16);
  },[fn,width,height]);
  return <canvas ref={ref} style={{width,height,borderRadius:8,border:'1px solid #e2e8f0'}} />;
};

// Attention heatmap mini
const MiniHeatmap: React.FC<{data: number[][]; labels: string[]; size?: number}> = ({data, labels, size=180}) => {
  const ref = useRef<HTMLCanvasElement>(null);
  const n = labels.length;
  useEffect(() => {
    const c = ref.current; if(!c) return;
    c.width = size; c.height = size;
    const ctx = c.getContext('2d')!;
    ctx.fillStyle = '#f8fafc'; ctx.fillRect(0,0,size,size);
    const off = 40, cs = (size-off)/n;
    for(let i=0;i<n;i++) for(let j=0;j<n;j++){
      const v = data[i]?.[j] || 0;
      ctx.fillStyle = `rgba(59,130,246,${v*0.9+0.05})`;
      ctx.fillRect(off+j*cs, off+i*cs, cs-1, cs-1);
      if(n<=6){ctx.fillStyle=v>0.3?'#fff':'#64748b';ctx.font=`bold ${Math.min(10,cs*0.35)}px monospace`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(v.toFixed(2),off+j*cs+cs/2,off+i*cs+cs/2);}
    }
    ctx.fillStyle='#64748b'; ctx.font='bold 9px sans-serif'; ctx.textAlign='right'; ctx.textBaseline='middle';
    for(let i=0;i<n;i++) ctx.fillText(labels[i].slice(0,5), off-3, off+i*cs+cs/2);
    ctx.textAlign='center'; ctx.textBaseline='bottom';
    for(let j=0;j<n;j++){ctx.save();ctx.translate(off+j*cs+cs/2,off-3);ctx.rotate(-0.4);ctx.fillText(labels[j].slice(0,5),0,0);ctx.restore();}
  },[data,labels,size,n]);
  return <canvas ref={ref} style={{width:size, height:size, borderRadius:8, border:'1px solid #e2e8f0'}} className="my-3" />;
};

// Convolution demo
const ConvDemo: React.FC = () => {
  const input = [[0,0,1,0,0],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[0,0,1,0,0]];
  const kernel = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]];
  const output: number[][] = [];
  for(let i=0;i<3;i++){const row:number[]=[];for(let j=0;j<3;j++){let s=0;for(let a=0;a<3;a++)for(let b=0;b<3;b++)s+=input[i+a][j+b]*kernel[a][b];row.push(s);}output.push(row);}
  return (
    <div className="my-4 flex items-center gap-4 flex-wrap justify-center">
      <GridVisual data={input} cellSize={32} label="Input 5×5" colorFn={v => v>0.5?'#3b82f6':'#1e293b'} />
      <span className="text-2xl text-yellow-500 font-bold">⊛</span>
      <GridVisual data={kernel} cellSize={32} label="Kernel 3×3" colorFn={v => v>0?`rgba(34,197,94,${v/8+0.2})`:`rgba(239,68,68,${Math.abs(v)/8+0.2})`} />
      <span className="text-2xl text-green-500 font-bold">=</span>
      <GridVisual data={output} cellSize={32} label="Output 3×3" colorFn={v => {const n=(v+8)/16;return `rgb(${Math.round(n*255)},${Math.round(n*200)},${Math.round((1-n)*255)})`}} />
    </div>
  );
};

// Pooling demo
const PoolDemo: React.FC = () => {
  const input = [[1,3,2,4],[5,6,1,2],[7,2,8,3],[4,1,5,6]];
  const maxOut = [[6,4],[7,8]];
  return (
    <div className="my-4 flex items-center gap-6 flex-wrap justify-center">
      <div>
        <GridVisual data={input} cellSize={36} label="Input 4×4" colorFn={v=>`rgba(59,130,246,${v/8+0.1})`} />
      </div>
      <div className="text-center">
        <span className="text-2xl text-orange-500 font-bold">Max Pool</span><br/>
        <span className="text-xs text-gray-500">2×2, stride 2</span>
      </div>
      <div>
        <GridVisual data={maxOut} cellSize={44} label="Output 2×2" colorFn={v=>`rgba(34,197,94,${v/8+0.2})`} />
      </div>
    </div>
  );
};

/* ═══════════════════════════════════════════════════════════
   LESSON DATA — each content item is a React element
   ═══════════════════════════════════════════════════════════ */
interface Lesson {
  id: string;
  title: string;
  description: string;
  icon: string;
  content: React.ReactNode;
}

const ALL_LESSONS: Record<string, Lesson[]> = {
  cnn: [
    { id:'cnn-1', title:'Images as Data', description:'How computers see images as matrices of numbers', icon:'🖼️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Images are grids of numbers called <b>pixels</b>. Grayscale images have one number per pixel (0=black, 1=white). Color images have <b>3 channels: Red, Green, Blue</b>.</p>
        <div className="flex gap-6 flex-wrap justify-center my-4">
          <GridVisual data={[[0,0,0,0,0],[0,0,1,0,0],[0,1,1,1,0],[0,0,1,0,0],[0,0,0,0,0]]} cellSize={32} label="Grayscale (1 channel)" colorFn={v=>v>0.5?'#fff':'#1e293b'} />
          <div className="text-center">
            <p className="text-xs font-bold text-red-500 mb-1">Red Channel</p>
            <GridVisual data={[[1,0,0],[0,1,0],[0,0,1]]} cellSize={28} colorFn={v=>`rgba(239,68,68,${v*0.8+0.1})`} />
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-green-500 mb-1">Green Channel</p>
            <GridVisual data={[[0,1,0],[1,0,1],[0,1,0]]} cellSize={28} colorFn={v=>`rgba(34,197,94,${v*0.8+0.1})`} />
          </div>
          <div className="text-center">
            <p className="text-xs font-bold text-blue-500 mb-1">Blue Channel</p>
            <GridVisual data={[[0,0,1],[0,0,0],[1,0,0]]} cellSize={28} colorFn={v=>`rgba(59,130,246,${v*0.8+0.1})`} />
          </div>
        </div>
        <InfoBox color="blue" title="💡 Key Insight">
          <p>Nearby pixels are related. A pixel next to a white pixel is likely also light. <b>CNNs exploit this spatial structure</b> through convolution.</p>
        </InfoBox>
        <MathBlock formula="Image: H × W × C" label="Image tensor shape (Height × Width × Channels)" />
      </div>
    )},
    { id:'cnn-2', title:'Convolution Operation', description:'How sliding filters detect patterns', icon:'🔍', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Convolution slides a small matrix (<b>kernel/filter</b>) across the image. At each position, it multiplies overlapping values and sums them into a single output value.</p>
        <ConvDemo />
        <MathBlock formula="Output[i,j] = Σₐ Σᵦ Input[i+a, j+b] × Kernel[a, b]" label="Convolution formula" />
        <InfoBox color="yellow" title="📐 Key Parameters">
          <p><b>Kernel Size:</b> 3×3, 5×5 (receptive field) · <b>Stride:</b> Step size (1=slide by 1 pixel) · <b>Padding:</b> Zeros around edges · <b>Filters:</b> Number of output channels</p>
        </InfoBox>
      </div>
    )},
    { id:'cnn-3', title:'Filters and Feature Maps', description:'How kernels detect edges and textures', icon:'🎨', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Different kernels detect different features. Each filter produces one <b>feature map</b> (output channel).</p>
        <div className="flex gap-4 flex-wrap justify-center">
          <div className="text-center">
            <GridVisual data={[[-1,0,1],[-2,0,2],[-1,0,1]]} cellSize={36} label="Vertical Edge" colorFn={v=>v>0?`rgba(34,197,94,${v/2+0.2})`:`rgba(239,68,68,${Math.abs(v)/2+0.2})`} />
          </div>
          <div className="text-center">
            <GridVisual data={[[-1,-2,-1],[0,0,0],[1,2,1]]} cellSize={36} label="Horizontal Edge" colorFn={v=>v>0?`rgba(34,197,94,${v/2+0.2})`:`rgba(239,68,68,${Math.abs(v)/2+0.2})`} />
          </div>
          <div className="text-center">
            <GridVisual data={[[0,-1,0],[-1,5,-1],[0,-1,0]]} cellSize={36} label="Sharpen" colorFn={v=>v>0?`rgba(250,204,21,${v/5+0.1})`:`rgba(239,68,68,${Math.abs(v)/5+0.2})`} />
          </div>
          <div className="text-center">
            <GridVisual data={[[.11,.11,.11],[.11,.11,.11],[.11,.11,.11]]} cellSize={36} label="Blur (1/9)" colorFn={()=>'rgba(59,130,246,0.4)'} />
          </div>
        </div>
        <InfoBox color="green" title="🧠 Learning"><p>During training, the network <b>learns</b> which kernels are useful. Early layers learn edges; deeper layers learn complex patterns like eyes, wheels, text.</p></InfoBox>
      </div>
    )},
    { id:'cnn-4', title:'Activation Functions', description:'Adding non-linearity with ReLU, Sigmoid, Tanh', icon:'⚡', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Without activations, stacking layers is just matrix multiplication — equivalent to a single layer. <b>Activations add non-linearity</b>.</p>
        <div className="flex gap-4 flex-wrap justify-center">
          <ActivationPlot fn="relu" /><ActivationPlot fn="sigmoid" /><ActivationPlot fn="tanh" />
        </div>
        <MathBlock formula="ReLU(x) = max(0, x)  |  σ(x) = 1/(1+e⁻ˣ)  |  tanh(x)" label="Common activation functions" />
        <InfoBox color="blue" title="🏆 ReLU is King"><p>ReLU is the most popular: fast to compute, avoids vanishing gradients. Sigmoid is used for output probabilities.</p></InfoBox>
      </div>
    )},
    { id:'cnn-5', title:'Pooling Layers', description:'Reducing dimensions while keeping features', icon:'⬇️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Pooling <b>reduces spatial size</b> of feature maps, keeping the most important information.</p>
        <PoolDemo />
        <InfoBox color="orange" title="Max vs Average Pooling">
          <p><b>Max Pool:</b> Takes the maximum value — preserves strongest detections.<br/><b>Average Pool:</b> Takes the mean — smoother, preserves general presence.</p>
        </InfoBox>
        <MathBlock formula="Input: H×W → Max Pool 2×2 → Output: H/2 × W/2" label="Size reduction (75% fewer values)" />
      </div>
    )},
    { id:'cnn-6', title:'CNN Architecture', description:'Full pipeline: Conv→Pool→Dense→Softmax', icon:'🏗️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">A complete CNN stacks multiple conv+pool blocks, then flattens and classifies.</p>
        <FlowDiagram steps={[{label:'Input',color:'bg-green-500'},{label:'Conv+ReLU',color:'bg-blue-500'},{label:'MaxPool',color:'bg-orange-500'},{label:'Conv+ReLU',color:'bg-blue-500'},{label:'MaxPool',color:'bg-orange-500'},{label:'Flatten',color:'bg-purple-500'},{label:'Dense',color:'bg-pink-500'},{label:'Softmax',color:'bg-red-500'}]} />
        <InfoBox color="purple" title="🏛️ Famous Architectures">
          <p><b>LeNet-5</b> (1998): Handwriting recognition · <b>AlexNet</b> (2012): Won ImageNet · <b>VGG</b> (2014): Very deep (16-19 layers) · <b>ResNet</b> (2015): Skip connections, 152 layers · <b>EfficientNet</b> (2019): Optimal scaling</p>
        </InfoBox>
      </div>
    )},
  ],
  ann: [
    { id:'ann-1', title:'What is a Neural Network?', description:'Neurons, weights, and layers', icon:'🧠', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">A neural network is inspired by the brain. <b>Artificial neurons</b> receive inputs, multiply by weights, sum them, add bias, and apply an activation.</p>
        <NeuronDiagram inputs={['x₁ × w₁','x₂ × w₂','x₃ × w₃']} output="ŷ" activation="σ" />
        <MathBlock formula="output = σ(w₁x₁ + w₂x₂ + w₃x₃ + bias)" label="Single neuron computation" />
        <FlowDiagram steps={[{label:'Input Layer',color:'bg-green-500'},{label:'Hidden Layer 1',color:'bg-blue-500'},{label:'Hidden Layer 2',color:'bg-blue-500'},{label:'Output Layer',color:'bg-purple-500'}]} />
      </div>
    )},
    { id:'ann-2', title:'Forward Propagation', description:'Data flowing through the network', icon:'➡️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Data enters the input layer and flows forward. Each neuron computes a <b>weighted sum + activation</b>.</p>
        <MathBlock formula="z = Σᵢ(wᵢ × xᵢ) + b  →  a = activation(z)" label="Per-neuron computation" />
        <FlowDiagram steps={[{label:'x = [0.5, 0.3]',color:'bg-green-500'},{label:'z = Wx + b',color:'bg-yellow-500'},{label:'a = ReLU(z)',color:'bg-blue-500'},{label:'z₂ = W₂a + b₂',color:'bg-yellow-500'},{label:'ŷ = σ(z₂)',color:'bg-purple-500'}]} />
        <InfoBox color="blue" title="📐 Shape Tracking"><p>Input: [2] → Hidden: [4] → Output: [1]. Weights matrix between layers: [4×2] then [1×4].</p></InfoBox>
      </div>
    )},
    { id:'ann-3', title:'Activation Functions', description:'Why non-linearity matters', icon:'⚡', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Without activations, a deep network is just one linear transformation. <b>Non-linearity</b> gives networks their power.</p>
        <div className="flex gap-4 flex-wrap justify-center">
          <ActivationPlot fn="relu" /><ActivationPlot fn="sigmoid" /><ActivationPlot fn="tanh" />
        </div>
        <InfoBox color="green" title="When to use what">
          <p><b>ReLU:</b> Hidden layers (fast, no vanishing gradients) · <b>Sigmoid:</b> Binary classification output · <b>Softmax:</b> Multi-class output (probabilities sum to 1)</p>
        </InfoBox>
      </div>
    )},
    { id:'ann-4', title:'Loss Functions & Training', description:'Measuring and reducing errors', icon:'📉', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300"><b>Loss</b> measures how wrong the predictions are. Training <b>minimizes</b> the loss.</p>
        <MathBlock formula="MSE = (1/n) Σ(yᵢ - ŷᵢ)²" label="Mean Squared Error (regression)" />
        <MathBlock formula="CE = -Σ yᵢ log(ŷᵢ)" label="Cross-Entropy (classification)" />
        <InfoBox color="yellow" title="⚙️ Learning Rate">
          <p>Controls step size. <span className="text-red-600 font-bold">Too large → overshooting</span>. <span className="text-blue-600 font-bold">Too small → slow convergence</span>. Typical: 0.001 to 0.1.</p>
        </InfoBox>
      </div>
    )},
    { id:'ann-5', title:'Backpropagation', description:'Gradients flowing backward', icon:'🔄', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Backprop computes the <b>gradient of loss with respect to each weight</b> using the chain rule.</p>
        <MathBlock formula="∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w" label="Chain rule of calculus" />
        <FlowDiagram steps={[{label:'Loss',color:'bg-red-500'},{label:'∂L/∂a₂',color:'bg-orange-500'},{label:'∂a₂/∂z₂',color:'bg-yellow-500'},{label:'∂z₂/∂w₂',color:'bg-green-500'},{label:'Update w₂',color:'bg-blue-500'}]} />
        <MathBlock formula="w_new = w_old − learning_rate × gradient" label="Weight update rule (gradient descent)" />
      </div>
    )},
    { id:'ann-6', title:'Decision Boundaries', description:'Separating classes in feature space', icon:'🎯', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">A single neuron draws a <b>linear boundary</b>. Hidden layers create <b>non-linear boundaries</b>.</p>
        <div className="flex gap-6 flex-wrap justify-center my-4">
          <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-200"><p className="font-bold text-sm mb-2">0 hidden layers</p><p className="text-xs text-gray-600">Linear boundary only</p><p className="text-xs text-red-500">❌ Cannot solve XOR</p></div>
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200"><p className="font-bold text-sm mb-2">1 hidden layer (4n)</p><p className="text-xs text-gray-600">Non-linear boundary</p><p className="text-xs text-green-600">✓ Solves XOR, circles</p></div>
          <div className="text-center p-4 bg-purple-50 rounded-lg border border-purple-200"><p className="font-bold text-sm mb-2">2+ hidden layers</p><p className="text-xs text-gray-600">Complex boundaries</p><p className="text-xs text-green-600">✓ Spirals, any shape</p></div>
        </div>
        <InfoBox color="blue" title="Universal Approximation"><p>A neural network with at least one hidden layer can approximate <b>any continuous function</b> — given enough neurons.</p></InfoBox>
      </div>
    )},
  ],
  rnn: [
    { id:'rnn-1', title:'Sequential Data', description:'Why order matters', icon:'📊', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Some data has a natural <b>order</b>: words in a sentence, stock prices over time. Standard neural networks ignore this order.</p>
        <FlowDiagram steps={[{label:'x₁',color:'bg-blue-500'},{label:'x₂',color:'bg-blue-500'},{label:'x₃',color:'bg-blue-500'},{label:'...',color:'bg-gray-400'},{label:'xₜ',color:'bg-blue-500'}]} />
        <InfoBox color="orange" title="📊 Examples"><p>Language: "The cat sat on the ___" · Time series: predict tomorrow's stock price · Music: generate the next note · Speech: convert audio to text</p></InfoBox>
      </div>
    )},
    { id:'rnn-2', title:'RNN Cell', description:'The recurrent hidden state', icon:'🔄', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">At each timestep, the RNN takes input xₜ and previous hidden state h<sub>t-1</sub>, producing new hidden state hₜ.</p>
        <MathBlock formula="hₜ = tanh(W_xh · xₜ + W_hh · hₜ₋₁ + b)" label="RNN hidden state update" />
        <FlowDiagram steps={[{label:'xₜ',color:'bg-green-500'},{label:'+ hₜ₋₁',color:'bg-purple-500'},{label:'→ tanh →',color:'bg-yellow-500'},{label:'hₜ',color:'bg-purple-500'},{label:'→ yₜ',color:'bg-blue-500'}]} />
        <InfoBox color="purple" title="🧠 The Hidden State is Memory"><p>hₜ encodes <b>everything seen so far</b>. The same weights W are shared across all timesteps.</p></InfoBox>
      </div>
    )},
    { id:'rnn-3', title:'Vanishing Gradients', description:'Why simple RNNs struggle', icon:'📉', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">During backprop through time, gradients are <b>multiplied at each step</b>. If &lt; 1, they shrink to zero.</p>
        <div className="my-4 flex items-center gap-2 justify-center">
          {[1,2,3,4,5].map(i => <div key={i} className="text-center"><div className="w-10 h-10 rounded-full flex items-center justify-center text-xs font-bold" style={{background:`rgba(239,68,68,${1-i*0.18})`,color:'#fff'}}>t={i}</div><p className="text-xs mt-1" style={{color:`rgba(239,68,68,${1-i*0.18})`}}>{(0.7**i).toFixed(2)}</p></div>)}
          <span className="text-gray-400 text-sm">→ 0.00</span>
        </div>
        <MathBlock formula="Gradient at t=1 ≈ 0.7⁵ = 0.168 → vanishes!" label="Gradient shrinkage over 5 timesteps" />
        <InfoBox color="red" title="❌ The Problem"><p>Early timesteps get almost <b>no gradient</b>. The network can't learn what happened 50 steps ago. This is why LSTM was invented.</p></InfoBox>
      </div>
    )},
    { id:'rnn-4', title:'LSTM Architecture', description:'Gates that control information', icon:'🧬', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">LSTM has a <b>cell state</b> (long-term memory) and <b>3 gates</b> controlling information flow.</p>
        <div className="flex gap-3 flex-wrap justify-center my-4">
          <div className="p-3 rounded-lg bg-red-50 border border-red-300 text-center w-40"><p className="font-bold text-red-700">🚪 Forget Gate</p><p className="text-xs text-gray-600 mt-1">f = σ(W·[h,x]+b)</p><p className="text-xs text-red-600">What to REMOVE</p></div>
          <div className="p-3 rounded-lg bg-green-50 border border-green-300 text-center w-40"><p className="font-bold text-green-700">🚪 Input Gate</p><p className="text-xs text-gray-600 mt-1">i = σ(W·[h,x]+b)</p><p className="text-xs text-green-600">What to ADD</p></div>
          <div className="p-3 rounded-lg bg-blue-50 border border-blue-300 text-center w-40"><p className="font-bold text-blue-700">🚪 Output Gate</p><p className="text-xs text-gray-600 mt-1">o = σ(W·[h,x]+b)</p><p className="text-xs text-blue-600">What to OUTPUT</p></div>
        </div>
        <MathBlock formula="cₜ = fₜ ⊙ cₜ₋₁ + iₜ ⊙ c̃ₜ" label="Cell state update (additive, not multiplicative!)" />
      </div>
    )},
    { id:'rnn-5', title:'LSTM Cell State', description:'The conveyor belt of memory', icon:'🔗', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">The cell state runs through the entire sequence like a <b>conveyor belt</b>. Information can flow unchanged.</p>
        <FlowDiagram steps={[{label:'c₀',color:'bg-yellow-500'},{label:'× forget',color:'bg-red-400'},{label:'+ input',color:'bg-green-400'},{label:'c₁',color:'bg-yellow-500'},{label:'× forget',color:'bg-red-400'},{label:'+ input',color:'bg-green-400'},{label:'c₂',color:'bg-yellow-500'}]} />
        <InfoBox color="yellow" title="✅ Why This Works"><p>The update is <b>additive</b> (c + ...) not multiplicative (c × ...). Gradients flow through the cell state without vanishing!</p></InfoBox>
        <MathBlock formula="hₜ = oₜ ⊙ tanh(cₜ)" label="Hidden state from cell state via output gate" />
      </div>
    )},
    { id:'rnn-6', title:'Applications & Variants', description:'GRU, Bidirectional, and more', icon:'🚀', content: (
      <div className="space-y-4">
        <div className="flex gap-4 flex-wrap justify-center">
          <InfoBox color="blue" title="🔀 GRU (Gated Recurrent Unit)"><p>Simplified LSTM: 2 gates instead of 3. Often works just as well with fewer parameters.</p></InfoBox>
          <InfoBox color="purple" title="↔️ Bidirectional RNN"><p>Processes sequence forward AND backward. Useful when context from both directions matters (e.g., translation).</p></InfoBox>
        </div>
        <InfoBox color="green" title="🌍 Real-World Applications"><p>Chatbots · Speech recognition · Music generation · Handwriting synthesis · Video captioning · DNA analysis · Stock prediction</p></InfoBox>
      </div>
    )},
  ],
  vae: [
    { id:'vae-1', title:'Autoencoders', description:'Compress and reconstruct', icon:'🔄', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">An autoencoder learns to <b>compress</b> (encode) data into a compact representation, then <b>reconstruct</b> (decode) it.</p>
        <FlowDiagram steps={[{label:'Input (784)',color:'bg-green-500'},{label:'Encoder',color:'bg-blue-500'},{label:'Latent (2)',color:'bg-yellow-500'},{label:'Decoder',color:'bg-pink-500'},{label:'Output (784)',color:'bg-green-500'}]} />
        <MathBlock formula="784 → 128 → 32 → 2 → 32 → 128 → 784" label="Bottleneck architecture" />
        <InfoBox color="blue" title="🔑 The Bottleneck"><p>The small latent code forces the network to learn the <b>most essential features</b>. This is unsupervised learning — no labels needed!</p></InfoBox>
      </div>
    )},
    { id:'vae-2', title:'The Latent Space', description:'Where similar data clusters', icon:'🗺️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">The latent space is the compressed representation. <b>Similar inputs map to nearby points.</b></p>
        <div className="my-4 bg-slate-100 rounded-lg p-4 text-center">
          <div className="inline-block relative" style={{width:200,height:200,background:'#0f172a',borderRadius:12}}>
            {[{x:30,y:40,c:'#ef4444',l:'3'},{x:50,y:50,c:'#ef4444',l:'3'},{x:40,y:35,c:'#ef4444',l:'3'},{x:150,y:130,c:'#3b82f6',l:'7'},{x:160,y:140,c:'#3b82f6',l:'7'},{x:140,y:145,c:'#3b82f6',l:'7'},{x:100,y:90,c:'#22c55e',l:'5'},{x:90,y:100,c:'#22c55e',l:'5'}].map((p,i) => (
              <div key={i} className="absolute w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white" style={{left:p.x,top:p.y,background:p.c}}>{p.l}</div>
            ))}
          </div>
          <p className="text-xs text-gray-500 mt-2">2D latent space: digits "3" cluster together, "7"s elsewhere</p>
        </div>
        <InfoBox color="green" title="✨ Generation"><p>Moving smoothly in latent space → smooth changes in output. This is what enables <b>generating new data</b>!</p></InfoBox>
      </div>
    )},
    { id:'vae-3', title:'Why Variational?', description:'Distributions, not just points', icon:'🎲', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Regular autoencoders map to <b>single points</b> → gaps in latent space. VAEs map to <b>distributions</b>.</p>
        <div className="flex gap-6 flex-wrap justify-center my-4">
          <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200 w-48"><p className="font-bold text-red-700 mb-2">Regular AE ❌</p><p className="text-xs text-gray-600">Scattered points with gaps. Decoding a gap point → garbage output.</p></div>
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200 w-48"><p className="font-bold text-green-700 mb-2">VAE ✅</p><p className="text-xs text-gray-600">Overlapping distributions. Any point can be decoded → smooth generation.</p></div>
        </div>
        <MathBlock formula="Encode: x → q(z|x) = N(μ, σ²)" label="VAE encodes to a distribution, not a point" />
      </div>
    )},
    { id:'vae-4', title:'Reparameterization Trick', description:'z = μ + σ × ε', icon:'🎯', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300"><b>Problem:</b> Can't backpropagate through random sampling. <b>Solution:</b> Reparameterize!</p>
        <MathBlock formula="z = μ + σ × ε,  where ε ~ N(0, 1)" label="The reparameterization trick" />
        <FlowDiagram steps={[{label:'μ (learnable)',color:'bg-purple-500'},{label:'+ σ × ε',color:'bg-yellow-500'},{label:'= z',color:'bg-green-500'}]} />
        <InfoBox color="yellow" title="🧠 Why It Works"><p>The randomness (ε) is <b>external</b> — not part of the computation graph. Gradients flow through μ and σ, making the whole thing trainable!</p></InfoBox>
      </div>
    )},
    { id:'vae-5', title:'VAE Loss Function', description:'Reconstruction + KL Divergence', icon:'📊', content: (
      <div className="space-y-4">
        <MathBlock formula="Loss = Reconstruction Loss + KL Divergence" label="VAE total loss (ELBO)" />
        <div className="flex gap-4 flex-wrap justify-center">
          <InfoBox color="blue" title="📏 Reconstruction Loss"><p>MSE between input and output. Ensures the latent code is <b>informative</b> enough to reproduce the input.</p></InfoBox>
          <InfoBox color="purple" title="📐 KL Divergence"><p>DKL(q(z|x) || N(0,1)). Ensures the latent distribution stays close to standard normal. <b>Regularizes</b> the latent space.</p></InfoBox>
        </div>
        <MathBlock formula="DKL = -½ Σ(1 + log(σ²) - μ² - σ²)" label="KL divergence formula (closed form for Gaussians)" />
      </div>
    )},
    { id:'vae-6', title:'Generation & Interpolation', description:'Creating new data', icon:'✨', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Once trained, sample z ~ N(0,1) → decode → <b>new image!</b></p>
        <FlowDiagram steps={[{label:'z ~ N(0,1)',color:'bg-yellow-500'},{label:'Decoder',color:'bg-pink-500'},{label:'New Image!',color:'bg-green-500'}]} />
        <p className="text-gray-700 dark:text-gray-300">Interpolate between two inputs by mixing their latent codes:</p>
        <FlowDiagram steps={[{label:'z_A',color:'bg-blue-500'},{label:'α·z_A + (1-α)·z_B',color:'bg-yellow-500'},{label:'z_B',color:'bg-purple-500'}]} />
        <InfoBox color="green" title="🌍 Applications"><p>Face generation · Drug molecule design · Music composition · Data augmentation · Anomaly detection (high reconstruction error = anomaly)</p></InfoBox>
      </div>
    )},
  ],
  transformers: [
    { id:'tf-1', title:'Beyond Sequences', description:'Why attention replaced RNNs', icon:'⚡', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">RNNs process one token at a time → bottleneck. Transformers process <b>ALL tokens simultaneously</b>.</p>
        <div className="flex gap-6 flex-wrap justify-center my-4">
          <div className="text-center p-4 bg-red-50 rounded-lg border border-red-200 w-48"><p className="font-bold text-red-700 mb-2">RNN ❌</p><p className="text-xs text-gray-600">Sequential (slow). Long-range dependencies are hard. Cannot parallelize.</p></div>
          <div className="text-center p-4 bg-green-50 rounded-lg border border-green-200 w-48"><p className="font-bold text-green-700 mb-2">Transformer ✅</p><p className="text-xs text-gray-600">Parallel (fast). Every token sees every other token directly. GPU-friendly.</p></div>
        </div>
        <InfoBox color="blue" title="💡 Key Paper"><p>"Attention Is All You Need" (Vaswani et al., 2017). This paper started the modern AI revolution — GPT, BERT, and all LLMs are Transformers.</p></InfoBox>
      </div>
    )},
    { id:'tf-2', title:'Token Embeddings', description:'Words → Vectors', icon:'📝', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Words are converted to dense vectors called <b>embeddings</b>. Similar words have similar vectors.</p>
        <FlowDiagram steps={[{label:'"the"',color:'bg-green-500'},{label:'→ lookup →',color:'bg-gray-400'},{label:'[0.2, -0.5, 0.8, ...]',color:'bg-blue-500'}]} />
        <MathBlock formula="Embedding Matrix: [vocab_size × d_model]" label="Each row is one token's learned vector" />
        <InfoBox color="green" title="📐 Dimensions"><p>GPT-2: d_model=768 · GPT-3: d_model=12,288 · Our simulator: d_model=4–16 for clarity.</p></InfoBox>
      </div>
    )},
    { id:'tf-3', title:'Positional Encoding', description:'Giving the model word order', icon:'🌊', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Transformers have no inherent order. <b>Positional encoding</b> adds position information via sine/cosine waves.</p>
        <MathBlock formula="PE(pos, 2i) = sin(pos / 10000^(2i/d))" label="Even dimensions use sine" />
        <MathBlock formula="PE(pos, 2i+1) = cos(pos / 10000^(2i/d))" label="Odd dimensions use cosine" />
        <InfoBox color="blue" title="🌊 Why Sine/Cosine?"><p>Each position gets a <b>unique signature</b>. Nearby positions have similar encodings. The model can learn relative positions from these wave patterns.</p></InfoBox>
      </div>
    )},
    { id:'tf-4', title:'Self-Attention (Q, K, V)', description:'The core of Transformers', icon:'🔑', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">Each token is projected into <b>Query</b>, <b>Key</b>, and <b>Value</b> vectors.</p>
        <div className="flex gap-3 flex-wrap justify-center my-4">
          <div className="p-3 rounded-lg bg-red-50 border border-red-300 text-center w-44"><p className="font-bold text-red-700">🔴 Query (Q)</p><p className="text-xs text-gray-600">"What am I looking for?"</p></div>
          <div className="p-3 rounded-lg bg-green-50 border border-green-300 text-center w-44"><p className="font-bold text-green-700">🟢 Key (K)</p><p className="text-xs text-gray-600">"What do I contain?"</p></div>
          <div className="p-3 rounded-lg bg-blue-50 border border-blue-300 text-center w-44"><p className="font-bold text-blue-700">🔵 Value (V)</p><p className="text-xs text-gray-600">"What info do I provide?"</p></div>
        </div>
        <MathBlock formula="Attention(Q,K,V) = softmax(Q·Kᵀ / √dₖ) × V" label="Scaled dot-product attention" />
        <MiniHeatmap data={[[0.1,0.7,0.1,0.1],[0.05,0.1,0.8,0.05],[0.2,0.2,0.1,0.5],[0.6,0.1,0.1,0.2]]} labels={['The','cat','sat','mat']} />
        <p className="text-xs text-gray-500 text-center">Example: "cat" strongly attends to "sat" (0.80)</p>
      </div>
    )},
    { id:'tf-5', title:'Multi-Head Attention', description:'Multiple attention patterns', icon:'👁️', content: (
      <div className="space-y-4">
        <p className="text-gray-700 dark:text-gray-300">One head learns one type of relationship. <b>Multiple heads</b> learn different relationships in parallel.</p>
        <div className="flex gap-3 flex-wrap justify-center my-4">
          {['Syntax','Semantics','Proximity','Coreference'].map((h,i) => (
            <div key={i} className="p-2 rounded-lg bg-yellow-50 border border-yellow-300 text-center w-28"><p className="font-bold text-yellow-700 text-xs">Head {i+1}</p><p className="text-xs text-gray-600">{h}</p></div>
          ))}
        </div>
        <FlowDiagram steps={[{label:'Head 1',color:'bg-yellow-500'},{label:'Head 2',color:'bg-orange-500'},{label:'Head 3',color:'bg-red-500'},{label:'Concat',color:'bg-purple-500'},{label:'W_O',color:'bg-blue-500'}]} />
        <MathBlock formula="MultiHead(Q,K,V) = Concat(head₁,...,headₙ) × W_O" label="Multi-head attention formula" />
      </div>
    )},
    { id:'tf-6', title:'Transformer Block', description:'Full block: Attention + FFN + Residuals', icon:'🏗️', content: (
      <div className="space-y-4">
        <FlowDiagram steps={[{label:'Input',color:'bg-green-500'},{label:'Multi-Head Attn',color:'bg-yellow-500'},{label:'Add & Norm',color:'bg-gray-500'},{label:'Feed-Forward',color:'bg-pink-500'},{label:'Add & Norm',color:'bg-gray-500'},{label:'Output',color:'bg-green-500'}]} />
        <div className="flex gap-4 flex-wrap justify-center">
          <InfoBox color="blue" title="➕ Residual Connection"><p><b>output = layer(x) + x</b>. Allows gradient to flow directly. Enables very deep networks (100+ layers).</p></InfoBox>
          <InfoBox color="purple" title="📏 Layer Norm"><p>Normalizes activations across features for each position. Stabilizes training.</p></InfoBox>
          <InfoBox color="orange" title="🧮 Feed-Forward"><p>Two linear layers with ReLU: <b>d_model → d_ff → d_model</b>. Applied per position independently.</p></InfoBox>
        </div>
        <MathBlock formula="FFN(x) = max(0, xW₁ + b₁)W₂ + b₂" label="Position-wise feed-forward network" />
      </div>
    )},
  ],
};

const TOPIC_GRADIENTS: Record<string, string> = {
  cnn:'from-green-50 via-white to-emerald-50', ann:'from-blue-50 via-white to-cyan-50',
  rnn:'from-orange-50 via-white to-amber-50', vae:'from-pink-50 via-white to-rose-50',
  transformers:'from-violet-50 via-white to-purple-50',
};

export const Lessons: React.FC = () => {
  const { topicId } = useParams();
  const navigate = useNavigate();
  const currentTopic = TOPICS.find(t => t.id === topicId);
  const { completedLessons, completeLesson } = useUserStore();
  const [selectedLesson, setSelectedLesson] = useState<Lesson | null>(null);
  const lessons = ALL_LESSONS[topicId || ''] || [];
  const gradient = TOPIC_GRADIENTS[topicId || ''] || 'from-gray-50 via-white to-gray-50';

  if (!lessons.length) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="max-w-md"><div className="p-8 text-center"><h2 className="text-2xl font-bold text-white mb-4">Coming Soon</h2><Button onClick={() => navigate('/topics')}>Back to Topics</Button></div></Card>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br from-blue-50 via-white to-green-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900`}>
      <Navbar actions={<><NavLink to="/topics">Topics</NavLink><NavLink to="/dashboard">Dashboard</NavLink><NavLink to={`/topics/${topicId}/lab`} primary>Go to Lab</NavLink></>} />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="mb-6 flex items-center gap-3">
          <Button variant="ghost" onClick={() => navigate('/topics')}><ArrowLeft className="w-4 h-4 mr-1" />Back</Button>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">{currentTopic?.name} Lessons</h1>
        </div>
        {!selectedLesson ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {lessons.map((lesson, idx) => {
              const isCompleted = completedLessons.includes(lesson.id);
              return (
                <motion.div key={lesson.id} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: idx * 0.1 }}>
                  <Card className="h-full hover:shadow-xl transition-shadow cursor-pointer">
                    <div onClick={() => setSelectedLesson(lesson)}>
                      <div className="flex items-start justify-between mb-4">
                        <div className="text-3xl">{lesson.icon}</div>
                        {isCompleted ? <CheckCircle className="w-6 h-6 text-green-600" /> : <Circle className="w-6 h-6 text-gray-300" />}
                      </div>
                      <h3 className="text-xl font-semibold text-gray-900 mb-2">{lesson.title}</h3>
                      <p className="text-gray-600 dark:text-gray-400 text-sm mb-4">{lesson.description}</p>
                      <Button variant="outline" size="sm" className="w-full">{isCompleted ? 'Review Lesson' : 'Start Lesson'}</Button>
                    </div>
                  </Card>
                </motion.div>
              );
            })}
          </div>
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <Card>
              <div className="mb-6"><Button variant="ghost" onClick={() => setSelectedLesson(null)}>← Back to Lessons</Button></div>
              <div className="flex items-center space-x-4 mb-6">
                <div className="text-4xl">{selectedLesson.icon}</div>
                <div><h2 className="text-3xl font-bold text-gray-900 dark:text-white">{selectedLesson.title}</h2><p className="text-gray-600 dark:text-gray-400">{selectedLesson.description}</p></div>
              </div>
              <div className="prose max-w-none">{selectedLesson.content}</div>
              <div className="mt-8 pt-6 border-t border-gray-200 dark:border-slate-700 flex items-center justify-between">
                <Link to={`/topics/${topicId}/lab/${selectedLesson.id}`}><Button variant="outline">🧪 Try in Lab →</Button></Link>
                {!completedLessons.includes(selectedLesson.id) ? (
                  <Button onClick={() => completeLesson(selectedLesson.id)}>Mark as Complete (+50 XP)</Button>
                ) : (
                  <div className="flex items-center text-green-600"><CheckCircle className="w-5 h-5 mr-2" /><span className="font-semibold">Completed</span></div>
                )}
              </div>
            </Card>
          </motion.div>
        )}
      </div>
    </div>
  );
};
