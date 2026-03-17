import React, { useRef, useState, useEffect, useCallback } from 'react';

/* ═══ MODEL LAYERS ═══ */
const LAYERS = [
  { label: 'Token Embedding', shape: '[50257×512]', params: '25.7M', color: '#22c55e', desc: 'Lookup table: every word → 512-dim vector', size: 1.0 },
  { label: 'Positional Encoding', shape: '[512×512]', params: '262K', color: '#06b6d4', desc: 'Sine/cosine waves for position info', size: 0.85 },
  { label: 'Encoder MHA ×6', shape: '6×4×[512×64]', params: '6.3M', color: '#f59e0b', desc: 'Self-attention: W_Q, W_K, W_V, W_O per layer', size: 1.1 },
  { label: 'Encoder FFN ×6', shape: '6×[512×2048×2]', params: '12.6M', color: '#3b82f6', desc: 'Feed-forward: expand→ReLU→compress per layer', size: 1.05 },
  { label: 'Decoder Masked MHA ×6', shape: '6×4×[512×64]', params: '6.3M', color: '#a855f7', desc: 'Masked self-attention (causal) per decoder layer', size: 1.0 },
  { label: 'Decoder Cross MHA ×6', shape: '6×4×[512×64]', params: '6.3M', color: '#ec4899', desc: 'Cross-attention: decoder→encoder per layer', size: 0.95 },
  { label: 'Decoder FFN ×6', shape: '6×[512×2048×2]', params: '12.6M', color: '#f472b6', desc: 'Decoder feed-forward per layer', size: 1.0 },
  { label: 'Linear Output', shape: '[512×50257]', params: '25.7M', color: '#c4b5fd', desc: 'Projects to vocabulary for prediction', size: 0.9 },
  { label: 'Softmax', shape: 'function', params: '0', color: '#86efac', desc: 'Converts logits → probabilities', size: 0.7 },
];

/* ═══ PARTICLES ═══ */
interface Particle { x: number; y: number; z: number; vx: number; vy: number; color: string; life: number; }

function createParticles(count: number): Particle[] {
  const colors = ['#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#ec4899', '#06b6d4'];
  return Array.from({ length: count }, () => ({
    x: (Math.random() - 0.5) * 200, y: Math.random() * 400 - 50, z: (Math.random() - 0.5) * 120,
    vx: (Math.random() - 0.5) * 0.3, vy: 0.4 + Math.random() * 0.6, color: colors[Math.floor(Math.random() * colors.length)], life: Math.random(),
  }));
}

/* ═══ MAIN COMPONENT ═══ */
export default function TransformerMemory3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotY, setRotY] = useState(25);
  const [rotX, setRotX] = useState(-18);
  const [zoom, setZoom] = useState(1.0);
  const [dragging, setDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(-1);
  const [inferStep, setInferStep] = useState(-1);
  const [autoInfer, setAutoInfer] = useState(false);
  const particlesRef = useRef<Particle[]>(createParticles(80));
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  const W = 700, H = 520;

  // 3D → 2D projection
  const project = useCallback((x: number, y: number, z: number) => {
    const ry = (rotY * Math.PI) / 180, rx = (rotX * Math.PI) / 180;
    const x1 = x * Math.cos(ry) - z * Math.sin(ry);
    const z1 = x * Math.sin(ry) + z * Math.cos(ry);
    const y1 = y * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = y * Math.sin(rx) + z1 * Math.cos(rx);
    const fov = 600 * zoom;
    const scale = fov / (fov + z2 + 300);
    return { sx: W / 2 + x1 * scale, sy: H / 2 + y1 * scale, scale, z: z2 };
  }, [rotY, rotX, zoom]);

  // Draw a 3D filled box face
  const drawFace = useCallback((ctx: CanvasRenderingContext2D, pts: { sx: number; sy: number }[], color: string, alpha: number) => {
    ctx.fillStyle = color;
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(pts[0].sx, pts[0].sy);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].sx, pts[i].sy);
    ctx.closePath();
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.strokeStyle = color;
    ctx.globalAlpha = alpha * 0.6;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }, []);

  // Render loop
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const ctx = c.getContext('2d')!;
    const dpr = window.devicePixelRatio || 1;
    c.width = W * dpr; c.height = H * dpr;
    ctx.scale(dpr, dpr);

    const draw = () => {
      timeRef.current += 0.016;
      const t = timeRef.current;
      ctx.clearRect(0, 0, W, H);

      // Background
      const bg = ctx.createLinearGradient(0, 0, 0, H);
      bg.addColorStop(0, '#020617'); bg.addColorStop(0.5, '#0a0a2e'); bg.addColorStop(1, '#020617');
      ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

      // Floor grid
      ctx.strokeStyle = '#1e293b44'; ctx.lineWidth = 0.5;
      for (let i = -150; i <= 150; i += 30) {
        const p1 = project(i, 200, -100), p2 = project(i, 200, 100);
        ctx.beginPath(); ctx.moveTo(p1.sx, p1.sy); ctx.lineTo(p2.sx, p2.sy); ctx.stroke();
        const p3 = project(-150, 200, i * 0.66), p4 = project(150, 200, i * 0.66);
        ctx.beginPath(); ctx.moveTo(p3.sx, p3.sy); ctx.lineTo(p4.sx, p4.sy); ctx.stroke();
      }

      // Glass box outline
      const bw = 160, bh = 360, bd = 100;
      const boxCorners = [
        project(-bw, -bh / 2 + 30, -bd), project(bw, -bh / 2 + 30, -bd),
        project(bw, -bh / 2 + 30, bd), project(-bw, -bh / 2 + 30, bd),
        project(-bw, bh / 2 + 30, -bd), project(bw, bh / 2 + 30, -bd),
        project(bw, bh / 2 + 30, bd), project(-bw, bh / 2 + 30, bd),
      ];
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      ctx.strokeStyle = '#3b82f633'; ctx.lineWidth = 1.5;
      edges.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(boxCorners[a].sx, boxCorners[a].sy); ctx.lineTo(boxCorners[b].sx, boxCorners[b].sy); ctx.stroke(); });

      // Glass faces (semi-transparent)
      drawFace(ctx, [boxCorners[0], boxCorners[1], boxCorners[5], boxCorners[4]], '#1a1a4e', 0.05);
      drawFace(ctx, [boxCorners[1], boxCorners[2], boxCorners[6], boxCorners[5]], '#1a1a4e', 0.04);
      drawFace(ctx, [boxCorners[4], boxCorners[5], boxCorners[6], boxCorners[7]], '#1a2a5e', 0.06);

      // CUDA cores at bottom
      for (let i = 0; i < 30; i++) {
        const cx = (i % 10 - 4.5) * 28, cz = (Math.floor(i / 10) - 1) * 35;
        const p = project(cx, 195, cz);
        const r = 4 * p.scale;
        const pulse = 0.4 + Math.sin(t * 3 + i * 0.4) * 0.3;
        ctx.fillStyle = `rgba(99,102,241,${pulse})`; ctx.beginPath(); ctx.arc(p.sx, p.sy, r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = `rgba(99,102,241,${pulse * 0.5})`; ctx.lineWidth = 0.5; ctx.stroke();
      }
      // CUDA label
      const cudaP = project(0, 210, 80);
      ctx.fillStyle = '#6366f1'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('CUDA Compute Cores', cudaP.sx, cudaP.sy);

      // Particles
      const particles = particlesRef.current;
      const flowing = inferStep >= 0;
      particles.forEach(p => {
        if (flowing) { p.y += p.vy * 1.5; p.x += p.vx; } else { p.y += p.vy * 0.3; }
        if (p.y > 200) { p.y = -50; p.x = (Math.random() - 0.5) * 200; p.z = (Math.random() - 0.5) * 80; }
        const pp = project(p.x, p.y, p.z);
        const size = Math.max(1, 2.5 * pp.scale);
        ctx.fillStyle = p.color; ctx.globalAlpha = flowing ? 0.7 : 0.2;
        ctx.beginPath(); ctx.arc(pp.sx, pp.sy, size, 0, Math.PI * 2); ctx.fill();
      });
      ctx.globalAlpha = 1;

      // Token input stream (left side)
      const inputTokens = ['I', 'love', 'India'];
      inputTokens.forEach((tok, i) => {
        const xOff = -220 - i * 40 + (flowing ? (t * 30 % 160) : 0);
        const p = project(xOff, 140, 0);
        if (p.sx > 50 && p.sx < W - 50) {
          ctx.fillStyle = '#22c55e44'; ctx.strokeStyle = '#22c55e88'; ctx.lineWidth = 1;
          const bx = p.sx - 18, by = p.sy - 8;
          ctx.beginPath(); ctx.roundRect(bx, by, 36, 16, 4); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#22c55e'; ctx.font = `bold ${9 * p.scale}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
          ctx.fillText(tok, p.sx, p.sy + 3);
        }
      });
      // Input label
      const inP = project(-220, 120, 0);
      ctx.fillStyle = '#22c55e'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('Token Input Packets', inP.sx, inP.sy);

      // Output stream (right side)
      const outputTokens = ['मुझे', 'भारत', 'पसंद', 'है'];
      outputTokens.forEach((tok, i) => {
        const xOff = 220 + i * 40 - (flowing ? (t * 25 % 180) : 0);
        const p = project(xOff, -140, 0);
        if (p.sx > 50 && p.sx < W - 50) {
          ctx.fillStyle = '#a855f744'; ctx.strokeStyle = '#a855f788'; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.roundRect(p.sx - 22, p.sy - 8, 44, 16, 4); ctx.fill(); ctx.stroke();
          ctx.fillStyle = '#a855f7'; ctx.font = `bold ${9 * p.scale}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
          ctx.fillText(tok, p.sx, p.sy + 3);
        }
      });
      const outP = project(220, -160, 0);
      ctx.fillStyle = '#a855f7'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('Prediction Probabilities', outP.sx, outP.sy);

      // LAYER BLOCKS — sorted by depth
      const layerRenders: { idx: number; y: number; avgZ: number }[] = [];
      LAYERS.forEach((layer, i) => {
        const yPos = 160 - i * 38;
        const p = project(0, yPos, 0);
        layerRenders.push({ idx: i, y: yPos, avgZ: p.z || 0 });
      });
      layerRenders.sort((a, b) => b.avgZ - a.avgZ);

      layerRenders.forEach(({ idx, y: yPos }) => {
        const layer = LAYERS[idx];
        const w = 110 * layer.size, h = 22, d = 55 * layer.size;
        const isHov = hovered === idx;
        const isActive = inferStep >= 0 && idx <= inferStep;
        const isCurrent = inferStep === idx;
        const bob = Math.sin(t * 1.5 + idx * 0.6) * 2;

        // 8 corners
        const corners = [
          project(-w, yPos + bob - h, -d), project(w, yPos + bob - h, -d),
          project(w, yPos + bob - h, d), project(-w, yPos + bob - h, d),
          project(-w, yPos + bob + h, -d), project(w, yPos + bob + h, -d),
          project(w, yPos + bob + h, d), project(-w, yPos + bob + h, d),
        ];

        const baseAlpha = isActive ? 0.75 : isHov ? 0.65 : 0.35;

        // Top face
        drawFace(ctx, [corners[0], corners[1], corners[5], corners[4]], layer.color, baseAlpha);
        // Front face
        drawFace(ctx, [corners[4], corners[5], corners[6], corners[7]], layer.color, baseAlpha * 0.7);
        // Right face
        drawFace(ctx, [corners[1], corners[2], corners[6], corners[5]], layer.color, baseAlpha * 0.5);

        // Glow for current inference step
        if (isCurrent) {
          ctx.shadowColor = layer.color; ctx.shadowBlur = 20;
          ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2.5; ctx.globalAlpha = 0.9;
          ctx.beginPath();
          ctx.moveTo(corners[0].sx, corners[0].sy);
          [1, 5, 4].forEach(i => ctx.lineTo(corners[i].sx, corners[i].sy));
          ctx.closePath(); ctx.stroke();
          ctx.shadowBlur = 0; ctx.globalAlpha = 1;
        }

        // Label
        const cx = (corners[0].sx + corners[1].sx + corners[5].sx + corners[4].sx) / 4;
        const cy = (corners[0].sy + corners[1].sy + corners[5].sy + corners[4].sy) / 4;
        ctx.fillStyle = isHov || isCurrent ? '#fff' : '#e2e8f0bb';
        ctx.font = `bold ${Math.max(8, 11 * corners[0].scale)}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(layer.label, cx, cy - 3);
        ctx.fillStyle = `${layer.color}cc`;
        ctx.font = `${Math.max(6, 8 * corners[0].scale)}px "JetBrains Mono", monospace`;
        ctx.fillText(`${layer.shape} · ${layer.params}`, cx, cy + 8);

        // Inference flow line
        if (isActive && idx < LAYERS.length - 1 && inferStep > idx) {
          const nextY = 160 - (idx + 1) * 38;
          const from = project(0, yPos + bob - h - 5, 0);
          const to = project(0, nextY + 25, 0);
          ctx.setLineDash([5, 3]);
          ctx.strokeStyle = '#facc1577'; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(from.sx, from.sy); ctx.lineTo(to.sx, to.sy); ctx.stroke();
          ctx.setLineDash([]);
        }
      });

      // Title
      ctx.fillStyle = '#e2e8f0'; ctx.font = `bold ${16 * zoom}px "DM Sans", sans-serif`; ctx.textAlign = 'center';
      ctx.fillText('Transformer Model in GPU Memory', W / 2, 22);
      ctx.fillStyle = '#64748b'; ctx.font = `${10 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('d_model=512 · 8 heads · 6 layers · ~65M parameters · ~260 MB VRAM', W / 2, 38);

      // GPU VRAM label
      const vramP = project(0, 210, -110);
      ctx.fillStyle = '#3b82f6'; ctx.font = `bold ${12 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('GPU VRAM', vramP.sx, vramP.sy);
      ctx.fillStyle = '#475569'; ctx.font = `${9 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('~260 MB (float32) · ~130 MB (float16)', vramP.sx, vramP.sy + 14);

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [rotY, rotX, zoom, hovered, inferStep, project, drawFace]);

  // Mouse handlers
  const onMouseDown = (e: React.MouseEvent) => {
    setDragging(true);
    setLastMouse({ x: e.clientX, y: e.clientY });
  };
  const onMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    if (dragging) {
      const dx = e.clientX - lastMouse.x, dy = e.clientY - lastMouse.y;
      setRotY(p => p + dx * 0.4);
      setRotX(p => Math.max(-50, Math.min(20, p + dy * 0.4)));
      setLastMouse({ x: e.clientX, y: e.clientY });
    } else {
      // Hover detection
      const mx = (e.clientX - rect.left) * (W / rect.width);
      const my = (e.clientY - rect.top) * (H / rect.height);
      let found = -1;
      for (let i = 0; i < LAYERS.length; i++) {
        const yPos = 160 - i * 38;
        const p = project(0, yPos, 0);
        if (Math.abs(mx - p.sx) < 90 * zoom && Math.abs(my - p.sy) < 16 * zoom) { found = i; break; }
      }
      setHovered(found);
    }
  };
  const onMouseUp = () => setDragging(false);
  const onWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    setZoom(p => Math.max(0.5, Math.min(2.2, p - e.deltaY * 0.001)));
  };

  // Auto inference
  useEffect(() => {
    if (autoInfer && inferStep < LAYERS.length - 1) {
      const t = setTimeout(() => setInferStep(p => p + 1), 700);
      return () => clearTimeout(t);
    } else if (inferStep >= LAYERS.length - 1) setAutoInfer(false);
  }, [autoInfer, inferStep]);

  const startInference = () => {
    setInferStep(-1);
    setTimeout(() => { setInferStep(0); setAutoInfer(true); }, 100);
  };

  return (
    <div style={{ position: 'relative', borderRadius: 14, overflow: 'hidden', background: '#020617' }}>
      <canvas ref={canvasRef}
        style={{ width: W, height: H, display: 'block', margin: '0 auto', cursor: dragging ? 'grabbing' : 'grab', maxWidth: '100%', borderRadius: 14 }}
        onMouseDown={onMouseDown} onMouseMove={onMouseMove} onMouseUp={onMouseUp} onMouseLeave={onMouseUp} onWheel={onWheel}
      />

      {/* Controls */}
      <div style={{ position: 'absolute', bottom: 10, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 6 }}>
        <button onClick={startInference}
          style={{ padding: '5px 14px', borderRadius: 7, fontSize: 11, fontWeight: 700, background: autoInfer ? '#dc2626' : '#16a34a', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace" }}>
          {autoInfer ? '⏸ Running...' : '▶ Run Inference'}
        </button>
        <button onClick={() => { setInferStep(-1); setAutoInfer(false); }}
          style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>↺ Reset</button>
        <button onClick={() => { setRotY(25); setRotX(-18); setZoom(1); }}
          style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>⊙ Reset View</button>
      </div>

      <p style={{ position: 'absolute', top: 48, right: 10, fontSize: 8, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>
        🖱️ Drag=rotate · Scroll=zoom
      </p>

      {/* Hover info panel */}
      {hovered >= 0 && LAYERS[hovered] && (
        <div style={{ position: 'absolute', top: 50, left: 10, padding: '10px 14px', borderRadius: 10, background: 'rgba(2,6,23,0.93)', border: `1.5px solid ${LAYERS[hovered].color}`, maxWidth: 260, backdropFilter: 'blur(8px)' }}>
          <p style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: LAYERS[hovered].color, fontWeight: 700, margin: 0 }}>{LAYERS[hovered].label}</p>
          <p style={{ fontSize: 10, color: '#c8d6e5', margin: '4px 0' }}>{LAYERS[hovered].desc}</p>
          <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace" }}>Shape: {LAYERS[hovered].shape}</span>
          <span style={{ fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace", marginLeft: 8 }}>{LAYERS[hovered].params} params</span>
        </div>
      )}

      {/* Inference status */}
      {inferStep >= 0 && (
        <div style={{ position: 'absolute', bottom: 44, left: '50%', transform: 'translateX(-50%)', padding: '4px 12px', borderRadius: 6, background: `${LAYERS[Math.min(inferStep, LAYERS.length - 1)].color}22`, border: `1px solid ${LAYERS[Math.min(inferStep, LAYERS.length - 1)].color}44`, whiteSpace: 'nowrap' }}>
          <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: LAYERS[Math.min(inferStep, LAYERS.length - 1)].color, fontWeight: 700 }}>
            ⚡ {inferStep >= LAYERS.length - 1 ? 'Inference complete! → "मुझे भारत पसंद है"' : `Processing: ${LAYERS[inferStep].label}`}
          </span>
        </div>
      )}
    </div>
  );
}
