import React, { useRef, useState, useEffect, useCallback } from 'react';

/* ═══ LAYER DATA — vertical slabs arranged front-to-back ═══ */
const SLABS = [
  { label: 'Input Embedding', shape: '[50257×512]', params: '25.7M', color: '#22c55e', glow: '#22c55e', type: 'grid', desc: 'Token lookup table — colored grid of 512-dim vectors' },
  { label: 'Positional Encoding', shape: '[512×512]', params: '262K', color: '#06b6d4', glow: '#06b6d4', type: 'wave', desc: 'Sine/cosine position signatures added to embeddings' },
  { label: 'Multi-Head Attention', shape: '4×[512×64]', params: '1.05M', color: '#f59e0b', glow: '#f59e0b', type: 'attention', desc: 'Q,K,V matrices — 8 heads of self-attention', sub: ['Query Tensor', 'Key Tensor', 'Value Tensor'] },
  { label: 'Feed-Forward Network', shape: '[512×2048×2]', params: '2.1M', color: '#3b82f6', glow: '#3b82f6', type: 'dense', desc: 'Two dense layers: expand 512→2048 → compress back' },
  { label: 'Encoder Stack ×6', shape: '6 blocks', params: '18.9M', color: '#2563eb', glow: '#2563eb', type: 'stack', desc: '6 identical encoder blocks (MHA + FFN + norms)' },
  { label: 'Decoder Masked MHA', shape: '4×[512×64]', params: '1.05M', color: '#a855f7', glow: '#a855f7', type: 'attention', desc: 'Causal masked self-attention — can only see past tokens' },
  { label: 'Cross-Attention', shape: '4×[512×64]', params: '1.05M', color: '#ec4899', glow: '#ec4899', type: 'cross', desc: 'Decoder queries attend to encoder keys/values' },
  { label: 'Decoder Stack ×6', shape: '6 blocks', params: '25.2M', color: '#7c3aed', glow: '#7c3aed', type: 'stack', desc: '6 decoder blocks (Masked MHA + Cross MHA + FFN)' },
  { label: 'Linear Output', shape: '[512×50257]', params: '25.7M', color: '#c4b5fd', glow: '#c4b5fd', type: 'dense', desc: 'Projects decoder output to vocabulary logits' },
  { label: 'Softmax Output Layer', shape: 'function', params: '0', color: '#86efac', glow: '#4ade80', type: 'softmax', desc: 'Converts logits to probability distribution' },
];

const W = 800, H = 550;

/* ═══ HELPER: hex to rgba ═══ */
function rgba(hex: string, a: number) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

/* ═══ MAIN COMPONENT ═══ */
export default function TransformerMemory3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotY, setRotY] = useState(28);
  const [rotX, setRotX] = useState(-22);
  const [zoom, setZoom] = useState(1.0);
  const [dragging, setDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(-1);
  const [inferStep, setInferStep] = useState(-1);
  const [autoInfer, setAutoInfer] = useState(false);
  const animRef = useRef(0);
  const tRef = useRef(0);

  // Particles
  const particlesRef = useRef(Array.from({ length: 100 }, () => ({
    x: -350 + Math.random() * 50, y: (Math.random() - 0.5) * 150,
    z: (Math.random() - 0.5) * 100, speed: 0.8 + Math.random() * 1.5,
    color: ['#22c55e', '#06b6d4', '#3b82f6', '#a855f7', '#ec4899', '#f59e0b'][Math.floor(Math.random() * 6)],
    size: 1.5 + Math.random() * 2,
  })));
  const outParticlesRef = useRef(Array.from({ length: 60 }, () => ({
    x: 300 + Math.random() * 50, y: (Math.random() - 0.5) * 120,
    z: (Math.random() - 0.5) * 80, speed: 0.6 + Math.random() * 1.2,
    color: ['#a855f7', '#c4b5fd', '#86efac', '#ec4899'][Math.floor(Math.random() * 4)],
    size: 1 + Math.random() * 2,
  })));

  // 3D projection
  const proj = useCallback((x: number, y: number, z: number) => {
    const ry = (rotY * Math.PI) / 180, rx = (rotX * Math.PI) / 180;
    const x1 = x * Math.cos(ry) - z * Math.sin(ry);
    const z1 = x * Math.sin(ry) + z * Math.cos(ry);
    const y1 = y * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = y * Math.sin(rx) + z1 * Math.cos(rx);
    const f = 700 * zoom;
    const s = f / (f + z2 + 400);
    return { sx: W / 2 + x1 * s, sy: H / 2 + y1 * s, s, z: z2 };
  }, [rotY, rotX, zoom]);

  // Draw
  useEffect(() => {
    const c = canvasRef.current; if (!c) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    c.width = W * dpr; c.height = H * dpr;
    const ctx = c.getContext('2d')!;
    ctx.scale(dpr, dpr);

    const render = () => {
      tRef.current += 0.016;
      const t = tRef.current;
      ctx.clearRect(0, 0, W, H);

      // BG gradient
      const bg = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W);
      bg.addColorStop(0, '#0f0a2a'); bg.addColorStop(0.5, '#060618'); bg.addColorStop(1, '#020210');
      ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

      // Stars
      for (let i = 0; i < 40; i++) {
        const sx = (Math.sin(i * 137.5) * 0.5 + 0.5) * W, sy = (Math.cos(i * 73.2) * 0.5 + 0.5) * H;
        ctx.fillStyle = `rgba(148,163,184,${0.1 + Math.sin(t + i) * 0.08})`;
        ctx.beginPath(); ctx.arc(sx, sy, 0.8, 0, Math.PI * 2); ctx.fill();
      }

      // ─── GLASS CUBE (GPU VRAM) ───
      const bx = 280, by = 180, bz = 160;
      const cubeCorners = [
        proj(-bx, -by, -bz), proj(bx, -by, -bz), proj(bx, -by, bz), proj(-bx, -by, bz),
        proj(-bx, by, -bz), proj(bx, by, -bz), proj(bx, by, bz), proj(-bx, by, bz),
      ];

      // Glass faces
      const drawGlassFace = (indices: number[], alpha: number) => {
        ctx.fillStyle = `rgba(20,20,80,${alpha})`;
        ctx.beginPath(); ctx.moveTo(cubeCorners[indices[0]].sx, cubeCorners[indices[0]].sy);
        indices.slice(1).forEach(i => ctx.lineTo(cubeCorners[i].sx, cubeCorners[i].sy));
        ctx.closePath(); ctx.fill();
      };
      drawGlassFace([0, 1, 5, 4], 0.04); // front
      drawGlassFace([1, 2, 6, 5], 0.03); // right
      drawGlassFace([4, 5, 6, 7], 0.06); // top
      drawGlassFace([3, 0, 4, 7], 0.03); // left
      drawGlassFace([0, 1, 2, 3], 0.05); // bottom

      // Glass edges — neon blue
      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
      ctx.strokeStyle = 'rgba(59,130,246,0.25)'; ctx.lineWidth = 1.2;
      edges.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(cubeCorners[a].sx, cubeCorners[a].sy); ctx.lineTo(cubeCorners[b].sx, cubeCorners[b].sy); ctx.stroke(); });

      // Subtle glass reflections
      ctx.strokeStyle = 'rgba(99,130,246,0.08)'; ctx.lineWidth = 0.5;
      for (let i = 0; i < 5; i++) {
        const yy = -by + (by * 2 / 5) * i;
        const p1 = proj(-bx, yy, -bz), p2 = proj(bx, yy, -bz);
        ctx.beginPath(); ctx.moveTo(p1.sx, p1.sy); ctx.lineTo(p2.sx, p2.sy); ctx.stroke();
      }

      // ─── INPUT PARTICLES (left side, flowing in) ───
      const flowing = inferStep >= 0;
      const particles = particlesRef.current;
      particles.forEach(p => {
        p.x += p.speed * (flowing ? 2.5 : 0.8);
        if (p.x > 280) { p.x = -380; p.y = (Math.random() - 0.5) * 150; p.z = (Math.random() - 0.5) * 100; }
        const pp = proj(p.x, p.y, p.z);
        const sz = p.size * pp.s;
        ctx.fillStyle = rgba(p.color, flowing ? 0.8 : 0.3);
        ctx.shadowColor = p.color; ctx.shadowBlur = flowing ? 6 : 2;
        ctx.beginPath(); ctx.arc(pp.sx, pp.sy, sz, 0, Math.PI * 2); ctx.fill();
      });
      ctx.shadowBlur = 0;

      // ─── SLABS (vertical plates, arranged left-to-right inside the cube) ───
      const slabW = 180, slabH = 260;
      const totalSlabs = SLABS.length;
      const spacing = (bx * 2 - 40) / totalSlabs;

      // Sort by depth
      const slabRenders = SLABS.map((slab, i) => {
        const xCenter = -bx + 30 + spacing * i + spacing / 2;
        const p = proj(xCenter, 0, 0);
        return { slab, i, xCenter, depth: p.z };
      });
      slabRenders.sort((a, b) => b.depth - a.depth);

      slabRenders.forEach(({ slab, i, xCenter }) => {
        const isHov = hovered === i;
        const isActive = inferStep >= 0 && i <= inferStep;
        const isCurrent = inferStep === i;
        const bob = Math.sin(t * 1.2 + i * 0.7) * 3;

        // Slab as a thin vertical plane
        const hw = 4; // half-width (thin)
        const hh = slabH / 2, hd = slabW / 2;
        const corners = [
          proj(xCenter - hw, -hh + bob, -hd), proj(xCenter + hw, -hh + bob, -hd),
          proj(xCenter + hw, -hh + bob, hd), proj(xCenter - hw, -hh + bob, hd),
          proj(xCenter - hw, hh + bob, -hd), proj(xCenter + hw, hh + bob, -hd),
          proj(xCenter + hw, hh + bob, hd), proj(xCenter - hw, hh + bob, hd),
        ];

        const alpha = isCurrent ? 0.85 : isActive ? 0.6 : isHov ? 0.55 : 0.25;

        // Front face (main visible face of the slab)
        ctx.fillStyle = rgba(slab.color, alpha);
        ctx.beginPath();
        ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[3].sx, corners[3].sy);
        ctx.lineTo(corners[7].sx, corners[7].sy); ctx.lineTo(corners[4].sx, corners[4].sy);
        ctx.closePath(); ctx.fill();

        // Neon edge glow
        ctx.strokeStyle = rgba(slab.glow, isHov || isCurrent ? 0.9 : 0.4);
        ctx.lineWidth = isCurrent ? 2.5 : isHov ? 2 : 1;
        ctx.shadowColor = slab.glow; ctx.shadowBlur = isCurrent ? 15 : isHov ? 8 : 0;
        ctx.stroke();
        ctx.shadowBlur = 0;

        // Side face (gives depth)
        ctx.fillStyle = rgba(slab.color, alpha * 0.4);
        ctx.beginPath();
        ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[1].sx, corners[1].sy);
        ctx.lineTo(corners[5].sx, corners[5].sy); ctx.lineTo(corners[4].sx, corners[4].sy);
        ctx.closePath(); ctx.fill();

        // Top face
        ctx.fillStyle = rgba(slab.color, alpha * 0.3);
        ctx.beginPath();
        ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[1].sx, corners[1].sy);
        ctx.lineTo(corners[2].sx, corners[2].sy); ctx.lineTo(corners[3].sx, corners[3].sy);
        ctx.closePath(); ctx.fill();

        // Grid pattern on front face (for embedding/attention types)
        if (slab.type === 'grid' || slab.type === 'attention' || slab.type === 'dense') {
          const rows = 8, cols = 6;
          for (let r = 0; r < rows; r++) for (let cc = 0; cc < cols; cc++) {
            const yy = -hh + bob + (hh * 2 / rows) * r + (hh * 2 / rows) * 0.5;
            const zz = -hd + (hd * 2 / cols) * cc + (hd * 2 / cols) * 0.5;
            const gp = proj(xCenter - hw - 1, yy, zz);
            const gs = Math.max(2, 5 * gp.s);
            const pulse = slab.type === 'attention' ? Math.sin(t * 3 + r + cc * 2) * 0.3 + 0.5 : 0.3 + Math.random() * 0.2;
            ctx.fillStyle = rgba(slab.color, pulse * (isActive ? 1.5 : 0.6));
            ctx.fillRect(gp.sx - gs / 2, gp.sy - gs / 2, gs, gs);
          }
        }

        // Wave pattern for softmax/PE
        if (slab.type === 'wave' || slab.type === 'softmax') {
          ctx.strokeStyle = rgba(slab.color, isActive ? 0.8 : 0.3); ctx.lineWidth = 1.5;
          ctx.beginPath();
          for (let z = -hd; z <= hd; z += 5) {
            const yy = Math.sin(z * 0.05 + t * 2) * 30 + bob;
            const wp = proj(xCenter - hw - 1, yy, z);
            z === -hd ? ctx.moveTo(wp.sx, wp.sy) : ctx.lineTo(wp.sx, wp.sy);
          }
          ctx.stroke();
        }

        // Label on the slab
        const labelP = proj(xCenter, -hh + bob - 15, 0);
        ctx.fillStyle = isHov || isCurrent ? '#fff' : rgba(slab.color, 0.9);
        ctx.font = `bold ${Math.max(7, 10 * labelP.s)}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(slab.label, labelP.sx, labelP.sy);

        // Sub-labels (Q, K, V for attention)
        if (slab.sub && (isHov || isActive)) {
          slab.sub.forEach((sub: string, si: number) => {
            const sy = -hh / 2 + bob + si * 35;
            const sp = proj(xCenter - hw - 2, sy, 0);
            ctx.fillStyle = rgba(slab.color, 0.7);
            ctx.font = `bold ${Math.max(6, 8 * sp.s)}px "JetBrains Mono", monospace`;
            ctx.fillText(sub, sp.sx, sp.sy);
          });
        }

        // Inference glow current
        if (isCurrent) {
          ctx.strokeStyle = '#facc15';
          ctx.lineWidth = 3;
          ctx.shadowColor = '#facc15'; ctx.shadowBlur = 20;
          ctx.beginPath();
          ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[3].sx, corners[3].sy);
          ctx.lineTo(corners[7].sx, corners[7].sy); ctx.lineTo(corners[4].sx, corners[4].sy);
          ctx.closePath(); ctx.stroke();
          ctx.shadowBlur = 0;
        }

        // Connection beams between slabs during inference
        if (isActive && i < totalSlabs - 1 && inferStep > i) {
          const nextX = -bx + 30 + spacing * (i + 1) + spacing / 2;
          const from = proj(xCenter + hw + 2, bob, 0);
          const to = proj(nextX - hw - 2, Math.sin(t * 1.2 + (i + 1) * 0.7) * 3, 0);
          ctx.strokeStyle = rgba('#facc15', 0.3); ctx.lineWidth = 1.5;
          ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(from.sx, from.sy); ctx.lineTo(to.sx, to.sy); ctx.stroke();
          ctx.setLineDash([]);
          // Beam glow
          const grd = ctx.createLinearGradient(from.sx, from.sy, to.sx, to.sy);
          grd.addColorStop(0, rgba(slab.color, 0.15)); grd.addColorStop(1, rgba(SLABS[i + 1].color, 0.15));
          ctx.strokeStyle = grd; ctx.lineWidth = 8; ctx.globalAlpha = 0.3;
          ctx.beginPath(); ctx.moveTo(from.sx, from.sy); ctx.lineTo(to.sx, to.sy); ctx.stroke();
          ctx.globalAlpha = 1;
        }
      });

      // ─── OUTPUT PARTICLES (right side, flowing out) ───
      const outP = outParticlesRef.current;
      outP.forEach(p => {
        p.x += p.speed * (flowing ? 2 : 0.5);
        if (p.x > W) { p.x = 280; p.y = (Math.random() - 0.5) * 120; p.z = (Math.random() - 0.5) * 80; }
        const pp = proj(p.x, p.y, p.z);
        ctx.fillStyle = rgba(p.color, flowing ? 0.7 : 0.2);
        ctx.shadowColor = p.color; ctx.shadowBlur = flowing ? 5 : 0;
        ctx.beginPath(); ctx.arc(pp.sx, pp.sy, p.size * pp.s, 0, Math.PI * 2); ctx.fill();
      });
      ctx.shadowBlur = 0;

      // ─── CUDA CORES (bottom) ───
      for (let i = 0; i < 24; i++) {
        const cx = -180 + (i % 8) * 52, cz = 100 + Math.floor(i / 8) * 30;
        const p = proj(cx, by + 20, cz);
        const r = 6 * p.s;
        const pulse = 0.3 + Math.sin(t * 2.5 + i * 0.5) * 0.25;
        // Cylinder shape
        ctx.fillStyle = rgba('#6366f1', pulse);
        ctx.beginPath(); ctx.ellipse(p.sx, p.sy, r * 1.3, r * 0.6, 0, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = rgba('#818cf8', pulse * 0.7); ctx.lineWidth = 0.8; ctx.stroke();
        // Glow
        ctx.fillStyle = rgba('#6366f1', pulse * 0.3);
        ctx.beginPath(); ctx.ellipse(p.sx, p.sy - 4 * p.s, r * 1.3, r * 0.6, 0, 0, Math.PI * 2); ctx.fill();
      }

      // ─── LABELS ───
      // GPU VRAM
      const vramP = proj(0, by + 50, 80);
      ctx.fillStyle = '#3b82f6'; ctx.font = `bold ${14 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('GPU VRAM', vramP.sx, vramP.sy);
      ctx.fillStyle = '#475569'; ctx.font = `${9 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('~260 MB (float32) · ~130 MB (float16)', vramP.sx, vramP.sy + 16);

      // CUDA
      const cudaP = proj(80, by + 50, 140);
      ctx.fillStyle = '#818cf8'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('CUDA compute core engines', cudaP.sx, cudaP.sy);

      // Token input
      const inLbl = proj(-350, -100, 0);
      ctx.fillStyle = '#22c55e'; ctx.font = `bold ${11 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('Token input', inLbl.sx, inLbl.sy);
      ctx.fillText('packets', inLbl.sx, inLbl.sy + 14);

      // Prediction output
      const outLbl = proj(380, -100, 0);
      ctx.fillStyle = '#a855f7'; ctx.font = `bold ${11 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('prediction token', outLbl.sx, outLbl.sy);
      ctx.fillText('probabilities', outLbl.sx, outLbl.sy + 14);

      // Title
      ctx.fillStyle = '#e2e8f0'; ctx.font = `bold ${15 * zoom}px "DM Sans", sans-serif`; ctx.textAlign = 'center';
      ctx.fillText('Transformer Model in GPU Memory', W / 2, 24);
      ctx.fillStyle = '#64748b'; ctx.font = `${9 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('d_model=512 · 8 heads · 6 encoder + 6 decoder layers · ~65M params', W / 2, 40);

      // Architecture mini-diagram (top-left)
      const mx = 28, my = 60;
      ctx.fillStyle = 'rgba(15,23,42,0.8)'; ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(mx, my, 115, 110, 6); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#94a3b8'; ctx.font = `bold 7px "JetBrains Mono", monospace`; ctx.textAlign = 'left';
      ctx.fillText('TRANSFORMER', mx + 6, my + 12);
      ctx.fillText('NETWORK', mx + 6, my + 21);
      const miniBlocks = [
        { l: 'Word ID', c: '#fca5a5', y: 95 }, { l: 'Pos Embed', c: '#67e8f9', y: 82 },
        { l: 'Multi-Head Attn', c: '#fdba74', y: 69 }, { l: 'Feed Forward', c: '#93c5fd', y: 56 },
      ];
      miniBlocks.forEach(b => {
        ctx.fillStyle = b.c + '44'; ctx.strokeStyle = b.c + '88'; ctx.lineWidth = 0.8;
        ctx.beginPath(); ctx.roundRect(mx + 6, my + (155 - b.y), 100, 11, 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = b.c; ctx.font = `bold 6px "JetBrains Mono", monospace`;
        ctx.fillText(b.l, mx + 10, my + (155 - b.y) + 8);
      });

      animRef.current = requestAnimationFrame(render);
    };

    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [rotY, rotX, zoom, hovered, inferStep, proj]);

  // Mouse
  const onDown = (e: React.MouseEvent) => { setDragging(true); setLastMouse({ x: e.clientX, y: e.clientY }); };
  const onMove = (e: React.MouseEvent) => {
    if (dragging) {
      setRotY(p => p + (e.clientX - lastMouse.x) * 0.35);
      setRotX(p => Math.max(-45, Math.min(15, p + (e.clientY - lastMouse.y) * 0.35)));
      setLastMouse({ x: e.clientX, y: e.clientY });
    } else {
      const rect = canvasRef.current?.getBoundingClientRect(); if (!rect) return;
      const mx = (e.clientX - rect.left) * (W / rect.width);
      let found = -1;
      const spacing = (280 * 2 - 40) / SLABS.length;
      for (let i = 0; i < SLABS.length; i++) {
        const xc = -280 + 30 + spacing * i + spacing / 2;
        const p = proj(xc, 0, 0);
        if (Math.abs(mx - p.sx) < 30 * zoom) { found = i; break; }
      }
      setHovered(found);
    }
  };
  const onUp = () => setDragging(false);
  const onWheel = (e: React.WheelEvent) => { e.preventDefault(); setZoom(p => Math.max(0.5, Math.min(2, p - e.deltaY * 0.001))); };

  // Inference
  useEffect(() => {
    if (autoInfer && inferStep < SLABS.length - 1) {
      const t = setTimeout(() => setInferStep(p => p + 1), 650);
      return () => clearTimeout(t);
    } else if (inferStep >= SLABS.length - 1) setAutoInfer(false);
  }, [autoInfer, inferStep]);

  return (
    <div style={{ position: 'relative', borderRadius: 14, overflow: 'hidden', background: '#020210' }}>
      <canvas ref={canvasRef} style={{ width: W, height: H, display: 'block', margin: '0 auto', cursor: dragging ? 'grabbing' : 'grab', maxWidth: '100%', borderRadius: 14 }}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp} onWheel={onWheel} />

      {/* Controls */}
      <div style={{ position: 'absolute', bottom: 10, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 6 }}>
        <button onClick={() => { setInferStep(-1); setTimeout(() => { setInferStep(0); setAutoInfer(true); }, 100); }}
          style={{ padding: '5px 14px', borderRadius: 7, fontSize: 11, fontWeight: 700, background: autoInfer ? '#dc2626' : '#16a34a', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace" }}>
          {autoInfer ? '⏸ Running...' : '▶ Run Inference'}
        </button>
        <button onClick={() => { setInferStep(-1); setAutoInfer(false); }} style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: 'rgba(15,23,42,0.8)', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>↺</button>
        <button onClick={() => { setRotY(28); setRotX(-22); setZoom(1); }} style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: 'rgba(15,23,42,0.8)', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>⊙</button>
      </div>

      <p style={{ position: 'absolute', top: 48, right: 10, fontSize: 8, color: '#47556988', fontFamily: "'JetBrains Mono', monospace" }}>drag=rotate · scroll=zoom</p>

      {/* Hover info */}
      {hovered >= 0 && SLABS[hovered] && (
        <div style={{ position: 'absolute', top: 55, right: 10, padding: '10px 14px', borderRadius: 10, background: 'rgba(2,2,16,0.93)', border: `1.5px solid ${SLABS[hovered].color}`, maxWidth: 240, backdropFilter: 'blur(8px)' }}>
          <p style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: SLABS[hovered].color, fontWeight: 700, margin: 0 }}>{SLABS[hovered].label}</p>
          <p style={{ fontSize: 10, color: '#c8d6e5', margin: '4px 0' }}>{SLABS[hovered].desc}</p>
          <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace" }}>{SLABS[hovered].shape}</span>
          <span style={{ fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace", marginLeft: 8 }}>{SLABS[hovered].params} params</span>
        </div>
      )}

      {/* Inference status */}
      {inferStep >= 0 && (
        <div style={{ position: 'absolute', bottom: 44, left: '50%', transform: 'translateX(-50%)', padding: '4px 12px', borderRadius: 6, background: `${SLABS[Math.min(inferStep, SLABS.length - 1)].color}22`, border: `1px solid ${SLABS[Math.min(inferStep, SLABS.length - 1)].color}44`, whiteSpace: 'nowrap' }}>
          <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: SLABS[Math.min(inferStep, SLABS.length - 1)].color, fontWeight: 700 }}>
            ⚡ {inferStep >= SLABS.length - 1 ? 'Complete! "I love India" → "मुझे भारत पसंद है"' : `${SLABS[inferStep].label}`}
          </span>
        </div>
      )}
    </div>
  );
}
