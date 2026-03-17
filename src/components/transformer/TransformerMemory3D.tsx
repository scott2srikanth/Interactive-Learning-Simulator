import React, { useRef, useState, useEffect, useCallback } from 'react';

const SLABS = [
  { label: 'Input Embedding', shape: '[50257×512]', params: '25.7M', color: '#22c55e', glow: '#22c55e', type: 'grid', desc: 'Token lookup table', phase: 'encoder' },
  { label: 'Positional Encoding', shape: '[512×512]', params: '262K', color: '#06b6d4', glow: '#06b6d4', type: 'wave', desc: 'Sine/cosine positions', phase: 'encoder' },
  { label: 'Multi-Head Attention', shape: '4×[512×64]', params: '1.05M', color: '#f59e0b', glow: '#f59e0b', type: 'attention', desc: 'Self-attention Q,K,V', phase: 'encoder', sub: ['Query', 'Key', 'Value'] },
  { label: 'Feed-Forward Net', shape: '[512×2048×2]', params: '2.1M', color: '#3b82f6', glow: '#3b82f6', type: 'ffn', desc: 'Expand→ReLU→Compress', phase: 'encoder' },
  { label: 'Encoder ×6', shape: '6 blocks', params: '18.9M', color: '#2563eb', glow: '#2563eb', type: 'stack', desc: '6 encoder layers', phase: 'encoder' },
  { label: 'Masked MHA', shape: '4×[512×64]', params: '1.05M', color: '#a855f7', glow: '#a855f7', type: 'masked', desc: 'Causal mask attention', phase: 'decoder' },
  { label: 'Cross-Attention', shape: '4×[512×64]', params: '1.05M', color: '#ec4899', glow: '#ec4899', type: 'cross', desc: 'Dec→Enc attention', phase: 'decoder' },
  { label: 'Decoder ×6', shape: '6 blocks', params: '25.2M', color: '#7c3aed', glow: '#7c3aed', type: 'stack', desc: '6 decoder layers', phase: 'decoder' },
  { label: 'Linear', shape: '[512×50257]', params: '25.7M', color: '#c4b5fd', glow: '#c4b5fd', type: 'linear', desc: 'Project to vocab', phase: 'decoder' },
  { label: 'Softmax', shape: 'function', params: '0', color: '#86efac', glow: '#4ade80', type: 'softmax', desc: 'Logits→Probabilities', phase: 'decoder' },
];

const TGT_TOKENS = ['मुझे', 'भारत', 'पसंद', 'है'];
const SRC_TOKENS = ['I', 'love', 'India'];
const W = 800, H = 560;

function rgba(hex: string, a: number) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}

export default function TransformerMemory3D() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [rotY, setRotY] = useState(28);
  const [rotX, setRotX] = useState(-22);
  const [zoom, setZoom] = useState(1.0);
  const [dragging, setDragging] = useState(false);
  const [lastMouse, setLastMouse] = useState({ x: 0, y: 0 });
  const [hovered, setHovered] = useState(-1);

  // Inference state
  const [inferStep, setInferStep] = useState(-1); // current slab index
  const [decoderPass, setDecoderPass] = useState(0); // which target token (0..3)
  const [generatedTokens, setGeneratedTokens] = useState<string[]>([]);
  const [inferPhase, setInferPhase] = useState<'idle' | 'encoder' | 'decoder' | 'done'>('idle');
  const [autoInfer, setAutoInfer] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [statusColor, setStatusColor] = useState('#94a3b8');

  const animRef = useRef(0);
  const tRef = useRef(0);

  const particlesRef = useRef(Array.from({ length: 100 }, () => ({
    x: -350 + Math.random() * 50, y: (Math.random() - 0.5) * 150,
    z: (Math.random() - 0.5) * 100, speed: 0.8 + Math.random() * 1.5,
    color: ['#22c55e', '#06b6d4', '#3b82f6', '#a855f7', '#ec4899', '#f59e0b'][Math.floor(Math.random() * 6)],
    size: 1.5 + Math.random() * 2,
  })));

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

  /* ═══ INFERENCE ENGINE ═══ */
  useEffect(() => {
    if (!autoInfer) return;

    if (inferPhase === 'encoder') {
      // Walk through encoder slabs (0..4)
      if (inferStep < 4) {
        const delay = inferStep < 0 ? 200 : 750;
        const t = setTimeout(() => {
          const next = inferStep + 1;
          setInferStep(next);
          setStatusText(`Encoder: ${SLABS[next].label}`);
          setStatusColor(SLABS[next].color);
        }, delay);
        return () => clearTimeout(t);
      } else {
        // Encoder done → start decoder pass 0
        const t = setTimeout(() => {
          setInferPhase('decoder');
          setDecoderPass(0);
          setInferStep(5); // first decoder slab
          setStatusText(`Decoder pass 1: generating "${TGT_TOKENS[0]}"...`);
          setStatusColor('#a855f7');
        }, 600);
        return () => clearTimeout(t);
      }
    }

    if (inferPhase === 'decoder') {
      // Walk through decoder slabs (5..9) for current token
      if (inferStep < 9) {
        const t = setTimeout(() => {
          const next = inferStep + 1;
          setInferStep(next);
          setStatusText(`Decoder pass ${decoderPass + 1}: ${SLABS[next].label}`);
          setStatusColor(SLABS[next].color);
        }, 550);
        return () => clearTimeout(t);
      } else {
        // Softmax done → emit token
        const t = setTimeout(() => {
          const newToken = TGT_TOKENS[decoderPass];
          setGeneratedTokens(prev => [...prev, newToken]);
          setStatusText(`✅ Generated: "${newToken}"`);
          setStatusColor('#22c55e');

          if (decoderPass < TGT_TOKENS.length - 1) {
            // More tokens to generate → restart decoder pass
            setTimeout(() => {
              setDecoderPass(prev => prev + 1);
              setInferStep(5); // reset to first decoder slab
              setStatusText(`Decoder pass ${decoderPass + 2}: generating "${TGT_TOKENS[decoderPass + 1]}"...`);
              setStatusColor('#a855f7');
            }, 800);
          } else {
            // All tokens generated
            setTimeout(() => {
              setInferPhase('done');
              setAutoInfer(false);
              setStatusText(`🎉 Complete! "I love India" → "${TGT_TOKENS.join(' ')}"`);
              setStatusColor('#22c55e');
            }, 600);
          }
        }, 500);
        return () => clearTimeout(t);
      }
    }
  }, [autoInfer, inferPhase, inferStep, decoderPass]);

  const startInference = () => {
    setInferStep(-1); setDecoderPass(0); setGeneratedTokens([]); setInferPhase('encoder'); setAutoInfer(true);
    setStatusText('Starting encoder...'); setStatusColor('#22c55e');
  };
  const resetInference = () => {
    setInferStep(-1); setDecoderPass(0); setGeneratedTokens([]); setInferPhase('idle'); setAutoInfer(false);
    setStatusText(''); setStatusColor('#94a3b8');
  };

  /* ═══ RENDER ═══ */
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

      // BG
      const bg = ctx.createRadialGradient(W / 2, H / 2, 50, W / 2, H / 2, W);
      bg.addColorStop(0, '#0f0a2a'); bg.addColorStop(0.5, '#060618'); bg.addColorStop(1, '#020210');
      ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);

      const flowing = inferPhase !== 'idle';
      const bx = 280, by = 180, bz = 160;

      // Glass cube
      const cc = [proj(-bx,-by,-bz),proj(bx,-by,-bz),proj(bx,-by,bz),proj(-bx,-by,bz),proj(-bx,by,-bz),proj(bx,by,-bz),proj(bx,by,bz),proj(-bx,by,bz)];
      const drawFace = (ids: number[], a: number) => { ctx.fillStyle = `rgba(20,20,80,${a})`; ctx.beginPath(); ctx.moveTo(cc[ids[0]].sx, cc[ids[0]].sy); ids.slice(1).forEach(i => ctx.lineTo(cc[i].sx, cc[i].sy)); ctx.closePath(); ctx.fill(); };
      drawFace([0,1,5,4], 0.04); drawFace([1,2,6,5], 0.03); drawFace([4,5,6,7], 0.06);
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(([a,b]) => {
        ctx.strokeStyle = 'rgba(59,130,246,0.2)'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cc[a].sx, cc[a].sy); ctx.lineTo(cc[b].sx, cc[b].sy); ctx.stroke();
      });

      // CUDA cores
      for (let i = 0; i < 24; i++) {
        const cx = -180 + (i % 8) * 52, cz = 100 + Math.floor(i / 8) * 30;
        const p = proj(cx, by + 20, cz); const r = 6 * p.s;
        const pulse = 0.3 + Math.sin(t * 2.5 + i * 0.5) * 0.25;
        ctx.fillStyle = rgba('#6366f1', flowing ? pulse * 1.5 : pulse);
        ctx.beginPath(); ctx.ellipse(p.sx, p.sy, r * 1.3, r * 0.6, 0, 0, Math.PI * 2); ctx.fill();
      }

      // Input particles
      particlesRef.current.forEach(p => {
        p.x += p.speed * (flowing ? 2.5 : 0.6);
        if (p.x > 280) { p.x = -380; p.y = (Math.random() - 0.5) * 150; p.z = (Math.random() - 0.5) * 100; }
        const pp = proj(p.x, p.y, p.z); const sz = p.size * pp.s;
        ctx.fillStyle = rgba(p.color, flowing ? 0.7 : 0.2);
        if (flowing) { ctx.shadowColor = p.color; ctx.shadowBlur = 4; }
        ctx.beginPath(); ctx.arc(pp.sx, pp.sy, sz, 0, Math.PI * 2); ctx.fill();
        ctx.shadowBlur = 0;
      });

      // ─── SLABS ───
      const slabH = 260, slabW = 180, hw = 4;
      const spacing = (bx * 2 - 40) / SLABS.length;
      const slabData = SLABS.map((slab, i) => {
        const xc = -bx + 30 + spacing * i + spacing / 2;
        const p = proj(xc, 0, 0);
        return { slab, i, xc, depth: p.z };
      });
      slabData.sort((a, b) => b.depth - a.depth);

      slabData.forEach(({ slab, i, xc }) => {
        const isHov = hovered === i;
        const isCurrent = inferStep === i;
        const isActive = inferStep >= i && ((slab.phase === 'encoder' && inferPhase !== 'idle') || (slab.phase === 'decoder' && inferPhase === 'decoder'));
        const bob = Math.sin(t * 1.2 + i * 0.7) * 3;
        const hh = slabH / 2, hd = slabW / 2;

        const corners = [
          proj(xc - hw, -hh + bob, -hd), proj(xc + hw, -hh + bob, -hd),
          proj(xc + hw, -hh + bob, hd), proj(xc - hw, -hh + bob, hd),
          proj(xc - hw, hh + bob, -hd), proj(xc + hw, hh + bob, -hd),
          proj(xc + hw, hh + bob, hd), proj(xc - hw, hh + bob, hd),
        ];

        const alpha = isCurrent ? 0.85 : isActive ? 0.5 : isHov ? 0.5 : 0.2;

        // Front face
        ctx.fillStyle = rgba(slab.color, alpha);
        ctx.beginPath(); ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[3].sx, corners[3].sy);
        ctx.lineTo(corners[7].sx, corners[7].sy); ctx.lineTo(corners[4].sx, corners[4].sy);
        ctx.closePath(); ctx.fill();
        // Edge
        ctx.strokeStyle = rgba(slab.glow, isCurrent ? 0.9 : isHov ? 0.7 : 0.3);
        ctx.lineWidth = isCurrent ? 2.5 : 1;
        if (isCurrent) { ctx.shadowColor = slab.glow; ctx.shadowBlur = 18; }
        ctx.stroke(); ctx.shadowBlur = 0;
        // Side + top
        ctx.fillStyle = rgba(slab.color, alpha * 0.35);
        ctx.beginPath(); ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[1].sx, corners[1].sy); ctx.lineTo(corners[5].sx, corners[5].sy); ctx.lineTo(corners[4].sx, corners[4].sy); ctx.closePath(); ctx.fill();
        ctx.fillStyle = rgba(slab.color, alpha * 0.25);
        ctx.beginPath(); ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[1].sx, corners[1].sy); ctx.lineTo(corners[2].sx, corners[2].sy); ctx.lineTo(corners[3].sx, corners[3].sy); ctx.closePath(); ctx.fill();

        /* ═══ PER-LAYER UNIQUE ANIMATION (when current) ═══ */
        if (isCurrent) {
          const faceX = xc - hw - 1;

          if (slab.type === 'grid') {
            // EMBEDDING: colored cells cascade from top-left, each cell a different color representing a word vector dimension
            const rows = 10, cols = 8;
            for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) {
              const yy = -hh + bob + (hh * 2 / rows) * r + 10;
              const zz = -hd + (hd * 2 / cols) * c + 10;
              const gp = proj(faceX, yy, zz);
              const gs = Math.max(3, 7 * gp.s);
              const wave = Math.sin(t * 5 - r * 0.3 - c * 0.3);
              const brightness = wave > 0 ? wave : 0;
              const hue = ((r * cols + c) * 25 + t * 40) % 360;
              ctx.fillStyle = `hsla(${hue}, 80%, ${40 + brightness * 40}%, ${0.5 + brightness * 0.4})`;
              ctx.fillRect(gp.sx - gs / 2, gp.sy - gs / 2, gs, gs);
            }
          }

          if (slab.type === 'wave') {
            // PE: multiple sine waves at different frequencies overlapping, sweeping across
            for (let w = 0; w < 4; w++) {
              ctx.strokeStyle = rgba(slab.color, 0.5 + w * 0.1); ctx.lineWidth = 1.5;
              ctx.beginPath();
              for (let z = -hd; z <= hd; z += 3) {
                const freq = 0.03 + w * 0.02;
                const amp = 25 + w * 10;
                const yy = Math.sin(z * freq + t * (2 + w * 0.5) + w) * amp + bob;
                const wp = proj(faceX, yy, z);
                z === -hd ? ctx.moveTo(wp.sx, wp.sy) : ctx.lineTo(wp.sx, wp.sy);
              }
              ctx.stroke();
            }
          }

          if (slab.type === 'attention') {
            // MHA: animated Q·K dot-product rays fanning out from center, then converging to V
            const numRays = 12;
            for (let r = 0; r < numRays; r++) {
              const angle = (r / numRays) * Math.PI * 2 + t * 1.5;
              const fromY = bob, fromZ = 0;
              const toY = Math.sin(angle) * hh * 0.7 + bob;
              const toZ = Math.cos(angle) * hd * 0.7;
              const fp = proj(faceX, fromY, fromZ), tp = proj(faceX, toY, toZ);
              const progress = (Math.sin(t * 4 + r) + 1) / 2;
              ctx.strokeStyle = rgba(slab.color, 0.3 + progress * 0.5);
              ctx.lineWidth = 1 + progress;
              ctx.beginPath(); ctx.moveTo(fp.sx, fp.sy); ctx.lineTo(tp.sx, tp.sy); ctx.stroke();
              // Dot at endpoint
              ctx.fillStyle = rgba('#facc15', progress);
              ctx.beginPath(); ctx.arc(tp.sx, tp.sy, 2 + progress * 2, 0, Math.PI * 2); ctx.fill();
            }
            // Q K V labels
            ['Q', 'K', 'V'].forEach((l, li) => {
              const lp = proj(faceX, -hh * 0.5 + bob + li * hh * 0.5, 0);
              ctx.fillStyle = ['#ef4444', '#22c55e', '#3b82f6'][li];
              ctx.font = `bold ${Math.max(10, 14 * lp.s)}px "JetBrains Mono", monospace`;
              ctx.textAlign = 'center'; ctx.fillText(l, lp.sx, lp.sy);
            });
          }

          if (slab.type === 'ffn') {
            // FFN: neurons expanding (512→2048) then compressing, with ReLU activation glow
            const layers = [4, 10, 4]; // visual neuron counts per column
            const colW = hd * 2 / (layers.length + 1);
            layers.forEach((n, col) => {
              for (let ni = 0; ni < n; ni++) {
                const yy = -hh * 0.6 + bob + (hh * 1.2 / n) * ni + (hh * 1.2 / n) * 0.5;
                const zz = -hd + colW * (col + 1);
                const np = proj(faceX, yy, zz);
                const activation = Math.max(0, Math.sin(t * 3 + ni + col * 2)); // ReLU
                const r = (3 + activation * 4) * np.s;
                ctx.fillStyle = rgba(col === 1 ? '#f59e0b' : '#3b82f6', 0.3 + activation * 0.6);
                ctx.beginPath(); ctx.arc(np.sx, np.sy, r, 0, Math.PI * 2); ctx.fill();
                // Connections to next layer
                if (col < layers.length - 1) {
                  const nextN = layers[col + 1];
                  for (let nj = 0; nj < nextN; nj++) {
                    const ny = -hh * 0.6 + bob + (hh * 1.2 / nextN) * nj + (hh * 1.2 / nextN) * 0.5;
                    const nz = -hd + colW * (col + 2);
                    const nnp = proj(faceX, ny, nz);
                    ctx.strokeStyle = rgba('#3b82f6', activation * 0.15);
                    ctx.lineWidth = 0.5; ctx.beginPath(); ctx.moveTo(np.sx, np.sy); ctx.lineTo(nnp.sx, nnp.sy); ctx.stroke();
                  }
                }
              }
            });
            // "ReLU" label
            const rlp = proj(faceX, bob, 0);
            ctx.fillStyle = '#f59e0b88'; ctx.font = `bold ${9 * rlp.s}px "JetBrains Mono", monospace`;
            ctx.textAlign = 'center'; ctx.fillText('ReLU', rlp.sx, rlp.sy);
          }

          if (slab.type === 'masked') {
            // MASKED MHA: triangular mask pattern, upper triangle fading/blocked
            const n = 6;
            const cellH = hh * 1.4 / n, cellW = hd * 1.4 / n;
            for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
              const yy = -hh * 0.7 + bob + cellH * r + cellH * 0.5;
              const zz = -hd * 0.7 + cellW * c + cellW * 0.5;
              const cp = proj(faceX, yy, zz);
              const allowed = c <= r;
              const sz = Math.max(3, 8 * cp.s);
              if (allowed) {
                const pulse = Math.sin(t * 4 + r - c) * 0.3 + 0.6;
                ctx.fillStyle = rgba('#a855f7', pulse);
              } else {
                ctx.fillStyle = rgba('#ef4444', 0.15 + Math.sin(t * 2) * 0.05);
              }
              ctx.fillRect(cp.sx - sz / 2, cp.sy - sz / 2, sz, sz);
              if (!allowed) {
                ctx.strokeStyle = rgba('#ef4444', 0.4); ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(cp.sx - sz / 2, cp.sy - sz / 2); ctx.lineTo(cp.sx + sz / 2, cp.sy + sz / 2); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(cp.sx + sz / 2, cp.sy - sz / 2); ctx.lineTo(cp.sx - sz / 2, cp.sy + sz / 2); ctx.stroke();
              }
            }
          }

          if (slab.type === 'cross') {
            // CROSS-ATTENTION: beams from right (encoder) connecting to left (decoder)
            const encPositions = SRC_TOKENS.map((_, ei) => ({ y: -hh * 0.4 + bob + ei * hh * 0.4, z: hd * 0.6 }));
            const decPositions = TGT_TOKENS.slice(0, decoderPass + 1).map((_, di) => ({ y: -hh * 0.4 + bob + di * hh * 0.3, z: -hd * 0.6 }));
            decPositions.forEach((dp, di) => {
              encPositions.forEach((ep, ei) => {
                const fp = proj(faceX, dp.y, dp.z), tp = proj(faceX, ep.y, ep.z);
                const strength = Math.sin(t * 3 + di + ei * 1.5) * 0.3 + 0.5;
                ctx.strokeStyle = rgba('#ec4899', strength * 0.6);
                ctx.lineWidth = strength * 2; ctx.beginPath(); ctx.moveTo(fp.sx, fp.sy); ctx.lineTo(tp.sx, tp.sy); ctx.stroke();
                ctx.fillStyle = rgba('#ec4899', strength);
                ctx.beginPath(); ctx.arc(tp.sx, tp.sy, 2 + strength * 2, 0, Math.PI * 2); ctx.fill();
              });
            });
            // Labels
            const elbl = proj(faceX, -hh * 0.8 + bob, hd * 0.6);
            ctx.fillStyle = '#22c55e'; ctx.font = `bold ${8 * elbl.s}px "JetBrains Mono", monospace`;
            ctx.textAlign = 'center'; ctx.fillText('Encoder K,V', elbl.sx, elbl.sy);
            const dlbl = proj(faceX, -hh * 0.8 + bob, -hd * 0.6);
            ctx.fillStyle = '#a855f7'; ctx.font = `bold ${8 * dlbl.s}px "JetBrains Mono", monospace`;
            ctx.fillText('Decoder Q', dlbl.sx, dlbl.sy);
          }

          if (slab.type === 'stack') {
            // STACK: mini block replicas cascading
            const numBlocks = 6;
            for (let b = 0; b < numBlocks; b++) {
              const yy = -hh * 0.6 + bob + (hh * 1.2 / numBlocks) * b;
              const shift = Math.sin(t * 2 + b * 0.8) * 5;
              const bp = proj(faceX, yy + shift, 0);
              const bw = 50 * bp.s, bh2 = 12 * bp.s;
              const blockAlpha = 0.3 + Math.sin(t * 3 + b) * 0.2;
              ctx.fillStyle = rgba(slab.color, blockAlpha);
              ctx.strokeStyle = rgba(slab.color, blockAlpha + 0.2); ctx.lineWidth = 1;
              ctx.beginPath(); ctx.roundRect(bp.sx - bw / 2, bp.sy - bh2 / 2, bw, bh2, 3); ctx.fill(); ctx.stroke();
              ctx.fillStyle = rgba('#fff', blockAlpha); ctx.font = `bold ${6 * bp.s}px "JetBrains Mono", monospace`;
              ctx.textAlign = 'center'; ctx.fillText(`Layer ${b + 1}`, bp.sx, bp.sy + 2);
            }
          }

          if (slab.type === 'linear') {
            // LINEAR: matrix multiply animation — expanding bar
            const barCount = 8;
            for (let b = 0; b < barCount; b++) {
              const yy = -hh * 0.6 + bob + (hh * 1.2 / barCount) * b + 8;
              const barLen = (Math.sin(t * 4 + b * 0.8) + 1) / 2 * hd * 1.2;
              const startZ = -hd * 0.6;
              for (let z = 0; z < barLen; z += 4) {
                const bp = proj(faceX, yy, startZ + z);
                const intensity = z / barLen;
                ctx.fillStyle = rgba('#c4b5fd', 0.2 + intensity * 0.6);
                ctx.fillRect(bp.sx - 2, bp.sy - 2, 4 * bp.s, 4 * bp.s);
              }
            }
          }

          if (slab.type === 'softmax') {
            // SOFTMAX: probability bars growing, with winner highlighted
            const probs = [0.05, 0.12, 0.03, 0.65, 0.08, 0.07];
            const barH = hh * 1.2 / probs.length;
            probs.forEach((p, pi) => {
              const yy = -hh * 0.6 + bob + barH * pi + barH * 0.5;
              const barMaxZ = hd * 1.0;
              const currentLen = barMaxZ * p * (Math.min(1, (Math.sin(t * 2) + 1) / 2 + 0.5));
              const startP = proj(faceX, yy, -hd * 0.5);
              const endP = proj(faceX, yy, -hd * 0.5 + currentLen);
              const isWinner = pi === 3;
              ctx.fillStyle = rgba(isWinner ? '#22c55e' : '#86efac', isWinner ? 0.8 : 0.35);
              if (isWinner) { ctx.shadowColor = '#22c55e'; ctx.shadowBlur = 8; }
              const barW = Math.abs(endP.sx - startP.sx);
              ctx.fillRect(startP.sx, startP.sy - 4 * startP.s, barW, 8 * startP.s);
              ctx.shadowBlur = 0;
              if (isWinner) {
                ctx.fillStyle = '#fff'; ctx.font = `bold ${8 * startP.s}px "JetBrains Mono", monospace`;
                ctx.textAlign = 'left'; ctx.fillText(`${(p * 100).toFixed(0)}%`, endP.sx + 4, endP.sy + 2);
              }
            });
          }
        } else if (!isCurrent) {
          // Idle pattern (subtle)
          if (slab.type === 'grid' || slab.type === 'attention' || slab.type === 'ffn' || slab.type === 'masked' || slab.type === 'cross') {
            const rows = 6, cols = 5;
            for (let r = 0; r < rows; r++) for (let c2 = 0; c2 < cols; c2++) {
              const yy = -hh + bob + (hh * 2 / rows) * r + 12;
              const zz = -hd + (hd * 2 / cols) * c2 + 12;
              const gp = proj(xc - hw - 1, yy, zz);
              ctx.fillStyle = rgba(slab.color, 0.1 + Math.random() * 0.08);
              ctx.fillRect(gp.sx - 2, gp.sy - 2, 4 * gp.s, 4 * gp.s);
            }
          }
          if (slab.type === 'wave' || slab.type === 'softmax') {
            ctx.strokeStyle = rgba(slab.color, 0.2); ctx.lineWidth = 1;
            ctx.beginPath();
            for (let z = -hd; z <= hd; z += 6) {
              const yy = Math.sin(z * 0.04 + t) * 20 + bob;
              const wp = proj(xc - hw - 1, yy, z);
              z === -hd ? ctx.moveTo(wp.sx, wp.sy) : ctx.lineTo(wp.sx, wp.sy);
            }
            ctx.stroke();
          }
        }

        // Label
        const labelP = proj(xc, -hh + bob - 14, 0);
        ctx.fillStyle = isCurrent ? '#fff' : isHov ? '#e2e8f0' : rgba(slab.color, 0.8);
        ctx.font = `bold ${Math.max(7, 9 * labelP.s)}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center'; ctx.textBaseline = 'bottom'; ctx.fillText(slab.label, labelP.sx, labelP.sy);

        // Yellow inference glow
        if (isCurrent) {
          ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2.5; ctx.shadowColor = '#facc15'; ctx.shadowBlur = 20;
          ctx.beginPath(); ctx.moveTo(corners[0].sx, corners[0].sy); ctx.lineTo(corners[3].sx, corners[3].sy);
          ctx.lineTo(corners[7].sx, corners[7].sy); ctx.lineTo(corners[4].sx, corners[4].sy); ctx.closePath(); ctx.stroke();
          ctx.shadowBlur = 0;
        }

        // Beam connections
        if (isActive && i < SLABS.length - 1 && inferStep > i) {
          const nextX = -bx + 30 + spacing * (i + 1) + spacing / 2;
          const from = proj(xc + hw + 2, bob, 0);
          const to = proj(nextX - hw - 2, Math.sin(t * 1.2 + (i + 1) * 0.7) * 3, 0);
          ctx.strokeStyle = rgba('#facc15', 0.25); ctx.lineWidth = 1.5;
          ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(from.sx, from.sy); ctx.lineTo(to.sx, to.sy); ctx.stroke();
          ctx.setLineDash([]);
        }
      });

      // ─── GENERATED TOKENS (right side, accumulating) ───
      generatedTokens.forEach((tok, i) => {
        const xOff = 310 + i * 55;
        const p = proj(xOff, -60 + i * 20, 0);
        ctx.fillStyle = rgba('#22c55e', 0.2); ctx.strokeStyle = rgba('#22c55e', 0.6); ctx.lineWidth = 1.5;
        ctx.shadowColor = '#22c55e'; ctx.shadowBlur = 8;
        ctx.beginPath(); ctx.roundRect(p.sx - 25, p.sy - 10, 50, 20, 6); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
        ctx.fillStyle = '#22c55e'; ctx.font = `bold ${11 * p.s}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center'; ctx.fillText(tok, p.sx, p.sy + 4);
      });

      // ─── INPUT TOKENS (left side) ───
      SRC_TOKENS.forEach((tok, i) => {
        const xOff = -340 - i * 45;
        const p = proj(xOff, -30 + i * 25, 0);
        ctx.fillStyle = rgba('#22c55e', 0.15); ctx.strokeStyle = rgba('#22c55e', 0.4); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.roundRect(p.sx - 20, p.sy - 8, 40, 16, 4); ctx.fill(); ctx.stroke();
        ctx.fillStyle = '#22c55e'; ctx.font = `bold ${9 * p.s}px "JetBrains Mono", monospace`;
        ctx.textAlign = 'center'; ctx.fillText(tok, p.sx, p.sy + 3);
      });

      // ─── LABELS ───
      const vramP = proj(0, by + 50, 80);
      ctx.fillStyle = '#3b82f6'; ctx.font = `bold ${13 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('GPU VRAM', vramP.sx, vramP.sy);
      ctx.fillStyle = '#475569'; ctx.font = `${8 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('~260 MB · CUDA compute core engines', vramP.sx, vramP.sy + 14);

      const inLbl = proj(-370, -100, 0);
      ctx.fillStyle = '#22c55e'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'center';
      ctx.fillText('Token input packets', inLbl.sx, inLbl.sy);

      const outLbl = proj(370, -100, 0);
      ctx.fillStyle = '#a855f7'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('prediction tokens', outLbl.sx, outLbl.sy);

      // Title
      ctx.fillStyle = '#e2e8f0'; ctx.font = `bold ${14 * zoom}px "DM Sans", sans-serif`; ctx.textAlign = 'center';
      ctx.fillText('Transformer Model in GPU Memory', W / 2, 22);
      ctx.fillStyle = '#64748b'; ctx.font = `${9 * zoom}px "JetBrains Mono", monospace`;
      ctx.fillText('d_model=512 · 8 heads · 6+6 layers · ~65M parameters', W / 2, 38);

      // Decoder pass indicator
      if (inferPhase === 'decoder') {
        ctx.fillStyle = '#a855f7'; ctx.font = `bold ${10 * zoom}px "JetBrains Mono", monospace`; ctx.textAlign = 'right';
        ctx.fillText(`Decoder pass ${decoderPass + 1}/${TGT_TOKENS.length}`, W - 15, H - 58);
        ctx.fillStyle = '#64748b'; ctx.font = `${8 * zoom}px "JetBrains Mono", monospace`;
        ctx.fillText(`generating: "${TGT_TOKENS[decoderPass]}"`, W - 15, H - 45);
      }

      animRef.current = requestAnimationFrame(render);
    };
    animRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animRef.current);
  }, [rotY, rotX, zoom, hovered, inferStep, inferPhase, decoderPass, generatedTokens, proj]);

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
      const spacing2 = (280 * 2 - 40) / SLABS.length;
      let found = -1;
      for (let i = 0; i < SLABS.length; i++) {
        const xc = -280 + 30 + spacing2 * i + spacing2 / 2;
        const p = proj(xc, 0, 0);
        if (Math.abs(mx - p.sx) < 25 * zoom) { found = i; break; }
      }
      setHovered(found);
    }
  };
  const onUp = () => setDragging(false);
  const onWheel = (e: React.WheelEvent) => { e.preventDefault(); setZoom(p => Math.max(0.5, Math.min(2, p - e.deltaY * 0.001))); };

  return (
    <div style={{ position: 'relative', borderRadius: 14, overflow: 'hidden', background: '#020210' }}>
      <canvas ref={canvasRef} style={{ width: W, height: H, display: 'block', margin: '0 auto', cursor: dragging ? 'grabbing' : 'grab', maxWidth: '100%', borderRadius: 14 }}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={onUp} onMouseLeave={onUp} onWheel={onWheel} />

      {/* Controls */}
      <div style={{ position: 'absolute', bottom: 10, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 6, zIndex: 10 }}>
        <button onClick={startInference} disabled={autoInfer}
          style={{ padding: '5px 14px', borderRadius: 7, fontSize: 11, fontWeight: 700, background: autoInfer ? '#475569' : '#16a34a', color: '#fff', border: 'none', cursor: autoInfer ? 'default' : 'pointer', fontFamily: "'JetBrains Mono', monospace", opacity: autoInfer ? 0.6 : 1 }}>
          {autoInfer ? '⏳ Running...' : '▶ Run Inference'}
        </button>
        <button onClick={resetInference} style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: 'rgba(15,23,42,0.8)', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>↺</button>
        <button onClick={() => { setRotY(28); setRotX(-22); setZoom(1); }} style={{ padding: '5px 10px', borderRadius: 7, fontSize: 11, background: 'rgba(15,23,42,0.8)', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer' }}>⊙</button>
      </div>

      <p style={{ position: 'absolute', top: 48, right: 10, fontSize: 8, color: '#47556966', fontFamily: "'JetBrains Mono', monospace" }}>drag=rotate · scroll=zoom</p>

      {/* Hover */}
      {hovered >= 0 && SLABS[hovered] && (
        <div style={{ position: 'absolute', top: 55, right: 10, padding: '10px 14px', borderRadius: 10, background: 'rgba(2,2,16,0.93)', border: `1.5px solid ${SLABS[hovered].color}`, maxWidth: 220, zIndex: 10 }}>
          <p style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: SLABS[hovered].color, fontWeight: 700, margin: 0 }}>{SLABS[hovered].label}</p>
          <p style={{ fontSize: 10, color: '#c8d6e5', margin: '3px 0' }}>{SLABS[hovered].desc}</p>
          <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace" }}>{SLABS[hovered].shape}</span>
          <span style={{ fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace", marginLeft: 8 }}>{SLABS[hovered].params}</span>
        </div>
      )}

      {/* Status */}
      {statusText && (
        <div style={{ position: 'absolute', bottom: 42, left: '50%', transform: 'translateX(-50%)', padding: '4px 14px', borderRadius: 6, background: `${statusColor}18`, border: `1px solid ${statusColor}44`, whiteSpace: 'nowrap', zIndex: 10 }}>
          <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: statusColor, fontWeight: 700 }}>⚡ {statusText}</span>
        </div>
      )}
    </div>
  );
}
