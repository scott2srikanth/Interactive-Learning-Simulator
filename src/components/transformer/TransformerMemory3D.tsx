import React, { useRef, useState, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, RoundedBox, Line } from '@react-three/drei';
import * as THREE from 'three';

/* ═══════════════════════════════════════════════════════════
   MODEL LAYERS — what lives in GPU VRAM
   ═══════════════════════════════════════════════════════════ */
const MODEL_LAYERS = [
  { id: 'emb', label: 'Token Embedding', shape: '[50257×512]', params: '25.7M', color: '#22c55e', y: -2.8, w: 3.2, h: 0.35, d: 2.0, desc: 'Lookup table: every word → 512-dim vector' },
  { id: 'pe', label: 'Positional Encoding', shape: '[512×512]', params: '262K', color: '#06b6d4', y: -2.2, w: 2.8, h: 0.25, d: 1.8, desc: 'Sine/cosine waves for position info' },
  { id: 'enc_mha1', label: 'Encoder MHA Layer 1', shape: '4×[512×64]', params: '1.05M', color: '#f59e0b', y: -1.5, w: 3.0, h: 0.4, d: 2.2, desc: 'W_Q, W_K, W_V, W_O matrices' },
  { id: 'enc_ffn1', label: 'Encoder FFN Layer 1', shape: '[512×2048]+[2048×512]', params: '2.1M', color: '#3b82f6', y: -0.9, w: 2.6, h: 0.35, d: 2.4, desc: 'Feed-forward: expand → ReLU → compress' },
  { id: 'enc_rest', label: 'Encoder Layers 2-6', shape: '×5 blocks', params: '15.75M', color: '#2563eb', y: -0.2, w: 3.4, h: 0.5, d: 2.0, desc: '5 more encoder blocks (same structure)' },
  { id: 'dec_mmha', label: 'Decoder Masked MHA', shape: '4×[512×64]', params: '1.05M', color: '#a855f7', y: 0.5, w: 2.8, h: 0.35, d: 2.2, desc: 'Masked self-attention (causal)' },
  { id: 'dec_xmha', label: 'Decoder Cross MHA', shape: '4×[512×64]', params: '1.05M', color: '#ec4899', y: 1.1, w: 2.6, h: 0.35, d: 2.0, desc: 'Cross-attention: decoder→encoder' },
  { id: 'dec_ffn', label: 'Decoder FFN Layer 1', shape: '[512×2048]+[2048×512]', params: '2.1M', color: '#f472b6', y: 1.7, w: 2.4, h: 0.3, d: 2.2, desc: 'Decoder feed-forward network' },
  { id: 'dec_rest', label: 'Decoder Layers 2-6', shape: '×5 blocks', params: '21M', color: '#7c3aed', y: 2.3, w: 3.2, h: 0.5, d: 2.0, desc: '5 more decoder blocks' },
  { id: 'linear', label: 'Linear Output', shape: '[512×50257]', params: '25.7M', color: '#c4b5fd', y: 3.0, w: 3.0, h: 0.3, d: 1.8, desc: 'Projects to vocabulary for prediction' },
  { id: 'softmax', label: 'Softmax', shape: 'function', params: '0 params', color: '#86efac', y: 3.5, w: 2.2, h: 0.2, d: 1.4, desc: 'Converts logits → probabilities' },
];

/* ═══════════════════════════════════════════════════════════
   GLASS BOX — the GPU VRAM container
   ═══════════════════════════════════════════════════════════ */
function GlassBox() {
  const ref = useRef<THREE.Mesh>(null);
  return (
    <mesh ref={ref} position={[0, 0.3, 0]}>
      <boxGeometry args={[5, 8, 4]} />
      <meshPhysicalMaterial
        color="#1a1a3e"
        transparent
        opacity={0.08}
        roughness={0.05}
        metalness={0.1}
        transmission={0.92}
        thickness={0.5}
        side={THREE.DoubleSide}
      />
      {/* Edge wireframe */}
      <lineSegments>
        <edgesGeometry args={[new THREE.BoxGeometry(5, 8, 4)]} />
        <lineBasicMaterial color="#3b82f6" transparent opacity={0.25} />
      </lineSegments>
    </mesh>
  );
}

/* ═══════════════════════════════════════════════════════════
   TENSOR BLOCK — a single weight matrix layer
   ═══════════════════════════════════════════════════════════ */
function TensorBlock({ layer, index, activeLayer, onHover }: any) {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);
  const isActive = activeLayer === index;
  const color = new THREE.Color(layer.color);

  useFrame((_, delta) => {
    if (meshRef.current) {
      // Gentle float
      meshRef.current.position.y = layer.y + Math.sin(Date.now() * 0.001 + index * 0.5) * 0.03;
      // Glow pulse when active
      const mat = meshRef.current.material as THREE.MeshStandardMaterial;
      if (isActive) {
        mat.emissiveIntensity = 0.8 + Math.sin(Date.now() * 0.005) * 0.4;
      } else if (hovered) {
        mat.emissiveIntensity = 0.5;
      } else {
        mat.emissiveIntensity = 0.15;
      }
    }
  });

  return (
    <group>
      <mesh
        ref={meshRef}
        position={[0, layer.y, 0]}
        onPointerOver={() => { setHovered(true); onHover(index); }}
        onPointerOut={() => { setHovered(false); onHover(-1); }}
      >
        <boxGeometry args={[layer.w, layer.h, layer.d]} />
        <meshStandardMaterial
          color={color}
          transparent
          opacity={isActive ? 0.85 : hovered ? 0.7 : 0.45}
          emissive={color}
          emissiveIntensity={0.15}
          roughness={0.3}
          metalness={0.6}
        />
      </mesh>
      {/* Wireframe outline */}
      <mesh position={[0, layer.y, 0]}>
        <boxGeometry args={[layer.w, layer.h, layer.d]} />
        <meshBasicMaterial color={layer.color} wireframe transparent opacity={isActive ? 0.6 : 0.2} />
      </mesh>
      {/* Label */}
      <Text
        position={[0, layer.y + layer.h / 2 + 0.12, layer.d / 2 + 0.01]}
        fontSize={0.14}
        color={hovered || isActive ? '#ffffff' : '#94a3b8'}
        anchorX="center"
        anchorY="bottom"
        font={undefined}
      >
        {layer.label}
      </Text>
      <Text
        position={[0, layer.y - layer.h / 2 - 0.08, layer.d / 2 + 0.01]}
        fontSize={0.1}
        color={layer.color}
        anchorX="center"
        anchorY="top"
        font={undefined}
      >
        {layer.shape} · {layer.params}
      </Text>
    </group>
  );
}

/* ═══════════════════════════════════════════════════════════
   DATA PARTICLES — flowing tokens through the model
   ═══════════════════════════════════════════════════════════ */
function DataParticles({ flowing, direction }: { flowing: boolean; direction: 'in' | 'out' | 'through' }) {
  const count = 120;
  const ref = useRef<THREE.Points>(null);
  const positions = useMemo(() => {
    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 3;
      arr[i * 3 + 1] = Math.random() * 8 - 4;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 2.5;
    }
    return arr;
  }, []);

  const colors = useMemo(() => {
    const arr = new Float32Array(count * 3);
    const palette = [[0.13, 0.76, 0.96], [0.39, 0.56, 0.94], [0.66, 0.33, 0.97], [0.13, 0.85, 0.39], [0.96, 0.62, 0.04]];
    for (let i = 0; i < count; i++) {
      const c = palette[Math.floor(Math.random() * palette.length)];
      arr[i * 3] = c[0]; arr[i * 3 + 1] = c[1]; arr[i * 3 + 2] = c[2];
    }
    return arr;
  }, []);

  useFrame(() => {
    if (!ref.current || !flowing) return;
    const pos = ref.current.geometry.attributes.position.array as Float32Array;
    for (let i = 0; i < count; i++) {
      if (direction === 'in') {
        pos[i * 3] += (0 - pos[i * 3]) * 0.01 + (Math.random() - 0.5) * 0.02;
        pos[i * 3 + 1] += 0.04;
        if (pos[i * 3 + 1] > 4) { pos[i * 3 + 1] = -4; pos[i * 3] = (Math.random() - 0.5) * 4 - 3; }
      } else if (direction === 'out') {
        pos[i * 3] += (3 - pos[i * 3]) * 0.005 + (Math.random() - 0.5) * 0.02;
        pos[i * 3 + 1] += 0.03;
        if (pos[i * 3 + 1] > 4.5) { pos[i * 3 + 1] = 3; pos[i * 3] = (Math.random() - 0.5) * 2; }
      } else {
        pos[i * 3 + 1] += 0.025;
        pos[i * 3] += (Math.random() - 0.5) * 0.01;
        if (pos[i * 3 + 1] > 4) pos[i * 3 + 1] = -4;
      }
    }
    ref.current.geometry.attributes.position.needsUpdate = true;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={count} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.06} transparent opacity={flowing ? 0.8 : 0.15} vertexColors sizeAttenuation />
    </points>
  );
}

/* ═══════════════════════════════════════════════════════════
   INPUT/OUTPUT STREAMS — token packets flowing in/out
   ═══════════════════════════════════════════════════════════ */
function TokenStream({ side, tokens, color, flowing }: any) {
  const x = side === 'in' ? -3.5 : 3.5;
  const direction = side === 'in' ? 1 : -1;
  const ref = useRef<THREE.Group>(null);

  useFrame(() => {
    if (ref.current && flowing) {
      ref.current.children.forEach((child, i) => {
        child.position.x += direction * 0.015;
        if (side === 'in' && child.position.x > -0.5) child.position.x = -5;
        if (side === 'out' && child.position.x < 0.5) child.position.x = 5;
      });
    }
  });

  return (
    <group ref={ref}>
      {tokens.map((t: string, i: number) => (
        <group key={i} position={[x - i * 0.8 * direction, -3.5, 0]}>
          <mesh>
            <boxGeometry args={[0.6, 0.25, 0.25]} />
            <meshStandardMaterial color={color} transparent opacity={0.7} emissive={color} emissiveIntensity={0.4} />
          </mesh>
          <Text position={[0, 0, 0.14]} fontSize={0.1} color="#fff" anchorX="center" anchorY="middle">{t}</Text>
        </group>
      ))}
    </group>
  );
}

/* ═══════════════════════════════════════════════════════════
   CUDA CORES — bottom of the GPU
   ═══════════════════════════════════════════════════════════ */
function CudaCores() {
  const ref = useRef<THREE.Group>(null);
  useFrame(() => {
    if (ref.current) {
      ref.current.children.forEach((child, i) => {
        const mat = (child as THREE.Mesh).material as THREE.MeshStandardMaterial;
        mat.emissiveIntensity = 0.3 + Math.sin(Date.now() * 0.003 + i * 0.4) * 0.3;
      });
    }
  });

  return (
    <group ref={ref} position={[0, -4.3, 0]}>
      {Array.from({ length: 40 }).map((_, i) => {
        const x = (i % 10 - 4.5) * 0.45;
        const z = (Math.floor(i / 10) - 1.5) * 0.6;
        return (
          <mesh key={i} position={[x, 0, z]}>
            <cylinderGeometry args={[0.12, 0.15, 0.25, 6]} />
            <meshStandardMaterial color="#6366f1" emissive="#6366f1" emissiveIntensity={0.3} metalness={0.8} roughness={0.2} />
          </mesh>
        );
      })}
      <Text position={[0, -0.3, 1.5]} fontSize={0.18} color="#6366f1" anchorX="center">CUDA Compute Cores</Text>
      <Text position={[0, -0.5, 1.5]} fontSize={0.12} color="#475569" anchorX="center">Parallel matrix multiplication engines</Text>
    </group>
  );
}

/* ═══════════════════════════════════════════════════════════
   INFERENCE FLOW LINE — animated dashed line through layers
   ═══════════════════════════════════════════════════════════ */
function InferenceFlow({ activeLayer }: { activeLayer: number }) {
  const ref = useRef<THREE.Line>(null);
  if (activeLayer < 0) return null;

  const points = MODEL_LAYERS.slice(0, activeLayer + 1).map(l => new THREE.Vector3(0, l.y, 0));
  if (points.length < 2) return null;

  return (
    <Line
      points={points}
      color="#facc15"
      lineWidth={3}
      dashed
      dashSize={0.15}
      gapSize={0.08}
    />
  );
}

/* ═══════════════════════════════════════════════════════════
   LABELS — GPU VRAM, titles
   ═══════════════════════════════════════════════════════════ */
function Labels() {
  return (
    <group>
      <Text position={[0, -4.7, 2.2]} fontSize={0.22} color="#3b82f6" anchorX="center" fontWeight="bold">GPU VRAM</Text>
      <Text position={[0, -4.95, 2.2]} fontSize={0.12} color="#475569" anchorX="center">~260 MB (float32) · ~130 MB (float16)</Text>
      <Text position={[-3.5, -3.0, 0]} fontSize={0.14} color="#22c55e" anchorX="center">Token Input</Text>
      <Text position={[-3.5, -3.2, 0]} fontSize={0.09} color="#22c55e" anchorX="center">Packets</Text>
      <Text position={[3.5, 3.8, 0]} fontSize={0.14} color="#a855f7" anchorX="center">Prediction</Text>
      <Text position={[3.5, 3.6, 0]} fontSize={0.09} color="#a855f7" anchorX="center">Token Probabilities</Text>
      <Text position={[0, 4.6, 0]} fontSize={0.2} color="#e2e8f0" anchorX="center" fontWeight="bold">Transformer Model in Memory</Text>
      <Text position={[0, 4.3, 0]} fontSize={0.11} color="#64748b" anchorX="center">d_model=512 · 8 heads · 6 layers · ~65M parameters</Text>
    </group>
  );
}

/* ═══════════════════════════════════════════════════════════
   MAIN SCENE
   ═══════════════════════════════════════════════════════════ */
function Scene({ activeLayer, setHoveredLayer, flowing }: any) {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[5, 5, 5]} intensity={1} color="#3b82f6" />
      <pointLight position={[-5, 5, -5]} intensity={0.7} color="#a855f7" />
      <pointLight position={[0, -3, 3]} intensity={0.5} color="#22c55e" />

      <GlassBox />

      {MODEL_LAYERS.map((layer, i) => (
        <TensorBlock key={layer.id} layer={layer} index={i} activeLayer={activeLayer} onHover={setHoveredLayer} />
      ))}

      <DataParticles flowing={flowing} direction="through" />
      <TokenStream side="in" tokens={['I', 'love', 'India']} color="#22c55e" flowing={flowing} />
      <TokenStream side="out" tokens={['मुझे', 'भारत', 'पसंद', 'है']} color="#a855f7" flowing={flowing} />

      <InferenceFlow activeLayer={activeLayer} />
      <CudaCores />
      <Labels />

      <OrbitControls
        enablePan={false}
        minDistance={4}
        maxDistance={16}
        minPolarAngle={0.3}
        maxPolarAngle={2.5}
        autoRotate={!flowing}
        autoRotateSpeed={0.5}
      />
    </>
  );
}

/* ═══════════════════════════════════════════════════════════
   EXPORTED COMPONENT
   ═══════════════════════════════════════════════════════════ */
export default function TransformerMemory3D() {
  const [activeLayer, setActiveLayer] = useState(-1);
  const [hoveredLayer, setHoveredLayer] = useState(-1);
  const [flowing, setFlowing] = useState(false);
  const [autoInfer, setAutoInfer] = useState(false);

  // Auto inference animation
  useEffect(() => {
    if (autoInfer && activeLayer < MODEL_LAYERS.length - 1) {
      const t = setTimeout(() => setActiveLayer(p => p + 1), 800);
      return () => clearTimeout(t);
    } else if (activeLayer >= MODEL_LAYERS.length - 1) {
      setAutoInfer(false);
    }
  }, [autoInfer, activeLayer]);

  const startInference = () => {
    setActiveLayer(-1);
    setFlowing(true);
    setTimeout(() => { setActiveLayer(0); setAutoInfer(true); }, 200);
  };

  const displayLayer = hoveredLayer >= 0 ? hoveredLayer : activeLayer >= 0 ? activeLayer : -1;

  return (
    <div style={{ position: 'relative', borderRadius: 16, overflow: 'hidden', background: '#020617' }}>
      <Canvas
        camera={{ position: [6, 2, 8], fov: 45 }}
        style={{ height: 520, background: 'linear-gradient(180deg, #020617 0%, #0a0a2e 50%, #020617 100%)' }}
        gl={{ antialias: true, alpha: true }}
      >
        <Scene activeLayer={activeLayer} setHoveredLayer={setHoveredLayer} flowing={flowing} />
      </Canvas>

      {/* Controls overlay */}
      <div style={{ position: 'absolute', bottom: 12, left: 0, right: 0, display: 'flex', justifyContent: 'center', gap: 8 }}>
        <button onClick={startInference}
          style={{ padding: '6px 16px', borderRadius: 8, fontSize: 11, fontWeight: 700, background: autoInfer ? '#dc2626' : '#16a34a', color: '#fff', border: 'none', cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace" }}>
          {autoInfer ? '⏸ Running...' : '▶ Run Inference'}
        </button>
        <button onClick={() => { setActiveLayer(-1); setFlowing(false); setAutoInfer(false); }}
          style={{ padding: '6px 12px', borderRadius: 8, fontSize: 11, background: '#1e293b', color: '#94a3b8', border: '1px solid #334155', cursor: 'pointer', fontFamily: "'JetBrains Mono', monospace" }}>
          ↺ Reset
        </button>
      </div>

      <p style={{ position: 'absolute', top: 8, right: 12, fontSize: 9, color: '#475569', fontFamily: "'JetBrains Mono', monospace" }}>
        🖱️ Drag to rotate · Scroll to zoom
      </p>

      {/* Info panel */}
      {displayLayer >= 0 && MODEL_LAYERS[displayLayer] && (
        <div style={{ position: 'absolute', top: 12, left: 12, padding: '10px 14px', borderRadius: 10, background: 'rgba(2,6,23,0.92)', border: `1.5px solid ${MODEL_LAYERS[displayLayer].color}`, maxWidth: 280, backdropFilter: 'blur(8px)' }}>
          <p style={{ fontSize: 13, fontFamily: "'JetBrains Mono', monospace", color: MODEL_LAYERS[displayLayer].color, fontWeight: 700, marginBottom: 4 }}>
            {MODEL_LAYERS[displayLayer].label}
          </p>
          <p style={{ fontSize: 10, color: '#c8d6e5', marginBottom: 6 }}>{MODEL_LAYERS[displayLayer].desc}</p>
          <div style={{ display: 'flex', gap: 10 }}>
            <span style={{ fontSize: 9, color: '#f59e0b', fontFamily: "'JetBrains Mono', monospace" }}>Shape: {MODEL_LAYERS[displayLayer].shape}</span>
            <span style={{ fontSize: 9, color: '#22c55e', fontFamily: "'JetBrains Mono', monospace" }}>{MODEL_LAYERS[displayLayer].params}</span>
          </div>
        </div>
      )}

      {/* Inference status */}
      {activeLayer >= 0 && (
        <div style={{ position: 'absolute', bottom: 48, left: '50%', transform: 'translateX(-50%)', padding: '4px 14px', borderRadius: 8, background: `${MODEL_LAYERS[Math.min(activeLayer, MODEL_LAYERS.length - 1)].color}22`, border: `1px solid ${MODEL_LAYERS[Math.min(activeLayer, MODEL_LAYERS.length - 1)].color}44` }}>
          <span style={{ fontSize: 10, fontFamily: "'JetBrains Mono', monospace", color: MODEL_LAYERS[Math.min(activeLayer, MODEL_LAYERS.length - 1)].color, fontWeight: 700 }}>
            ⚡ {activeLayer >= MODEL_LAYERS.length - 1 ? 'Inference complete! → "मुझे भारत पसंद है"' : `Processing: ${MODEL_LAYERS[activeLayer].label}`}
          </span>
        </div>
      )}
    </div>
  );
}
