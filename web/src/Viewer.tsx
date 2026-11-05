import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { bytes, fileUrl, useFetched } from "./api";
import type { Camera, Export, Model, State } from "./api";

const AUTO = "auto";

export function Viewer({ state, focus }: { state: State; focus: string | null }) {
  const live = state.current?.live;
  // Models: the live mapper snapshot while solving, else sparse/0.
  const models = useMemo(() => {
    const list: { path: string; label: string; key: number }[] = [];
    if (live?.model) list.push({ path: live.model.path, label: `live: ${live.model.registered} registered`, key: live.model.mtime });
    if (state.model) list.push({ path: state.model.path, label: `${state.model.path} · ${state.model.registered} registered`, key: state.model.mtime });
    return list;
  }, [live?.model, state.model]);
  const splats = useMemo(() => {
    const list: (Export & { label: string })[] = [];
    for (const e of [...(live?.exports ?? [])].reverse()) list.push({ ...e, label: `live: step ${e.iteration ?? "?"} · ${bytes(e.size)}` });
    for (const e of [...state.exports].reverse()) {
      const run = e.run.split("/").slice(1).join("/");
      list.push({ ...e, label: `${run} · ${e.iteration ?? e.path.split("/").pop()}${e.marker ? ` · ${e.marker}` : ""}` });
    }
    return list;
  }, [live?.exports, state.exports]);

  const [modelPick, setModelPick] = useState(AUTO);
  const [splatPick, setSplatPick] = useState(AUTO);
  const [show, setShow] = useState({ cameras: true, points: true, splat: true, grid: true });
  const [pointSize, setPointSize] = useState(1);
  const [splatStatus, setSplatStatus] = useState("");

  const modelChoice = modelPick === AUTO ? models[0] : models.find((m) => m.path === modelPick) ?? models[0];
  const splatChoice = splatPick === AUTO ? splats[0] : splats.find((s) => s.path === splatPick) ?? splats[0];
  const model = useFetched<Model>(modelChoice ? "model" : null, { path: modelChoice?.path ?? "" }, modelChoice?.key);
  const frame = useMemo(() => (model ? sceneFrame(model) : null), [model]);

  return (
    <div className="viewer">
      <aside className="side">
        <div>
          <h3>COLMAP model</h3>
          <select value={modelPick} onChange={(e) => setModelPick(e.target.value)}>
            <option value={AUTO}>newest{models[0] ? ` (${models[0].label})` : ""}</option>
            {models.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
          </select>
        </div>
        <div>
          <h3>Splat</h3>
          <select value={splatPick} onChange={(e) => setSplatPick(e.target.value)}>
            <option value={AUTO}>newest{splats[0] ? ` (${splats[0].label})` : ""}</option>
            {splats.map((s) => <option key={s.path} value={s.path}>{s.label}</option>)}
          </select>
        </div>
        <div>
          <h3>Show</h3>
          {(Object.keys(show) as (keyof typeof show)[]).map((key) => (
            <label key={key}>
              <input type="checkbox" checked={show[key]} onChange={() => setShow({ ...show, [key]: !show[key] })} />
              {key}
            </label>
          ))}
          <label>
            points
            <input type="range" min={0.2} max={4} step={0.1} value={pointSize} onChange={(e) => setPointSize(Number(e.target.value))} />
          </label>
        </div>
        {model && (
          <div>
            <h3>Reconstruction</h3>
            <dl className="kv">
              <dt>cameras</dt><dd>{model.cameras.length}</dd>
              <dt>points</dt><dd>{model.stats.points.toLocaleString()}</dd>
              <dt>shown</dt><dd>{(model.points.xyz.length / 3).toLocaleString()}</dd>
              <dt>reproj. error</dt><dd>{model.stats.mean_error.toFixed(2)} px</dd>
              <dt>track length</dt><dd>{model.stats.mean_track.toFixed(1)}</dd>
            </dl>
          </div>
        )}
        <p className="step-sub" style={{ whiteSpace: "normal" }}>
          Drag to orbit, right-drag to pan, scroll to zoom. Up is taken from the cameras' level right axes.
        </p>
      </aside>
      <div className="canvas">
        {!model && !splatChoice ? (
          <div className="empty">Nothing to show yet: no COLMAP model and no splat.</div>
        ) : (
          <Canvas gl={{ antialias: false }} camera={{ fov: 55, near: 0.01, far: 5000 }} dpr={[1, 2]}>
            <color attach="background" args={["#120e1c"]} />
            <SparkSetup />
            {frame && <FitCamera frame={frame} />}
            <group quaternion={frame?.rotation} position={frame?.offset}>
              {model && show.points && <Points model={model} size={pointSize * (frame?.scale ?? 1) * 0.004} />}
              {model && show.cameras && <Frustums cameras={model.cameras} depth={(frame?.scale ?? 1) * 0.06} focus={focus} />}
              {show.splat && splatChoice && <Splat path={splatChoice.path} onStatus={setSplatStatus} />}
            </group>
            {show.grid && frame && (
              <gridHelper args={[frame.scale * 8, 40, "#ff7edb", "#4a2d6b"]} position={[0, frame.floor, 0]} />
            )}
            <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
          </Canvas>
        )}
        {(splatStatus || focus) && (
          <div className="overlay-note">{[splatStatus, focus && `camera: ${focus}`].filter(Boolean).join(" · ")}</div>
        )}
      </div>
    </div>
  );
}

interface Frame {
  rotation: THREE.Quaternion; // COLMAP world → viewer (up = +Y)
  offset: THREE.Vector3; // puts the subject at the origin
  scale: number; // typical camera distance from the subject
  eye: THREE.Vector3; // a starting viewpoint
  floor: number;
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? 0;
}

/** Turn the model so its up is +Y and centre it on its points. */
function sceneFrame(model: Model): Frame {
  const rotation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(...model.up).normalize(), new THREE.Vector3(0, 1, 0));
  const turn = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z).applyQuaternion(rotation);
  const xyz = model.points.xyz;
  const pts: THREE.Vector3[] = [];
  for (let i = 0; i < xyz.length; i += 3 * Math.max(1, Math.floor(xyz.length / 3 / 5000))) pts.push(turn(xyz[i], xyz[i + 1], xyz[i + 2]));
  const cams = model.cameras.map((c) => turn(...c.center));
  const source = pts.length ? pts : cams;
  const centre = new THREE.Vector3(median(source.map((p) => p.x)), median(source.map((p) => p.y)), median(source.map((p) => p.z)));
  const scale = Math.max(1e-3, median(cams.map((c) => c.distanceTo(centre))) || 1);
  const ys = source.map((p) => p.y - centre.y).sort((a, b) => a - b);
  const floor = ys[Math.floor(ys.length * 0.03)] ?? -scale * 0.5;
  // Start behind a typical camera, a little higher.
  const typical = cams.length ? cams[Math.floor(cams.length / 2)].clone().sub(centre) : new THREE.Vector3(0, 0, scale);
  const eye = typical.multiplyScalar(1.6).add(new THREE.Vector3(0, scale * 0.4, 0));
  return { rotation, offset: centre.clone().negate(), scale, eye, floor };
}

function FitCamera({ frame }: { frame: Frame }) {
  const camera = useThree((s) => s.camera);
  // SAFETY: the only default controls are the <OrbitControls makeDefault> below.
  const controls = useThree((s) => s.controls) as unknown as { target: THREE.Vector3; update: () => void } | null;
  const fitted = useRef<number | null>(null);
  useEffect(() => {
    // Fit once per scene scale, so a live model update does not yank the view.
    if (fitted.current !== null && Math.abs(fitted.current - frame.scale) / frame.scale < 0.5) return;
    fitted.current = frame.scale;
    camera.position.copy(frame.eye);
    camera.near = frame.scale / 1000;
    camera.far = frame.scale * 200;
    camera.updateProjectionMatrix();
    controls?.target.set(0, 0, 0);
    controls?.update();
  }, [frame, camera, controls]);
  return null;
}

function SparkSetup() {
  const gl = useThree((s) => s.gl);
  const spark = useMemo(() => new SparkRenderer({ renderer: gl }), [gl]);
  return <primitive object={spark} />;
}

function Points({ model, size }: { model: Model; size: number }) {
  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(model.points.xyz, 3));
    g.setAttribute("color", new THREE.Uint8BufferAttribute(model.points.rgb, 3, true));
    return g;
  }, [model]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <points geometry={geometry}>
      <pointsMaterial size={size} vertexColors sizeAttenuation />
    </points>
  );
}

/** Camera frustums as one line set; the focused image's camera in pink. */
function Frustums({ cameras, depth, focus }: { cameras: Camera[]; depth: number; focus: string | null }) {
  const [all, focused] = useMemo(() => {
    const build = (list: Camera[], scale: number) => {
      const vertices: number[] = [];
      for (const c of list) {
        const r = c.rotation;
        const world = (x: number, y: number, z: number) => [
          r[0] * x + r[1] * y + r[2] * z + c.center[0],
          r[3] * x + r[4] * y + r[5] * z + c.center[1],
          r[6] * x + r[7] * y + r[8] * z + c.center[2],
        ];
        const d = depth * scale;
        const w = (c.width / 2 / c.fx) * d;
        const h = (c.height / 2 / c.fy) * d;
        const corners = [world(-w, -h, d), world(w, -h, d), world(w, h, d), world(-w, h, d)];
        const apex = world(0, 0, 0);
        for (let i = 0; i < 4; i++) vertices.push(...apex, ...corners[i], ...corners[i], ...corners[(i + 1) % 4]);
        vertices.push(...world(-w * 0.5, -h * 1.15, d), ...world(0, -h * 1.5, d)); // a tick for image-up
        vertices.push(...world(0, -h * 1.5, d), ...world(w * 0.5, -h * 1.15, d));
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
      return g;
    };
    const hit = cameras.filter((c) => c.name === focus);
    return [build(cameras, 1), hit.length ? build(hit, 2.5) : null];
  }, [cameras, depth, focus]);
  useEffect(() => () => { all.dispose(); focused?.dispose(); }, [all, focused]);
  return (
    <>
      <lineSegments geometry={all}>
        <lineBasicMaterial color="#36f9f6" transparent opacity={0.55} />
      </lineSegments>
      {focused && (
        <lineSegments geometry={focused}>
          <lineBasicMaterial color="#ff7edb" />
        </lineSegments>
      )}
    </>
  );
}

/** A trained splat; a new one replaces the shown one only once it has loaded. */
function Splat({ path, onStatus }: { path: string; onStatus: (status: string) => void }) {
  const [mesh, setMesh] = useState<SplatMesh | null>(null);
  useEffect(() => {
    let cancelled = false;
    onStatus(`loading ${path.split("/").pop()}…`);
    const next = new SplatMesh({
      url: fileUrl(path),
      fileName: "export.ply",
      onProgress: (event) => {
        if (!cancelled && event.total) onStatus(`loading ${path.split("/").pop()} ${Math.round((100 * event.loaded) / event.total)}%`);
      },
    });
    next.initialized.then(
      () => {
        if (cancelled) return next.dispose();
        setMesh(next); // the effect below disposes the one it replaces
        onStatus(`${path.split("/").pop()} · ${next.packedSplats?.numSplats?.toLocaleString() ?? "?"} splats`);
      },
      (error: unknown) => !cancelled && onStatus(`could not load ${path}: ${String(error)}`),
    );
    return () => {
      cancelled = true;
    };
  }, [path, onStatus]);
  useEffect(() => () => mesh?.dispose(), [mesh]);
  return mesh ? <primitive object={mesh} /> : null;
}
