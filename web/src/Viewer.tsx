import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import type { MutableRefObject } from "react";
import { Canvas, useFrame, useLoader, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { bytes, fileUrl, thumbUrl, useFetched } from "./api";
import type { Camera, Export, Model, State } from "./api";

const AUTO = "auto";
const NONE = "none";

type Layers = { cameras: boolean; points: boolean; splat: boolean; grid: boolean };
type Splat = Export & { label: string };

/** Camera pose shared by the two views of a comparison; the view last touched leads. */
interface Sync {
  leader: number;
  position: THREE.Vector3;
  target: THREE.Vector3;
  fov: number;
  set: boolean;
}

/** A request to put the viewer camera where a photo was taken; `n` makes repeated requests distinct. */
interface LookRequest {
  name: string;
  n: number;
}

export function Viewer({ state, focus }: { state: State; focus: string | null }) {
  const live = state.current?.live;
  // Models: the live mapper snapshot while solving, else sparse/0.
  const models = useMemo(() => {
    const list: { path: string; label: string; key: number }[] = [];
    if (live?.model) list.push({ path: live.model.path, label: `solving: ${live.model.registered} placed`, key: live.model.mtime });
    if (state.model) list.push({ path: state.model.path, label: `${state.model.path}: ${state.model.registered} placed`, key: state.model.mtime });
    return list;
  }, [live?.model, state.model]);
  const splats = useMemo(() => {
    const list: Splat[] = [];
    for (const e of [...(live?.exports ?? [])].reverse()) list.push({ ...e, label: `training: step ${e.iteration ?? "?"}, ${bytes(e.size)}` });
    for (const e of [...state.exports].reverse()) {
      const run = e.run.split("/").slice(1).join("/");
      list.push({ ...e, label: `${run}: ${e.iteration ? `step ${e.iteration}` : e.path.split("/").pop()}${e.marker ? `, ${e.marker}` : ""}` });
    }
    return list;
  }, [live?.exports, state.exports]);

  const [modelPick, setModelPick] = useState(AUTO);
  const [splatPick, setSplatPick] = useState(AUTO);
  const [comparePick, setComparePick] = useState(NONE);
  const [show, setShow] = useState<Layers>({ cameras: true, points: true, splat: true, grid: true });
  const [pointSize, setPointSize] = useState(1);
  const [selected, setSelected] = useState<string | null>(focus);
  const [look, setLook] = useState<LookRequest | null>(null);
  const sync = useRef<Sync>({ leader: 0, position: new THREE.Vector3(), target: new THREE.Vector3(), fov: 55, set: false });
  useEffect(() => setSelected(focus), [focus]);

  const modelChoice = modelPick === AUTO ? models[0] : models.find((m) => m.path === modelPick) ?? models[0];
  const splatChoice = splatPick === AUTO ? splats[0] : splats.find((s) => s.path === splatPick) ?? splats[0];
  const compareChoice = splats.find((s) => s.path === comparePick) ?? null;
  const model = useFetched<Model>(modelChoice ? "model" : null, { path: modelChoice?.path ?? "" }, modelChoice?.key);
  const frame = useMemo(() => (model ? sceneFrame(model) : null), [model]);
  const camera = model?.cameras.find((c) => c.name === selected) ?? null;

  const views = compareChoice ? [splatChoice, compareChoice] : [splatChoice];
  return (
    <div className="viewer">
      <aside className="side">
        <section>
          <span className="label">model</span>
          <select value={modelPick} onChange={(e) => setModelPick(e.target.value)}>
            <option value={AUTO}>{live?.model ? "follow the solve" : "newest"}</option>
            {models.map((m) => <option key={m.path} value={m.path}>{m.label}</option>)}
          </select>
        </section>
        <section>
          <span className="label">splat</span>
          <select value={splatPick} onChange={(e) => setSplatPick(e.target.value)}>
            <option value={AUTO}>{state.current?.step === "train" ? "follow the training" : "newest"}</option>
            {splats.map((s) => <option key={s.path} value={s.path}>{s.label}</option>)}
          </select>
        </section>
        <section>
          <span className="label">compare with</span>
          <select value={comparePick} onChange={(e) => setComparePick(e.target.value)} disabled={splats.length < 2}>
            <option value={NONE}>nothing</option>
            {splats.map((s) => <option key={s.path} value={s.path}>{s.label}</option>)}
          </select>
        </section>
        <section>
          <span className="label">layers</span>
          {(Object.keys(show) as (keyof Layers)[]).map((key) => (
            <label key={key}>
              <input type="checkbox" checked={show[key]} onChange={() => setShow({ ...show, [key]: !show[key] })} />
              {key}
            </label>
          ))}
          <label>
            point size
            <input type="range" min={0.2} max={4} step={0.1} value={pointSize} onChange={(e) => setPointSize(Number(e.target.value))} />
          </label>
        </section>
        {camera ? (
          <section>
            <span className="label">camera</span>
            <img className="side-photo" src={thumbUrl(`images/${camera.name}`, 384, 0)} alt={camera.name} />
            <dl className="kv">
              <dt>photo</dt><dd title={camera.name}>{camera.name}</dd>
              <dt>sees</dt><dd>{camera.points.toLocaleString()} points</dd>
              <dt>size</dt><dd>{camera.width}×{camera.height}</dd>
            </dl>
            <div className="buttons">
              <button className="button" onClick={() => setLook({ name: camera.name, n: (look?.n ?? 0) + 1 })}>Look through it</button>
              <a className="button" href={fileUrl(`images/${camera.name}`)} target="_blank" rel="noreferrer">Open photo</a>
              <button className="link-button" onClick={() => setSelected(null)}>clear</button>
            </div>
          </section>
        ) : model && show.cameras ? (
          <p className="label">Click a camera to see its photo.</p>
        ) : null}
        {model && (
          <section>
            <span className="label">reconstruction</span>
            <dl className="kv">
              <dt>cameras</dt><dd>{model.cameras.length.toLocaleString()}</dd>
              <dt>points</dt><dd>{model.stats.points.toLocaleString()}</dd>
              {model.points.xyz.length / 3 < model.stats.points && <><dt>drawn</dt><dd>{(model.points.xyz.length / 3).toLocaleString()}</dd></>}
              <dt>reprojection</dt><dd>{model.stats.mean_error.toFixed(2)} px</dd>
              <dt>track length</dt><dd>{model.stats.mean_track.toFixed(1)}</dd>
            </dl>
          </section>
        )}
      </aside>
      {!model && !splatChoice ? (
        <div className="canvas">
          <div className="empty">Cameras appear here once COLMAP places them, and the splat once Brush exports one.</div>
        </div>
      ) : (
        <div className="canvases">
          {views.map((splat, id) => (
            <View
              key={id} id={id} sync={compareChoice ? sync : null} model={model} frame={frame} show={show} pointSize={pointSize}
              splat={splat ?? null} selected={selected} onPick={setSelected} look={look} label={compareChoice ? splat?.label : undefined}
            />
          ))}
          <div className="canvas-hint">click a camera · drag to orbit · right-drag to pan · scroll to zoom</div>
        </div>
      )}
    </div>
  );
}

function View({ id, sync, model, frame, show, pointSize, splat, selected, onPick, look, label }: {
  id: number; sync: MutableRefObject<Sync> | null; model: Model | null; frame: Frame | null; show: Layers; pointSize: number;
  splat: Splat | null; selected: string | null; onPick: (name: string) => void; look: LookRequest | null; label?: string;
}) {
  const [status, setStatus] = useState("");
  const group = useRef<THREE.Group>(null);
  const photo = model?.cameras.find((c) => c.name === selected);
  // Looking through a camera hides its photo, so the splat shows from where the photo was taken; touching the view ends it.
  const [looking, setLooking] = useState(false);
  useEffect(() => setLooking(look !== null), [look]);
  const lead = () => {
    if (sync) sync.current.leader = id;
  };
  return (
    <div className="canvas" onPointerDown={() => { lead(); setLooking(false); }} onWheel={lead}>
      <Canvas gl={{ antialias: false }} camera={{ fov: 55, near: 0.01, far: 5000 }} dpr={[1, 2]}>
        <color attach="background" args={["#171520"]} />
        <SparkSetup />
        {frame && <FitCamera frame={frame} />}
        {sync && <SyncCamera id={id} sync={sync} />}
        <group ref={group} quaternion={frame?.rotation} position={frame?.offset}>
          {model && show.points && <Points model={model} size={pointSize * (frame?.scale ?? 1) * 0.004} />}
          {model && show.cameras && <Frustums cameras={model.cameras} depth={(frame?.scale ?? 1) * 0.06} focus={selected} />}
          {photo && show.cameras && !looking && (
            <Suspense fallback={null}>
              <Photo camera={photo} depth={(frame?.scale ?? 1) * 0.15} />
            </Suspense>
          )}
          {show.splat && splat && <SplatView path={splat.path} onStatus={setStatus} />}
        </group>
        {model && show.cameras && <CameraPicker cameras={model.cameras} group={group} onPick={onPick} />}
        {model && frame && <LookThrough request={look} cameras={model.cameras} group={group} scale={frame.scale} />}
        {show.grid && frame && <gridHelper args={[frame.scale * 8, 40, "#34294f", "#34294f"]} position={[0, frame.floor, 0]} />}
        <OrbitControls makeDefault enableDamping dampingFactor={0.12} />
      </Canvas>
      {label && <div className="canvas-label">{label}</div>}
      {status && <div className="canvas-note">{status}</div>}
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

type Controls = { target: THREE.Vector3; update: () => void };

function useControls(): Controls | null {
  // SAFETY: the only default controls are the <OrbitControls makeDefault> in View.
  return useThree((s) => s.controls) as unknown as Controls | null;
}

function FitCamera({ frame }: { frame: Frame }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const controls = useControls();
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

/** Keep the two views of a comparison on the same camera. */
function SyncCamera({ id, sync }: { id: number; sync: MutableRefObject<Sync> }) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const controls = useControls();
  useFrame(() => {
    if (!controls) return;
    const shared = sync.current;
    if (shared.leader === id || !shared.set) {
      shared.position.copy(camera.position);
      shared.target.copy(controls.target);
      shared.fov = camera.fov;
      shared.set = true;
    } else if (!camera.position.equals(shared.position) || !controls.target.equals(shared.target) || camera.fov !== shared.fov) {
      camera.position.copy(shared.position);
      controls.target.copy(shared.target);
      camera.fov = shared.fov;
      camera.updateProjectionMatrix();
      controls.update();
    }
  });
  return null;
}

function cameraWorld(camera: Camera, group: THREE.Group, x: number, y: number, z: number): THREE.Vector3 {
  const r = camera.rotation;
  const local = new THREE.Vector3(
    r[0] * x + r[1] * y + r[2] * z + camera.center[0],
    r[3] * x + r[4] * y + r[5] * z + camera.center[1],
    r[6] * x + r[7] * y + r[8] * z + camera.center[2],
  );
  return group.localToWorld(local);
}

/** A click (not a drag) picks the camera whose centre is nearest on screen, within 14 px. */
function CameraPicker({ cameras, group, onPick }: { cameras: Camera[]; group: MutableRefObject<THREE.Group | null>; onPick: (name: string) => void }) {
  const { camera, gl, size } = useThree();
  useEffect(() => {
    const element = gl.domElement;
    let down: [number, number] | null = null;
    const press = (e: PointerEvent) => (down = [e.clientX, e.clientY]);
    const release = (e: PointerEvent) => {
      if (!down || !group.current || Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 4) return;
      const rect = element.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      group.current.updateWorldMatrix(true, false);
      let best: [number, string] | null = null;
      for (const c of cameras) {
        const p = cameraWorld(c, group.current, 0, 0, 0).project(camera);
        if (p.z > 1) continue; // behind the viewer
        const d = Math.hypot(((p.x + 1) / 2) * size.width - x, ((1 - p.y) / 2) * size.height - y);
        if (d < 14 && (!best || d < best[0])) best = [d, c.name];
      }
      if (best) onPick(best[1]);
    };
    element.addEventListener("pointerdown", press);
    element.addEventListener("pointerup", release);
    return () => {
      element.removeEventListener("pointerdown", press);
      element.removeEventListener("pointerup", release);
    };
  }, [cameras, camera, gl, size, group, onPick]);
  return null;
}

/** Put the viewer where the photo was taken, looking where it looked, with its vertical field of view. */
function LookThrough({ request, cameras, group, scale }: {
  request: LookRequest | null; cameras: Camera[]; group: MutableRefObject<THREE.Group | null>; scale: number;
}) {
  const camera = useThree((s) => s.camera) as THREE.PerspectiveCamera;
  const controls = useControls();
  useEffect(() => {
    const c = cameras.find((c) => c.name === request?.name);
    if (!c || !group.current || !controls) return;
    group.current.updateWorldMatrix(true, false);
    camera.position.copy(cameraWorld(c, group.current, 0, 0, 0));
    controls.target.copy(cameraWorld(c, group.current, 0, 0, scale * 0.5));
    camera.fov = THREE.MathUtils.radToDeg(2 * Math.atan(c.height / 2 / c.fy));
    camera.updateProjectionMatrix();
    controls.update();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [request]);
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

/** The selected camera's photo, hung in its frustum at `depth`, as the camera saw it. */
function Photo({ camera, depth }: { camera: Camera; depth: number }) {
  const texture = useLoader(THREE.TextureLoader, thumbUrl(`images/${camera.name}`, 512, 0));
  const matrix = useMemo(() => {
    const r = camera.rotation;
    const [cx, cy, cz] = camera.center;
    return new THREE.Matrix4().set(r[0], r[1], r[2], cx, r[3], r[4], r[5], cy, r[6], r[7], r[8], cz, 0, 0, 0, 1);
  }, [camera]);
  texture.colorSpace = THREE.SRGBColorSpace;
  const w = (camera.width / camera.fx) * depth;
  const h = (camera.height / camera.fy) * depth;
  return (
    <group matrixAutoUpdate={false} matrix={matrix}>
      {/* camera y points down, so the plane is flipped to show the photo upright */}
      <mesh position={[0, 0, depth]} scale={[w, -h, 1]}>
        <planeGeometry />
        <meshBasicMaterial map={texture} side={THREE.DoubleSide} transparent opacity={0.92} toneMapped={false} />
      </mesh>
    </group>
  );
}

/** A trained splat; a new one replaces the shown one only once it has loaded. */
function SplatView({ path, onStatus }: { path: string; onStatus: (status: string) => void }) {
  const [mesh, setMesh] = useState<SplatMesh | null>(null);
  useEffect(() => {
    let cancelled = false;
    onStatus(`loading ${path.split("/").pop()}`);
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
        onStatus(`${path.split("/").pop()}, ${next.packedSplats?.numSplats?.toLocaleString() ?? "?"} splats`);
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
