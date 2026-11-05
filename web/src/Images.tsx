import { useMemo, useState } from "react";
import { thumbUrl, useFetched } from "./api";
import type { ImageInfo, Model, State } from "./api";

type Overlay = "none" | "mask" | "exclude";
type Sort = "name" | "sharpness" | "points";
type Filter = "all" | "unregistered" | "registered";

export function Images({ state, onShow }: { state: State; onShow: (name: string) => void }) {
  const images = useFetched<ImageInfo[]>("images", {}, [state.images.count, state.masks.count, state.masks.exclude]);
  const model = useFetched<Model>(state.model ? "model" : null, { path: state.model?.path ?? "" }, state.model?.mtime);
  const [overlay, setOverlay] = useState<Overlay>(state.masks.count ? "mask" : state.masks.exclude ? "exclude" : "none");
  const [sort, setSort] = useState<Sort>("name");
  const [filter, setFilter] = useState<Filter>("all");
  const [size, setSize] = useState(state.images.panos ? 110 : 190);
  const [open, setOpen] = useState<ImageInfo | null>(null);

  const cameras = useMemo(() => new Map(model?.cameras.map((c) => [c.name, c]) ?? []), [model]);
  const shown = useMemo(() => {
    let list = images ?? [];
    if (model && filter !== "all") list = list.filter((i) => cameras.has(i.name) === (filter === "registered"));
    const key: Record<Sort, (i: ImageInfo) => number | string> = {
      name: (i) => i.name,
      sharpness: (i) => i.sharpness,
      points: (i) => cameras.get(i.name)?.points ?? -1,
    };
    return [...list].sort((a, b) => (key[sort](a) < key[sort](b) ? -1 : key[sort](a) > key[sort](b) ? 1 : 0));
  }, [images, model, cameras, filter, sort]);

  if (!images) return <div className="empty">{state.images.count ? `Reading ${state.images.count} images…` : "No images yet."}</div>;
  const aspect = state.images.size ? state.images.size[0] / state.images.size[1] : 1;
  const registered = model ? images.filter((i) => cameras.has(i.name)).length : null;
  const pano = state.images.panos > 0 && sort === "name";
  const groups = pano ? groupBy(shown, (i) => i.frame) : [["", shown] as [string, ImageInfo[]]];

  const tile = (image: ImageInfo) => {
    const camera = cameras.get(image.name);
    const maskPath = overlay === "mask" ? image.mask : overlay === "exclude" ? image.exclude : false;
    return (
      <div key={image.name} className={`tile ${model && !camera ? "unregistered" : ""}`} onClick={() => setOpen(image)} title={image.name}>
        <div className="thumb" style={{ "--aspect": aspect } as React.CSSProperties}>
          <img loading="lazy" src={thumbUrl(`images/${image.name}`, size > 200 ? 384 : 256, image.mtime)} alt="" />
          {maskPath && <img className="mask" loading="lazy" src={thumbUrl(maskPath, 256, image.mtime)} alt="" />}
        </div>
        <div className="tile-info">
          <span className="name">{pano ? image.view : image.name}</span>
          {model && (camera ? <span className="reg">{camera.points}</span> : <span className="unreg">✗</span>)}
        </div>
      </div>
    );
  };

  return (
    <div className="images">
      <div className="toolbar">
        <span>
          <b style={{ color: "var(--cyan)" }}>{images.length}</b> images
          {registered !== null && <> · <b style={{ color: "var(--green)" }}>{registered}</b> registered</>}
          {state.images.panos > 0 && <> · {state.images.panos} panoramas × {state.images.views} views</>}
        </span>
        <label>overlay
          <select value={overlay} onChange={(e) => setOverlay(e.target.value as Overlay)}>
            <option value="none">none</option>
            <option value="mask" disabled={!state.masks.count}>subject mask</option>
            <option value="exclude" disabled={!state.masks.exclude}>exclude mask</option>
          </select>
        </label>
        <label>sort
          <select value={sort} onChange={(e) => setSort(e.target.value as Sort)}>
            <option value="name">name</option>
            <option value="sharpness">sharpness</option>
            <option value="points" disabled={!model}>3D points seen</option>
          </select>
        </label>
        <label>show
          <select value={filter} onChange={(e) => setFilter(e.target.value as Filter)} disabled={!model}>
            <option value="all">all</option>
            <option value="unregistered">unregistered</option>
            <option value="registered">registered</option>
          </select>
        </label>
        <label>size <input type="range" min={70} max={360} value={size} onChange={(e) => setSize(Number(e.target.value))} /></label>
      </div>
      <div className="grid-scroll">
        {groups.map(([frame, list]) => (
          <div key={frame} className={pano ? "frame-row" : ""}>
            {pano && (
              <div className="frame-label" title={frame} style={{ width: size * 2 }}>
                {list[0]?.pano && <img loading="lazy" src={thumbUrl(list[0].pano, 512, list[0].mtime)} alt="" />}
                {frame}
              </div>
            )}
            <div
              className="grid"
              style={{ gridTemplateColumns: pano ? `repeat(${state.images.views}, minmax(0, 1fr))` : `repeat(auto-fill, minmax(${size}px, 1fr))` }}
            >
              {list.map(tile)}
            </div>
          </div>
        ))}
      </div>
      {open && <Lightbox image={open} camera={cameras.get(open.name)} modelLoaded={!!model} onClose={() => setOpen(null)} onShow={onShow} />}
    </div>
  );
}

function Lightbox({ image, camera, modelLoaded, onClose, onShow }: {
  image: ImageInfo; camera: Model["cameras"][number] | undefined; modelLoaded: boolean; onClose: () => void; onShow: (name: string) => void;
}) {
  return (
    <div className="lightbox" onClick={onClose}>
      <div className="lightbox-images" onClick={(e) => e.stopPropagation()}>
        <img src={thumbUrl(`images/${image.name}`, 1024, image.mtime)} alt={image.name} />
        {image.mask && <img src={thumbUrl(image.mask, 1024, image.mtime)} alt="mask" />}
        {image.exclude && <img src={thumbUrl(image.exclude, 1024, image.mtime)} alt="exclude mask" />}
      </div>
      <dl className="kv" onClick={(e) => e.stopPropagation()}>
        <dt>image</dt><dd>{image.name}</dd>
        <dt>sharpness</dt><dd>{image.sharpness.toFixed(0)}</dd>
        {image.masks !== undefined && <><dt>mask keeps</dt><dd>{(image.masks * 100).toFixed(1)}%</dd></>}
        {image["masks-exclude"] !== undefined && <><dt>exclude keeps</dt><dd>{(image["masks-exclude"] * 100).toFixed(1)}%</dd></>}
        {modelLoaded && <><dt>registered</dt><dd>{camera ? `yes, sees ${camera.points} points` : "no"}</dd></>}
      </dl>
      {camera && <button className="button" onClick={() => onShow(image.name)}>show camera in 3D</button>}
    </div>
  );
}

function groupBy<T>(items: T[], key: (item: T) => string): [string, T[]][] {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const k = key(item);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k)!.push(item);
  }
  return [...groups];
}
