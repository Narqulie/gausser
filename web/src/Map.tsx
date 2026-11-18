import { useEffect, useMemo, useRef, useState } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { fileUrl, useFetched } from "./api";
import type { GeoData, MapRun, State } from "./api";

const BASES = {
  satellite: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", "Esri World Imagery"],
  map: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png", "© OpenStreetMap"],
} as const;
type Base = keyof typeof BASES;
type Layer = "ortho" | "dsm";

/** A georeferenced scene on a web map: cameras and points as solve places them, then the orthophoto and DSM of `map`. */
export function Map({ state }: { state: State }) {
  const solving = state.current?.step === "solve";
  const geo = useFetched<GeoData>("geo", {}, solving ? Math.floor(state.now / 10) : [state.model?.mtime, state.geo?.origin]);
  const maps = state.maps;
  const [runPick, setRunPick] = useState<string | null>(null);
  const run = maps.find((m) => m.path === runPick) ?? maps[maps.length - 1];
  const [base, setBase] = useState<Base>("satellite");
  const [layer, setLayer] = useState<Layer>("ortho");
  const [opacity, setOpacity] = useState(1);
  const [show, setShow] = useState({ cameras: true, points: false });

  const element = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const tiles = useRef<L.TileLayer | null>(null);
  const overlay = useRef<L.ImageOverlay | null>(null);
  const markers = useRef<{ cameras: L.LayerGroup; points: L.LayerGroup } | null>(null);
  const fitted = useRef(false);

  useEffect(() => {
    const m = L.map(element.current!, { maxZoom: 23, preferCanvas: true, zoomControl: true });
    L.control.scale({ imperial: false }).addTo(m);
    markers.current = { cameras: L.layerGroup().addTo(m), points: L.layerGroup() };
    map.current = m;
    return () => {
      m.remove();
      map.current = null;
    };
  }, []);

  useEffect(() => {
    const [url, attribution] = BASES[base];
    tiles.current?.remove();
    tiles.current = L.tileLayer(url, { maxNativeZoom: 19, maxZoom: 23, attribution }).addTo(map.current!);
    tiles.current.bringToBack();
  }, [base]);

  useEffect(() => {
    overlay.current?.remove();
    overlay.current = null;
    if (!run?.bounds_latlon) return;
    const image = `${run.path}/${layer === "ortho" ? "ortho-web.webp" : "dsm-web.webp"}`;
    overlay.current = L.imageOverlay(fileUrl(image), run.bounds_latlon, { opacity }).addTo(map.current!);
    if (!fitted.current) {
      map.current!.fitBounds(run.bounds_latlon, { padding: [30, 30] });
      fitted.current = true;
    }
    // opacity has its own effect, so changing it does not reload the image
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run?.path, run?.mtime, layer]);

  useEffect(() => overlay.current?.setOpacity(opacity) && undefined, [opacity]);

  useEffect(() => {
    if (!geo || !markers.current) return;
    const { cameras, points } = markers.current;
    cameras.clearLayers();
    for (const c of geo.cameras) {
      const oblique = c.pitch !== undefined && c.pitch > -75;
      const colour = oblique ? "#ff7edb" : "#36f9f6";
      const where = L.latLng(c.lat, c.lon);
      if (oblique && c.heading !== undefined) {
        const ahead = destination(where, c.heading, 6);
        L.polyline([where, ahead], { color: colour, weight: 1.5, opacity: 0.9, interactive: false }).addTo(cameras);
      }
      L.circleMarker(where, { radius: 3.5, color: colour, weight: 1, fillOpacity: 0.9 })
        .bindTooltip(`${c.name}${c.pitch !== undefined ? ` · ${c.pitch.toFixed(0)}°` : ""}`)
        .addTo(cameras);
    }
    points.clearLayers();
    if (geo.points) {
      const { latlon, rgb } = geo.points;
      for (let i = 0; i < latlon.length / 2; i++) {
        L.circleMarker([latlon[2 * i], latlon[2 * i + 1]], {
          radius: 1.2, stroke: false, fillOpacity: 0.9, interactive: false,
          fillColor: `rgb(${rgb[3 * i]},${rgb[3 * i + 1]},${rgb[3 * i + 2]})`,
        }).addTo(points);
      }
    }
    if (!fitted.current && geo.cameras.length) {
      map.current!.fitBounds(L.latLngBounds(geo.cameras.map((c) => [c.lat, c.lon])), { padding: [40, 40] });
      fitted.current = true;
    }
  }, [geo]);

  useEffect(() => {
    if (!markers.current || !map.current) return;
    for (const key of ["cameras", "points"] as const) {
      if (show[key]) markers.current[key].addTo(map.current);
      else markers.current[key].remove();
    }
  }, [show]);

  const counts = useMemo(() => {
    const cams = geo?.cameras ?? [];
    const oblique = cams.filter((c) => c.pitch !== undefined && c.pitch > -75).length;
    return { nadir: cams.length - oblique, oblique };
  }, [geo]);

  return (
    <div className="geomap">
      <div className="toolbar">
        <label>base
          <select value={base} onChange={(e) => setBase(e.target.value as Base)}>
            <option value="satellite">satellite</option>
            <option value="map">map</option>
          </select>
        </label>
        {maps.length > 0 && run && (
          <>
            <label>run
              <select value={run.path} onChange={(e) => setRunPick(e.target.value)}>
                {[...maps].reverse().map((m) => <option key={m.path} value={m.path}>{m.path.split("/").pop()} · {m.kind ?? "splat"}</option>)}
              </select>
            </label>
            <span className="segmented">
              {(["ortho", "dsm"] as const).map((l) => (
                <button key={l} className={layer === l ? "active" : ""} onClick={() => setLayer(l)}>{l === "ortho" ? "orthophoto" : "surface (DSM)"}</button>
              ))}
            </span>
            <label>opacity
              <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
            </label>
          </>
        )}
        <label><input type="checkbox" checked={show.cameras} onChange={(e) => setShow({ ...show, cameras: e.target.checked })} /> cameras</label>
        {geo?.points && (
          <label><input type="checkbox" checked={show.points} onChange={(e) => setShow({ ...show, points: e.target.checked })} /> points</label>
        )}
      </div>
      <div className="geomap-body">
        <div ref={element} className="geomap-map" />
        <MapLegend state={state} geo={geo} run={run} layer={layer} counts={counts} />
      </div>
    </div>
  );
}

function MapLegend({ state, geo, run, layer, counts }: {
  state: State; geo: GeoData | null; run: MapRun | undefined; layer: Layer; counts: { nadir: number; oblique: number };
}) {
  const fit = state.geo;
  return (
    <aside className="geomap-legend">
      <div className="legend-row"><i style={{ background: "var(--solve)" }} />{counts.nadir} straight down</div>
      {counts.oblique > 0 && <div className="legend-row"><i style={{ background: "var(--train)" }} />{counts.oblique} oblique</div>}
      {geo?.source === "gps" && <p className="muted">Positions from the photos' GPS; the model is not georeferenced yet.</p>}
      {fit && (
        <dl>
          <dt>CRS</dt><dd>{fit.crs}</dd>
          <dt>fit to GPS</dt><dd>{fit.images} cameras, {fit.rms_horizontal_m} m horizontal, {fit.rms_vertical_m} m vertical RMS</dd>
        </dl>
      )}
      {run ? (
        <>
          <dl>
            <dt>cell</dt><dd>{(run.resolution_m * 100).toFixed(0)} cm · {run.size[0].toLocaleString()} × {run.size[1].toLocaleString()}</dd>
            {!state.map_current && run === state.maps[state.maps.length - 1] && <><dt>state</dt><dd className="stale">older than the newest splat</dd></>}
          </dl>
          {layer === "dsm" && run.dsm_range_m && (
            <div className="dsm-ramp">
              <span className="ramp" />
              <span>{run.dsm_range_m[0].toFixed(1)} m</span><span>{run.dsm_range_m[1].toFixed(1)} m</span>
            </div>
          )}
          <div className="downloads">
            <a href={fileUrl(`${run.path}/ortho.tif`)} download>ortho.tif</a>
            <a href={fileUrl(`${run.path}/dsm.tif`)} download>dsm.tif</a>
          </div>
        </>
      ) : (
        <p className="muted">{fit ? "No orthophoto yet: `gausser map` (or `run --map`) after training." : ""}</p>
      )}
    </aside>
  );
}

/** The point `metres` from `from` along a compass heading (flat-earth; fine over a few metres). */
function destination(from: L.LatLng, heading: number, metres: number): L.LatLng {
  const rad = (heading * Math.PI) / 180;
  const north = metres * Math.cos(rad), east = metres * Math.sin(rad);
  return L.latLng(from.lat + north / 111_320, from.lng + east / (111_320 * Math.cos((from.lat * Math.PI) / 180)));
}
