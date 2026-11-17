# gausser

Photos or video in, Gaussian splats out.

**Full documentation:** [`docs/index.html`](docs/index.html): pipeline, capture types, scene state, monitor, command reference, architecture, troubleshooting.

One command wraps the steps:

```
frames → mask → solve → [dense] → train → lod   [mesh] [map]
(video)  (SAM3)  (COLMAP) (OpenMVS) (Brush/Msplat) (splat-transform) (OpenMVS) (orthophoto)
```

`dense`, `mesh` and `map` are optional (`run --dense`, `run --mesh`, `run --map`).

## Quick start

The guided way. It reads the scene, recommends the next step, asks only what that
step needs, and prints the equivalent command before running it:

```bash
gausser clip.mp4           # makes clip/ next to the video, then guides you
gausser ~/photos/statue    # a folder of photos; offers to move them into images/
gausser                    # the current folder
```

The direct way, for scripts and overnight runs:

```bash
mkdir -p statue/images && cd statue
cp ~/photos/statue/*.jpg images/        # or: gausser frames walkaround.mp4
gausser run --prompt "statue" --monitor  # mask → solve → train, watched in the browser
gausser view                             # open the result in SuperSplat
```

`--monitor` (on `run`, `frames`, `mask`, `solve`, `train` and `lod`) opens the scene in the web monitor at
http://127.0.0.1:8737/: each step's progress and live log, the COLMAP cameras and points as the mapper adds
them, and the splat as Brush trains it (every `--export-every` steps, 5000 by default). Click a camera to see
its photo or look through it, compare two splats side by side, show a `gausser mesh` result in the same
frame (alone or over the splat), and compare training runs. With
`train --eval-split N`, the Renders tab shows Brush's render of each held-out photo next to the photo, with PSNR.
A toast (and, if you allow it, a desktop notification) tells you when a step ends.

One monitor serves every scene gausser has worked on: http://127.0.0.1:8737/ lists them with every step
that ran. The list lives in `~/.local/share/gausser/history.jsonl` (`GAUSSER_HOME` moves it). The monitor
is read-only, stays up as a viewer after a run, and quits 10 minutes after its last tab closes.
`gausser monitor` opens the current scene in it (`--all`: the list).

A scene is a folder with its photos in `images/`. If the photos are in the
folder itself, `gausser run --import` moves them into `images/`.

`gausser run` is unattended. It skips steps that are already up to date, so you
can rerun it after a failure or after adding photos. Use `--from STEP` to redo a
step and everything after it. For example, `gausser run --from solve --matcher sequential`.
If COLMAP places fewer than half the images, `run` stops before training: the
images don't overlap enough. `gausser train` trains on them anyway.

While a command runs, gausser keeps the Mac awake with `caffeinate`, because a
sleeping Mac pauses training. Closing a laptop's lid still puts it to sleep unless
an external display is attached.

Without `--prompt`, no new masks are made, but existing masks are still used. Add
`--video clip.mp4` to extract frames first, `--dense` to start Brush from a dense point cloud,
`--lod` to finish with a streamed LOD set and `--mesh` to also build a textured mesh.

## Commands

| Command | Does | Writes |
|---|---|---|
| `gausser frames SOURCE [--range START-END] [--spacing low\|medium\|high] [--min-frames N] [--max-frames N] [--every N] [--overlap PERCENT]` | Picks sharp frames from a video wherever the camera has moved on; 360° frames (equirect, or a QooCam 8K raw video) become 14 views each; an iPhone LiDAR scan folder adds ARKit poses and LiDAR points; a DJI video's `.SRT` gives each frame its GPS | `images/` (and `panos/`, `arkit/`, `frames.json`) |
| `gausser mask PROMPT [--miss drop]` | SAM3 masks the subject | `masks/`, `colmap/masks/` |
| `gausser mask PROMPT --exclude [--dilation N]` | SAM3 blacks out every instance of PROMPT (e.g. people), except inside `masks/` | `masks-exclude/` |
| `gausser solve [--no-masks]` | COLMAP finds the camera poses; images with GPS put the model in a map projection | `database.db`, `sparse/0/` (and `sparse/geo.json`) |
| `gausser dense [--resolution-level N]` | OpenMVS estimates a depth map per image (on the GPU) and fuses them into a dense point cloud | `dense/` |
| `gausser crop [--box=x,y,z,X,Y,Z --up=x,y,z \| --clear]` | Sets the box `train` and `mesh` keep to; draw it in the monitor (3D tab, crop), which shows this command | `sparse/crop.json` |
| `gausser train [--no-masks \| --exclude-masks] [--init auto\|dense\|sparse] [--adaptive [--max-iterations N]] [-- args]` | Brush (default) or `--trainer msplat` trains splats; Brush starts from `dense/` when it exists; args after `--` go to Brush | `runs/brush/<time>/exports/*.ply` |
| `gausser mesh [--faces N] [--no-refine] [--texture-size N]` | OpenMVS meshes, refines and textures the dense cloud | `runs/mesh/<time>/mesh.glb` |
| `gausser map [PLY] [--resolution M]` | Renders the splat straight down into an orthophoto and a surface model (DSM) in the scene's CRS, and a web map with the orthophoto over it | `runs/map/<time>/{ortho.tif,dsm.tif,map.html}` |
| `gausser lod [PLY]` | splat-transform builds streamed LOD (newest export by default) | `runs/lod/<time>/` |
| `gausser align MOVING.ply --to REF… -o OUT.ply [--no-scale]` | Moves a re-trained splat into the coordinate frame of a published one (scale, rotation, translation fitted to the splat centres) | `OUT.ply` |
| `gausser view [PLY]` | Opens a splat (newest export by default) in a local SuperSplat | — |
| `gausser orbit [PLY] [--seconds 12] [--size 1920x1080] [--background jutaka] [--target X Y Z --radius R --height H] [-o OUT.mp4]` | SuperSplat renders a 360° orbit video (newest export by default) | `runs/orbit/<time>/orbit.mp4` |
| `gausser status` | Shows what the scene has and which tools are installed | — |
| `gausser monitor [--all] [--no-open]` | Opens the scene (or the list of all scenes) in the web monitor: steps, logs, stats, images, COLMAP cameras, splats, meshes, eval renders | — |

`gausser frames` needs no tuning for a walk-around video. One pass decodes the video (ffmpeg) and scores every
frame for focus and camera motion (OpenCV optical flow); a frame is picked each time the view has moved on by a
set share of the image, and the sharpest frame of that stretch is the one kept. Several spacings are tracked at
once, so the result is moved to a finer or wider spacing if it would be under `--min-frames` (150) or over
`--max-frames` (600); `--spacing low|medium|high` moves the starting point. A second pass writes only the picked
frames, full size, as `images/frame_<number>.jpg`, with `frames.json` listing each one. Still stretches give a frame
every 3 s. HDR (HLG/PQ) video is converted to SDR. `--every N` takes the sharpest of every N video frames instead.
A folder of photos is copied whole; a folder of videos (subfolders included) is processed video by video (`p01_`, `p02_` … prefixes).

`gausser dense` undistorts the images (`colmap image_undistorter`) into `dense/images/`, then OpenMVS
(`InterfaceCOLMAP`, `DensifyPointCloud`) estimates a depth map per image by patch-match stereo (Metal on Apple
Silicon) and keeps the points that at least two images agree on: `dense/scene_dense.ply`, tens of millions of points
with the images that saw them. `--resolution-level 1` (default) works on half-size images. IMG_8779 (350 iPhone
4K frames): 47 s of depth maps, 4 min in all, 32M points. Depth maps are deleted after fusion; `dense/` still takes
about the size of `images/` plus 2 GB.
`gausser train` then gives Brush 500,000 random points of the cloud instead of COLMAP's sparse points (it writes them
as `points3D.bin` into Brush's own folder; `sparse/` is untouched) and marks the run `init-dense` or `init-sparse`.
`--init sparse` keeps COLMAP's points. `auto` (default) uses the dense cloud only for unmasked training: the cloud
ignores masks, and a masked run would keep the background points it starts from. `run --dense` on a trained scene
builds `dense/` and then trains again.

`gausser mesh` runs `ReconstructMesh` (Delaunay tetrahedra, graph cut on what the cameras see through),
`RefineMesh` (moves the surface until the photos agree; CPU only, the slow part, `--no-refine` skips it) and
`TextureMesh` (a photo per triangle, seams levelled) into `runs/mesh/<time>/mesh.glb` with its textures beside it.
`--faces 30000` decimates before texturing. IMG_8779: 45 min with refinement; at 30k faces the mesh was 10 MB and
scored 21.0 dB PSNR against the photos, better than meshes made from the splat.

Every command takes `-s/--scene DIR` (default: current folder) and `--help`.
Only one command at a time can change a scene; a second one stops with an error
instead of overwriting the first one's work. `align` needs no scene at all. `status`, `view`, `orbit` and `monitor` always run
(`orbit` only adds a new run folder).

`gausser orbit` needs `sparse/0` and a Brush export. The camera circles the point nearest all
camera viewing axes, at the median camera distance times `--zoom` (1.3) and aimed `--drop` (0.35, splat
units) below that point, with "up" from the cameras (normal to their level right axes; Brush's `Vertical axis` guess can be far off on top-down-heavy drone sets). It drives a visible Chromium
window through Playwright (installed by `scripts/install-macos.sh` into `~/.local/share/gausser/node`;
override with `GAUSSER_NODE_DIR`, and `GAUSSER_CHROME` picks another browser executable), so leave the window alone
while it renders.
`--background` takes `#rrggbb`, `jutaka` (the public-art viewer's sky gradient) or your own
`linear-gradient(to bottom, rgb(r,g,b) P%, …)`. A gradient renders the orbit over black and over white, then ffmpeg
recovers the splat's alpha and blends it over the gradient (h264, crf 16).

## Drone mapping

A straight-down (nadir) drone flight gives an orthophoto and a surface model:

```bash
mkdir field && cd field
gausser frames ~/Desktop/DJI_0003.MP4 --overlap 80   # or put drone photos in images/
gausser run --map                                    # solve (georeferenced) → train → map
open runs/map/*/map.html
```

- **GPS.** Drone photos carry GPS in EXIF. For video, turn on *Video Subtitles* in DJI Fly: the drone then
  writes `DJI_xxxx.SRT` next to each `.MP4`, one position per frame. `frames` reads it and writes each picked
  frame's position into the frame's EXIF, so frames and photos look the same to the rest of the pipeline.
- **`--overlap 80`** picks a frame whenever the ground has moved 20% of the frame's short side since the last
  one, so consecutive frames share at least 80%, however long the flight. It replaces `--spacing` and the
  frame-count limits.
- **solve** matches each image with its GPS neighbours (`spatial` matching) and then fits a similarity from
  the COLMAP cameras onto their GPS positions. The model is moved into EPSG:3067 (ETRS-TM35FIN) in Finland,
  else the WGS 84 UTM zone, with gausser's axes kept: x east, y down, z north, in metres from the origin in
  `sparse/geo.json`. Splats trained afterwards are in that frame too. The log and `gausser status` give the
  cameras' RMS distance from their GPS; a consumer drone's GPS is good to a metre or two, and heights are
  the drone's GPS altitude, not survey heights.
- **map** renders the newest splat with an orthographic camera looking down: every Gaussian projects onto
  the ground as a 2D Gaussian and is blended top first, on a grid of `--resolution` metres (3 cm by
  default). The DSM is the height where each cell becomes half covered. Both are Cloud Optimized GeoTIFFs;
  `map.html` lays the orthophoto over satellite imagery or OpenStreetMap with the camera positions.
- **Flying.** Fly parallel lines with 70% side overlap, at a steady height and speed, camera at −90°, shutter
  1/500 s or faster. Nadir-only flights let the model bend into a shallow bowl; a crosshatch or a −45° pass
  holds it flat. Snow, water and shadows give COLMAP little texture to match.

## Leaving people out

`mask --exclude` masks transient things (passers-by, the operator of a 360° camera) without touching the
subject masks or the COLMAP model:

```bash
gausser mask person --exclude --dilation 8   # masks-exclude/: black = every person, grown by 8 px
gausser train --exclude-masks                # Brush ignores those pixels
```

`masks-exclude/<stem>.png` is black over every instance of the prompt grown by `--dilation` px and white
elsewhere and wherever the subject mask in `masks/` is white, since SAM3 may call a bronze statue a
"person". Use 8 px: people's edges and shadows smear. In a 360° scene `solve` also masks those pixels'
features, since the operator moves with the camera.

A masked run on an unmasked solve (`mask --miss drop`, `solve --no-masks`, `train`) starts from the whole
scene's COLMAP points and keeps opaque background splats that the masked loss never removes, unless Brush
trains the outside of the masks transparent: `gausser train -- --alpha-mode transparent`. For cleaner
subject edges, mask with `--dilation -2`: it shrinks the Brush masks by 2 px, so edge pixels mixed with
background are not trained (COLMAP masks are unchanged).

## 360° video

`gausser frames` recognises stitched 360° frames by their 2:1 (equirectangular) shape. It keeps them in
`panos/` and writes 14 pinhole views of each into `images/` as `cNN_<frame>.jpg` (4 directions at
-35°, 0° and 35° pitch plus straight up and straight down, 90° wide, so a 5760-pixel-wide frame gives
1440×1440 views). This is COLMAP's own panorama workflow (`pycolmap.panorama`). From there every
step works as usual, and `solve` treats the 14 views of one frame as a camera rig with known,
fixed rotations, so COLMAP only solves one pose per frame.

```bash
gausser frames pano.mp4                          # equirectangular video
gausser mask person --exclude --dilation 8       # the operator moves with the camera
gausser solve                                    # also masks the operator's features
gausser train --exclude-masks
```

In a 360° scene `solve` also applies `masks-exclude/` to COLMAP: the operator stays in the same
place in every frame and would otherwise match as a fixed object, so `mask --exclude` deletes the model.

A Kandao QooCam 8K raw `.MOV` needs no stitching. `frames` reads the camera's own lens calibration
from the file, puts the two 3840×3840 fisheye streams side by side and selects frames from that. Each
lens then gives 7 views (its axis and a ring of 6 at 55°), each a real pinhole camera at its lens's
centre: there is no seam, and `solve` knows the 2.95 cm between the lenses, so the model comes out in
metres. `frames` writes `masks-exclude/` too: black outside the lens circle and in a 35° cone around
the grip (the hand or a stick); `mask --exclude` keeps those black, and `train` uses them even without
`--exclude-masks`. `--range` keeps only the passes you want (seconds or m:ss, repeatable):

```bash
gausser frames Q360_….MOV --range 1.5-32.5 --range 0:57-1:20
```

A folder of QooCam 8K photos (3840×7680, lens 0 above lens 1, calibration in each JPEG) works the
same way: every photo is kept. Stand still for each shot, about 20 cm apart.

```bash
gausser frames /Volumes/QooCam\ 8K/PHOTO/Kitchen
```

## iPhone LiDAR scans

The Camera app keeps no depth. Record with [Stray Scanner](https://apps.apple.com/app/stray-scanner/id1557051662)
on an iPhone or iPad with LiDAR, then AirDrop the scan folder (`rgb.mp4`, `odometry.csv`, `depth/`,
`confidence/`) to the Mac and give `frames` the folder:

```bash
gausser frames ~/Downloads/a9aca2e1bd       # frames from rgb.mp4, plus arkit/
gausser solve                               # snapped to ARKit: metres, gravity +y
gausser train                               # Brush starts from the LiDAR points
```

`frames` selects frames from `rgb.mp4` as for any video (1920×1440, lower than 4K video) and writes
`arkit/`: each frame's ARKit pose, its LiDAR depth map (high-confidence pixels) and `points.ply`, the
LiDAR points of all of them at one per 1 cm voxel. `solve` runs COLMAP as usual, starting from ARKit's
focal length, then moves the model onto ARKit's frame and scales it by the LiDAR depths, and prints how
well the two agree. `train --init auto` then starts Brush from the LiDAR points unless it gets masks
(`--init lidar|sparse` chooses).

## Re-splatting a published artwork

Camera presets and hand-placed geometry (water nodes, say) are expressed in the published splat's
coordinates, but a new COLMAP solve lands at an arbitrary position, rotation and scale. Fit the new
splat onto the published one before swapping it in:

```bash
gausser align runs/brush/<time>/exports/export_30000.ply \
    --to public/splats/artwork-lod/0_0/meta.json public/splats/artwork-lod/0_1/meta.json -o artwork-aligned.ply
```

`--to` takes anything `splat-transform` reads (.ply, .sog, the LOD child `meta.json` files; several are
concatenated). gausser lets `splat-transform` decode the reference, because SOG means are log-encoded.
It fits a similarity to the opaque splat centres: many starting rotations and scales, trimmed ICP from
each, then a refinement on the best. The reference may be a cropped or cleaned part of the new splat
(not the other way round: reference points look for their nearest new splat), and neither needs a known up
axis. It prints scale, rotation, translation, the trimmed RMS and the share of reference splats within 2% of
the reference's extent: look for an RMS far below the splat size and ~100% inliers. `splat-transform`
applies the result (quaternions and spherical harmonics rotate with it), and gausser reads the output back
and compares it with the planned means. The Brush `Vertical axis` header is rotated along. `orbit` follows the scene's cameras, so render
orbits of the unaligned splat. Expect about a minute for million-splat clouds. `--no-scale` fits rotation and translation only.

## How state works

The files are the state. There is no database or manifest.

- Each step writes into `.gausser/tmp/` and moves the result into place only when
  it succeeds. A failed step leaves the previous output untouched.
- A step counts as up to date when its output matches its input:
  - masks: every image has a mask;
  - solve: the COLMAP database lists exactly the current images;
  - train: the newest export is newer than the model and the crop;
  - dense: `dense/scene_dense.ply` is newer than the model;
  - lod: the newest LOD set is newer than the newest export;
  - mesh: the newest mesh is newer than the dense cloud and the crop.

  `gausser status` marks anything out of date.
- New frames delete the masks, the COLMAP model and `dense/`, new masks delete the
  COLMAP model and `dense/`, and a new model deletes `dense/`, because each was built from what changed. A `solve --no-masks`
  model (marked `sparse/no-masks`) never read the masks, so new masks keep it. Training and mesh runs in
  `runs/` are never deleted.
- `.gausser/tmp/` keeps failed attempts for inspection. `gausser status` shows their
  size; delete the folder whenever nothing is running.
- Brush gets a folder that holds only links to `images/`, `sparse/` and `masks/`.
  Brush scans everything in the folder it is given and would otherwise pick up
  stale COLMAP models and images from `runs/` or `.gausser/`.

## Notes that cost time to learn

- **Masks.** Brush reads `masks/<stem>.png`. COLMAP reads `colmap/masks/<name><ext>.png`;
  the double extension is required. If SAM3 finds nothing in an image, that image
  gets a white mask, so it is kept whole and the run continues (`--miss keep`, the
  default). With `--miss drop` its Brush mask is black, so `train` leaves that image out,
  and its COLMAP mask stays white, so it still registers. Use `drop` when a whole-image
  fallback would train people, water or sky into the subject. Try
  `--threshold 0.3` if many images miss. `--no-masks` on `solve` and `train` ignores
  existing masks; masks are still deleted when new ones are made.
- **Matching.** `--matcher auto` uses exhaustive matching up to 200 images and
  sequential matching with loop detection above that. Exhaustive matching on 500+
  images runs for hours. Loop detection also matches distant images that look
  alike, so a walk that ends where it started, or a set of unordered photos, still
  connects. COLMAP downloads the vocabulary tree it needs once, into `~/.cache/colmap`.
  For large unordered photo sets, `--matcher vocab_tree` is the dedicated option.
- **COLMAP on macOS.** Homebrew COLMAP has no CUDA, so GPU is off on macOS. Frames
  of one video share one camera; each source video in `frames.json` gets its own, since
  crowd captures mix phones and lenses. Photos share one camera. If COLMAP splits the scene, the
  largest model becomes `sparse/0`.
- **Brush.** `--max-splats` defaults to 1.5M, because unbounded growth stalls
  Metal on Apple Silicon. Brush exports at the final step too: 5000 iterations with
  `--export-every 2500` writes `export_2500.ply` and `export_5000.ply`.
  Masks are confirmed in use: with all-black masks Brush stops adding splats and
  their opacity falls to zero. Masked-out areas are ignored during training, not
  made transparent.
- **Quality.** `gausser train --eval-split 8` holds out every 8th image and
  prints `Eval iter N: PSNR …, ssim …` every 1000 steps and at the end. Compare
  runs with the same split.
- **Adaptive steps.** `train --adaptive` trains `--iterations` (30000) first, then 5000 more steps at a time
  while the held-out PSNR gains at least 0.1 dB, up to `--max-iterations` (60000). It holds out every 20th
  image unless `--eval-split` says otherwise. Brush has no checkpoints: each continuation starts from the last
  export (an `init.ply` in Brush's folder) with `--start-iter`, so splat growth stays off, and its mean learning
  rate restarts at 2e-7, where the first run's schedule ended. A continuation that lowers the PSNR leaves no export.
- **Crop.** `gausser crop` keeps work to a box around the subject. Brush starts only from the points inside it
  (the dense cloud is cropped before it is thinned to 500k), the run's exports keep only the splats whose
  centres are inside (Brush still trains on whole images, so the background is fitted and then cut off), and
  `mesh` cuts the surface to the box before refining, so RefineMesh and TextureMesh only work on the subject.
  The box is axis-aligned in the model's level frame (the shortest turn of the cameras' up onto +y, as the
  monitor shows it). A new crop makes `train` and `mesh` out of date; a new solve removes it with `sparse/`.
  Write negative numbers with `=` (`--box=-1,0,-1,1,2,1`).
- **Msplat** is experimental. It has no mask input, so with masks gausser
  composites each image onto black and trains against a black background. The
  masked-out areas become empty space (so `--miss drop` images become all black
  there; prefer Brush for that). Ctrl+C saves a checkpoint; continue with
  `gausser train --resume <run>`.

## Install (macOS, Apple Silicon)

```bash
scripts/install-macos.sh
```

The script is safe to rerun. It installs:

- COLMAP, ffmpeg (7 or newer: `frames` reads its filter graph from a file), Python 3.13, Node and Rust (Homebrew);
- splat-transform (npm);
- a `.venv` with gausser, SAM3 (patched) and Msplat;
- Brush at a pinned commit (`cargo install`);
- OpenMVS at a pinned commit (`scripts/install-openmvs.sh`: vcpkg builds its dependencies, about an hour the first
  time; the tools are linked into `~/.local/bin`) for `dense` and `mesh`;
- a SuperSplat build for `gausser view` and `gausser orbit`, and Playwright with Chromium for `orbit`;
- the monitor page (`web/`, built into `src/gausser/web/`);
- `gausser` in `~/.local/bin`.

It ends with `gausser status`. After you reinstall SAM3 by hand, rerun the script:
`status` reports an unpatched SAM3.

SAM3 is pinned to commit `f6e51f5`; the older pin `d7a33bb` no longer exists
upstream. Upstream SAM3 assumes CUDA: it imports `triton` and hardcodes
`device="cuda"`. `patches/sam3-macos.patch` adds CPU and MPS fallbacks; it is the
same patch the earlier Automask-GS setup used.

Gausser loads `~/.cache/gausser/sam3.pt` when it exists and no `--sam-checkpoint`
is given. Otherwise it downloads the gated model from Hugging Face. Gausser sets
`PYTORCH_ENABLE_MPS_FALLBACK=1` itself; without it, SAM3 misses every image on
Apple GPUs.

## Development

```bash
pytest -q && ruff check src tests && mypy src
```
