# gausser

Photos or video in, Gaussian splats out. One command wraps five steps:

```
frames → mask → solve → train → lod
(video)  (SAM3)  (COLMAP) (Brush/Msplat) (splat-transform)
```

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
gausser run --prompt "statue"            # mask → solve → train
gausser view                             # open the result in SuperSplat
```

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
`--video clip.mp4` to extract frames first, and `--lod` to finish with a streamed
LOD set.

## Commands

| Command | Does | Writes |
|---|---|---|
| `gausser frames SOURCE [-- args]` | Sharp Frames picks sharp frames from a video; 360° frames become 14 views each | `images/` (and `panos/`) |
| `gausser mask PROMPT [--miss drop]` | SAM3 masks the subject | `masks/`, `colmap/masks/` |
| `gausser mask PROMPT --exclude [--dilation N]` | SAM3 blacks out every instance of PROMPT (e.g. people), except inside `masks/` | `masks-exclude/` |
| `gausser solve [--no-masks]` | COLMAP finds the camera poses | `database.db`, `sparse/0/` |
| `gausser train [--no-masks \| --exclude-masks] [-- args]` | Brush (default) or `--trainer msplat` trains splats; args after `--` go to Brush | `runs/brush/<time>/exports/*.ply` |
| `gausser lod [PLY]` | splat-transform builds streamed LOD (newest export by default) | `runs/lod/<time>/` |
| `gausser subset PATTERN… -o NEW_SCENE` | Copies the images (and masks) whose file names match the globs into a new scene, with the COLMAP model filtered to them | `NEW_SCENE/{images,masks,colmap/masks,sparse/0,database.db}` |
| `gausser composite [--statue PLY] [--context PLY] [--radius N] [--column R [--floor F]] [--cells N] [--slack N] [--grow N] [-o OUT]` | Puts a masked statue splat into its `--no-masks` or `--exclude-masks` context splat (newest runs by default) | `runs/composite/<time>/composite.ply` |
| `gausser align MOVING.ply --to REF… -o OUT.ply [--no-scale]` | Moves a re-trained splat into the coordinate frame of a published one (scale, rotation, translation fitted to the splat centres) | `OUT.ply` |
| `gausser view [PLY]` | Opens a splat (newest export by default) in a local SuperSplat | — |
| `gausser orbit [PLY] [--seconds 12] [--size 1920x1080] [--background jutaka] [--target X Y Z --radius R --height H] [-o OUT.mp4]` | SuperSplat renders a 360° orbit video (newest export by default) | `runs/orbit/<time>/orbit.mp4` |
| `gausser status` | Shows what the scene has and which tools are installed | — |

Every command takes `-s/--scene DIR` (default: current folder) and `--help`.
Only one command at a time can change a scene; a second one stops with an error
instead of overwriting the first one's work. `align` needs no scene at all. `status`, `view`, `orbit`, `composite` and `subset` always run
(`orbit` and `composite` only add a new run folder; `subset` only reads the source scene).

`gausser orbit` needs `sparse/0` and a Brush export. The camera circles the point nearest all
camera viewing axes, at the median camera distance times `--zoom` (1.3) and aimed `--drop` (0.35, splat
units) below that point, with "up" from the cameras (normal to their level right axes; Brush's `Vertical axis` guess can be far off on top-down-heavy drone sets). It drives a visible Chromium
window through Playwright (installed by `scripts/install-macos.sh` into `~/.local/share/gausser/node`;
override with `GAUSSER_NODE_DIR`, and `GAUSSER_CHROME` picks another browser executable), so leave the window alone
while it renders.
`--background` takes `#rrggbb`, `jutaka` (the public-art viewer's sky gradient) or your own
`linear-gradient(to bottom, rgb(r,g,b) P%, …)`. A gradient renders the orbit over black and over white, then ffmpeg
recovers the splat's alpha and blends it over the gradient (h264, crf 16).

## Subject plus context from one solve

Mask the subject, but solve COLMAP on the whole images: masks that restrict COLMAP
to the subject lose registrations and leave no background points. Then train twice on
that one model, so the two splats share coordinates and can be merged later:

```bash
gausser mask statue --miss drop --dilation -2   # frames SAM3 misses are ignored by Brush, not trained whole
gausser solve --no-masks          # COLMAP sees the whole image
gausser mask person --exclude --dilation 8      # optional: people to leave out of the context (needs masks/)
gausser train --exclude-masks     # context splat without them (or `train --no-masks` to keep everything)
gausser train                     # statue splat: masked
gausser composite                 # statue into context, as one splat
gausser composite --column 0.35   # statue + plinth only (cylinder of 0.35 statue heights, 0.3 below the feet)
```

`mask --exclude` writes `masks-exclude/<stem>.png` (black = every instance of the prompt grown by
`--dilation` px, white elsewhere and wherever the subject mask in `masks/` is white, since SAM3 may call a
bronze statue a "person"). Use 8 px: people's edges and shadows smear. It leaves `masks/` and COLMAP alone.
`train --exclude-masks` gives Brush those masks, and leaves an `excluded` marker that `composite` also accepts
for the context run (newest `unmasked` or `excluded` run).

`gausser composite` finds the two runs itself: `train` leaves an empty `masked` or `unmasked`
file in each Brush run folder (older runs have none; pass `--statue` and `--context`). The
masked run starts from the whole scene's COLMAP points, so it also keeps opaque background splats.
Composite first keeps only statue splats that fall inside the masks in at least 80% of the views
that see them (a visual hull; frames dropped with `--miss drop` don't vote), then removes the context
splats inside the statue's volume and appends the statue. `--cells N` (default 256) sets the carve voxels along the statue's longest side; raise it for
spread-out statues on a big scene. `--slack N` (default 2) grows the masks by N px at 1/4 size before the hull
filter, and `--grow N` (default 2) carves N voxels beyond the statue's hull; lower them if the statue loses
edges, raise them if a soft shell of the statue remains in the context. The output keeps the context's PLY header,
so `gausser orbit` still works. For cleaner edges, mask with `--dilation -2`: it shrinks the Brush
masks by 2 px, so edge pixels mixed with background are not trained (COLMAP masks are unchanged).

`gausser run --prompt statue --no-masks` makes the masks and then solves and trains
without them (the first two steps of the recipe, minus `--miss drop`).

## 360° video

`gausser frames` recognises 360° frames by their 2:1 (equirectangular) shape. It keeps them in
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

Cameras that record raw dual fisheye, like the QooCam 8K (two 3840×3840 streams), need stitching
first. The camera vendor's app (Kandao Studio) stitches with calibration. A rough ffmpeg stitch for
the QooCam 8K, with the lens field of view fitted by eye on the seam:

```bash
ffmpeg -i raw.MOV -filter_complex \
  "[0:0][0:1]hstack,v360=input=dfisheye:output=e:ih_fov=196:iv_fov=196:roll=90:w=5760:h=2880" \
  -an -c:v hevc_videotoolbox -q:v 75 -tag:v hvc1 pano.mp4
```

## Mixing captures

Photos from several captures (say sunny drone frames `frame_*.jpg` and iPhone frames from another
day, `iphone_*.jpg`) go into one `images/` and get one joint solve, so every camera shares one
coordinate frame. Train each lighting on its own, then composite the splats:

```bash
gausser solve --no-masks
gausser subset 'frame_*' -o ../sunny          # copies images and masks, filters sparse/0
gausser subset 'iphone_*' -o ../overcast
gausser train -s ../sunny --no-masks
gausser train -s ../overcast
gausser composite --statue ../overcast/runs/brush/<time>/exports/export_30000.ply \
                  --context ../sunny/runs/brush/<time>/exports/export_30000.ply -s ../overcast
```

`subset` copies (no links: scenes often live on exFAT), keeps all cameras and 3D points, and
leaves the source untouched. It refuses a new scene that already has `images/`.

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
  - train: the newest export is newer than the model;
  - lod: the newest LOD set is newer than the newest export.

  `gausser status` marks anything out of date.
- New frames delete the masks and the COLMAP model, and new masks delete the
  COLMAP model, because both were built from what changed. A `solve --no-masks`
  model (marked `sparse/no-masks`) never read the masks, so new masks keep it. Training runs in
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
  default). With `--miss drop` its Brush mask is black, so Brush ignores that image,
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
- **COLMAP on macOS.** Homebrew COLMAP has no CUDA, so GPU is off on macOS. All
  images share one camera (`single_camera=1`). If COLMAP splits the scene, the
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

- COLMAP, ffmpeg, Python 3.13, Node and Rust (Homebrew);
- splat-transform (npm);
- a `.venv` with gausser, Sharp Frames, SAM3 (patched) and Msplat;
- Brush at a pinned commit (`cargo install`);
- a SuperSplat build for `gausser view` and `gausser orbit`, and Playwright with Chromium for `orbit`;
- `gausser` and `sharp-frames` in `~/.local/bin`.

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
