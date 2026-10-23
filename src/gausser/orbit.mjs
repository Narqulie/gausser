// Render a 360° orbit video of a splat with SuperSplat's own video renderer (run by `gausser orbit`).
//   node orbit.mjs URL ORBIT.json OUT.mp4 SECONDS WIDTH HEIGHT [#rrggbb=FILE.mp4 ...]
// Each trailing #rrggbb=FILE renders the orbit once more over that background colour (default: one render to OUT.mp4).
// URL is a `gausser view` address; ORBIT.json holds target/up/radius/height in PLY coordinates.
// Env: GAUSSER_NODE_DIR (where playwright-core is installed), GAUSSER_CHROME (browser executable override).
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

const { chromium } = createRequire(`${process.env.GAUSSER_NODE_DIR}/`)('playwright-core');

const [url, orbitPath, out, seconds, width, height, ...passes] = process.argv.slice(2);
const orbit = JSON.parse(readFileSync(orbitPath, 'utf8'));
const browser = await chromium.launch({
    headless: false, // watch it work; the video renderer needs a real GPU context anyway
    executablePath: process.env.GAUSSER_CHROME || undefined,
    args: ['--enable-unsafe-webgpu']
});
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 }, acceptDownloads: true });
page.on('console', m => m.type() === 'error' && console.log('[page]', m.text()));
await page.goto(url);
await page.waitForFunction(() => window.scene?.elements?.some(e => e.type === 'splat'), null, { timeout: 300000 });
console.log('splat loaded');

const fps = 30;
const frames = Math.round(Number(seconds) * fps);
const keys = 36;
await page.evaluate(async ({ orbit, frames, keys, fps }) => {
    const scene = window.scene;
    const events = scene.events;
    const splat = scene.elements.find(e => e.type === 'splat');
    const Vec3 = scene.camera.position.constructor;
    let m = splat.entity.getWorldTransform().data; // column-major 4x4
    const toWorld = ([x, y, z], w = 1) => [
        m[0] * x + m[4] * y + m[8] * z + m[12] * w,
        m[1] * x + m[5] * y + m[9] * z + m[13] * w,
        m[2] * x + m[6] * y + m[10] * z + m[14] * w
    ];
    // SuperSplat's camera keeps world Y up, so turn the splat until the scene's up is world Y
    // (otherwise the orbit rolls whenever the scene's up is not the PLY's -y).
    {
        const [ux, uy, uz] = toWorld(orbit.up, 0);
        const l = Math.hypot(ux, uy, uz);
        const angle = Math.acos(Math.max(-1, Math.min(1, uy / l))) * 180 / Math.PI;
        if (angle > 0.01) {
            const Quat = splat.entity.getLocalRotation().constructor;
            const axis = Math.hypot(uz, ux) > 1e-9 ? new Vec3(-uz, 0, ux).normalize() : new Vec3(1, 0, 0);
            const turn = new Quat().setFromAxisAngle(axis, angle);
            splat.move(undefined, new Quat().mul2(turn, splat.entity.getLocalRotation()));
            m = splat.entity.getWorldTransform().data;
        }
    }
    const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
    const norm = a => { const l = Math.hypot(...a); return a.map(v => v / l); };

    const t = orbit.target;
    const up = orbit.up;
    // Two horizontal axes perpendicular to up.
    const e1 = norm(cross(up, Math.abs(up[0]) < 0.9 ? [1, 0, 0] : [0, 0, 1]));
    const e2 = cross(up, e1);
    const target = toWorld(t);

    events.fire('timeline.setFrameRate', fps);
    events.fire('timeline.setFrames', frames);
    events.fire('timeline.setLoop', true);
    for (let k = 0; k < keys; k++) {
        const a = (2 * Math.PI * k) / keys;
        const p = t.map((v, i) => v + orbit.radius * (Math.cos(a) * e1[i] + Math.sin(a) * e2[i]) + orbit.height * up[i]);
        const pos = toWorld(p);
        events.fire('timeline.setFrame', Math.round((k * frames) / keys));
        events.fire('camera.setPose', { position: new Vec3(...pos), target: new Vec3(...target) }, 0);
        await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
        events.fire('track.addKey', Math.round((k * frames) / keys));
    }
    events.fire('timeline.setFrame', 0);
}, { orbit, frames, keys, fps });
console.log(`${keys} keys over ${frames} frames`);

for (const pass of passes.length ? passes : [`=${out}`]) {
    const [colour, file] = pass.split('=');
    await page.evaluate(colour => {
        if (!colour) return;
        const events = window.scene.events;
        const Color = events.invoke('bgClr').constructor;
        const n = i => parseInt(colour.slice(1 + 2 * i, 3 + 2 * i), 16) / 255;
        events.fire('setBgClr', new Color(n(0), n(1), n(2), 1));
    }, colour);
    const download = page.waitForEvent('download', { timeout: 0 });
    await page.evaluate(({ frames, fps, width, height }) => window.scene.events.invoke('render.video', {
        startFrame: 0, endFrame: frames - 1, frameRate: fps,
        width, height, bitrate: 40_000_000,
        transparentBg: false, showDebug: false,
        format: 'mp4', codec: 'h264', projection: 'standard', levelHorizon: false
    }, null), { frames, fps, width: Number(width), height: Number(height) });
    await (await download).saveAs(file);
    console.log('saved', file);
}
await browser.close();
