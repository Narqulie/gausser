#!/usr/bin/env python3
"""Build gausser's documentation site: docs/src/*.html (hand-written) → docs/*.html.

Run it after every change to commands, options, modules, steps or tools (AGENTS.md):

    .venv/bin/python scripts/build_docs.py

Pages are fragments with a first-line `<!-- title: … -->`. A `<!-- gen:NAME -->` line is replaced by
generated HTML: the command reference from `cli.build_parser()`, the module map and its import graph
from the source, the tool table from `scene.TOOLS`, and the charts below. The build fails when a
command, a step or a module is not documented, so the site cannot silently fall behind the code.
"""

from __future__ import annotations

import argparse
import ast
import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gausser import cli  # noqa: E402
from gausser.scene import TOOLS  # noqa: E402

SRC, OUT, PACKAGE = ROOT / "docs" / "src", ROOT / "docs", ROOT / "src" / "gausser"
PAGES = [  # file, nav label
    ("index", "Overview"),
    ("pipeline", "Pipeline steps"),
    ("captures", "Capture types"),
    ("state", "Scene and state"),
    ("tools", "View, orbit, align, crop"),
    ("monitor", "Monitor"),
    ("commands", "Command reference"),
    ("architecture", "Architecture"),
    ("troubleshooting", "Troubleshooting"),
]
E = html.escape


# ---------------------------------------------------------------- charts

def flow_svg(nodes: dict[str, tuple], edges: list[tuple], width: int, height: int, title: str) -> str:
    """Boxes and arrows. node: (x, y, w, h, colour token, lines, dashed); edge: (from, to[, label])."""
    marker = "arrow-" + re.sub(r"\W+", "-", title.lower())  # ids must be unique on a page with several charts
    out = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" aria-label="{E(title)}" xmlns="http://www.w3.org/2000/svg">',
        f"<defs><marker id=\"{marker}\" viewBox=\"0 0 10 10\" refX=\"9\" refY=\"5\" markerWidth=\"7\" markerHeight=\"7\" orient=\"auto\">"
        "<path d=\"M0,0 L10,5 L0,10 z\" style=\"fill:var(--muted)\"/></marker></defs>",
    ]
    for edge in edges:
        a, b = nodes[edge[0]], nodes[edge[1]]
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        if bx >= ax + aw:  # left to right
            x1, y1, x2, y2 = ax + aw, ay + ah / 2, bx, by + bh / 2
            mid = (x1 + x2) / 2
            path = f"M{x1},{y1} C{mid},{y1} {mid},{y2} {x2 - 2},{y2}"
        elif by >= ay + ah:  # top to bottom
            x1, y1, x2, y2 = ax + aw / 2, ay + ah, bx + bw / 2, by
            mid = (y1 + y2) / 2
            path = f"M{x1},{y1} C{x1},{mid} {x2},{mid} {x2},{y2 - 2}"
        else:  # bottom to top
            x1, y1, x2, y2 = ax + aw / 2, ay, bx + bw / 2, by + bh
            mid = (y1 + y2) / 2
            path = f"M{x1},{y1} C{x1},{mid} {x2},{mid} {x2},{y2 + 2}"
        out.append(f'<path d="{path}" class="edge" marker-end="url(#{marker})"/>')
        if len(edge) > 2:
            out.append(f'<text x="{(x1 + x2) / 2}" y="{(y1 + y2) / 2 - 5}" class="edge-label">{E(edge[2])}</text>')
    for x, y, w, h, colour, lines, dashed in nodes.values():
        out.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="node{" dashed" if dashed else ""}" '
            f'style="stroke:var(--{colour})"/>'
        )
        top = y + h / 2 - (len(lines) - 1) * 8
        for i, line in enumerate(lines):
            cls = "node-title" if i == 0 else "node-sub"
            out.append(f'<text x="{x + w / 2}" y="{top + i * 16 + 4}" class="{cls}" style="fill:var(--{colour if i == 0 else "muted"})">{E(line)}</text>')
    out.append("</svg>")
    return "\n".join(out)


def pipeline_chart() -> str:
    """Inputs, the steps (cli.STEPS, all of them) and what each makes."""
    w, h = 132, 58
    col = lambda i: 20 + i * 168  # noqa: E731
    nodes = {
        "photos": (col(0), 20, w, h, "muted", ["photos", "images/"], False),
        "video": (col(0), 96, w, h, "muted", ["video", "phone, 360°, drone"], False),
        "scan": (col(0), 172, w, h, "muted", ["LiDAR scan", "Stray Scanner"], False),
        "frames": (col(1), 96, w, h, "frames", ["frames", "ffmpeg + OpenCV"], False),
        "mask": (col(2), 20, w, h, "mask", ["mask", "SAM3, optional"], True),
        "solve": (col(2), 136, w, h, "solve", ["solve", "COLMAP"], False),
        "dense": (col(3), 20, w, h, "solve", ["dense", "OpenMVS, optional"], True),
        "train": (col(3), 136, w, h, "train", ["train", "Brush / Msplat"], False),
        "lod": (col(4), 20, w, h, "lod", ["lod", "splat-transform"], True),
        "map": (col(4), 96, w, h, "lod", ["map", "ortho + DSM"], True),
        "mesh": (col(4), 172, w, h, "lod", ["mesh", "OpenMVS"], True),
        "use": (col(5), 136, w, h, "muted", ["view · orbit", "align · monitor"], False),
    }
    edges = [
        ("video", "frames"), ("scan", "frames"), ("photos", "mask"), ("photos", "solve"), ("frames", "mask"),
        ("frames", "solve"), ("mask", "solve", "masks"), ("solve", "dense"), ("solve", "train"),
        ("dense", "train"), ("dense", "mesh"), ("train", "lod"), ("train", "map"), ("train", "use"),
    ]
    missing = set(cli.STEPS) - set(nodes)
    if missing:
        raise SystemExit(f"build_docs: pipeline_chart lacks the steps {sorted(missing)}")
    return flow_svg(nodes, edges, col(5) + w + 20, 250, "gausser pipeline")


def decision_chart() -> str:
    """Which capture goes which way through `frames`."""
    w, h = 190, 58
    rows = [
        ("Phone video walk-around", "frames VIDEO", "sharp frames by motion", "train", "frames"),
        ("Folder of photos", "images/ (or run --import)", "used whole", "train", "frames"),
        ("360° equirect video", "frames VIDEO", "14 pinhole views, rig", "train", "solve"),
        ("QooCam 8K raw .MOV / photos", "frames MOV|FOLDER", "2 fisheyes, calibrated rig", "train", "solve"),
        ("iPhone LiDAR (Stray Scanner)", "frames SCAN_FOLDER", "ARKit poses + LiDAR points", "train", "mask"),
        ("DJI drone video + .SRT", "frames VIDEO --overlap 80", "GPS in EXIF, georeferenced", "map", "lod"),
    ]
    nodes, edges = {}, []
    for i, (capture, command, effect, end, colour) in enumerate(rows):
        y = 16 + i * 72
        nodes[f"c{i}"] = (20, y, w + 30, h, "muted", [capture], False)
        nodes[f"f{i}"] = (290, y, w + 30, h, "frames", [command.split(" ", 1)[0], command.split(" ", 1)[1] if " " in command else ""], False)
        nodes[f"e{i}"] = (560, y, w + 40, h, colour, [effect.split(",")[0], effect.split(",", 1)[1].strip() if "," in effect else ""], False)
        nodes[f"r{i}"] = (840, y, 150, h, "train" if end == "train" else "lod", ["run" + (" --map" if end == "map" else ""), end], False)
        edges += [(f"c{i}", f"f{i}"), (f"f{i}", f"e{i}"), (f"e{i}", f"r{i}")]
    return flow_svg(nodes, edges, 1010, 16 + len(rows) * 72, "Which capture takes which path")


# What deletes (✕) or makes out of date (○) what. Mirrors steps.py remove() calls and Scene.*_done.
INVALIDATION_COLUMNS = ["masks/", "masks-exclude/", "database + sparse/", "dense/", "arkit/", "train", "mesh", "lod / map"]
INVALIDATION_ROWS = [
    ("frames (new images)", ["✕", "✕", "✕", "✕", "✕", "○", "○", "○"]),
    ("mask (subject)", ["new", "", "✕¹", "✕¹", "", "○", "○", "○"]),
    ("mask --exclude", ["", "new", "✕²", "✕²", "", "", "", ""]),
    ("solve", ["", "", "new", "✕", "", "○", "○", "○"]),
    ("crop", ["", "", "", "", "", "○", "○", ""]),
    ("dense", ["", "", "", "new", "", "", "○", ""]),
    ("train (new export)", ["", "", "", "", "", "new", "", "○"]),
]


def invalidation_table() -> str:
    head = "".join(f"<th>{E(c)}</th>" for c in INVALIDATION_COLUMNS)
    rows = []
    for change, cells in INVALIDATION_ROWS:
        tds = "".join(
            f'<td class="cell {"del" if c.startswith("✕") else "stale" if c == "○" else "new" if c == "new" else ""}">{E(c)}</td>'
            for c in cells
        )
        rows.append(f"<tr><th>{E(change)}</th>{tds}</tr>")
    return f'<table class="matrix"><tr><th>change ↓ · affects →</th>{head}</tr>{"".join(rows)}</table>'


def scene_tree() -> str:
    """The scene folder, with the path attributes Scene defines."""
    from gausser.scene import Scene

    scene = Scene(ROOT)
    attrs = {name: value for name, value in vars(scene).items() if isinstance(value, Path) and name != "root"}
    notes = {
        "images": "the photos or frames every step reads", "panos": "360° source frames (images/ then holds their views)",
        "masks": "Brush masks, <stem>.png, white = train", "masks_exclude": "black = people, operator, lens edge",
        "colmap_masks": "COLMAP masks, <name><ext>.png", "database": "COLMAP features and matches",
        "sparse": "COLMAP model sparse/0, geo.json, crop.json, no-masks", "crop": "the crop box (gausser crop)",
        "dense": "OpenMVS undistorted images and dense cloud", "arkit": "iPhone scan: poses.json, points.ply, depth/",
        "runs": "brush/, msplat/, lod/, mesh/, map/, orbit/ <time>/ (never deleted)",
        "tmp": "staging of running or failed steps", "logs": "one log per step (monitor)", "history": "start/end record per step",
    }
    missing = set(attrs) - set(notes)
    if missing:
        raise SystemExit(f"build_docs: scene_tree lacks notes for Scene.{sorted(missing)}")
    lines = []
    for name, path in attrs.items():
        rel = path.relative_to(ROOT).as_posix() + ("" if path.suffix else "/")
        lines.append(f'<tr><td><code>{E(rel)}</code></td><td><code>scene.{name}</code></td><td>{E(notes[name])}</td></tr>')
    return '<table><tr><th>path</th><th>attribute</th><th>holds</th></tr>' + "".join(lines) + "</table>"


# ---------------------------------------------------------------- generated reference

def _option(action: argparse.Action) -> str:
    names = ", ".join(action.option_strings) or action.metavar or action.dest
    if isinstance(names, tuple):
        names = " ".join(names)
    default = action.default
    shown = "" if default in (None, False, argparse.SUPPRESS) or isinstance(action, argparse._HelpAction) else f"{default}"
    choices = " | ".join(map(str, action.choices)) if action.choices and not isinstance(action.choices, dict) else ""
    help_text = (action.help or "") % vars(action) if action.help and "%(" in action.help else action.help or ""
    return (
        f"<tr><td><code>{E(str(names))}</code></td><td>{E(choices)}</td><td><code>{E(shown)}</code></td>"
        f"<td>{E(help_text)}</td></tr>"
    )


def commands_reference() -> str:
    parser = cli.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    helps = {c.dest: c.help for c in sub._choices_actions}
    parts = ['<nav class="toc">' + " ".join(f'<a href="#cmd-{n}">{n}</a>' for n in sub.choices) + "</nav>"]
    for name, command in sub.choices.items():
        usage = command.format_usage().removeprefix("usage: ").strip()
        parts.append(f'<section class="command" id="cmd-{name}"><h2>gausser {name}</h2>')
        parts.append(f"<p>{E(helps.get(name) or '')}</p>")
        if command.description:
            parts.append(f'<p class="muted">{E(command.description)}</p>')
        parts.append(f"<pre>{E(usage)}</pre>")
        for group in command._action_groups:
            actions = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
            if not actions:
                continue
            parts.append(f"<h3>{E(group.title or '')}</h3><table><tr><th>option</th><th>choices</th><th>default</th><th>meaning</th></tr>")
            parts += [_option(a) for a in actions]
            parts.append("</table>")
        parts.append("</section>")
    return "\n".join(parts)


def _modules() -> dict[str, dict]:
    modules = {}
    for path in sorted(PACKAGE.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text())
        doc = (ast.get_docstring(tree) or "").split("\n\n")[0].replace("\n", " ")
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1:
                if node.module:
                    imports.add(node.module.split(".")[0])
                else:
                    imports.update(alias.name for alias in node.names)
        public = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
        tools = sorted(set(re.findall(r'require_tool\("([^"]+)"\)', path.read_text())))
        modules[path.stem] = {
            "doc": doc, "lines": len(path.read_text().splitlines()), "imports": sorted(imports - {path.stem}),
            "public": public, "tools": tools,
        }
    return modules


def modules_table() -> str:
    rows = []
    for name, m in _modules().items():
        rows.append(
            f'<tr id="mod-{name}"><td><code>{name}.py</code></td><td>{m["lines"]}</td><td>{E(m["doc"])}</td>'
            f'<td>{", ".join(f"<code>{i}</code>" for i in m["imports"])}</td>'
            f'<td class="small">{", ".join(E(p) for p in m["public"][:14])}{" …" if len(m["public"]) > 14 else ""}</td></tr>'
        )
    return ('<table><tr><th>module</th><th>lines</th><th>what it is</th><th>imports</th><th>public functions and classes</th></tr>'
            + "".join(rows) + "</table>")


def module_graph() -> str:
    """Layered import graph: a module sits one row above everything it imports."""
    modules = _modules()
    depth: dict[str, int] = {}

    def level(name: str, seen: tuple[str, ...] = ()) -> int:
        if name not in depth:
            deps = [d for d in modules[name]["imports"] if d in modules and d not in seen and d != "scene"]
            depth[name] = 1 + max((level(d, (*seen, name)) for d in deps), default=0)
        return depth[name]

    depth["scene"] = 0
    for name in modules:
        level(name)
    depth["cli"] = 1 + max(depth[n] for n in modules if n != "cli")  # cli imports nearly everything
    rows: dict[int, list[str]] = {}
    for name in sorted(modules, key=lambda n: (depth[n], n)):
        rows.setdefault(depth[name], []).append(name)
    w, h, gap_x, gap_y = 112, 34, 14, 66
    width = max(len(r) for r in rows.values()) * (w + gap_x) + 20
    top = max(rows)
    nodes = {}
    colours = {"cli": "train", "guide": "train", "scene": "solve", "steps": "frames", "monitor": "lod"}
    for d, names in rows.items():
        offset = (width - len(names) * (w + gap_x)) / 2
        for i, name in enumerate(names):
            nodes[name] = (offset + i * (w + gap_x), 16 + (top - d) * (h + gap_y), w, h, colours.get(name, "muted"), [name], False)
    # cli imports nearly every module and nearly every module imports scene; those edges are left out.
    edges = [
        (name, dep) for name, m in modules.items() for dep in m["imports"] if dep in modules and "cli" != name and dep != "scene"
    ]
    svg = flow_svg(nodes, edges, int(width), 16 + (top + 1) * (h + gap_y), "module import graph")
    return svg.replace('class="edge"', 'class="edge thin"')


def tools_table() -> str:
    users: dict[str, list[str]] = {}
    for name, m in _modules().items():
        for tool in m["tools"]:
            users.setdefault(tool, []).append(name)
    rows = "".join(
        f'<tr><td><code>{E(t)}</code></td><td>{", ".join(f"<code>{u}.py</code>" for u in users.get(t, []))}</td><td>{E(hint)}</td></tr>'
        for t, hint in TOOLS.items()
    )
    return "<table><tr><th>tool</th><th>run by</th><th>install</th></tr>" + rows + "</table>"


GENERATED = {
    "commands": commands_reference, "modules": modules_table, "module-graph": module_graph, "tools": tools_table,
    "pipeline-chart": pipeline_chart, "decision-chart": decision_chart, "invalidation": invalidation_table,
    "scene-tree": scene_tree,
}


# ---------------------------------------------------------------- pages

def layout(name: str, title: str, body: str) -> str:
    nav = "\n".join(
        f'<a href="{page}.html"{" class=\"here\"" if page == name else ""}>{E(label)}</a>' for page, label in PAGES
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{E(title)} · gausser</title>
<link rel="stylesheet" href="assets/docs.css">
</head>
<body>
<!-- Generated by scripts/build_docs.py from docs/src/{name}.html. Edit the source, then rebuild. -->
<aside>
<a class="brand" href="index.html">gausser</a>
<nav>
{nav}
</nav>
<p class="side-note">Reports: <a href="reports/2026-11-14-drone-mapping.html">drone mapping</a>,
<a href="reports/2026-11-14-crop-adaptive.html">crop + adaptive</a>, <a href="mapping/flight-patterns.html">flight patterns</a></p>
</aside>
<main>
<h1>{E(title)}</h1>
{body}
</main>
</body>
</html>
"""


def build() -> list[str]:
    problems = []
    texts = {}
    architecture_prose = ""
    for name, _ in PAGES:
        source = (SRC / f"{name}.html").read_text()
        match = re.match(r"<!-- title: (.+?) -->\n", source)
        if not match:
            raise SystemExit(f"build_docs: {name}.html must start with <!-- title: … -->")
        body = source[match.end():]
        for key, make in GENERATED.items():
            body = body.replace(f"<!-- gen:{key} -->", make())
        unknown = re.findall(r"<!-- gen:([\w-]+) -->", body)
        if unknown:
            problems.append(f"{name}.html: unknown generated sections {unknown}")
        texts[name] = (match.group(1), body)
        if name == "architecture":
            architecture_prose = source

    # Everything the code offers must be explained on a hand-written page, not only in the reference.
    prose = "".join(body for name, (_, body) in texts.items() if name not in ("commands", "architecture"))
    for command in cli.COMMANDS:
        if f"gausser {command}" not in prose:
            problems.append(f"command `gausser {command}` is not explained on any hand-written page")
    for module, info in _modules().items():
        if f"<code>{module}.py</code>" not in architecture_prose:
            problems.append(f"module {module}.py is not placed in the layers on docs/src/architecture.html")
        if not info["doc"]:
            problems.append(f"module {module}.py has no docstring (the module table shows it)")
    for href in re.findall(r'(?:href|src)="([^"#:]+)', "".join(b for _, b in texts.values())):
        if href not in {f"{page}.html" for page, _ in PAGES} and not (OUT / href).exists():
            problems.append(f"broken link or image: {href}")
    if problems:
        return problems
    for name, (title, body) in texts.items():
        (OUT / f"{name}.html").write_text(layout(name, title, body))
    return []


if __name__ == "__main__":
    errors = build()
    if errors:
        print("build_docs: the docs are behind the code:\n  " + "\n  ".join(errors), file=sys.stderr)
        sys.exit(1)
    print(f"Built {len(PAGES)} pages in {OUT}")
