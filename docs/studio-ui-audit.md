# Gausser Studio UI audit and visual-system plan

_Date: 2026-08-26_

## Evidence and design constraints

- The app targets macOS 15 because MetalSplatter requires it, and is built on macOS 26/Xcode 26.6. `glassEffect`, `GlassEffectContainer`, and the glass button style are therefore available behind macOS 26 availability checks; macOS 15 receives a standard-material fallback.
- Apple’s **Adopting Liquid Glass**, **Applying Liquid Glass to custom views**, HIG Materials guidance, and WWDC25 *Build an AppKit app with the new design* establish the relevant hierarchy: use standard system navigation, toolbar, sidebar, controls, and materials first; reserve Liquid Glass for the top-level navigation and important interactive controls; group adjacent glass controls; avoid glass behind ordinary information or form content; test reduced transparency/motion and dynamic appearances.
- A SwiftUI workspace must not use a free-scrolling `Form` as the only action surface. The prior StageWorkspace made its Validate/Confirm actions part of its scroll content, so tall option catalogues could put the action below the visible sheet.

## Findings

| Severity | Surface | Finding | Resolution |
| --- | --- | --- | --- |
| Blocker | `StageWorkspace` | Validation and run controls were in the scrolling form, and sheet sizing was unconstrained. | Scrollable content with a persistent bottom action bar; desktop-sized sheet frame; cancel action in the toolbar. |
| High | Pipeline cards | A stage’s configuration action was disabled by runtime readiness, conflating “can inspect/configure” with “can launch”. | Settings remain inspectable; engine resolution remains the authority that accepts or rejects a run. |
| High | Source controls | Long explanatory copy, policy picker, and import actions were forced into one horizontal line. | Two-row responsive source panel. |
| High | Visual hierarchy | Generic `GroupBox` and quaternary rounded rectangles made stage state, actions, and output context visually indistinguishable. | A small Studio visual system: project hero, semantic readiness badge, stage icon, grouped status/action card, and content cards. |
| Medium | App chrome | The app did not deliberately adopt the macOS 26 toolbar/sidebar hierarchy. | Unified toolbar/split-view navigation; grouped glass only for primary project and stage actions. |
| Medium | Information density | Logs and artifact rows could extend the project page indefinitely. | Bounded log console and resilient, vertical artifact cards. |
| Medium | Copy and typography | Internal keys were title-cased (`Colmap`) and several stateful labels did not explain why a stage was unavailable. | Human stage-title map, monospaced artifact/path styling, direct prerequisite text, and consistent sentence-style helper copy. |

## Executed visual model

1. **Project overview** is the content layer: a clear project hero, source actions, stage cards, telemetry, reconstruction, artifacts, and logs.
2. **Stage configuration** is a dedicated desktop workspace: a scrollable content area with presets, grouped fields, help, warnings, output context, and a persistent validation/run bar.
3. **Glass** is limited to the active controls and short status/action groups. It is not used as a backdrop for forms, logs, or artifact data.
4. **Engine authority remains unchanged**: Python still resolves options and validates all prerequisites; the native client only enables a clearly structured route to that resolution.
5. **Compatibility remains explicit**: native macOS 26 visual effects are availability-gated; existing macOS 15 functionality still uses system materials.

## Follow-up consistency pass

A second full pass aligned every workflow page and review screen with the same
content measure, 28-point edge inset, 18-point card inset, and standard
material card surface. It resolves the narrow-window defects identified during
live use:

- `StudioScrollPage` is the shared scroll/padding container for overview,
  sources, training, publishing, and run review.
- `StudioActionBar`, placed through `safeAreaInset`, keeps configuration and
  review actions persistently visible above scrolled forms and long commands.
  `ViewThatFits` changes the footer from horizontal to vertical before buttons
  can compress or disappear in a split-view detail column.
- Content cards, reconstruction output, stage cards, trainer choices, and
  artifact rows now share a restrained standard-material surface. Liquid Glass
  remains limited to the single primary action in a footer, with the existing
  macOS 15 fallback.
- The project list uses the standard sidebar style and full-row pipeline hit
  targets. Native file panels, alerts, and confirmation dialogs remain system
  presentation rather than custom modal imitations.

These are presentation-only changes; no engine request, command, manifest, or
stage-readiness semantics changed. Visual layout has no meaningful headless
unit test, so Swift build/test plus source review are the deterministic gates;
manual narrow-window exercise remains a release check.

## Verification requirements

- Existing Python and Swift suites stay green.
- Add model regression coverage when visual control-flow changes touch lifecycle or action eligibility.
- Release build, developer package, `plutil`, and strict ad-hoc signature checks must pass.
- Review the final diff for unwanted changes to engine ownership, stage option semantics, and macOS 15 fallback behavior.
