# Automated evaluation of gaze-tracking changes — proposal

Date: 2026-07-05
Status: proposal / early planning (prototyping may happen on this branch; the harness
itself will probably move to its own repository)

## 1. Problem and goal

We are about to iterate on the gaze-tracking algorithm — multi-pose calibration, a
pose-conditioned residual correction, possibly swapping the base estimator (see
`docs/research/webeyetrack-accuracy-investigation.md`). Today the only way to judge a
change is to sit in front of the laptop, calibrate, wave one's head around and form an
impression. That is slow, unrepeatable, and biased — useless for comparing algorithm
variants or catching regressions.

**Goal: a harness where evaluating an algorithm change costs zero human time per
iteration** — record ground-truth sessions *once*, then replay them through any candidate
tracker and get comparable numbers.

### What the application actually needs (from RollScore)

RollScore follows the reading position in *both axes*: while a system is being read, the
score scrolls smoothly forward — keeping the active staff system touching the top of the
screen — as the gaze approaches the right screen edge; a gaze move *down and back to the
left edge* signals activation of the next system, which is immediately snapped into view
if it is partially hidden at the bottom. So vertical accuracy identifies *which* system
is active, while horizontal position drives progress along the line and, combined with
the vertical drop, triggers the next-system transition. The research memo and the
staff-system detection spike give concrete acceptance targets:

- **≥ 95 % correct staff-system identification** within the operating pose envelope
  (initially ~±10–15° yaw, ±8–10° pitch, plus modest translation), with almost all
  residual errors on an *adjacent* system.
- **Vertical p90 error < ½ the neighboring-system centerline distance.** On the La Maja
  test piece, systems are ~340–560 px tall on a 2160-px-wide canvas; on a ~1080-px
  viewport showing 3–4 systems that means centerlines ~200–300 px apart, so vertical p90
  must stay under roughly **10–14 % of viewport height**. (The WebGazer spike measured
  smoothed p90 = 18 % — the gap we need to close.)
- **Reliable next-system transitions**: the down-and-left return sweep must be detected
  promptly, with no missed or spurious activations — so horizontal accuracy near the
  left and right screen edges and responsiveness to that saccade matter, not just
  static accuracy.
- **Stability over a session**: drift over 20+ minutes, cross-session repeatability,
  and graceful behavior through blinks and off-score glances.

### Requirements for the harness

| # | Requirement |
|---|---|
| R1 | **Deterministic replay** — same input corpus + same code ⇒ same metrics (within a documented tolerance) |
| R2 | **Tracker-agnostic** — one adapter interface; WebEyeTrack (JS and Python), WebGazer, and any future estimator plug in |
| R3 | **Calibration in the loop** — the corpus contains labeled calibration segments that are fed to the tracker's own calibration/adaptation API, then evaluation runs on the remaining segments |
| R4 | **Pose-conditioned reporting** — errors broken down by head-pose sector, with *held-out* sectors (train the correction on part of the pose range, test on the rest) — this is the memo's go/no-go diagnostic |
| R5 | **Task-level metric** — staff-system classification accuracy, not just cm/degrees |
| R6 | **CI-runnable** — regression gates against a stored baseline on every PR |
| R7 | Latency/throughput measured (separately — replay is offline, so speed needs its own micro-benchmark) |

## 2. Ground truth: what "correct" means

All non-synthetic approaches share one assumption: **the subject is actually looking at
the on-screen target**. That is never perfectly true (saccade latency ~200 ms, attention
lapses, blinks). Mitigations, baked into the corpus format rather than each analysis:

- Distinguish **dwell** segments (static target, generous settle time; only the last
  ~70 % of each dwell is scored) from **pursuit** segments (moving target; score with a
  fixed latency allowance, e.g. best alignment within 0–300 ms lag).
- Blink/eyes-closed frames are *labeled*, not silently dropped — blink handling is
  itself a behavior we want to evaluate (WebEyeTrack used to emit screen-center on
  blinks).
- Multiple short recordings beat one long one: attention compliance decays.

## 3. Candidate approaches

### A. Pre-recorded real-session replay corpus ("record once, replay forever")

This is the idea in the task description, refined. Record webcam video while a scripted
target moves on screen; store the video + a per-frame ground-truth log + geometry
metadata; replay the frames offline through each candidate tracker.

**Recording protocol** (one "session" ≈ 3–5 min, several scenes):

1. **Calibration scene** — 9-point (3×3) grid, target positions and timestamps logged.
   This segment is *replayed into the tracker's calibration API*, not scored.
2. **Sweep scene** — smooth-pursuit target covering the screen (including edges and the
   bottom band where WebEyeTrack historically fails), head held still.
3. **Head-sweep scenes** — the memo's fixed-target diagnostic: fixate a static target
   while slowly sweeping yaw, then pitch, then translation through the operating
   envelope. Repeat at a few target positions (center, top, bottom).
4. **Multi-pose calibration scene** — the same 9 points revisited with the head turned
   (feeds future multi-pose calibration APIs; scored as evaluation data for trackers
   that don't use it).
5. **Reading scene** — the target replays a realistic music-reading pattern over a
   scrolling score: pursuit left→right along each system, then a *down-and-left
   saccade* to the start of the next — exercising exactly the right-edge arrival and
   return-sweep signals the reader's transition logic depends on; closest proxy for
   the actual task.

**Ground-truth capture details:** record in the browser so the target renderer and the
camera share one clock — log target position with `performance.now()` and stamp each
video frame via `requestVideoFrameCallback` (gives per-frame capture timestamps);
record `MediaRecorder` VP9/H.264 at high bitrate (compression-artifact sensitivity gets
checked once, see §6). Store screen dimensions (px and cm), camera position relative to
the screen, and approximate viewing distance in `meta.json`. Optionally also record a
head-pose reference track (e.g. run MediaPipe at record time) as *diagnostic* ground
truth for pose-sector bucketing — it need not be perfect, only consistent.

**Feasibility is already proven in-repo:** upstream's
`python/scripts/ml_routines/eval_eye_of_the_typer.py` replays pre-recorded `.webm`
webcam videos with Tobii ground truth through the Python pipeline offline — our harness
is the same pattern with our own corpus, protocol, and metrics.

- **Pros:** exactly our deployment conditions (camera above screen at the instrument,
  lighting, glasses); direct A/B of algorithm variants on identical input; enables R3
  and R4 precisely; near-zero marginal cost per iteration; corpus grows over time.
- **Cons:** ground-truth validity rests on compliance (§2 mitigations); population is
  initially N=1 → overfitting risk to one face (mitigate: record family/friends, add
  approach B); videos of faces are private data → corpus lives in private storage, not
  in a public repo.

### B. Public dataset benchmarks

Run the tracker on established datasets. Best fits:

- **EYEDIAP** — *videos* with screen targets and explicit static/moving head-pose
  sessions; the closest public analog to our protocol. Upstream already has
  `scripts/preprocessing/eyediap.py`. Free academic license request required.
- **Eye of the Typer** — webcam `.webm` + Tobii ground truth; upstream eval script
  exists and gives numbers comparable to the WebEyeTrack paper.
- **MPIIFaceGaze** — static images; **caution: the default model was trained on it**,
  so it can only serve as a sanity check, never a headline metric.
- **GazeCapture** — huge population, but phone/tablet geometry.

- **Pros:** population diversity (glasses, ethnicity, lighting) that our own corpus
  can't reach; comparability with published numbers; catches "works on our face only"
  overfitting.
- **Cons:** none match our exact geometry/task; mostly images or short clips, weak for
  calibration-in-the-loop and drift; licensing friction; large downloads → not for
  every-PR CI, more like a weekly/pre-release job.

### C. Synthetic parametric rendering

Render a rigged 3D head (Blender/FLAME, UnityEyes-style, or Microsoft-Face-Synthetics
pipeline) looking at known screen points on a virtual monitor; sweep gaze × pose ×
lighting with *perfect* ground truth.

- **Pros:** perfect, dense ground truth including head pose; arbitrary pose coverage;
  no privacy issues; ideal for *unit-testing the geometry* (does the pose-compensation
  math cancel a pure rotation?).
- **Cons:** the appearance-based CNN sees a domain gap — absolute error on renders
  does not predict absolute error on real faces, and tuning against synthetic quirks is
  a real risk; highest build effort of all options.
- **Verdict:** not for accuracy evaluation. Possibly later as a small suite of
  geometry regression tests (e.g. "pure 10° yaw with fixed gaze must move predicted
  PoG < X"), where only *relative* behavior matters.

### D. Live semi-automated protocol (human in the loop, metrics automated)

Extend rollscore's existing spike harness (`web/spike/gaze-accuracy.html`): a scripted
target course, automatic metric computation, archived results per run.

- **Pros:** cheapest to build (mostly exists); the only approach that measures the
  *live* system — real frame rates, camera auto-exposure, latency.
- **Cons:** not repeatable (every run is new input) → can't isolate an algorithm change
  from run-to-run variance; needs a human per data point; not CI-able.
- **Verdict:** keep, but re-cast as the **recording front-end for approach A** — every
  live run also captures video + ground truth and becomes a corpus session. Its live
  metrics remain useful as a final acceptance check and for R7 (latency).

### E. Task-level simulation in RollScore

rollscore's `GazeSource` abstraction already has a `ScriptedGazeSource` that replays
`{x, y, confidence, t}` traces through the scroll controller with no camera or DOM.
Feed it the *tracker-output traces produced by approach A replays*, run the staff-system
classifier and scroll controller, and measure what actually matters: system-ID accuracy,
missed/spurious next-system activations, scroll stability, freeze/coast behavior.

- **Pros:** measures the end-to-end task (R5); decouples cleanly — gaze-error traces
  can even be synthesized from a fitted error model for cheap Monte-Carlo tuning of the
  controller (smoothing, dead-zones) without any video processing.
- **Cons:** evaluates the *consumer* of gaze, not the tracker; needs approach A (or D)
  output as input.
- **Verdict:** adopt as the top layer over A. Not a substitute for tracker metrics —
  controller smoothing can mask tracker regressions, so both layers gate.

### Comparison

| | A. Recorded replay | B. Public datasets | C. Synthetic | D. Live protocol | E. Task simulation |
|---|---|---|---|---|---|
| Matches deployment conditions | ●●● | ● | ○ | ●●● | ●●● (given A input) |
| Ground-truth quality | ●● (compliance) | ●●● (Tobii) / ●● | ●●● perfect | ●● | inherits A |
| Head-pose coverage & control | ●●● (scripted scenes) | ●● (EYEDIAP) | ●●● | ●● | n/a |
| Population diversity | ● (N≈1 at first) | ●●● | ●● | ● | n/a |
| Deterministic / CI-able | ●●● | ●●● | ●●● | ○ | ●●● |
| Human cost per algorithm iteration | ~0 | ~0 | ~0 | high | ~0 |
| Build effort | medium | low–medium (scripts exist) | high | low (exists) | low (exists) |
| Blind spots | one population; compression | wrong geometry/task | domain gap | variance swamps signal | masks tracker error |

No single approach satisfies all requirements; A is the core, D is its recording
front-end, E sits on top, B guards against N=1 overfitting, C is deferred.

## 4. Recommended architecture (layered)

```
┌──────────────────────────────────────────────────────────────┐
│  L4  CI & reporting: baseline JSONs, tolerance gates,        │
│      HTML report (plots per scene / pose sector / tracker)   │
├──────────────────────────────────────────────────────────────┤
│  L3  Metrics engine (pure Python, tracker-agnostic):         │
│      error stats, H/V split, pose sectors, drift, jitter,    │
│      staff-system classification proxy                       │
├──────────────────────────────────────────────────────────────┤
│  L2  Replay adapters → standard gaze-trace JSONL:            │
│      • WebEyeTrack-Py  (process_frame/adapt_from_frames)     │
│      • WebEyeTrack-JS  (Playwright: step(frame, t))          │
│      • WebGazer        (Playwright: fake-video-capture)      │
│      • <future tracker>                                      │
├──────────────────────────────────────────────────────────────┤
│  L1  Corpus: sessions = video + targets.jsonl + meta.json    │
│      + segment labels (calib / dwell / pursuit / head-sweep) │
├──────────────────────────────────────────────────────────────┤
│  L0  Recording harness (browser; grown from rollscore spike) │
└──────────────────────────────────────────────────────────────┘
```

### Corpus session format (L1)

```
corpus/
  sessions/
    2026-07-10_antti_laptop_piano_01/
      meta.json        # schema ver, screen px+cm, camera offset cm, distance cm,
                       # camera model, fps, lighting note, subject pseudonym, glasses
      video.webm       # camera recording, high bitrate
      frames.jsonl     # one line per video frame: {frame_idx, t}   (from rVFC)
      targets.jsonl    # {t, x_px, y_px, segment_id}
      segments.jsonl   # {segment_id, kind: calib|dwell|pursuit|head_sweep|reading,
                       #  scene, pose_hint: {yaw_range, pitch_range}, score: bool}
      pose_ref.jsonl   # optional diagnostic head-pose track (recorded MediaPipe)
```

Replay output, one file per (session × tracker × code-version):

```
{t, frame_idx, pred_x, pred_y, raw_x, raw_y, blink, head_pose: {yaw,pitch,roll,tx,ty,tz},
 tracker_state: {...}}   # + run-level: tracker id, git sha, backend, seed, versions
```

### Replay adapters (L2)

- **WebEyeTrack-Python** — first prototype. `WebEyeTrack.process_frame()` accepts
  frames directly and `adapt_from_frames()` covers R3; the Eye-of-the-Typer script is
  the template. Decode `video.webm` with OpenCV/PyAV, feed frames with corpus
  timestamps.
- **WebEyeTrack-JS** — the real deployment target. Crucial finding:
  `WebEyeTrack.step(frame: ImageData, timestamp)` is a pure frame-in/gaze-out API with
  an *explicit timestamp* (webcam handling lives in a separate `WebcamClient`). So the
  adapter is a Playwright-driven page that decodes the video (WebCodecs/`<video>`+
  canvas), calls `step()` frame-by-frame with **corpus timestamps** (virtual clock — the
  Kalman filter and click-TTL logic then behave deterministically), and calls `adapt()`
  for calibration segments. No fake camera, no real-time races, no dropped frames.
  Run TFJS on the WASM/CPU backend for determinism (WebGL is not bit-stable).
- **WebGazer** — owns `getUserMedia` internally, so use Chromium's
  `--use-fake-device-for-media-stream --use-file-for-fake-video-capture=<file>.y4m/.mjpeg`
  and real-time playback. Inherently timing-jittery → run it as a *noisy baseline*
  (report averages over k runs), not as a gated metric.

**Staged replay caching:** cache intermediate artifacts per session keyed by content
hash + stage version — decoded frames → face landmarks → eye patches/embeddings. A
change that only touches calibration logic then re-runs in seconds instead of re-running
MediaPipe over the whole corpus.

### Metrics engine (L3)

Per session × tracker, per scored segment, on smoothed *and* raw output:

- Error: px, % of viewport, cm, and visual degrees (geometry from `meta.json`);
  mean / median / p90; **horizontal and vertical separately** (vertical decides which
  system is active — the upstream paper never reported the split; horizontal drives
  line progress and the next-system trigger, so report it broken out near the left
  and right screen edges too).
- **Pose-sector breakdown**: bucket by yaw×pitch from the pose reference; report
  per-sector vertical p90; support **held-out-sector evaluation** (calibrate/fit on
  sectors X, evaluate on sectors Y) — the memo's go/no-go criterion for the
  pose-residual approach.
- Drift: error trend over session time; first-vs-second-half delta.
- Jitter: detrended residual std (rollscore's spike metric, kept compatible).
- Blink robustness: prediction behavior in labeled blink windows (no center-snap).
- Calibration benefit: error with vs without applying the calibration segment.
- **Staff-system classification proxy**: project vertical predictions onto the La Maja
  page geometry (system centerlines from `gazescroll/systems.py` ground truth) →
  % correct system, % adjacent-system errors. This makes R5 a number, not a feeling.
- **Transition-event detection**: on reading-scene segments, precision/recall and
  latency of detected right-edge arrivals and down-left return sweeps (the
  next-system activation signal) against the scripted ground-truth event times.

### CI (L4)

- Small **smoke corpus** (1–2 short sessions, ~1 min) runs per-PR: Python adapter +
  JS adapter, metrics diffed against a committed baseline JSON with explicit tolerances
  (see §6). Full corpus + dataset benchmarks (B) run nightly/weekly or on demand.
- Report artifact: one HTML page per run — per-scene error plots, pose-sector heatmap,
  tracker-vs-tracker table. (rollscore's dataviz conventions can be reused.)

### Repository & privacy

The harness (L0–L4 code) lives in a **new repo** as planned. The corpus contains face
videos → keep it out of any public repo; store in a private location (private GitHub
repo with LFS, or plain object storage + DVC) referenced by the harness. WebEyeTrack/
rollscore repos carry only code and, at most, a tiny synthetic/consenting smoke fixture.

## 5. Phased roadmap

1. **Phase 0 — prove the loop (~days).** Extend the rollscore spike page to record
   video + `targets.jsonl` + `frames.jsonl` (scenes 1–3 only). Record one session.
   Write the Python replay adapter (crib from `eval_eye_of_the_typer.py`) + a minimal
   metrics script (vertical p90 overall and per pose sector). Deliverable: one number
   before/after a WebEyeTrack tweak, from one command.
2. **Phase 1 — JS adapter.** Playwright + `step()` virtual-clock replay; verify
   Python-vs-JS parity on the same session (they share weights; divergence is itself a
   bug we currently can't see).
3. **Phase 2 — harden.** Segment schema, staged caching, baseline gates in CI on a
   smoke corpus, HTML report.
4. **Phase 3 — broaden.** More subjects/lighting/cameras; head-sweep + multi-pose +
   reading scenes; EYEDIAP / Eye-of-the-Typer as the population check; wire tracker
   traces into rollscore's `ScriptedGazeSource` for task-level gating.
5. **Phase 4 — optional.** Synthetic geometry regression tests; error-model Monte
   Carlo for controller tuning.

Phase 0 deliberately front-loads the riskiest assumptions: that compliance-based ground
truth is clean enough, and that offline replay reproduces live behavior.

## 6. Determinism & known pitfalls

- **TFJS backend**: WASM/CPU for gated runs; WebGL only for informational speed runs.
- **Virtual clock**: all time-dependent logic (Kalman, click-TTL, adaptation history)
  must see corpus timestamps, never wall clock. The `step(frame, timestamp)` /
  Python API signatures already allow this; audit for stray `Date.now()`/`time.time()`.
- **Seeded adaptation**: `adapt()` runs SGD — fix seeds (upstream eval scripts already
  seed numpy/TF).
- **First-frame iris scale**: WebEyeTrack freezes metric scale from the first frame —
  corpus sessions must start with a clean frontal dwell, and the harness should assert
  a face was found in frame 0.
- **Video compression**: measure once whether high-bitrate VP9 vs lossless frames
  changes metrics materially; if yes, store lossless (bigger corpus, same harness).
- **Cross-machine float drift**: pin all model/library versions per run and use
  tolerance bands (e.g. gate on "vertical p90 worsens by > 0.5 % viewport") rather
  than exact-match baselines.
- **N=1 overfitting**: never tune on the whole corpus — keep held-out sessions (and
  held-out pose sectors) that are only ever evaluated, plus the approach-B population
  check.

## 7. Open questions

1. Corpus hosting: private GitHub+LFS vs object storage+DVC (decide when the corpus
   outgrows a few hundred MB).
2. ~~Should the recording harness live in rollscore or in the new eval repo from day
   one?~~ **Decided 2026-07-05: start inside rollscore.** Built on rollscore branch
   `claude/gaze-eval-recording-harness`: `web/spike/record-session.html` +
   `record-scenes.js` (scenes 1–3: lead-in, 9-point calibration, pursuit sweep,
   fixed-target head sweeps), producing `video.webm` + `session.json`
   (schema `rollscore-gaze-session/1`), with vitest unit tests and a headless
   fake-camera Playwright smoke (`web/tools/record_session_smoke.py`).
3. Record `pose_ref.jsonl` with which pose estimator — the same MediaPipe the tracker
   uses (consistent but correlated) or an independent one (e.g. OpenFace) for
   bucketing? Phase 0: MediaPipe, revisit if pose-sector results look suspicious.
4. How much of the WebGazer path is worth building, given the research memo already
   ruled it out as the base tracker? (Suggestion: only the fake-camera baseline run,
   for a one-time comparison number.)
