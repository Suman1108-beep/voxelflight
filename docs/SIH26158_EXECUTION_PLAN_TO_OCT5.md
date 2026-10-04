# SIH26158 — execution plan and acceptance gates through 5 October

**Written 3 October 2026, Asia/Kolkata.** The team reports an extension to 5 October; confirm the exact organizer cutoff. This is a **priority plan, not a promise that every target can be reached in two days**. Begin with the [current-status handoff](SIH26158_READ_THIS_FIRST.md) and [run evidence](SIH26158_RESULTS_EVIDENCE_LEDGER.md). Preserve all saved run outputs and dirty working trees. A100 access, pretrained weights, and a public URL are useful but are not substitutes for ground truth or a durable service.

## Decision: what to optimize first

The rubric allocates **50% to accuracy + completeness**, **20% to speed**, and only **5% to UI**. The next two days should be spent on **one frozen, independently scored, georeferenced, unseen-video pipeline**, not a new login screen, a GAN, training a foundation model from scratch, or 15k more Gaussian iterations. Keep the already polished public saved demo, but make the new-upload story honest.

### P0 — must-have submission integrity (first block)

1. **Freeze one baseline commit and run manifest.** Record code revision, checkpoint hashes, dataset segment, input MP4 hash/duration, telemetry/calibration hashes, A100/CUDA/CPU/RAM, exact CLI, output checksums, and evaluator version. Mark which previous segments were used for tuning. Do not reuse Zurich GT in construction.
2. **Resolve the product path.** Test the public `apiOrigin` DNS/HTTPS, authenticated Google session, job create/status/result, ownership isolation, and upload limit. Its temporary hostname failed to resolve from this Mac on 3 Oct. Either configure durable HTTPS ingress + A100/other stable worker and demonstrate a fresh upload, **or** label the site a saved-demo/preview and provide a reproducible offline model CLI. Do not display a “ready” A100 badge without a completed job.
3. **Choose one reportable pipeline.** The 666.7 s baseline passes speed on one flight but drifts; the DPVO-conditioned path improves trajectory shape yet its 4072 s fusion fails speed. Optimize and rerun the *same selected path* end-to-end, rather than taking the best metric from each and presenting an impossible composite model.
4. **Build a requirement matrix for final slides/README.** Mark each item demonstrated, partial, or open, with artifact links. Update every older “10-minute untested” statement now superseded by the benchmark while retaining the geometry caveats. Do not alter historical raw reports.

### P1 — decisive modeling experiments (A100)

| Experiment | Why | Method / controls | Go/no-go |
|---|---|---|---|
| Profile 67-minute DPVO refusion | New conditioned fusion alone misses 15 min and its cause is unknown. | Capture per-stage wall/CPU/GPU utilization, input depths/poses, TSDF frame integration time, Open3D/thread contention and memory. Compare same 360 cached views with baseline; no reference positions. | If corrected pipeline cannot get total <900 s, do not call it the default speed-compliant path. |
| Pose/window anchoring | 10-minute camera shape 3.94 m after DPVO vs short-flight 0.3 m; drift compounds across windows. | Shared cross-window camera constraints, GPS as noisy absolute factor, loop/anchor checks, robust losses. Keep one global coordinate frame. Compare on distinct holdout segment with frozen settings. | Improve held-out trajectory **and** mesh surface/coverage; do not tune on final reference. |
| Geometry/coverage ablation | Fusion removes valid surfaces or creates floating fragments. | Same frozen depth/poses, compare no-filter, calibrated confidence filter, 5 cm/6 cm voxels, visibility-masked evaluation; inspect connected components and 0.5 m/1 m precision-recall. | Select by F1 and visible coverage, not prettiest screenshot or maximum triangle count. |
| Independent UAV surface test | Zurich has camera GT but no surveyed surface. | Obtain a lawful bounded UAV+LiDAR/survey set (e.g. UseGeo/UAVFF3D if accessible), or organizer eval if released. Freeze output first; use official evaluator if available, otherwise clearly labeled metric proxy. | Report surface precision/completeness and distance quantiles; if unavailable, state “≤1 m unverified.” |
| Metric/geographic audit | Onboard GPS biases direct accuracy by many metres. | Verify timestamp offset, intrinsics, sensor extrinsics/lever arm, axis handedness, altitude datum, UTM zone; optional RTK/PPK as extra branch. | Direct absolute location measured separately from post-hoc Sim3. A UTM label alone does not pass. |

**Do not overfit the deadline:** GeoFF3D/VGGT-Long/UAVFF3D are promising research paths, but integrating a new paper without compatible weights, data/license, inference, and held-out scoring may consume the remaining time and make the pipeline less reliable. Make one controlled ablation at a time. An A100 has compute, not guaranteed training labels.

### P2 — end-to-end product and formats

- The locally written `scripts/run_a100_pipeline.py` is an **orchestrator prototype**: video + telemetry + optional calibration, optional DPVO, MapAnything, TSDF, optional geospatial transfer. It is not yet verified end-to-end on the A100, and no A100 `scripts/` directory was present at the audited remote project root. Upload/reproduce it only after code review and small fixture tests. Its use of pretrained weights and no reference inputs is appropriate for blind evaluation.
- Exercise **one arbitrary fresh MP4 and one MOV**, each with GPS/metadata; use a deliberately malformed file, GPS timestamp gaps, missing calibration, portrait/rotated metadata, VFR/compression, and a bounded 4K smoke. Log clear errors instead of silently accepting wrong coordinates.
- Time **all** stages from submitted video available on disk to final downloadable outputs: decode, selection, odometry, inference, fusion, georeference, texture/material generation (if claimed), and export. Report upload/queue separately if the organizer excludes them; never hide them in a reviewer demo.
- Export and independently reopen **OBJ, PLY, LAS, GeoTIFF, GLB/glTF, and FBX** if the annex is enforced. Current FBX is missing. Audit units, axis, CRS, texture/material files, no empty output, and viewer measurements against known scale. Do not describe an occupancy GeoTIFF as an accurate terrain DEM.
- Upgrade the backend only if there is time for an actually proven flow: stable HTTPS endpoint, resumable/appropriately bounded upload, authenticated owner-scoped durable job record, queue limit, restart handling, immutable run manifest, artifact downloads, and real progress/errors. A temporary tunnel or backend health endpoint is insufficient.
- Review mobile and desktop UI after the model flow works: labels must say “saved demo,” “new upload preview,” “local vs absolute coordinates,” “surface accuracy not yet verified” when true. The optional appearance comparison must not masquerade as the exported mesh.

## 48-hour concrete sequence

| Order | Work package | Deliverable that proves it |
|---|---|---|
| 1 | Save/copy current A100 run reports and artifact hashes; fix our own ledger. | This handoff + report hashes + unchanged raw JSON. |
| 2 | Profile the DPVO refusion regression and run one short controlled test. | Per-stage profiler/timing table and reason for 4072 s; selected configuration. |
| 3 | Freeze candidate pipeline; run an untouched/least-tuned flight once. | Input manifest, one-command log, end-to-end time, mesh/point cloud, GPS exports. |
| 4 | Score geometry if lawful independent UAV reference is available; otherwise ETH3D proxy plus explicit SIH limitation. | Precision/recall/F1, distance distribution, reference/alignment protocol, coverage visualization. |
| 5 | Perform absolute georeference audit with available GPS and, if supplied, RTK/anchors. | Direct metre-error report distinct from Sim3 and surface metrics. |
| 6 | Test site API and one new upload end-to-end; otherwise provide explicit offline CLI/demo. | Browser recording or logs showing input, job ID, output ownership and download; no mocked completion. |
| 7 | Verify all export formats, 4K smoke, packaging and restart. | Reopen-test matrix, failure logs, updated README/slides/demo. |
| 8 | Freeze/publish only after QA; confirm submission link access and deadline. | Public repo/site, downloadable deck/video, final results table consistent across them. |

## Numerical acceptance dashboard

| Gate | Target | Current honest state | Evidence required to mark pass |
|---|---:|---|---|
| Ten-minute processing | **<900 s** | Baseline **666.7 s on one derived Zurich flight**; conditioned DPVO fusion **4072.4 s** alone. | One candidate, one full timed run on unseen/held-out video; all claimed stages included. |
| Spatial accuracy | **≤1 m** | **Unverified surface**; direct absolute camera error historically 4.445–8.649 m depending run. | Independent surveyed surface and organizer alignment; absolute reference separately if required. |
| Visible-scene completeness | Entire visible scene | Partial; baseline 3065 mesh components, largest 52.2%; DPVO 734, largest 60.9%. | Visibility-masked coverage/recall across façades, roof, ground, vegetation, obstacles. |
| New-input generality | Arbitrary supported drone video + GPS | Derived Zurich MP4 and selected datasets tested; generic MOV/4K not proven. | Fresh vendor files and failure-case suite. |
| Formats | OBJ, PLY, LAS, GeoTIFF, GLB/glTF, FBX | Most export paths exist across different runs; FBX missing; all final outputs not jointly verified. | Independent reopen of *one candidate run's* every requested output. |
| Public model availability | Real upload -> finished model | Frontend HTTP 200; configured temporary processing hostname did not resolve from Mac; A100 not site worker. | Authenticated external-user end-to-end smoke, secure artifacts, restart. |

## Definition of done for a truthful submission

**Minimum credible prototype:** an accessible repository and demo; actual moving-video+GPS inference with a real exported model; provenance and reproducibility instructions; at least one full measured runtime; honest, run-specific evaluation; clear disclosure that ≤1 m and completeness are not verified if they are not. **Full problem compliance** requires the organizer's independent metric-surface and coverage tests, robust unseen-video input, all required formats, and runtime on the final pipeline. No amount of UI polish changes that.

## Team runbook / handoff safeguards

- The public code checkout is `work/sih3d-gpu/deliverables/github-submission-20260910/`; its remote is `https://github.com/Suman1108-beep/voxelflight.git`. The active `viewer/` checkout has local modifications. Review diffs and selectively integrate; do not overwrite or assume later research is already published.
- The A100 workspace audited read-only is `/workspace/voxelflight_a100_20260928`. It contains `datasets/`, `runs/`, `cache/`, `thirdparty/DPVO/dpvo.pth` and model environment. It is **not** a durable public host. Use the user's private local bridge instructions to reconnect; keep encrypted token/private key **outside GitHub and these docs**. Do not paste raw secrets into issue reports.
- Preserve immutable raw evaluation JSON, screenshots and meshes before another run. Create new run IDs; never overwrite a baseline to make a chart look better.
- The public frontend has a 60 MiB upload cap in its existing configuration, potentially too small for a 10-minute/4K clip. Any increase needs server bounds, storage, abuse control and an actual processing capacity test.
- Check paper/model/dataset licenses before pushing checkpoints or claiming commercial/defence suitability. Source code, pretrained weights and public data may have different terms.
- If time runs out, submit the **tested** branch, not a half-integrated new model. Change all public claims to match that branch's actual evidence.

## Outstanding decisions requiring explicit evidence

1. Does the organizer's “≤1 m spatial accuracy” refer to registered/local surface, absolute geolocation, or both? Obtain scoring script/rules; until then report both separately.
2. Will the organizer provide a mandatory fixed input/output schema, official flight and ground-truth mesh/LiDAR? Adapt parser/evaluator only after receipt and do not train on final labels.
3. Is FBX actually enforced for every submitted output or a list of acceptable formats? The supplied table lists it; implement if possible, disclose otherwise.
4. Can the team provide a durable A100 worker and HTTPS ingress for public uploads by the deadline? If not, clearly provide a local/offline tested inference workflow and a saved public viewer without claiming live GPU processing.
5. What is the exact 5 October deadline hour/time zone and required form/repository/deck/video links? Confirm from organizer communication rather than relying on the extension date alone.
