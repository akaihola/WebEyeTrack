---
name: webeyetrack-accuracy-investigation
description: "Root causes of WebEyeTrack head-pose drift & bottom-edge bias, fixes applied 2026-07-04, and improvement/alternatives research"
metadata: 
  node_type: memory
  type: project
  originSessionId: e0be87d9-3c8c-4df6-9e2e-79ecaeeda622
---

Investigation (2026-07-04) into two symptoms of the Python demo: gaze follows head rotation/tilt, and bottom-edge gaze predicted too high.

**Root causes found:**
- Head pose is NOT geometrically compensated: `head_vector` + `face_origin_3d` are just 6 of 518 inputs to a Dense(16,16,2) MLP (`blazegaze.py:213-217`). The full geometric pipeline (`compute_pog`, `screen_plane_intersection` in `model_based.py:401-939`) is dead code.
- Calibration (`adapt()` in `webeyetrack.py:320`) fine-tunes the whole MLP + fits a 2×3 affine on samples from ONE head pose → correction invalid after head motion.
- Default model `blazegaze_mpiifacegaze.keras` was trained with MPIIFaceGaze ground-truth `face_origin_3d`, but inference computes it via MediaPipe reconstruction → train/inference feature mismatch.
- Blink frames return `norm_pog=(0,0)` = screen CENTER (coords centered at 0, range ±0.5).
- Metric scale from hard-coded 1.2 cm iris, frozen at first frame; two inconsistent camera models (60° FOV perspective matrix vs f=width intrinsics); inconsistent Euler sign conventions across call sites.

**Fixes applied to demo (uncommitted as of 2026-07-04):** symmetric ±0.5 clamp (was y±0.45), blink frames filtered from both calibration paths, 3×3 calibration grid at {−0.45,0,0.45} (was 4 corners ±0.4), up to 5 samples/point for affine fit (was 1), config.yml calib/kalman settings wired into WebEyeTrackConfig (were dead).

**Paper:** arXiv:2508.19544 (Davalos et al.). No head-pose ablation, no vertical/horizontal error split. GitHub RedForestAI/WebEyeTrack lightly maintained; issue #6 = closest analog (calibration degradation, JS lacks MAML). Our symptoms not filed upstream.

**Second opinion / component decision:** RollScore needs accurate vertical gaze to identify the staff system, not merely horizontal sweep detection. Fork and improve WebEyeTrack's browser implementation: retain its eye encoder and pose inputs, expose the pose per frame, and add RollScore-specific dynamic multi-pose calibration plus a small regularized pose-conditioned residual correction. Do not start with WebGazer.js (no pose input), the dormant geometric pipeline, full-network retraining, or a ground-up model. Use a fixed-target head-sweep diagnostic and held-out pose sectors to verify that drift is smooth and correctable before committing to the fork.

**Revised improvement estimate:** target about 20-30% lower overall error under moderate pose variation; 30-50% may be achievable only for the explicitly head-correlated component. The previous 30-50% overall estimate was too confident. Evaluate staff-system classification directly: provisional target >=95% correct within the operating pose envelope, almost all residual errors adjacent, and vertical p90 error comfortably below half the neighboring-system centerline distance. If WebEyeTrack's residual is not repeatable because occlusion/embedding instability dominates, replace only the base estimator with a wide-pose model while retaining calibration and staff classification; a model trained from scratch is the last resort.

**Best alternatives (deep-research verified):** FalchLucas/WebCamGazeEstimation (3.2°→5.1° under ±30° yaw, but n=1, no license); OpenFace 3.0 (2.56° MPIIGaze, 38ms CPU, maintained); L2CS-Net & yakhyo/gaze-estimation (MIT, Gaze360-trained, ONNX, no screen mapping). Browser: no verified drop-in beats WebEyeTrack; path = ONNX Runtime Web port. Full report: scratchpad deep-research-result.json (session e0be87d9).
