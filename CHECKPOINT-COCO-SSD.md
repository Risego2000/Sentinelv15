# SENTINEL V15 - Checkpoint Documentation
**Date:** 2026-01-10 06:31 CET
**Status:** STABLE - COCO-SSD Implementation

## System State

### Current Detection Engine
- **Model:** TensorFlow.js COCO-SSD
- **Version:** Latest (loaded from CDN)
- **Inference:** Client-side, CPU-based
- **Performance:** ~3-5 FPS during inference (Frame Skip: 4)

### Tracking System
- **Algorithm:** Custom Kalman-style tracking with IoU-based duplicate detection
- **Stability:** ✅ Excellent (no flicker, no phantom boxes)
- **Precision:** ✅ Good (boxes ~200px max, proper vehicle coverage)
- **Persistence:** 30-45 frames minimum

### Parameters (Motor de Inferencia)
```
Umbral Confianza: 0.35
Frame Skip: 4
Persistencia: 45 frames
```

### Recent Fixes Applied
1. ✅ Eliminated giant phantom boxes (max dimension: 200px)
2. ✅ IoU-based duplicate detection (threshold: 0.2)
3. ✅ Confidence gating for new tracks (min: 0.4)
4. ✅ Dimension clamping (max 25% change per frame)
5. ✅ Opacity hysteresis (95% after 5 frames)
6. ✅ Enhanced size similarity in matching

### Known Issues
- None critical
- FPS drops to 3-5 during heavy inference (expected behavior)

## Rollback Instructions

If YOLOv8 migration fails or causes issues:

1. Restore backup file:
   ```bash
   Copy-Item "index.tsx.backup-coco-ssd-*" "index.tsx"
   ```

2. The backup contains the fully functional COCO-SSD implementation with all tracking fixes

3. Refresh browser to load restored code

## Files Backed Up
- `index.tsx` → `index.tsx.backup-coco-ssd-YYYYMMDD-HHMMSS`

## Next Steps
- Migrate to YOLOv8 for improved accuracy
- Maintain all tracking logic
- Test thoroughly before removing COCO-SSD

---
**Backup created successfully. Safe to proceed with YOLOv8 migration.**
