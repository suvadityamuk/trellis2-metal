"""Offline: GPU wavefront TELEA (build/glb_test --dump DIR) against OpenCV cv2.inpaint(INPAINT_TELEA), the call the official
to_glb makes (radius 3 on base colour, radius 1 on metallic / roughness / alpha). Run with ref/venv; not used at runtime.

  python tools/check_telea.py DIR/telea_real [--out bench/x.json]

OpenCV fills in fast-marching order (a priority queue on the distance map T); the kernel fills in wavefronts of the
4-connected level, with the level standing in for T. Texels next to a chart are filled from the same known texels by
both; far from the charts the two orders differ, which nothing samples (only a bilinear / mip footprint next to a chart
can be seen). Hence the report splits the holes by level.
"""
import json, sys, time
import numpy as np, cv2


def main():
    base = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    m = json.load(open(base + ".json")); W, H = m["W"], m["H"]; P = W * H
    src = np.fromfile(base + ".in.u8", np.uint8).reshape(6, H, W)
    got = np.fromfile(base + ".out.u8", np.uint8).reshape(6, H, W)
    lv = np.fromfile(base + ".lv.i32", np.int32).reshape(H, W)
    hole = lv != 0
    reach = lv > 0  # holes the wavefront reached (all of them unless nothing is known)
    assert np.array_equal(got[:, ~hole], src[:, ~hole]), "known texels were modified"
    mask = hole.astype(np.uint8)
    t = time.perf_counter()
    ref = np.empty_like(src)
    ref[:3] = cv2.inpaint(np.ascontiguousarray(src[:3].transpose(1, 2, 0)), mask, 3, cv2.INPAINT_TELEA).transpose(2, 0, 1)
    for c in (3, 4, 5):
        ref[c] = cv2.inpaint(np.ascontiguousarray(src[c]), mask, 1, cv2.INPAINT_TELEA)
    t_cv = time.perf_counter() - t
    res = {"W": W, "H": H, "holes": int(hole.sum()), "levels": int(lv.max()), "cv2_seconds_4_inpaints": round(t_cv, 3), "rows": {}}
    print(f"{base}: {W}x{H}, {hole.sum()} hole texels ({100 * hole.mean():.1f}%), {lv.max()} levels; cv2.inpaint x4 on this Mac (naive CPU): {t_cv:.2f} s")
    d = np.abs(got.astype(np.int16) - ref.astype(np.int16))
    for name, sel in [("level 1 (touching a chart)", lv == 1), ("levels 1-2", (lv >= 1) & (lv <= 2)), ("levels 1-4", (lv >= 1) & (lv <= 4)), ("levels 1-8", (lv >= 1) & (lv <= 8)), ("all holes", reach)]:
        if not sel.any():
            continue
        row = {}
        for cn, cs in [("base rgb", slice(0, 3)), ("metallic", slice(3, 4)), ("roughness", slice(4, 5)), ("alpha", slice(5, 6))]:
            x = d[cs][:, sel]
            row[cn] = {"mean": round(float(x.mean()), 3), "p99": float(np.percentile(x, 99)), "max": int(x.max()), "equal_pct": round(100 * float((x == 0).mean()), 2), "within1_pct": round(100 * float((x <= 1).mean()), 2)}
        res["rows"][name] = {"texels": int(sel.sum()), **row}
        print(f"  {name:27s} {int(sel.sum()):9d} texels | " + " | ".join(f"{k}: mean {v['mean']:.2f} p99 {v['p99']:.0f} max {v['max']} (<=1: {v['within1_pct']:.1f}%)" for k, v in row.items()))
    if out:
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
