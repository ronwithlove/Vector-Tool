"""标签图 → 叠放的矢量形状 → SVG。

叠放思路（设计师手绘习惯）：
    按“填满孔洞后的面积”从大到小排序，大块在下、小块在上。
    每个色块只需描它对“下层”可见的边界；与“上层”相邻的部分可以随意外扩，
    反正会被盖住。这样相邻色块之间没有缝隙，路径也更简单。
"""
from __future__ import annotations

import math
import time

import cv2
import numpy as np
from skimage.measure import find_contours

from .pipeline import Params, Segmentation
from . import fit as F


# ------------------------------------------------------------------ 区域

def _fill_holes(m: np.ndarray) -> np.ndarray:
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m.astype(np.uint8)
    inv = (1 - pad).astype(np.uint8)
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(inv, mask, (0, 0), 2)
    return (inv[1:-1, 1:-1] != 2)


def extract_regions(labels: np.ndarray, n: int):
    H, W = labels.shape
    regions = []
    rank = np.zeros((H, W), np.int32)
    tmp = []
    for c in range(n):
        m = (labels == c).astype(np.uint8)
        cnt, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        for i in range(1, cnt):
            x, y, w, h, area = [int(v) for v in stats[i]]
            sub = cc[y:y + h, x:x + w] == i
            filled = _fill_holes(sub)
            tmp.append(dict(color=c, x=x, y=y, w=w, h=h, area=area,
                            filled_area=int(filled.sum()), sub=sub,
                            border=(x == 0 or y == 0 or x + w == W or y + h == H)))
    tmp.sort(key=lambda r: -r["filled_area"])
    for k, r in enumerate(tmp):
        r["order"] = k
        view = rank[r["y"]:r["y"] + r["h"], r["x"]:r["x"] + r["w"]]
        view[r["sub"]] = k
        regions.append(r)
    return regions, rank


# ------------------------------------------------------------------ 轮廓

def region_outline(r, rank, sigma, expand):
    """返回该区域外轮廓（工作坐标，像素中心为 +0.5）的点列表，可能多条。"""
    H, W = rank.shape
    pad = expand + int(3 * sigma) + 3
    x0, y0 = r["x"] - pad, r["y"] - pad
    x1, y1 = r["x"] + r["w"] + pad, r["y"] + r["h"] + pad
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    crop_rank = np.full((y1 - y0, x1 - x0), -1, np.int32)
    crop_rank[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0] = rank[cy0:cy1, cx0:cx1]
    k = r["order"]
    # 先填自身孔洞（孔洞里只会是上层区域），再外扩
    own = _fill_holes(crop_rank == k).astype(np.uint8)
    # 外扩到上层区域（以及画布外，避免画布边缘露白）
    if expand > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * expand + 1, 2 * expand + 1))
        dil = cv2.dilate(own, ker)
        above = (crop_rank > k) | (crop_rank < 0)
        own = own | (dil & above.astype(np.uint8))
    # 外扩后不再填洞：外扩可能把一圈细环闭合，里面若是下层色块必须保留为洞
    filled = (own > 0).astype(np.float32)
    if sigma > 0:
        filled = cv2.GaussianBlur(filled, (0, 0), sigma)
    contours = find_contours(np.pad(filled, 1), 0.5)
    out = []
    for c in contours:
        if len(c) < 6:
            continue
        if np.linalg.norm(c[0] - c[-1]) > 1e-6:
            continue
        pts = np.stack([c[:, 1] - 1 + x0 + 0.5, c[:, 0] - 1 + y0 + 0.5], axis=1)[:-1]
        area = _signed_area(pts)
        out.append((area, pts))
    return [p for a, p in out if abs(a) > 4]


def _signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def resample(pts: np.ndarray, step: float) -> np.ndarray:
    closed = np.vstack([pts, pts[:1]])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))]
    total = d[-1]
    n = max(8, int(total / step))
    t = np.linspace(0, total, n, endpoint=False)
    x = np.interp(t, d, closed[:, 0])
    y = np.interp(t, d, closed[:, 1])
    return np.stack([x, y], axis=1)


def find_corners(pts: np.ndarray, arm: int, angle_deg: float):
    n = len(pts)
    if n < 2 * arm + 3:
        return []
    a = np.roll(pts, arm, axis=0) - pts
    b = np.roll(pts, -arm, axis=0) - pts
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    cosv = np.clip(np.sum(a * b, axis=1) / (na * nb), -1, 1)
    turn = 180 - np.degrees(np.arccos(cosv))
    cand = turn > angle_deg
    corners = []
    i = 0
    if cand.all():
        return []
    # 从一个非候选点开始，保证不会把跨越首尾的候选段切开
    start = int(np.argmin(cand))
    idx = [(start + j) % n for j in range(n)]
    run = []
    for j in idx + [idx[0]]:
        if cand[j]:
            run.append(j)
        elif run:
            best = max(run, key=lambda q: turn[q])
            corners.append(best)
            run = []
    corners = sorted(set(corners))
    return corners


# ------------------------------------------------------------------ 拟合一条闭合轮廓

def _sharpen(pts, c, arm, maxdist):
    """模糊会把尖角磨圆：用尖角两侧较远处的走向作直线求交，还原尖角位置。"""
    n = len(pts)
    i0, i1 = pts[(c - 2 * arm) % n], pts[(c - arm) % n]
    o0, o1 = pts[(c + arm) % n], pts[(c + 2 * arm) % n]
    d_in, d_out = F._norm(i1 - i0), F._norm(o1 - o0)
    cross = d_in[0] * d_out[1] - d_in[1] * d_out[0]
    if abs(cross) < 0.25:
        return pts[c]
    M = np.array([d_in, d_out]).T
    t, u = np.linalg.solve(M, o0 - i1)
    q = i1 + t * d_in
    if t < 0 or u < 0 or np.linalg.norm(q - pts[c]) > maxdist:
        return pts[c]
    return q


def fit_closed(pts, s, p: Params):
    tol = p.tolerance * s
    step = max(0.5, 0.5 * s)
    pts = resample(pts, step)
    n = len(pts)
    # 沿轮廓做高斯平滑，抹掉压缩造成的小抖动（尖角之后会用切线求交还原）
    sg = p.smooth * s / step
    if sg > 0.3 and n > 8:
        rad = int(3 * sg) + 1
        if 2 * rad + 1 < n:
            kx = np.exp(-0.5 * (np.arange(-rad, rad + 1) / sg) ** 2)
            kx /= kx.sum()
            ext = np.vstack([pts[-rad:], pts, pts[:rad]])
            pts = np.stack([np.convolve(ext[:, 0], kx, "valid"), np.convolve(ext[:, 1], kx, "valid")], axis=1)
    span = 2.5 * s
    # 测尖角的“臂长”要超出平滑磨圆的范围，否则直角会被量成 45°
    arm = max(2, int(round((2.0 + 2.5 * p.smooth) * s / step)))
    line_abs = max(0.3, 0.45 * p.tolerance) * s
    line_rel = 0.005
    min_line = 10.0 * s
    # 小形状里的直线段也要能识别（按形状尺寸缩放）
    size = float(max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
    min_line = min(min_line, max(3.0 * s, 0.25 * size))

    corners = find_corners(pts, arm, p.corner_angle)

    if not corners and p.detect_circles:
        e = F.try_ellipse(pts, max(0.6 * s, tol), p.snap_angle, rel=0.06)
        if e is not None:
            return e, []

    if not corners:
        # 无尖角：在曲率最小（最平直）的地方切开，保证首尾切向连续
        i0 = 0
        t = F._norm(pts[(i0 + 1) % n] - pts[i0 - 1])
        ring = np.vstack([pts[i0:], pts[:i0 + 1]])
        segs = F.fit_open(ring, tol, line_abs, line_rel, min_line, t1=t, t2=-t, span=span)
        return segs, []

    segs = []
    corner_vertices = []
    m = len(corners)
    rr = (1.0 + 2.0 * p.smooth) * s   # 模糊+平滑磨圆的范围
    sharp = [_sharpen(pts, c, arm, rr) for c in corners]
    trim = rr
    for k in range(m):
        a = corners[k]
        b = corners[(k + 1) % m]
        if b <= a:
            chunk = np.vstack([pts[a:], pts[:b + 1]])
        else:
            chunk = pts[a:b + 1]
        qa, qb = sharp[k], sharp[(k + 1) % m]
        if len(chunk) > 6:
            da = np.linalg.norm(chunk - chunk[0], axis=1)
            db = np.linalg.norm(chunk - chunk[-1], axis=1)
            keep = (da >= trim) & (db >= trim)
            keep[0] = keep[-1] = False
            mid = chunk[keep]
            chunk = np.vstack([qa[None], mid, qb[None]])
        else:
            chunk = chunk.copy()
            chunk[0], chunk[-1] = qa, qb
        corner_vertices.append(len(segs))
        segs += F.fit_open(chunk, tol, line_abs, line_rel, min_line, span=span)
    return segs, corner_vertices


# ------------------------------------------------------------------ 规整

def _seg_end_dir(seg):
    """离开终点的反向切向（从终点指回曲线）"""
    if seg[0] == "L":
        return F._norm(seg[1] - seg[2])
    v = seg[3] - seg[4]
    if np.linalg.norm(v) < 1e-9:
        v = seg[2] - seg[4]
    return F._norm(v)


def _seg_start_dir(seg):
    if seg[0] == "L":
        return F._norm(seg[2] - seg[1])
    v = seg[2] - seg[1]
    if np.linalg.norm(v) < 1e-9:
        v = seg[3] - seg[1]
    return F._norm(v)


def _move_vertex(segs, i, newp):
    """把第 i 段起点（=第 i-1 段终点）移到 newp，相邻控制柄一起平移。"""
    n = len(segs)
    cur = list(segs[i])
    prev = list(segs[i - 1 if i > 0 else n - 1])
    delta = newp - cur[1]
    cur[1] = newp.copy()
    if cur[0] == "C":
        cur[2] = cur[2] + delta
    prev[-1] = newp.copy()
    if prev[0] == "C":
        prev[3] = prev[3] + delta
    segs[i] = tuple(cur)
    segs[i - 1 if i > 0 else n - 1] = tuple(prev)


def regularize(segs, corner_vertices, s, p: Params):
    n = len(segs)
    if n < 2:
        return segs
    segs = list(segs)
    if not p.regularize:
        return segs
    # 接近水平/垂直的直线吸附
    tan_snap = math.tan(math.radians(p.snap_angle))
    fixed_x = {}
    fixed_y = {}
    allp = np.array([sg[1] for sg in segs])
    size = float(max(np.ptp(allp[:, 0]), np.ptp(allp[:, 1])))
    min_snap = min(12.0 * s, 0.25 * size)   # 小形状（如 ® 里的 R）按比例放宽
    for i, sg in enumerate(segs):
        if sg[0] != "L":
            continue
        d = sg[2] - sg[1]
        # 太短的直线不吸附：曲线上偶然的一小段“直”不应被拉成水平/垂直
        if np.linalg.norm(d) < min_snap:
            continue
        if abs(d[0]) > 1e-9 and abs(d[1]) / abs(d[0]) < tan_snap:
            y = (sg[1][1] + sg[2][1]) / 2
            fixed_y[i] = y
            fixed_y[(i + 1) % n] = y
        elif abs(d[1]) > 1e-9 and abs(d[0]) / abs(d[1]) < tan_snap:
            x = (sg[1][0] + sg[2][0]) / 2
            fixed_x[i] = x
            fixed_x[(i + 1) % n] = x
    for vi in set(fixed_x) | set(fixed_y):
        v = segs[vi][1].copy()
        if vi in fixed_x:
            v[0] = fixed_x[vi]
        if vi in fixed_y:
            v[1] = fixed_y[vi]
        _move_vertex(segs, vi, v)
    return segs


# ------------------------------------------------------------------ SVG

def _fmt(v):
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def segs_to_d(segs, s):
    if not segs:
        return "", []
    k = 1.0 / s
    p0 = segs[0][1] * k
    d = [f"M{_fmt(p0[0])} {_fmt(p0[1])}"]
    anchors = [p0]
    last = len(segs) - 1
    for i, sg in enumerate(segs):
        if i == last and sg[0] == "L":
            break  # 最后一段直线回到起点，用 Z 闭合即可，不多出一个重合锚点
        if sg[0] == "L":
            q = sg[2] * k
            d.append(f"L{_fmt(q[0])} {_fmt(q[1])}")
        else:
            c1, c2, q = sg[2] * k, sg[3] * k, sg[4] * k
            d.append(f"C{_fmt(c1[0])} {_fmt(c1[1])} {_fmt(c2[0])} {_fmt(c2[1])} {_fmt(q[0])} {_fmt(q[1])}")
        anchors.append(q)
    d.append("Z")
    if segs[-1][0] != "L":
        anchors = anchors[:-1]
    return "".join(d), anchors


def vectorize_segmentation(seg: Segmentation, p: Params):
    t0 = time.time()
    s = seg.scale
    W, H = seg.orig_size
    regions, rank = extract_regions(seg.labels, len(seg.colors_hex))
    sigma = max(0.8, 0.6 * s)
    expand = max(2, int(round(1.5 * s)))

    shapes = []
    total_anchors = 0
    bg_color = None
    for r in regions:
        color = seg.colors_hex[r["color"]]
        if r["order"] == 0 and r["border"]:
            # 最大的、贴边的色块作为背景矩形
            bg_color = color
            continue
        ds, anchors = [], []
        for pts in region_outline(r, rank, sigma, expand):
            segs, cv = fit_closed(pts, s, p)
            segs = regularize(segs, cv, s, p)
            d, an = segs_to_d(segs, s)
            if d:
                ds.append(d)
                anchors += an
        if not ds:
            continue
        total_anchors += len(anchors)
        shapes.append(dict(d="".join(ds), fill=color,
                           anchors=[[round(float(a[0]), 2), round(float(a[1]), 2)] for a in anchors]))

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">']
    if bg_color and not p.remove_bg:
        parts.append(f'<rect width="{W}" height="{H}" fill="{bg_color}"/>')
    for sh in shapes:
        parts.append(f'<path fill="{sh["fill"]}" fill-rule="evenodd" d="{sh["d"]}"/>')
    parts.append("</svg>")
    stats = dict(colors=len(seg.colors_hex), shapes=len(shapes) + (1 if bg_color and not p.remove_bg else 0),
                 anchors=total_anchors, trace_ms=int((time.time() - t0) * 1000))
    return "\n".join(parts), shapes, stats
