"""曲线拟合：直线检测、三次贝塞尔拟合（Schneider 算法）、椭圆识别。

所有坐标为 float64 的 (N, 2) 数组 (x, y)。
段的表示：
    ("L", p0, p3)
    ("C", p0, c1, c2, p3)
"""
from __future__ import annotations

import math
import numpy as np


def _norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v * 0


# ------------------------------------------------------------------ 直线

def line_dev(pts: np.ndarray) -> float:
    """点到首尾连线的最大距离。"""
    a, b = pts[0], pts[-1]
    d = b - a
    L = np.linalg.norm(d)
    if L < 1e-9:
        return float(np.max(np.linalg.norm(pts - a, axis=1)))
    n = np.array([-d[1], d[0]]) / L
    return float(np.max(np.abs((pts - a) @ n)))


def find_straight_runs(pts: np.ndarray, abs_tol: float, rel_tol: float, min_len: float):
    """贪心找出足够长、足够直的子段，返回 [(i, j), ...]（含端点下标）。"""
    n = len(pts)
    runs = []
    i = 0
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    while i < n - 2:
        # 先以 min_len 为起步长度
        j = int(np.searchsorted(seg, seg[i] + min_len))
        if j >= n:
            break
        if not _straight(pts[i:j + 1], abs_tol, rel_tol):
            i += 1
            continue
        # 指数 + 二分扩展
        step = max(1, (j - i) // 2)
        while j + step < n and _straight(pts[i:j + step + 1], abs_tol, rel_tol):
            j += step
            step *= 2
        while step > 1:
            step //= 2
            if j + step < n and _straight(pts[i:j + step + 1], abs_tol, rel_tol):
                j += step
        runs.append((i, j))
        i = j
    return runs


def _straight(p, abs_tol, rel_tol):
    L = np.linalg.norm(p[-1] - p[0])
    if L < 1e-6:
        return False
    d = line_dev(p)
    return d <= abs_tol and d <= rel_tol * L


# ------------------------------------------------------------------ 贝塞尔

def _bez(ctrl, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return (mt ** 3) * ctrl[0] + 3 * (mt ** 2) * t * ctrl[1] + 3 * mt * (t ** 2) * ctrl[2] + (t ** 3) * ctrl[3]


def _bez_d1(ctrl, t):
    t = np.asarray(t)[:, None]
    mt = 1 - t
    return 3 * (mt ** 2) * (ctrl[1] - ctrl[0]) + 6 * mt * t * (ctrl[2] - ctrl[1]) + 3 * (t ** 2) * (ctrl[3] - ctrl[2])


def _bez_d2(ctrl, t):
    t = np.asarray(t)[:, None]
    return 6 * (1 - t) * (ctrl[2] - 2 * ctrl[1] + ctrl[0]) + 6 * t * (ctrl[3] - 2 * ctrl[2] + ctrl[1])


def _chord_param(pts):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return d / d[-1] if d[-1] > 0 else np.linspace(0, 1, len(pts))


def _generate(pts, u, t1, t2):
    p0, p3 = pts[0], pts[-1]
    A1 = t1[None, :] * (3 * (1 - u) ** 2 * u)[:, None]
    A2 = t2[None, :] * (3 * (1 - u) * u ** 2)[:, None]
    C00 = np.sum(A1 * A1)
    C01 = np.sum(A1 * A2)
    C11 = np.sum(A2 * A2)
    base = _bez(np.array([p0, p0, p3, p3]), u)
    tmp = pts - base
    X0 = np.sum(A1 * tmp)
    X1 = np.sum(A2 * tmp)
    det = C00 * C11 - C01 * C01
    seg_len = np.linalg.norm(p3 - p0)
    eps = 1e-6 * seg_len
    if abs(det) > 1e-12:
        al = (X0 * C11 - X1 * C01) / det
        ar = (C00 * X1 - C01 * X0) / det
    else:
        al = ar = 0
    if al < eps or ar < eps or al > seg_len * 2 or ar > seg_len * 2:
        al = ar = seg_len / 3.0
    return np.array([p0, p0 + t1 * al, p3 + t2 * ar, p3])


def _max_err(pts, ctrl, u):
    d = np.linalg.norm(_bez(ctrl, u) - pts, axis=1)
    i = int(np.argmax(d))
    return float(d[i]), i


def _reparam(pts, ctrl, u):
    q = _bez(ctrl, u)
    q1 = _bez_d1(ctrl, u)
    q2 = _bez_d2(ctrl, u)
    num = np.sum((q - pts) * q1, axis=1)
    den = np.sum(q1 * q1, axis=1) + np.sum((q - pts) * q2, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = np.where(np.abs(den) > 1e-12, u - num / den, u)
    nu = np.clip(nu, 0, 1)
    # 保持单调
    nu = np.maximum.accumulate(nu)
    return nu


def fit_cubic(pts: np.ndarray, t1: np.ndarray, t2: np.ndarray, tol: float, depth=0):
    """t1：起点处指向曲线内部的单位切向；t2：终点处指向曲线内部（向后）的单位切向。"""
    n = len(pts)
    if n == 2 or depth > 12:
        L = np.linalg.norm(pts[-1] - pts[0]) / 3.0
        return [("C", pts[0], pts[0] + t1 * L, pts[-1] + t2 * L, pts[-1])]
    u = _chord_param(pts)
    ctrl = _generate(pts, u, t1, t2)
    err, split = _max_err(pts, ctrl, u)
    if err < tol:
        return [("C", *ctrl)]
    if err < tol * 4:
        for _ in range(12):
            u = _reparam(pts, ctrl, u)
            ctrl = _generate(pts, u, t1, t2)
            err, split = _max_err(pts, ctrl, u)
            if err < tol:
                return [("C", *ctrl)]
    split = min(max(split, 1), n - 2)
    tc = _norm(pts[split - 1] - pts[split + 1])
    if np.linalg.norm(tc) < 1e-9:
        tc = _norm(pts[split - 1] - pts[split])
    left = fit_cubic(pts[:split + 1], t1, tc, tol, depth + 1)
    right = fit_cubic(pts[split:], -tc, t2, tol, depth + 1)
    return left + right


def end_tangent(pts: np.ndarray, at_start: bool, span: float) -> np.ndarray:
    """用端点附近一段点估计切向（指向曲线内部）。"""
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if at_start:
        k = int(np.searchsorted(seg, min(span, seg[-1] * 0.4)))
        k = max(1, min(k, len(pts) - 1))
        return _norm(np.mean(pts[1:k + 1], axis=0) - pts[0])
    else:
        target = seg[-1] - min(span, seg[-1] * 0.4)
        k = int(np.searchsorted(seg, target))
        k = max(0, min(k, len(pts) - 2))
        return _norm(np.mean(pts[k:-1], axis=0) - pts[-1])


def fit_open(pts: np.ndarray, tol: float, line_abs: float, line_rel: float, min_line: float,
             t1=None, t2=None, span=3.0):
    """拟合一段开放曲线（两端是尖角或已知切向）。混合直线段与曲线段。"""
    if len(pts) < 2:
        return []
    if len(pts) == 2 or line_dev(pts) <= line_abs:
        return [("L", pts[0], pts[-1])]
    runs = find_straight_runs(pts, line_abs, line_rel, min_line)
    segs = []
    cursor = 0
    cur_t = t1
    for (i, j) in runs:
        if i > cursor:
            d = _norm(pts[j] - pts[i])
            segs += _fit_curve(pts[cursor:i + 1], cur_t, -d, tol, span)
        segs.append(("L", pts[i], pts[j]))
        cur_t = _norm(pts[j] - pts[i])
        cursor = j
    if cursor < len(pts) - 1:
        segs += _fit_curve(pts[cursor:], cur_t, t2, tol, span)
    return segs


def _fit_curve(pts, t1, t2, tol, span):
    if len(pts) < 2:
        return []
    if t1 is None:
        t1 = end_tangent(pts, True, span)
    if t2 is None:
        t2 = end_tangent(pts, False, span)
    if len(pts) <= 3:
        L = np.linalg.norm(pts[-1] - pts[0])
        if L < 1e-9:
            return []
        return [("C", pts[0], pts[0] + t1 * L / 3, pts[-1] + t2 * L / 3, pts[-1])]
    return fit_cubic(pts, t1, t2, tol)


# ------------------------------------------------------------------ 椭圆

KAPPA = 0.5522847498


def try_ellipse(pts: np.ndarray, tol: float, snap_deg: float, rel: float = 0.0):
    """整条闭合轮廓能否用一个椭圆（或正圆）表示。返回段列表或 None。"""
    if len(pts) < 12:
        return None
    (cx, cy), (w, h), ang = __import__("cv2").fitEllipse(pts.astype(np.float32))
    a, b = w / 2.0, h / 2.0
    if a < 1 or b < 1:
        return None
    th = math.radians(ang)
    c, s = math.cos(th), math.sin(th)
    d = pts - np.array([cx, cy])
    x = d[:, 0] * c + d[:, 1] * s
    y = -d[:, 0] * s + d[:, 1] * c
    r = np.sqrt((x / a) ** 2 + (y / b) ** 2)
    dev = np.max(np.abs(r - 1)) * min(a, b)
    # 小形状允许按比例放宽：20px 的圆差 1px 也应当是圆
    if dev > max(tol, rel * min(a, b)):
        return None
    # 规整：接近正圆 → 正圆；接近轴对齐 → 轴对齐
    if abs(a - b) / max(a, b) < 0.04:
        a = b = (a + b) / 2.0
        th = 0.0
    else:
        deg = ang % 90
        if deg < snap_deg or deg > 90 - snap_deg:
            th = math.radians(round(ang / 90.0) * 90.0)
    return ellipse_segments(cx, cy, a, b, th)


def ellipse_segments(cx, cy, a, b, th):
    c, s = math.cos(th), math.sin(th)

    def P(x, y):
        return np.array([cx + x * c - y * s, cy + x * s + y * c])
    k = KAPPA
    pts = [(a, 0), (0, b), (-a, 0), (0, -b)]
    hands = [((a, k * b), (k * a, b)), ((-k * a, b), (-a, k * b)),
             ((-a, -k * b), (-k * a, -b)), ((k * a, -b), (a, -k * b))]
    segs = []
    for i in range(4):
        p0 = P(*pts[i])
        p3 = P(*pts[(i + 1) % 4])
        c1 = P(*hands[i][0])
        c2 = P(*hands[i][1])
        segs.append(("C", p0, c1, c2, p3))
    return segs
