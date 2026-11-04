"""位图 → 干净色块标签图。

步骤：读图 → 统一工作尺寸 → 保边降噪 → Lab 空间量化 + ΔE 合并 →
清理抗锯齿细条和小碎块 → 输出 (labels, palette)。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

import math

import cv2
import numpy as np


@dataclass
class Params:
    max_colors: int = 12          # 最多保留的颜色数
    merge_de: float = 10.0        # ΔE 小于此值的颜色强制合并
    min_area: float = 40.0        # 最小色块面积（原图像素²）
    strip_width: float = 2.0      # 宽度小于此值的细条视为抗锯齿过渡带（原图像素）
    denoise: int = 2              # 降噪强度 0-5
    tolerance: float = 1.0        # 曲线拟合容差（原图像素）
    smooth: float = 1.0           # 轮廓平滑度（原图像素）
    corner_angle: float = 50.0    # 转角大于此角度视为尖角（度）
    regularize: bool = True       # 几何规整：直线吸附水平/垂直
    snap_angle: float = 4.0       # 偏离水平/垂直多少度以内吸附
    detect_circles: bool = True   # 识别正圆/椭圆
    remove_bg: bool = False       # 去掉背景色块
    palette: Optional[list] = None  # 用户锁定的调色板 [{"match": "#rrggbb", "out": "#rrggbb"}]

    @classmethod
    def from_dict(cls, d: dict | None) -> "Params":
        p = cls()
        for k, v in (d or {}).items():
            if hasattr(p, k) and v is not None:
                cur = getattr(p, k)
                if isinstance(cur, bool):
                    v = bool(v)
                elif isinstance(cur, int) and not isinstance(cur, bool):
                    v = int(v)
                elif isinstance(cur, float):
                    v = float(v)
                setattr(p, k, v)
        return p

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- 读图与缩放

def decode_image(data: bytes) -> np.ndarray:
    """读取任意格式图片为 BGR uint8；透明部分合成到白底。"""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("无法读取图片")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        rgb = img[:, :, :3].astype(np.float32)
        img = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img


def working_scale(h: int, w: int) -> float:
    """小图放大到约 1600px 再处理（细节和小符号更稳），大图缩到 2400px 以内。"""
    m = max(h, w)
    if m < 1600:
        return min(4.0, 1600.0 / m)
    if m > 2400:
        return 2400.0 / m
    return 1.0


def prepare(img: np.ndarray, p: Params):
    """返回 (工作尺寸图像 BGR, 缩放系数 s)。"""
    h, w = img.shape[:2]
    s = working_scale(h, w)
    if s != 1.0:
        interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=interp)
    # 保边降噪：先轻微中值去 JPEG 块噪，再多次双边滤波抹平纸纹/压缩色差
    if p.denoise > 0:
        k = 3 if s < 2 else 5
        img = cv2.medianBlur(img, k)
        for _ in range(p.denoise):
            img = cv2.bilateralFilter(img, d=max(5, int(3 * s) | 1), sigmaColor=18 + 6 * p.denoise,
                                      sigmaSpace=2.0 * s)
    return img, s


def guided_filter(I: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """引导滤波（He et al.）。用亮度当引导去修正色度：
    JPEG 的 4:2:0 色度下采样会让色度边界呈 8~16px 的块状波浪，
    但亮度边界是准的；让色度边界跟着亮度走，色块边缘就直了。"""
    k = (2 * r + 1, 2 * r + 1)
    box = lambda x: cv2.boxFilter(x, -1, k)
    mI, mp = box(I), box(src)
    vI = box(I * I) - mI * mI
    cIp = box(I * src) - mI * mp
    a = cIp / (vI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


# ---------------------------------------------------------------- 颜色

def bgr_to_lab(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    return cv2.cvtColor(f, cv2.COLOR_BGR2Lab)  # L 0-100, a/b ≈ -127..127


def hex_to_lab(hx: str) -> np.ndarray:
    hx = hx.lstrip("#")
    rgb = np.array([[[int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16)]]], np.uint8)  # BGR
    return bgr_to_lab(rgb)[0, 0]


def bgr_to_hex(c) -> str:
    b, g, r = [int(round(float(x))) for x in c]
    return "#{:02x}{:02x}{:02x}".format(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))


def _looks_mixed(c, cnt, centers, counts):
    """候选色是否像两种已选主色的混合（抗锯齿过渡），且峰值远低于两端主色。"""
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            a, b = centers[i], centers[j]
            ab = b - a
            L2 = float(ab @ ab)
            if L2 < 1e-6:
                continue
            t = float((c - a) @ ab) / L2
            if 0.05 < t < 0.95 and np.linalg.norm(a + t * ab - c) < max(5.0, 0.12 * math.sqrt(L2)):
                if cnt < 0.12 * min(counts[i], counts[j]):
                    return True
    return False


def build_palette(lab: np.ndarray, p: Params, min_px: float) -> np.ndarray:
    """找颜色直方图的峰值作为主色。

    扁平插画/Logo 的每种颜色在 Lab 直方图里都是一个尖峰，
    而抗锯齿混色、压缩色差是散开的“雾”，计数低。
    所以按峰值高度挑选，并要求和已选颜色差距 ≥ merge_de：
    面积小但颜色明确的（耳朵内侧、眼睛）能保住，边缘混色不会占名额。
    """
    px = lab.reshape(-1, 3)
    step = 2.0
    lo = np.array([0.0, -128.0, -128.0])
    dims = np.array([51, 129, 129])
    idx = np.clip(((px - lo) / step).astype(np.int32), 0, dims - 1)
    flat = (idx[:, 0] * dims[1] + idx[:, 1]) * dims[2] + idx[:, 2]
    hist = np.bincount(flat, minlength=int(np.prod(dims))).astype(np.float32).reshape(dims)
    # 3x3x3 盒式平滑，让同一种颜色的轻微抖动汇成一个峰
    sm = np.zeros_like(hist)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                sm += np.roll(hist, (dz, dy, dx), axis=(0, 1, 2))
    order = np.argsort(-sm.ravel())[:4000]
    centers, counts = [], []
    for o in order:
        cnt = sm.ravel()[o]
        if cnt < min_px:
            break
        z, rem = divmod(int(o), int(dims[1] * dims[2]))
        y, x = divmod(rem, int(dims[2]))
        c = lo + (np.array([z, y, x]) + 0.5) * step
        if centers and np.min(np.linalg.norm(np.array(centers) - c, axis=1)) < p.merge_de:
            continue
        if _looks_mixed(c, cnt, centers, counts):
            continue
        centers.append(c)
        counts.append(cnt)
        if len(centers) >= p.max_colors:
            break
    if not centers:
        centers = [px.mean(axis=0)]
    C = np.array(centers, np.float32)
    # 用每个中心附近的“核心”像素均值微调（避开边缘混色）；在抽样像素上做，速度快
    rng = np.random.default_rng(0)
    sub = px[rng.choice(len(px), min(len(px), 200000), replace=False)].astype(np.float32)
    nearest = lambda C: np.argmin((C * C).sum(1)[None, :] - 2.0 * (sub @ C.T), axis=1)
    for _ in range(2):
        lbl = nearest(C)
        for i in range(len(C)):
            sel = sub[lbl == i]
            if len(sel) == 0:
                continue
            d = np.linalg.norm(sel - C[i], axis=1)
            core = sel[d < max(3.0, p.merge_de * 0.5)]
            if len(core) > 10:
                C[i] = core.mean(axis=0)
        # 微调后可能有颜色靠得太近：去掉像素少的那个
        while len(C) > 1:
            d = np.linalg.norm(C[:, None] - C[None], axis=2)
            np.fill_diagonal(d, np.inf)
            i, j = np.unravel_index(np.argmin(d), d.shape)
            if d[i, j] >= p.merge_de:
                break
            cnt = np.bincount(lbl, minlength=len(C))
            C = np.delete(C, j if cnt[i] >= cnt[j] else i, axis=0)
            lbl = nearest(C)
    return C


def assign(lab: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """每个像素归到 Lab 距离最近的调色板颜色（矩阵运算，避免逐色循环）。"""
    h, w = lab.shape[:2]
    px = lab.reshape(-1, 3).astype(np.float32)
    C = np.asarray(pal, np.float32)
    d = (C * C).sum(1)[None, :] - 2.0 * (px @ C.T)
    return np.argmin(d, axis=1).astype(np.int32).reshape(h, w)


# ---------------------------------------------------------------- 清理碎片

def fill_unassigned(labels: np.ndarray) -> np.ndarray:
    """把 -1 像素填成离它最近的已分配像素的颜色。"""
    hole = labels < 0
    if not hole.any():
        return labels
    src = np.where(hole, 255, 0).astype(np.uint8)  # 0 = 已分配
    _, lab = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    lut = np.zeros(lab.max() + 1, np.int32)
    lut[lab[~hole]] = labels[~hole]
    out = labels.copy()
    out[hole] = lut[lab[hole]]
    return out


def clean_labels(labels: np.ndarray, n_colors: int, s: float, p: Params) -> np.ndarray:
    labels = labels.copy()
    # 1) 细条：对每种颜色做开运算，开运算抹掉的部分就是抗锯齿过渡带 / 描边残影
    r = max(1, int(round(p.strip_width * s / 2)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for c in range(n_colors):
        m = (labels == c).astype(np.uint8)
        if not m.any():
            continue
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
        labels[(m > 0) & (opened == 0)] = -1
    labels = fill_unassigned(labels)

    # 2) 小块：面积不到阈值的连通区域并入周围
    min_px = max(4, p.min_area * s * s)
    for _ in range(2):
        changed = False
        for c in range(n_colors):
            m = (labels == c).astype(np.uint8)
            if not m.any():
                continue
            n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
            small = np.where(stats[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
            if len(small):
                kill = np.isin(cc, small)
                labels[kill] = -1
                changed = True
        if not changed:
            break
        labels = fill_unassigned(labels)
    # 3) 众数平滑一遍，去掉边界上的单像素毛刺
    kk = max(3, int(round(1.5 * s)) | 1)
    for _ in range(2):
        labels = mode_filter(labels, n_colors, kk)
    return labels


def mode_filter(labels: np.ndarray, n: int, k: int) -> np.ndarray:
    best = np.zeros(labels.shape, np.float32) - 1
    out = labels.copy()
    for c in range(n):
        m = (labels == c).astype(np.float32)
        if not m.any():
            continue
        v = cv2.boxFilter(m, -1, (k, k), normalize=False)
        upd = v > best
        out[upd] = c
        best[upd] = v[upd]
    return out


def drop_transitional(labels, pal, s, p: Params, max_width=5.0):
    """去掉“过渡色”：颜色上介于另外两种颜色之间（像混出来的），
    且在图上只以窄带形式出现（最宽处 < max_width 原图像素）。
    典型例子：白和米色之间的奶油色描边、红和黄之间的橙色细带。"""
    n = len(pal)
    dropped = []
    for c in range(n):
        m = (labels == c).astype(np.uint8)
        if not m.any():
            continue
        # 颜色是否像两色混合
        is_mix = False
        for i in range(n):
            for j in range(i + 1, n):
                if c in (i, j):
                    continue
                a, b = pal[i], pal[j]
                ab = b - a
                L2 = float(ab @ ab)
                if L2 < 1e-6:
                    continue
                t = float((pal[c] - a) @ ab) / L2
                if 0.08 < t < 0.92:
                    d = np.linalg.norm(a + t * ab - pal[c])
                    if d < max(6.0, 0.6 * p.merge_de, 0.18 * math.sqrt(L2)):
                        is_mix = True
                        break
            if is_mix:
                break
        if not is_mix:
            continue
        # 平均宽度 ≈ 2 × 面积 / 边界像素数（细长条的平均宽度很小，即使局部有鼓包）
        cnt, cc = cv2.connectedComponents(m, connectivity=8)
        edge = (m > 0) & (cv2.erode(m, np.ones((3, 3), np.uint8)) == 0)
        area = np.bincount(cc.ravel(), minlength=cnt).astype(np.float32)
        per = np.bincount(cc[edge], minlength=cnt).astype(np.float32) + 1
        width = 2 * area / per / s
        narrow = np.where(width < max_width)[0]
        narrow = narrow[narrow > 0]
        if len(narrow):
            if not dropped:
                labels = labels.copy()
            labels[np.isin(cc, narrow)] = -1
            dropped.append(c)
    if dropped:
        labels = fill_unassigned(labels)
    return labels, dropped


# ---------------------------------------------------------------- 主流程

@dataclass
class Segmentation:
    labels: np.ndarray            # 工作尺寸的标签图
    colors_hex: list              # 每个标签的输出颜色
    match_hex: list               # 每个标签的识别颜色（调色板确认用）
    shares: list                  # 每个标签的面积占比
    scale: float
    orig_size: tuple              # (w, h)
    work_bgr: np.ndarray = field(repr=False, default=None)


_prep_cache: dict = {}


def _prepared(img: np.ndarray, p: Params):
    """预处理只和降噪强度有关，拖其他滑杆时直接复用。"""
    key = (id(img), img.shape, p.denoise)
    hit = _prep_cache.get(key)
    if hit is not None and hit[0] is img:
        return hit[1], hit[2], hit[3]
    work, s = prepare(img, p)
    lab = bgr_to_lab(work)
    if p.denoise > 0:
        I = lab[:, :, 0] / 100.0
        r = max(2, int(round(3 * s)))
        for ch in (1, 2):
            lab[:, :, ch] = guided_filter(I, lab[:, :, ch], r, 2e-3)
    if len(_prep_cache) > 6:
        _prep_cache.clear()
    _prep_cache[key] = (img, work, s, lab)
    return work, s, lab


def segment(img: np.ndarray, p: Params) -> Segmentation:
    H, W = img.shape[:2]
    work, s, lab = _prepared(img, p)

    out_override = None
    if p.palette:
        pal = np.array([hex_to_lab(e["match"]) for e in p.palette], np.float32)
        out_override = [e.get("out") or e["match"] for e in p.palette]
    else:
        pal = build_palette(lab, p, min_px=max(30.0, 0.5 * p.min_area * s * s))

    labels = assign(lab, pal)
    labels = clean_labels(labels, len(pal), s, p)
    if not p.palette:
        labels, dropped = drop_transitional(labels, pal, s, p)
        if dropped:
            labels = clean_labels(labels, len(pal), s, p)

    # 用区域内部（腐蚀后）像素的中位数作为真实颜色，避开边缘混色
    n = len(pal)
    used = np.bincount(labels.ravel(), minlength=n)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    colors, match = [], []
    for c in range(n):
        if used[c] == 0:
            colors.append(None)
            match.append(None)
            continue
        m = (labels == c).astype(np.uint8)
        inner = cv2.erode(m, ker) > 0
        if inner.sum() < 20:
            inner = m > 0
        med = np.median(work[inner], axis=0)
        hx = bgr_to_hex(med)
        match.append(p.palette[c]["match"] if p.palette else hx)
        colors.append(out_override[c] if out_override else hx)

    # 压缩掉未使用的标签
    remap = -np.ones(n, np.int32)
    new_colors, new_match, shares = [], [], []
    total = labels.size
    for c in np.argsort(-used):
        if used[c] == 0:
            continue
        remap[c] = len(new_colors)
        new_colors.append(colors[c])
        new_match.append(match[c])
        shares.append(float(used[c]) / total)
    labels = remap[labels]
    return Segmentation(labels, new_colors, new_match, shares, s, (W, H), work)
