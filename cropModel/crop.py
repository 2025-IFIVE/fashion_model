from __future__ import annotations
import os
import numpy as np
import cv2  # 메인에서 import 경로 유지 필요
from PIL import Image
import torch
from transformers import pipeline, CLIPSegProcessor, CLIPSegForImageSegmentation

# rembg를 메인에서 import하므로 여기서도 re-export
import rembg  # type: ignore

# =========================
# ▶ 환경/튜닝 파라미터 (필요시 ENV로 조절)
# =========================
MAX_IMAGE_SIZE      = int(os.getenv("CROP_MAX_IMAGE_SIZE", "2048"))  # 내부 처리 해상도 상한
TOP_ONLY            = os.getenv("CROP_TOP_ONLY", "false").lower() == "true"
BOTTOM_ONLY         = os.getenv("CROP_BOTTOM_ONLY", "false").lower() == "true"

MASK_THRESHOLD      = float(os.getenv("CROP_MASK_THRESHOLD", "0.35"))
CLOTH_BOOST         = float(os.getenv("CROP_CLOTH_BOOST", "1.3"))
NEGATIVE_PENALTY    = float(os.getenv("CROP_NEG_PENALTY", "0.6"))  # print2.py 기본값과 동일

SMOOTH_KERNEL       = int(os.getenv("CROP_SMOOTH_KERNEL", "3"))
FEATHER_SIGMA       = float(os.getenv("CROP_FEATHER_SIGMA", "0.8"))
RMBG_WEIGHT         = float(os.getenv("CROP_RMBG_WEIGHT", "0.8"))

ENABLE_MORPH_FIX    = os.getenv("CROP_ENABLE_MORPH", "true").lower() == "true"
ENABLE_EDGE_REFINE  = os.getenv("CROP_ENABLE_EDGE_REFINE", "true").lower() == "true"
MIN_AREA_RATIO      = float(os.getenv("CROP_MIN_AREA_RATIO", "0.001"))

ONEPIECE_OVERLAP_THR   = float(os.getenv("CROP_ONEPIECE_IOU", "0.15"))
ONEPIECE_CROSSRATE_THR = float(os.getenv("CROP_ONEPIECE_CROSS", "0.25"))
ONEPIECE_NEAR_BAND     = float(os.getenv("CROP_ONEPIECE_NEAR", "0.08"))

SKIN_STRONG_GAMMA   = float(os.getenv("CROP_SKIN_GAMMA", "1.2"))
SKIN_STRONG_THRESH  = float(os.getenv("CROP_SKIN_THRESH", "0.40"))
SKIN_POST_KERNEL    = int(os.getenv("CROP_SKIN_POST_K", "5"))

# 프롬프트들
CLOTH_PROMPTS = [
    "clothing", "clothes", "garment", "fabric", "textile",
    "shirt", "t-shirt", "blouse", "top", "upper body clothing",
    "pants", "trousers", "jeans", "bottom", "lower body clothing",
    "dress", "skirt", "shorts",
    "jacket", "coat", "blazer", "cardigan", "sweater", "hoodie",
    "sleeve", "collar", "button", "zipper", "pocket"
]
TOP_PROMPTS = [
    "top clothing", "upper body clothing",
    "shirt", "t-shirt", "blouse", "sweater", "hoodie", "cardigan",
    "jacket", "coat", "blazer", "vest"
]
BOTTOM_PROMPTS = [
    "bottom clothing", "lower body clothing",
    "pants", "trousers", "jeans", "slacks",
    "shorts", "denim shorts", "hotpants",
    "skirt", "mini skirt"
]
DRESS_PROMPTS = [
    "dress", "one-piece dress", "long dress", "midi dress", "gown",
    "casual dress", "summer dress", "knit dress", "jumpsuit", "romper"
]

# 🔹 프린팅(그래픽/로고/텍스트) 보호용 프롬프트 (print2.py 반영)
PRINT_PROMPTS = [
    "graphic print", "illustration", "cartoon", "anime", "comic",
    "logo", "text print", "graphic tee"
]

# 🔹 부정 프롬프트를 real human으로 제한(프린팅을 피부로 오검출 방지)
NEGATIVE_PROMPTS = [
    "real human skin", "naked human skin",
    "real human face", "ear", "neck", "neck skin",
    "arm", "forearm", "elbow", "hand", "finger", "wrist",
    "leg", "thigh", "knee", "calf", "ankle", "foot", "toe",
    "belly", "abdomen", "stomach", "midriff", "collarbone skin", "clavicle skin",
    "body parts without clothing"
]

# =========================
# ▶ 전역(지연 로딩) 리소스
# =========================
_device_idx = 0 if torch.cuda.is_available() else -1
_clipseg_proc: CLIPSegProcessor | None = None
_clipseg_model: CLIPSegForImageSegmentation | None = None
_rmbg_pipe = None  # print2.py처럼 항상 사용 (GPU/CPU 무관)

def _lazy_load_models():
    global _clipseg_proc, _clipseg_model, _rmbg_pipe
    if _clipseg_proc is None or _clipseg_model is None:
        _clipseg_proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
        _clipseg_model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
        _clipseg_model = _clipseg_model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    # ✅ GPU/CPU와 상관없이 RMBG 로드 (print2.py와 동일한 동작 보장)
    if _rmbg_pipe is None:
        try:
            _rmbg_pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4",
                                  trust_remote_code=True, device=_device_idx)
        except Exception:
            _rmbg_pipe = None  # 네트워크/환경 이슈 시 graceful fallback

# =========================
# ▶ 유틸
# =========================
def _resize_if_large(img: np.ndarray, max_size: int = MAX_IMAGE_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_size:
        return img
    scale = max_size / float(max(h, w))
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

def _to_rgb_pil(image: np.ndarray) -> Image.Image:
    """ BGR/RGBA/GRAY 모두 안전하게 PIL RGB로 변환 """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # 먼저 BGR
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _alpha_from_rgba(image: np.ndarray) -> np.ndarray | None:
    if image.ndim == 3 and image.shape[2] == 4:
        a = image[:, :, 3].astype(np.float32) / 255.0
        return np.clip(a, 0, 1)
    return None

def _postprocess_mask01(mask01: np.ndarray, kernel: int) -> np.ndarray:
    m = (mask01 * 255).astype(np.uint8)
    if ENABLE_MORPH_FIX and kernel and kernel >= 3 and kernel % 2 == 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)
        if ENABLE_EDGE_REFINE:
            m = cv2.medianBlur(m, kernel)
    return m.astype(np.float32) / 255.0

def _remove_small(mask01: np.ndarray, min_area_ratio: float) -> np.ndarray:
    if min_area_ratio <= 0: return mask01
    m = (mask01 * 255).astype(np.uint8)
    H, W = m.shape
    min_area = int(H * W * min_area_ratio)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            m[lab == i] = 0
    return m.astype(np.float32) / 255.0

def _close_small_holes(mask01: np.ndarray, ksize: int = 5) -> np.ndarray:
    """프린팅 내부의 작은 구멍(핀홀) 메움 — print2.py 반영"""
    m = (mask01 * 255).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m.astype(np.float32) / 255.0

def _feather_alpha(rgba: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0: return rgba
    a = rgba[:, :, 3].astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = rgba.copy()
    out[:, :, 3] = np.clip(a, 0, 255).astype(np.uint8)
    return out

def _compose_rgba(image_bgr: np.ndarray, mask01: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    a = (np.clip(mask01, 0, 1) * 255).astype(np.uint8)
    rgba = np.dstack([rgb, a])
    # OpenCV 관례 맞춰 BGRA로 바꿔 저장하고 싶다면 아래 한 줄 사용
    rgba_bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    return rgba_bgra

def _predict_clipseg( proc, model, image_pil, prompts, boost_factor=1.0 ):
    if not prompts:
        return []
    inputs = proc(text=prompts, images=[image_pil] * len(prompts),
                  padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        probs = torch.sigmoid(outputs.logits).squeeze(1).cpu().numpy()

    W, H = image_pil.size
    out = []
    for m in probs:
        resized = cv2.resize(m, (W, H), interpolation=cv2.INTER_CUBIC)
        if boost_factor != 1.0:
            resized = np.power(resized, 1.0 / boost_factor)
        out.append(resized.astype(np.float32))
    return out

def _adaptive_threshold(mask: np.ndarray, base: float) -> np.ndarray:
    hist, _ = np.histogram(mask.flatten(), 50, [0, 1])
    total = mask.size
    sum_total = np.sum(np.arange(len(hist)) * hist)
    best, max_var = base, 0.0
    for t in np.linspace(0.1, 0.9, 20):
        idx = int(t * len(hist))
        w0 = np.sum(hist[:idx]); w1 = total - w0
        if w0 == 0 or w1 == 0: continue
        sum0 = np.sum(np.arange(idx) * hist[:idx])
        mu0 = sum0 / w0; mu1 = (sum_total - sum0) / w1
        vb = w0 * w1 * (mu0 - mu1) ** 2
        if vb > max_var:
            max_var, best = vb, t
    thr = 0.7 * base + 0.3 * best
    return (mask >= thr).astype(np.float32)

def _binarize(soft: np.ndarray, base: float = 0.33) -> np.ndarray:
    return _adaptive_threshold(np.clip(soft, 0, 1).astype(np.float32), base)

def _smooth1d(x: np.ndarray, k: int = 9) -> np.ndarray:
    k = max(3, k | 1)
    return cv2.GaussianBlur(x.astype(np.float32), (k,1), 0, borderType=cv2.BORDER_REPLICATE)

def _find_boundary(top_map: np.ndarray, bot_map: np.ndarray):
    H, _ = top_map.shape
    top_row = _smooth1d(top_map.sum(axis=1, keepdims=True), k=max(7, H//80)).ravel()
    bot_row = _smooth1d(bot_map.sum(axis=1, keepdims=True), k=max(7, H//80)).ravel()
    d = top_row - bot_row
    d_s = _smooth1d(d.reshape(-1,1), k=max(7, H//80)).ravel()

    zeros = []
    for y in range(1, H):
        if (d_s[y-1] <= 0 and d_s[y] > 0) or (d_s[y-1] >= 0 and d_s[y] < 0):
            zeros.append(y)
    if not zeros:
        y_star = int(np.argmax(np.abs(np.gradient(d_s))))
        conf = 0.3
        return y_star, conf
    grad = np.abs(np.gradient(d_s))
    y_star = max(zeros, key=lambda y: grad[y])
    conf = float(grad[y_star] / (np.max(np.abs(d_s)) + 1e-6))
    conf = np.clip(conf, 0.0, 1.0)
    return int(y_star), conf

def _gate(H: int, y_star: int, band: float = 0.06, invert: bool = False) -> np.ndarray:
    y = np.arange(H, dtype=np.float32)
    sigma = max(3.0, H * band)
    z = (y - y_star) / sigma
    gate_top = 1.0 / (1.0 + np.exp(+z))
    gate_bot = 1.0 / (1.0 + np.exp(-z))
    return (gate_bot if invert else gate_top)[:, None]

def _decide_focus_by_area(top_map: np.ndarray, bot_map: np.ndarray) -> str:
    area_top = int((_binarize(top_map, 0.33) > 0).sum())
    area_bot = int((_binarize(bot_map, 0.33) > 0).sum())
    return 'top' if area_top >= area_bot else 'bottom'

def _is_onepiece(top_map: np.ndarray, bot_map: np.ndarray, y_star: int) -> bool:
    H, _ = top_map.shape
    top_bin = _binarize(top_map, 0.33)
    bot_bin = _binarize(bot_map, 0.33)
    inter = np.logical_and(top_bin>0, bot_bin>0).sum()
    uni   = np.logical_or (top_bin>0, bot_bin>0).sum()
    iou = (inter / max(uni, 1)) if uni else 0.0

    band = max(4, int(H * ONEPIECE_NEAR_BAND))
    y0, y1 = max(0, y_star-band), min(H, y_star+band)
    band_top = top_map[y0:y1, :]
    band_bot = bot_map[y0:y1, :]
    both = np.logical_and(band_top >= band_top.mean(), band_bot >= band_bot.mean()).sum()
    total = band_top.size
    cross_rate = both / max(total, 1)
    return (iou >= ONEPIECE_OVERLAP_THR) and (cross_rate >= ONEPIECE_CROSSRATE_THR)

def _keep_component_crossing(mask01: np.ndarray, y_star: int, min_area_ratio=0.002) -> np.ndarray:
    H, W = mask01.shape[:2]
    area_min = int(H * W * min_area_ratio)
    m = (mask01 * 255).astype(np.uint8)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    best_idx, best_area = None, 0
    for i in range(1, num):
        x,y,w,h,area = stats[i]
        if area < area_min: continue
        touch = (y <= y_star <= (y+h-1))
        has_up   = (m[:y_star, :][lab[:y_star,:]==i] > 0).any()
        has_down = (m[y_star:, :][lab[y_star:,:]==i] > 0).any()
        if touch and has_up and has_down and area > best_area:
            best_area, best_idx = area, i
    keep = np.zeros_like(m)
    if best_idx is not None:
        keep[lab == best_idx] = 255
    elif num > 1:
        idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        keep[lab == idx] = 255
    return keep.astype(np.float32) / 255.0

def _refine_with_crf_stub(image_rgb: np.ndarray, prob_fg: np.ndarray, iters: int = 5) -> np.ndarray:
    # print2.py에서도 ENABLE_CRF=False라서 동일 동작
    return prob_fg

# =========================
# ▶ 외부 공개 함수 (FastAPI 메인과 호환)
# =========================
def apply_grabcut(image: np.ndarray) -> np.ndarray:
    _lazy_load_models()

    # 1) 해상도 제한 & 색공간 정리 (원본 보존용)
    img_in = image
    img_in = _resize_if_large(img_in, MAX_IMAGE_SIZE)
    img_bgr = _ensure_bgr(img_in)  # 내부 통일: BGR

    # 2) RMBG 알파 가져오기 (print2.py처럼 항상 시도)
    a01 = _alpha_from_rgba(img_in)
    if a01 is None and _rmbg_pipe is not None:
        pil_rgba = _rmbg_pipe(_to_rgb_pil(img_bgr))
        a01 = _to01_from_pil_rgba(pil_rgba)
    if a01 is None:  # 안전장치
        a01 = np.ones(img_bgr.shape[:2], np.float32)
    a01 = np.clip(a01 * RMBG_WEIGHT, 0, 1)

    # 3) CLIPSeg 맵
    pil_rgb = _to_rgb_pil(img_bgr)
    top_maps = _predict_clipseg(_clipseg_proc, _clipseg_model, pil_rgb, TOP_PROMPTS, boost_factor=CLOTH_BOOST)
    bot_maps = _predict_clipseg(_clipseg_proc, _clipseg_model, pil_rgb, BOTTOM_PROMPTS, boost_factor=CLOTH_BOOST)
    top_map = np.maximum.reduce(top_maps) if top_maps else np.zeros(pil_rgb.size[::-1], np.float32)
    bot_map = np.maximum.reduce(bot_maps) if bot_maps else np.zeros_like(top_map)

    # 부정(피부/신체) + 프린팅 맵
    neg_masks = _predict_clipseg(_clipseg_proc, _clipseg_model, pil_rgb, NEGATIVE_PROMPTS)
    neg_map = np.maximum.reduce(neg_masks) if neg_masks else np.zeros_like(top_map)

    print_maps = _predict_clipseg(_clipseg_proc, _clipseg_model, pil_rgb, PRINT_PROMPTS, boost_factor=1.2)
    print_map  = np.maximum.reduce(print_maps) if print_maps else np.zeros_like(top_map)

    # 프린팅이 강할수록 부정맵 영향 약화(0.3~1.0 게이트) — print2.py 동일식
    neg_gate = np.clip(1.0 - 0.7 * print_map, 0.3, 1.0)

    # 4) 경계/원피스 판정
    y_star, _ = _find_boundary(top_map, bot_map)
    is_one = _is_onepiece(top_map, bot_map, y_star)

    # 5) 타깃 결정(원피스/상의/하의)
    if is_one:
        dress_maps = _predict_clipseg(_clipseg_proc, _clipseg_model, pil_rgb, DRESS_PROMPTS, boost_factor=1.5)
        dress_map = np.maximum.reduce(dress_maps) if dress_maps else np.zeros_like(top_map)
        pos_soft = np.maximum.reduce([dress_map, top_map, bot_map])

        # 프린팅 보호 반영 (0.9 * neg_map * neg_gate)
        raw = a01 * (pos_soft * (1.0 - 0.9 * neg_map * neg_gate))
        raw = _refine_with_crf_stub(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                    np.clip(raw, 0, 1).astype(np.float32), iters=5)

        mask01 = _adaptive_threshold(np.clip(raw, 0, 1), max(0.32, MASK_THRESHOLD - 0.03))
        mask01 = _postprocess_mask01(mask01, SMOOTH_KERNEL)
        mask01 = _remove_small(mask01, max(0.0015, MIN_AREA_RATIO))
        mask01 = _keep_component_crossing(mask01, y_star, min_area_ratio=0.002)

        # 강한 피부만 제거 (회색화 방지)
        skin_strong = np.clip(np.power(neg_map, SKIN_STRONG_GAMMA), 0, 1)
        skin_bin1   = _adaptive_threshold(skin_strong, SKIN_STRONG_THRESH)
        skin_bin1   = _postprocess_mask01(skin_bin1, SKIN_POST_KERNEL)
        mask01      = np.clip(mask01 * (1.0 - skin_bin1), 0, 1)
        mask01      = _postprocess_mask01(mask01, SMOOTH_KERNEL)

        # 프린팅 내부 핀홀 닫기 — print2.py 동일
        mask01 = _close_small_holes(mask01, ksize=5)

    else:
        if TOP_ONLY and not BOTTOM_ONLY:
            target = 'top'
        elif BOTTOM_ONLY and not TOP_ONLY:
            target = 'bottom'
        else:
            target = _decide_focus_by_area(top_map, bot_map)

        H, _ = top_map.shape
        if target == 'top':
            gate = _gate(H, y_star, band=0.06, invert=False)
            pos_soft = top_map * gate
        else:
            gate = _gate(H, y_star, band=0.06, invert=True)
            pos_soft = bot_map * gate

        # 프린팅 보호 반영 (NEGATIVE_PENALTY * neg_map * neg_gate)
        raw = a01 * (pos_soft * (1.0 - NEGATIVE_PENALTY * neg_map * neg_gate))
        raw = _refine_with_crf_stub(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                                    np.clip(raw, 0, 1).astype(np.float32), iters=5)

        mask01 = _adaptive_threshold(np.clip(raw, 0, 1), MASK_THRESHOLD)
        mask01 = _postprocess_mask01(mask01, SMOOTH_KERNEL)
        mask01 = _remove_small(mask01, MIN_AREA_RATIO)

        # 프린팅 내부 핀홀 닫기 — print2.py 동일
        mask01 = _close_small_holes(mask01, ksize=5)

    # 6) 합성 (RGBA->BGRA) & feather
    rgba = _compose_rgba(img_bgr, mask01)
    rgba = _feather_alpha(rgba, FEATHER_SIGMA)
    return rgba


# =========================
# ▶ 내부 보조 (정의 순서상 마지막에)
# =========================
def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)  # 이미 BGRA일 가능성, 안전하게 BGRA2BGR
    # Heuristic: 대부분 OpenCV로 읽으면 BGR이므로 그대로 사용
    return image

def _to01_from_pil_rgba(pil_rgba: Image.Image) -> np.ndarray:
    arr = np.array(pil_rgba)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return np.clip(arr[:, :, 3].astype(np.float32) / 255.0, 0, 1)
    # 방어로직
    return np.ones(arr.shape[:2], np.float32)
