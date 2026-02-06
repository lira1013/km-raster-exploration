from flask import Flask, request, jsonify
from flask.json.provider import DefaultJSONProvider
import cv2
import numpy as np
import base64
import os
import logging
import requests
from functools import wraps
from skimage.morphology import thin
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
import copy
import time
from collections import defaultdict
import pandas as pd

# ------------------ 配置与初始化 ------------------
class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, obj):
        if isinstance(obj, (np.generic, np.ndarray)):
            return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
        return super().default(obj)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)
app.config['API_KEY'] = os.getenv('API_KEY', 'km_service_2026_secure_key')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB限制

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger()
logging.getLogger('werkzeug').setLevel(logging.INFO)

# 核心配置（严格按要求调整，剔除错误参数）
CONFIG = {
    # Step1: 安全裁剪参数
    "PAD": 10,  # bbox扩边像素
    # Step2: 灰度增强参数
    "MIN_LINE_AREA": 5,           # 仅保留面积过滤
    "CLAHE_CLIP_LIMIT": 2.0,      # 保守CLAHE参数（≤2.0）
    "ADAPTIVE_THRESH_CONST": 5,   # 灰度自适应阈值常数项
    "HSV_V_THRESH": 180,          # HSV V通道最低亮度阈值
    # Step3: 拓扑保留参数
    "DILATE_KERNEL_SIZE": 2,      # 轻量膨胀核
    "CONNECTIVITY": 8,            # 连通域分析连接方式
    # Step4: 起始区合并参数
    "START_REGION_RATIO": 0.1,    # 起始区占宽度比例（10%）
    "MERGE_DISTANCE": 2,          # 起始区合并像素距离
    "START_REGION_SAMPLE_POINTS": 50,  # 新增：起始区采样点数
    # Step5: 曲线提取核心参数
    "MIN_HORIZONTAL_LENGTH": 6,   # 水平段最小长度
    "MIN_VERTICAL_LENGTH": 2,
    "H_MIN": 3,                   # 竖线去噪最小高度
    "RUN_BATCH_SIZE": 200,        # 新增：Run分批处理大小
    # Step7: 兜底重试参数
    "FEEDBACK_RETRY_TIMES": 1,    # 最多1次兜底重试
    "ADAPTIVE_THRESH_CONST_RETRY": 3,  # 重试时的阈值常数项
    # Sanity Check参数
    "SANITY_CHECK_MIN_PIXEL": 20, # 起始区最小像素数
    "SANITY_CHECK_MIN_WIDTH": 3,  # 起始区最小曲线宽度
    # 其他保留核心参数
    "INVALID_SURVIVAL_THRESH": 0.05,
    "Y_MATCH_TOLERANCE": 8.0,
    "N_MAX": 50,
    "EPS_ERROR_BASE": 0.05,
    "EPS_ERROR_SCALE": 0.002,
    "MIN_N0": 3,
    "MAX_EVENT_PROCESS": 30,
    "REQUEST_TIMEOUT": 10,
    "REQUEST_RETRIES": 2,
    "MAX_PROCESS_TIME": 15,
}

# ------------------ 核心新增：CoordinateMapper类（Step1核心） ------------------
class CoordinateMapper:
    def __init__(self, offset_x, offset_y, x_mapping, y_mapping):
        self.ox = offset_x  # 裁剪区域左上角全局X坐标
        self.oy = offset_y  # 裁剪区域左上角全局Y坐标
        self.x_k = x_mapping["k"]  # X轴像素→时间斜率
        self.x_b = x_mapping["b"]  # X轴截距
        self.y_k = y_mapping["k"]  # Y轴像素→生存率斜率
        self.y_b = y_mapping["b"]  # Y轴截距
    
    def local_to_global(self, x, y):
        """局部像素坐标→全局像素坐标"""
        return x + self.ox, y + self.oy
    
    def global_to_local(self, x, y):
        """全局像素坐标→局部像素坐标"""
        return x - self.ox, y - self.oy
    
    def local_to_math(self, x, y):
        """局部像素坐标→数学坐标（时间/生存率）"""
        time = self.x_k * x + self.x_b
        survival = self.y_k * y + self.y_b
        return {"time": round(time,4), "survival": round(survival,4)}

# ------------------ 核心新增：AdaptiveEnhancer类（Step2简化版） ------------------
class AdaptiveEnhancer:
    def __init__(self, config):
        self.config = config
    
    def enhance_gray(self, gray_img):
        """仅灰度图轻量化增强"""
        # CLAHE增强（保守参数）
        clahe = cv2.createCLAHE(
            clipLimit=self.config["CLAHE_CLIP_LIMIT"], 
            tileGridSize=(8,8)  # 固定8×8，不动态调整
        )
        enhanced = clahe.apply(gray_img)
        # 自适应二值化
        bw = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            self.config["ADAPTIVE_THRESH_CONST"]
        )
        return bw
    
    def enhance_gray_retry(self, gray_img):
        """重试时降低阈值常数项"""
        clahe = cv2.createCLAHE(
            clipLimit=self.config["CLAHE_CLIP_LIMIT"], 
            tileGridSize=(8,8)
        )
        enhanced = clahe.apply(gray_img)
        bw = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            self.config["ADAPTIVE_THRESH_CONST_RETRY"]  # 从5→3
        )
        return bw

# ------------------ 数据结构定义 ------------------
@dataclass
class VerticalRun:
    x: int
    y_start: int
    y_end: int
    height: int
    status: str = "unknown"
    curve_id: Optional[int] = None

@dataclass
class CurveBundle:
    curves: List["KMCurve"]
    x_range: Tuple[int, int]
    first_split_x: int

@dataclass
class KMState:
    n: int
    s_recon: float
    error: float
    path: List[Dict]
    valid: bool = True

@dataclass
class KMCurve:
    curve_id: int
    segments: List[Dict]
    min_x: int
    max_x: int
    y_values: List[float]
    total_length: float
    pixel_mask: np.ndarray
    all_x: List[int]
    all_y: List[int]
    group_name: str = ""
    last_known_y: Optional[float] = None

# ------------------ 核心工具函数 ------------------
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key != app.config['API_KEY']:
            logger.warning(f"Unauthorized access from {request.remote_addr}")
            return jsonify({"status": "failed", "error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def check_process_time(start_time):
    if time.time() - start_time > CONFIG["MAX_PROCESS_TIME"]:
        raise TimeoutError(f"Processing time exceeded {CONFIG['MAX_PROCESS_TIME']} seconds")

def download_image_from_url(url):
    """从URL下载图片"""
    session = requests.Session()
    retry = requests.adapters.HTTPAdapter(max_retries=CONFIG["REQUEST_RETRIES"])
    session.mount('http://', retry)
    session.mount('https://', retry)
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0 Safari/537.36'}
    try:
        response = session.get(url, headers=headers, timeout=CONFIG["REQUEST_TIMEOUT"], verify=False)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')
    except Exception as e:
        logger.error(f"Download image failed: {str(e)}")
        raise

def load_image_from_input(data, start_time):
    """加载图片（支持URL/Base64）"""
    check_process_time(start_time)
    img_base64 = None
    if 'image_url' in data and data['image_url']:
        logger.info(f"Loading image from URL: {data['image_url'][:50]}...")
        img_base64 = download_image_from_url(data['image_url'])
    elif 'image_base64' in data and data['image_base64']:
        logger.info("Loading image from Base64")
        img_base64 = data['image_base64'].split(',')[-1]
    if not img_base64:
        raise ValueError("No valid image: image_url or image_base64 required")
    
    img_bytes = base64.b64decode(img_base64)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed (invalid Base64/URL)")
    logger.info(f"Image loaded successfully, actual pixel size: {img.shape[1]}x{img.shape[0]}")
    return img

def safe_thin(bw01, max_iter=2):
    """安全骨架化"""
    try:
        skel = thin(bw01, max_iter=max_iter)
    except TypeError:
        try:
            skel = thin(bw01, max_num_iter=max_iter)
        except TypeError:
            skel = thin(bw01)
    return skel

def get_bbox(pos):
    """获取边界框"""
    if len(pos) < 4:
        return 0, 0, 0, 0
    xs = pos[::2]
    ys = pos[1::2]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

# ------------------ Step1: 安全裁剪 + 全局坐标统一 ------------------
def safe_crop_image(img, plot_area, config):
    """
    安全裁剪：基于plot_area生成bbox并扩边，执行硬裁剪，返回裁剪图和偏移量
    """
    # 生成目标区域bbox
    x_min, y_min, x_max, y_max = plot_area["x_min"], plot_area["y_min"], plot_area["x_max"], plot_area["y_max"]
    
    # 扩边处理
    pad = config["PAD"]
    x_min = max(0, x_min - pad)
    y_min = max(0, y_min - pad)
    x_max = min(img.shape[1], x_max + pad)
    y_max = min(img.shape[0], y_max + pad)
    
    # 硬裁剪（仅坐标平移，非重采样）
    cropped_img = img[y_min:y_max, x_min:x_max].copy()
    
    # 保存裁剪偏移量
    offset_x = x_min
    offset_y = y_min
    
    logger.info(f"Safe crop completed - ROI: {x_max-x_min}x{y_max-y_min}, offset: ({offset_x}, {offset_y})")
    return cropped_img, offset_x, offset_y

# ------------------ Step2: 灰度优先的轻量化增强 ------------------
def gray_based_enhancement(img, enhancer, config, retry=False):
    """
    灰度优先的图像增强：
    1. 灰度域二值化（核心）
    2. HSV V通道辅助掩码
    3. 轻量化CLAHE增强
    """
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE增强（仅用于骨架化）
    clahe = cv2.createCLAHE(
        clipLimit=config["CLAHE_CLIP_LIMIT"],
        tileGridSize=(8,8)
    )
    clahe_gray = clahe.apply(gray)
    
    # 自适应阈值二值化
    if retry:
        bw = enhancer.enhance_gray_retry(gray)
    else:
        bw = enhancer.enhance_gray(gray)
    
    # Otsu阈值作为兜底（备用）
    _, otsu_bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # HSV V通道辅助掩码
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    v_mask = (v_channel > config["HSV_V_THRESH"]).astype(np.uint8) * 255
    
    # 最终二值图 = 灰度自适应阈值结果 ∩ HSV V通道掩码
    final_bw = cv2.bitwise_and(bw, v_mask)
    
    return final_bw, clahe_gray

# ------------------ Step3: 拓扑保留的连通域提取 ------------------
def topology_preserved_extraction(bw_img, config):
    """
    拓扑保留的连通域提取：
    1. 骨架化
    2. 轻量膨胀
    3. 连通域分析（剔除Canny/形态学闭运算）
    """
    # 骨架化
    skel = safe_thin((bw_img > 0).astype(np.uint8), max_iter=2)
    skel_uint8 = (skel * 255).astype(np.uint8)
    
    # 轻量膨胀（修复轻微断裂）
    kernel_size = config["DILATE_KERNEL_SIZE"]
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_skel = cv2.dilate(skel_uint8, kernel, iterations=1)
    
    # 连通域提取
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        dilated_skel, 
        connectivity=config["CONNECTIVITY"]
    )
    
    # 过滤小面积噪声（仅保留面积过滤）
    valid_labels = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= config["MIN_LINE_AREA"]:
            valid_labels.append(label)
    
    logger.info(f"Topology extraction completed - total labels: {num_labels}, valid labels: {len(valid_labels)}")
    return dilated_skel, labels, stats, valid_labels

# ------------------ Step4: 起始区一致性合并（优化版） ------------------
def merge_start_region_curves(labels, stats, valid_labels, config, img_width):
    """
    起始区一致性合并（优化版）：
    1. 向量化距离计算，移除开方运算
    2. 限制合并闭包关系，按min_x排序保证稳定性
    3. 仅采样50个点，降低计算量
    """
    start_time = time.time()
    # 确定起始区范围
    first_split_x = int(img_width * config["START_REGION_RATIO"])
    start_region_mask = np.zeros(labels.shape, dtype=np.bool_)
    start_region_mask[:, :first_split_x] = True
    
    # 预生成：label→起始区像素坐标（仅存x,y，采样50个点）
    label_coords = {}
    for label in valid_labels:
        # 只提取起始区内的像素
        coords = np.where((labels == label) & start_region_mask)
        if len(coords[0]) > 0:
            # 转置为(N,2)的坐标矩阵，采样前50个点
            points = np.vstack([coords[1], coords[0]]).T[:config["START_REGION_SAMPLE_POINTS"]]
            label_coords[label] = points
        else:
            label_coords[label] = np.array([])
    
    # 按label的min_x排序，保证合并顺序稳定
    def get_label_min_x(label):
        coords = np.where(labels == label)
        return np.min(coords[1]) if len(coords[1]) > 0 else 0
    sorted_labels = sorted(valid_labels, key=get_label_min_x)
    
    # 合并逻辑：使用并查集，仅直接比较，不级联扩展
    parent = {label: label for label in sorted_labels}
    
    def find(u):
        """并查集查找（带路径压缩）"""
        if parent[u] != u:
            parent[u] = find(parent[u])
        return parent[u]
    
    def union(u, v):
        """并查集合并"""
        u_root = find(u)
        v_root = find(v)
        if u_root != v_root:
            parent[v_root] = u_root
    
    # 批量计算距离（向量化，比较距离平方）
    merge_distance_sq = config["MERGE_DISTANCE"] **2  # 距离平方，移除开方
    label_list = sorted_labels
    
    for i in range(len(label_list)):
        label1 = label_list[i]
        coords1 = label_coords[label1]
        if len(coords1) == 0:
            continue
        
        # 只和后续label比较（避免重复计算）
        for j in range(i+1, len(label_list)):
            label2 = label_list[j]
            coords2 = label_coords[label2]
            if len(coords2) == 0:
                continue
            
            # 向量化计算最小距离平方
            diff = coords1[:, np.newaxis, :] - coords2[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            min_dist_sq = np.min(dist_sq)
            
            # 仅当直接距离满足时合并，不级联
            if min_dist_sq <= merge_distance_sq:
                union(label1, label2)
    
    # 生成最终的合并映射
    merged_label_map = {}
    curve_id = 0
    root_to_cid = {}
    
    for label in sorted_labels:
        root = find(label)
        if root not in root_to_cid:
            root_to_cid[root] = curve_id
            curve_id += 1
        merged_label_map[label] = root_to_cid[root]
    
    # 生成合并后的曲线掩码
    merged_masks = {}
    for curve_id in set(merged_label_map.values()):
        curve_labels = [l for l, cid in merged_label_map.items() if cid == curve_id]
        mask = np.zeros(labels.shape, dtype=np.uint8)
        for l in curve_labels:
            mask[labels == l] = 1
        merged_masks[curve_id] = mask
    
    merge_time = round(time.time() - start_time, 2)
    logger.info(f"Start region merge completed - original labels: {len(valid_labels)}, merged curves: {len(merged_masks)}, time: {merge_time}s")
    return merged_masks, first_split_x, merged_label_map

# ------------------ 新增：Step4后Sanity Check ------------------
def sanity_check_start_region_curves(merged_masks, merged_label_map, labels, first_split_x, config):
    """
    Step4后只读式一致性校验：
    1. 检查起始区像素数
    2. 检查曲线宽度
    3. 检查Run关联（这里先检查曲线基础属性）
    """
    start_region_mask = np.zeros(labels.shape, dtype=np.bool_)
    start_region_mask[:, :first_split_x] = True
    
    sanity_results = {
        "low_pixel_curves": [],
        "narrow_curves": [],
        "empty_curves": []
    }
    
    for curve_id, mask in merged_masks.items():
        # 1. 起始区像素数校验
        start_pixels = np.sum(mask & start_region_mask)
        if start_pixels < config["SANITY_CHECK_MIN_PIXEL"]:
            sanity_results["low_pixel_curves"].append({
                "curve_id": curve_id,
                "start_pixel_count": int(start_pixels),
                "threshold": config["SANITY_CHECK_MIN_PIXEL"]
            })
        
        # 2. 曲线宽度校验
        if start_pixels > 0:
            coords = np.where(mask & start_region_mask)
            if len(coords[0]) > 0:
                x_coords = coords[1]
                width = np.max(x_coords) - np.min(x_coords) + 1
                if width < config["SANITY_CHECK_MIN_WIDTH"]:
                    sanity_results["narrow_curves"].append({
                        "curve_id": curve_id,
                        "width": int(width),
                        "threshold": config["SANITY_CHECK_MIN_WIDTH"]
                    })
        
        # 3. 空曲线校验
        if np.sum(mask) == 0:
            sanity_results["empty_curves"].append(curve_id)
    
    # 仅记录日志，不影响流程
    logger.info(f"Sanity Check Results - low pixel curves: {len(sanity_results['low_pixel_curves'])}, narrow curves: {len(sanity_results['narrow_curves'])}, empty curves: {len(sanity_results['empty_curves'])}")
    if sanity_results["low_pixel_curves"]:
        logger.warning(f"Low pixel curves: {[item['curve_id'] for item in sanity_results['low_pixel_curves']]}")
    if sanity_results["narrow_curves"]:
        logger.warning(f"Narrow curves: {[item['curve_id'] for item in sanity_results['narrow_curves']]}")
    return sanity_results

# ------------------ Step5: 拓扑优先的Run解析与事件判定（优化版） ------------------
def extract_and_classify_runs(skel_img, merged_masks, merged_label_map, first_split_x, config, labels):
    """
    Run解析优化版：
    1. 预生成curve_id映射矩阵，O(1)查找
    2. 预提取水平段，向量化检查
    3. 分批处理Run，降低循环复杂度
    """
    start_time = time.time()
    h, w = skel_img.shape[:2]
    
    # 优化1：预生成curve_id映射矩阵（直接通过坐标查curve_id）
    curve_id_map = np.full(labels.shape, -1, dtype=np.int32)
    for label in merged_label_map:
        curve_id = merged_label_map[label]
        curve_id_map[labels == label] = curve_id
    
    # 优化2：预提取所有曲线的水平段（按curve_id分组）
    curve_horizontal_y = defaultdict(set)
    row_has_horizontal = np.zeros(h, dtype=np.bool_)
    for curve_id, mask in merged_masks.items():
        for y in range(h):
            row = mask[y, :]
            if np.sum(row) >= config["MIN_HORIZONTAL_LENGTH"]:
                curve_horizontal_y[curve_id].add(y)
                row_has_horizontal[y] = True
    
    # 向量化提取竖段（原有逻辑保留）
    runs = []
    ys, xs = np.where(skel_img == 255)
    
    if len(xs) > 0:
        df = pd.DataFrame({'x': xs, 'y': ys})
        grouped = df.groupby('x')
        
        for x, group in grouped:
            y_coords = sorted(group['y'].values)
            if not y_coords:
                continue
            
            y_start = y_coords[0]
            for i in range(1, len(y_coords)):
                if y_coords[i] != y_coords[i-1] + 1:
                    height = y_coords[i-1] - y_start + 1
                    if height >= config["H_MIN"]:
                        runs.append(VerticalRun(
                            x=int(x),
                            y_start=int(y_start),
                            y_end=int(y_coords[i-1]),
                            height=height
                        ))
                    y_start = y_coords[i]
            
            height = y_coords[-1] - y_start + 1
            if height >= config["H_MIN"]:
                runs.append(VerticalRun(
                    x=int(x),
                    y_start=int(y_start),
                    y_end=int(y_coords[-1]),
                    height=height
                ))
    
    # 优化3：分批处理Run，O(1)查找curve_id
    processed_runs = []
    batch_size = config["RUN_BATCH_SIZE"]
    shared_run_map = defaultdict(list)  # 保存起始区shared run
    
    for batch_idx in range(0, len(runs), batch_size):
        batch_runs = runs[batch_idx:batch_idx+batch_size]
        
        for run in batch_runs:
            # 起始区run处理：保存不丢弃
            if run.x < first_split_x:
                run.status = "shared"
                # 找到关联的curve_id（基于y_mid）
                y_mid = (run.y_start + run.y_end) // 2
                if 0 <= y_mid < h and 0 <= run.x < w:
                    curve_id = curve_id_map[y_mid, run.x]
                    if curve_id != -1:
                        shared_run_map[curve_id].append(run)
                processed_runs.append(run)
                continue
            
            # O(1)查找curve_id
            y_mid = (run.y_start + run.y_end) // 2
            if y_mid < 0 or y_mid >= h or run.x < 0 or run.x >= w:
                run.status = "ambiguous"
                processed_runs.append(run)
                continue
            
            curve_id = curve_id_map[y_mid, run.x]
            if curve_id == -1:
                run.status = "ambiguous"
                processed_runs.append(run)
                continue
            
            run.curve_id = curve_id
            
            # 优化4：向量化检查上下水平段
            has_upper = False
            has_lower = False
            upper_y = run.y_start - 1
            lower_y = run.y_end + 1
            
            # 上端检查（边界保护）
            if 0 <= upper_y < h:
                # 检查当前曲线的水平段
                upper_ys = [y for y in [upper_y-1, upper_y, upper_y+1] if 0 <= y < h]
                has_upper = any(y in curve_horizontal_y[curve_id] for y in upper_ys)
                
                # 骨架兜底（向量化）
                if not has_upper:
                    x_min = max(0, run.x - 1)
                    x_max = min(w - 1, run.x + 1)
                    upper_slice = skel_img[upper_y:upper_y+1, x_min:x_max+1]
                    has_upper = np.any(upper_slice == 255)
            
            # 下端检查（边界保护）
            if 0 <= lower_y < h:
                lower_ys = [y for y in [lower_y-1, lower_y, lower_y+1] if 0 <= y < h]
                has_lower = any(y in curve_horizontal_y[curve_id] for y in lower_ys)
                
                # 骨架兜底（向量化）
                if not has_lower:
                    x_min = max(0, run.x - 1)
                    x_max = min(w - 1, run.x + 1)
                    lower_slice = skel_img[lower_y:lower_y+1, x_min:x_max+1]
                    has_lower = np.any(lower_slice == 255)
            
            # 分类判定
            if has_upper and has_lower:
                run.status = "event"
            elif has_upper or has_lower:
                run.status = "censor"
            else:
                run.status = "ambiguous"
            
            processed_runs.append(run)
    
    # 统计结果
    event_count = len([r for r in processed_runs if r.status == "event"])
    censor_count = len([r for r in processed_runs if r.status == "censor"])
    shared_count = len([r for r in processed_runs if r.status == "shared"])
    run_time = round(time.time() - start_time, 2)
    
    logger.info(f"Run extraction completed - total runs: {len(processed_runs)}, events: {event_count}, censors: {censor_count}, shared: {shared_count}, time: {run_time}s")
    return processed_runs, curve_horizontal_y, shared_run_map, h, w

# ------------------ 轴标定与坐标转换 ------------------
def calibrate_axis(anchors, offset_x, offset_y, start_time):
    """轴标定：基于局部坐标"""
    check_process_time(start_time)
    
    def extract_val(t):
        val = t.get("value_norm") if t.get("value_norm") is not None else t.get("value")
        return float(val) if val is not None else None

    try:
        axis_mapping = {}
        for ax in ["x", "y"]:
            ticks = []
            for a in anchors:
                if a.get("axis") == ax and a.get("role") == f"axis_tick_{ax}":
                    val = extract_val(a)
                    if val is not None:
                        # 转换为局部坐标
                        bbox = get_bbox(a["pos"])
                        px_x = (bbox[0] + bbox[2])/2 - offset_x
                        px_y = (bbox[1] + bbox[3])/2 - offset_y
                        px_coord = px_x if ax == "x" else px_y
                        ticks.append((px_coord, val))
            
            if len(ticks) < 2:
                raise ValueError(f"Insufficient {ax}-axis ticks (need ≥2, got {len(ticks)})")
            
            px_coords, vals = zip(*ticks[:8])
            k, b = np.polyfit(px_coords, vals, 1)
            
            if ax == "x" and k <= 0:
                raise ValueError(f"X-axis slope invalid (must >0, got {k:.4f})")
            if ax == "y" and k >= 0:
                raise ValueError(f"Y-axis slope invalid (must <0, got {k:.4f})")
            
            axis_mapping[ax] = {"k": float(k), "b": float(b)}
        
        logger.info(f"Axis calibration success - X(k={axis_mapping['x']['k']:.6f}), Y(k={axis_mapping['y']['k']:.6f})")
        return axis_mapping
    except Exception as e:
        logger.error(f"Axis calibration failed: {str(e)}")
        raise

# ------------------ KM风险集计算 ------------------
def compute_delta_i(events: List[Dict]) -> List[Dict]:
    for idx, event in enumerate(events):
        S_b, S_a = event["survival_before"], event["survival_after"]
        S_b, S_a = max(0.01, S_b), max(0.0, S_a)
        p_i = S_a / S_b
        delta_i = 1 - p_i
        event.update({"p_i": round(p_i, 6), "delta_i": round(delta_i, 6), "idx": idx})
    return events

def count_censors_between(censors: List[Dict], t_prev: float, t_curr: float) -> int:
    return len([c for c in censors if t_prev < c["time"] <= t_curr])

def dp_km_search_optimized(events: List[Dict], censors: List[Dict], start_time) -> Tuple[Optional[KMState], Dict]:
    diagnostics = {"valid": False, "min_N0": None, "total_events_processed": len(events), "processing_time": 0.0}
    if not events:
        return None, diagnostics
    
    events_with_delta = compute_delta_i(events)
    best_state, min_error = None, float('inf')
    
    for N0 in range(CONFIG["MIN_N0"], min(CONFIG["N_MAX"], 51)):
        check_process_time(start_time)
        eps_error = max(CONFIG["EPS_ERROR_BASE"], CONFIG["EPS_ERROR_SCALE"] * N0)
        current_states = [KMState(n=N0, s_recon=1.0, error=0.0, path=[], valid=True)]
        event_processed, t_prev = True, 0.0
        
        for event in events_with_delta[:20]:
            check_process_time(start_time)
            next_states = []
            t_curr, delta_i, S_target = event["time"], event["delta_i"], event["survival_after"]
            c_i = count_censors_between(censors, t_prev, t_curr)
            
            for state in current_states[:50]:
                if not state.valid:
                    continue
                
                n_curr = state.n
                ideal_d = n_curr * delta_i
                
                for d_i in range(max(1, round(ideal_d - 1)), min(n_curr, round(ideal_d + 1)) + 1):
                    actual_delta = d_i / n_curr
                    new_s = state.s_recon * (1 - actual_delta)
                    error = abs(new_s - S_target)
                    
                    if error > eps_error and event["idx"] < len(events_with_delta)-1:
                        continue
                    
                    new_n = max(1, n_curr - d_i - c_i)
                    new_path = state.path + [{
                        "time": t_curr, "n_i": n_curr, "d_i": d_i, "c_i": c_i,
                        "survival_before": event["survival_before"], "survival_after": event["survival_after"],
                        "actual_delta": actual_delta
                    }]
                    next_states.append(KMState(n=new_n, s_recon=new_s, error=state.error + error, path=new_path, valid=True))
            
            if not next_states:
                event_processed = False
                break
            
            current_states = sorted(next_states, key=lambda x: x.error)[:20]
            t_prev = t_curr
        
        if event_processed and current_states:
            for state in current_states:
                if state.error < min_error:
                    min_error = state.error
                    best_state = state
                    diagnostics["valid"] = True
                    diagnostics["min_N0"] = N0
        
        if best_state:
            break
    
    diagnostics["processing_time"] = time.time() - start_time
    return best_state, diagnostics

def compute_km_risk_table(events: List[Dict], censors: List[Dict], start_time) -> List[Dict]:
    if len(events) == 0:
        logger.info("No events found, skip KM risk table computation")
        return []
    
    try:
        best_state, _ = dp_km_search_optimized(events, censors, start_time)
        risk_table = []
        
        if best_state and best_state.path:
            for entry in best_state.path[:50]:
                risk_table.append({
                    "time": round(entry["time"], 4), "n_i": entry["n_i"], "d_i": entry["d_i"],
                    "c_i": entry["c_i"], "survival": round(entry["survival_after"], 4)
                })
        
        logger.info(f"Generated KM risk table with {len(risk_table)} rows")
        return risk_table
    except Exception as e:
        logger.error(f"Compute risk table failed: {str(e)}")
        return []

# ------------------ 核心API路由（集成优化） ------------------
@app.route("/digitize_km", methods=["POST"])
@require_api_key
def digitize_km():
    start_time = time.time()
    request_id = f"req_{int(time.time() * 1000)}"
    logger.info(f"=== Start processing request {request_id} ===")
    
    try:
        # 1. 基础校验
        if not request.is_json:
            raise ValueError("Request must be JSON format")
        data = request.get_json()
        
        required_fields = ["plot_area", "clean_anchors"]
        missing = [f for f in required_fields if f not in data]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")
        
        # 2. 加载图片
        img = load_image_from_input(data, start_time)
        
        # 3. Step1: 安全裁剪 + 全局坐标统一
        plot_area = data["plot_area"]
        cropped_img, offset_x, offset_y = safe_crop_image(img, plot_area, CONFIG)
        
        # 4. 初始化增强器
        enhancer = AdaptiveEnhancer(CONFIG)
        
        # 5. Step2: 灰度优先的轻量化增强
        bw_img, clahe_gray = gray_based_enhancement(cropped_img, enhancer, CONFIG)
        
        # 6. Step3: 拓扑保留的连通域提取
        skel_img, labels, stats, valid_labels = topology_preserved_extraction(bw_img, CONFIG)
        
        # 7. Step7: 极简反馈兜底
        retry = False
        if len(valid_labels) == 0 and CONFIG["FEEDBACK_RETRY_TIMES"] > 0:
            logger.info("No curves detected, retrying with lower threshold...")
            bw_img, clahe_gray = gray_based_enhancement(cropped_img, enhancer, CONFIG, retry=True)
            skel_img, labels, stats, valid_labels = topology_preserved_extraction(bw_img, CONFIG)
            retry = True
        
        # 8. Step4: 起始区一致性合并（优化版）
        merged_masks, first_split_x, merged_label_map = merge_start_region_curves(
            labels, stats, valid_labels, CONFIG, cropped_img.shape[1]
        )
        
        # 9. Step4后Sanity Check（只读，不影响流程）
        sanity_check_start_region_curves(merged_masks, merged_label_map, labels, first_split_x, CONFIG)
        
        # 10. Step5: 拓扑优先的Run解析与事件判定（优化版）
        vertical_runs, curve_horizontal_y, shared_run_map, h, w = extract_and_classify_runs(
            skel_img, merged_masks, merged_label_map, first_split_x, CONFIG, labels
        )
        
        # 11. 轴标定
        axis_mapping = calibrate_axis(data["clean_anchors"], offset_x, offset_y, start_time)
        
        # 12. 初始化CoordinateMapper
        coord_mapper = CoordinateMapper(
            offset_x=offset_x,
            offset_y=offset_y,
            x_mapping=axis_mapping["x"],
            y_mapping=axis_mapping["y"]
        )
        
        # 13. 转换为数学坐标（修复逻辑错误）
        curve_events = defaultdict(list)
        curve_censors = defaultdict(list)
        
        for run in vertical_runs:
            # 处理shared run，用于校准初始值
            if run.status == "shared" and run.curve_id is not None:
                # 记录shared run的坐标，用于校准初始survival
                math_coords = coord_mapper.local_to_math(run.x, (run.y_start + run.y_end)/2)
                shared_run_map[run.curve_id].append({
                    "time": math_coords["time"],
                    "survival": math_coords["survival"]
                })
                continue
            
            if run.curve_id is None:
                continue
            
            # 局部坐标转数学坐标
            math_coords = coord_mapper.local_to_math(run.x, (run.y_start + run.y_end)/2)
            
            if run.status == "event":
                # Event：边界保护，确保坐标有效
                upper_y = run.y_start - 1
                lower_y = run.y_end + 1
                
                # 边界校验
                upper_y = max(0, min(h-1, upper_y))
                lower_y = max(0, min(h-1, lower_y))
                
                upper_survival = coord_mapper.local_to_math(run.x, upper_y)["survival"]
                lower_survival = coord_mapper.local_to_math(run.x, lower_y)["survival"]
                
                # 确保survival_before ≥ survival_after
                survival_before = max(upper_survival, lower_survival)
                survival_after = min(upper_survival, lower_survival)
                
                curve_events[run.curve_id].append({
                    "time": math_coords["time"],
                    "survival_before": survival_before,
                    "survival_after": survival_after
                })
            
            elif run.status == "censor":
                # Censor：使用当前曲线的水平段，避免跨曲线错误
                censor_y = None
                upper_y_candidate = run.y_start - 1
                if 0 <= upper_y_candidate < h and upper_y_candidate in curve_horizontal_y[run.curve_id]:
                    censor_y = upper_y_candidate
                else:
                    lower_y_candidate = run.y_end + 1
                    if 0 <= lower_y_candidate < h and lower_y_candidate in curve_horizontal_y[run.curve_id]:
                        censor_y = lower_y_candidate
                
                # 兜底
                if censor_y is None:
                    censor_y = (run.y_start + run.y_end) // 2
                
                censor_survival = coord_mapper.local_to_math(run.x, censor_y)["survival"]
                
                curve_censors[run.curve_id].append({
                    "time": math_coords["time"],
                    "survival": censor_survival
                })
        
        # 14. 计算KM风险表
        final_groups = []
        for curve_id, mask in merged_masks.items():
            events = curve_events.get(curve_id, [])
            censors = curve_censors.get(curve_id, [])
            risk_table = compute_km_risk_table(events, censors, start_time)
            
            # 去重排序
            unique_events = {round(e["time"], 2): e for e in events}
            unique_censors = {round(c["time"], 2): c for c in censors}
            
            # 加入shared run统计
            shared_runs = shared_run_map.get(curve_id, [])
            
            final_groups.append({
                "group_name": f"Group_{curve_id + 1}",
                "curve_id": curve_id,
                "events": sorted(unique_events.values(), key=lambda x: x["time"]),
                "censors": sorted(unique_censors.values(), key=lambda x: x["time"]),
                "shared_runs": shared_runs,  # 保留shared run信息
                "risk_table": risk_table,
                "event_count": len(events),
                "censor_count": len(censors),
                "shared_count": len(shared_runs)
            })
        
        # 15. 构造返回结果
        processing_time = round(time.time() - start_time, 2)
        response = {
            "status": "success",
            "request_id": request_id,
            "data": {
                "curve_count": len(merged_masks),
                "groups": final_groups,
                "processing_time": processing_time,
                "stats": {
                    "total_events": sum([g["event_count"] for g in final_groups]),
                    "total_censors": sum([g["censor_count"] for g in final_groups]),
                    "total_shared_runs": sum([g["shared_count"] for g in final_groups]),
                    "plot_area": plot_area,
                    "crop_offset": {"x": offset_x, "y": offset_y},
                    "first_split_x": first_split_x,
                    "retry_performed": retry
                }
            }
        }
        
        logger.info(f"=== Request {request_id} processed successfully (time: {processing_time}s) ===")
        return jsonify(response), 200

    except TimeoutError as e:
        logger.error(f"=== Request {request_id} failed: Timeout ===")
        return jsonify({
            "status": "failed", "request_id": request_id,
            "data": {"curve_count": 0, "groups": [], "error": str(e), "processing_time": round(time.time()-start_time, 2)}
        }), 408
    except Exception as e:
        logger.error(f"=== Request {request_id} failed: {str(e)} ===", exc_info=True)
        return jsonify({
            "status": "failed", "request_id": request_id,
            "data": {"curve_count": 0, "groups": [], "error": str(e), "processing_time": round(time.time()-start_time, 2)}
        }), 500

# ------------------ 全局异常处理器 ------------------
@app.errorhandler(Exception)
def handle_all_exceptions(e):
    request_id = f"req_{int(time.time() * 1000)}"
    logger.error(f"=== Unhandled exception {request_id}: {str(e)} ===", exc_info=True)
    return jsonify({
        "status": "failed", "request_id": request_id,
        "data": {"curve_count": 0, "groups": [], "error": f"Server error: {str(e)}"}
    }), 500

# ------------------ 启动入口 ------------------
if __name__ == "__main__":
    logger.info(f"KM Digitize Service Starting - Optimized Version")
    app.run(host='0.0.0.0', port=8888, debug=False, use_reloader=False)