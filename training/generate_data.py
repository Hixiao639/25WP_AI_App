import os
import cv2
import numpy as np
import random
from tqdm import tqdm

# ================= 配置区域 =================
RAW_DATA_DIR = "./raw_data/extracted_images" 
OUTPUT_DIR = "./yolo_math_dataset"

CLASS_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, 'times': 12, 'div': 13,
    '(': 14, ')': 15
}

IMG_SIZE = 640       
NUM_SAMPLES = 2000   
SYMBOL_SIZE = 80     # 稍微调大一点，方便识别

# ===========================================

def create_structure():
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def load_symbol_paths():
    symbol_dict = {}
    if not os.path.exists(RAW_DATA_DIR): return None
    for folder in os.listdir(RAW_DATA_DIR):
        key = folder
        if folder == 'add': key = '+'
        elif folder == 'sub': key = '-'
        elif folder == 'mul': key = 'times'
        if key in CLASS_MAP:
            p = os.path.join(RAW_DATA_DIR, folder)
            symbol_dict[key] = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(('.png', '.jpg'))]
    return symbol_dict

def generate_sample(sample_id, symbol_dict, split='train'):
    # 创建纯白画布
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    labels = []
    
    # 随机生成 3 到 7 个字符
    for _ in range(random.randint(3, 7)):
        cls_name = random.choice(list(symbol_dict.keys()))
        if not symbol_dict[cls_name]: continue
        
        img = cv2.imread(random.choice(symbol_dict[cls_name]))
        if img is None: continue

        # --- 核心改进：智能预处理 ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 如果背景是黑的，反转成白底黑字
        if np.mean(gray) < 127:
            gray = cv2.bitwise_not(gray)
        # 强化对比度，滤除杂色边缘
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # 缩放
        h, w = binary.shape
        scale = SYMBOL_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        char_img = cv2.resize(binary, (new_w, new_h))
        
        # 随机位置 (x轴不重叠的简易逻辑：这里仅示范随机，实际可按序排列)
        x = random.randint(0, IMG_SIZE - new_w)
        y = random.randint(IMG_SIZE//4, 3*IMG_SIZE//4)

        # --- 核心改进：像素级融合 (只粘贴黑色笔迹) ---
        roi = canvas[y:y+new_h, x:x+new_w]
        char_3ch = cv2.cvtColor(char_img, cv2.COLOR_GRAY2BGR)
        # bitwise_and：白色(255) & 任何色 = 任何色；黑色(0) & 任何色 = 黑色
        canvas[y:y+new_h, x:x+new_w] = cv2.bitwise_and(roi, char_3ch)

        # 记录标签
        x_c, y_c = (x + new_w/2)/IMG_SIZE, (y + new_h/2)/IMG_SIZE
        nw, nh = new_w/IMG_SIZE, new_h/IMG_SIZE
        labels.append(f"{CLASS_MAP[cls_name]} {x_c} {y_c} {nw} {nh}")

    # 保存
    name = f"{split}_{sample_id}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', split, f"{name}.jpg"), canvas)
    with open(os.path.join(OUTPUT_DIR, 'labels', split, f"{name}.txt"), 'w') as f:
        f.write("\n".join(labels))

if __name__ == "__main__":
    create_structure()
    symbols = load_symbol_paths()
    if symbols:
        for s in ['train', 'val']:
            num = int(NUM_SAMPLES * 0.8) if s == 'train' else int(NUM_SAMPLES * 0.2)
            for i in tqdm(range(num), desc=f"Generating {s}"):
                generate_sample(i, symbols, s)
        
        with open("math_data.yaml", "w") as f:
            f.write(f"path: {os.path.abspath(OUTPUT_DIR)}\ntrain: images/train\nval: images/val\nnc: {len(CLASS_MAP)}\nnames: {list(CLASS_MAP.keys())}")