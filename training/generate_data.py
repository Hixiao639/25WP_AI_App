import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import shutil

# ================= 配置区域 =================
# 原始数据集路径 (请修改为你解压的路径)
RAW_DATA_DIR = "./raw_data/extracted_images" 
# 输出 YOLO 数据集的路径
OUTPUT_DIR = "./yolo_math_dataset"

# 定义映射关系：文件夹名 -> YOLO 类别 ID
# 注意：YOLO 类别必须从 0 开始的整数
CLASS_MAP = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
    '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    '+': 10, '-': 11, 'times': 12, 'div': 13, # 根据你的数据集文件夹名修改 key
    '(': 14, ')': 15
}
# 反向映射用于显示
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}

# 生成设置
IMG_SIZE = 640       # 画板大小
NUM_SAMPLES = 2000   # 生成多少张训练图
MIN_SYMBOLS = 3      # 每张图最少几个字符
MAX_SYMBOLS = 6      # 每张图最多几个字符
SYMBOL_SIZE = 64     # 字符缩放大小 (约数)

# ===========================================

def create_yolo_structure():
    """创建 datasets/images 和 datasets/labels 文件夹结构"""
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

def load_symbol_paths():
    """加载所有字符图片的路径"""
    symbol_dict = {}
    valid_classes = list(CLASS_MAP.keys())
    
    print("正在加载图片路径...")
    for folder_name in os.listdir(RAW_DATA_DIR):
        # 处理文件夹名映射 (比如有些数据集 + 号文件夹叫 'add')
        key_name = folder_name
        if folder_name == 'add': key_name = '+'
        elif folder_name == 'sub': key_name = '-'
        elif folder_name == 'mul': key_name = 'times'
        elif folder_name == 'div': key_name = 'div' # 注意数据集里面可能是 div 或 forward_slash
        
        if key_name in valid_classes:
            folder_path = os.path.join(RAW_DATA_DIR, folder_name)
            images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg'))]
            symbol_dict[key_name] = images
            
    return symbol_dict

def overlap(box1, box2):
    """简单的防重叠检测"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    if (x1 < x2 + w2 and x1 + w1 > x2 and
        y1 < y2 + h2 and y1 + h1 > y2):
        return True
    return False

def generate_sample(sample_id, symbol_dict, split='train'):
    """生成单张合成图和对应的标签文件"""
    # 1. 创建白底画布
    canvas = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    
    num_items = random.randint(MIN_SYMBOLS, MAX_SYMBOLS)
    labels = [] # 存储 (class_id, x_center, y_center, w, h)
    placed_boxes = [] # 存储 (x, y, w, h) 用于防重叠
    
    # 为了模拟算式，我们可以让它们在水平方向上大致排列，也可以完全随机
    # 这里使用简单的随机放置+防重叠
    
    for _ in range(num_items):
        # 随机选一个类别
        cls_name = random.choice(list(CLASS_MAP.keys()))
        if not symbol_dict[cls_name]: continue
        
        # 随机选一张图
        img_path = random.choice(symbol_dict[cls_name])
        img = cv2.imread(img_path)
        if img is None: continue
        
        # 二值化处理并反转（数据集通常是黑底白字，我们需要白底黑字或者保持一致，这里假设变成黑字）
        # 如果原图是白字黑底：
        img = cv2.bitwise_not(img) 
        
        # 缩放
        h, w = img.shape[:2]
        scale = SYMBOL_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        
        # 尝试放置（尝试10次如果不重叠）
        for _ in range(10):
            # 简单策略：让y轴在大致中间波动，x轴随机
            x = random.randint(0, IMG_SIZE - new_w)
            y = random.randint(IMG_SIZE//3, 2*IMG_SIZE//3) 
            
            current_box = (x, y, new_w, new_h)
            
            is_overlap = False
            for pb in placed_boxes:
                if overlap(current_box, pb):
                    is_overlap = True
                    break
            
            if not is_overlap:
                # 贴图 (处理透明度或直接覆盖)
                # 这里简化：直接替换像素 (假设背景纯白)
                roi = canvas[y:y+new_h, x:x+new_w]
                # 简单的掩膜融合，保证黑色笔迹保留
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
                mask_inv = cv2.bitwise_not(mask)
                
                # 将字符贴上去
                canvas[y:y+new_h, x:x+new_w] = img # 简单覆盖
                
                # 计算 YOLO 坐标 (归一化 0-1)
                x_center = (x + new_w / 2) / IMG_SIZE
                y_center = (y + new_h / 2) / IMG_SIZE
                w_norm = new_w / IMG_SIZE
                h_norm = new_h / IMG_SIZE
                
                labels.append(f"{CLASS_MAP[cls_name]} {x_center} {y_center} {w_norm} {h_norm}")
                placed_boxes.append(current_box)
                break

    # 保存图片
    file_name = f"{split}_{sample_id}"
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'images', split, file_name + '.jpg'), canvas)
    
    # 保存标签
    with open(os.path.join(OUTPUT_DIR, 'labels', split, file_name + '.txt'), 'w') as f:
        f.write('\n'.join(labels))

if __name__ == "__main__":
    create_yolo_structure()
    symbols = load_symbol_paths()
    
    if not symbols:
        print("错误：未找到图片，请检查 RAW_DATA_DIR 路径是否正确")
    else:
        print("开始生成训练集...")
        for i in tqdm(range(int(NUM_SAMPLES * 0.8))):
            generate_sample(i, symbols, 'train')
            
        print("开始生成验证集...")
        for i in tqdm(range(int(NUM_SAMPLES * 0.2))):
            generate_sample(i, symbols, 'val')
            
        print(f"完成！数据已保存在 {OUTPUT_DIR}")
        
        # 生成 data.yaml
        yaml_content = f"""
path: ../{OUTPUT_DIR} # dataset root dir
train: images/train
val: images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
        """
        with open("math_data.yaml", "w") as f:
            f.write(yaml_content)
        print("已生成 math_data.yaml 配置文件")