import cv2
import numpy as np
from ultralytics import YOLO
import os

# ================= 配置区域 =================
MODEL_PATH = 'runs/detect/math_yolo_run/weights/best.pt' # 你的模型路径
TARGET_SIZE = 640                                       # 训练时的 imgsz
CONF_THRESHOLD = 0.25                                   # 置信度阈值
# ===========================================

def resize_and_pad(img, size=640, pad_color=(255, 255, 255)):
    """
    将图片等比例缩放，并填充成正方形，防止变形。
    """
    h, w = img.shape[:2]
    sh, sw = size, size
    interp = cv2.INTER_AREA if h > sh or w > sw else cv2.INTER_CUBIC
    
    # 计算缩放比例
    aspect = w / h
    if aspect > 1: # 横图
        new_w = sw
        new_h = np.round(sw / aspect).astype(int)
        pad_vert = (sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # 竖图
        new_h = sh
        new_w = np.round(sh * aspect).astype(int)
        pad_horz = (sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # 正方形
        new_h, new_w = sh, sw
        pad_top, pad_bot, pad_left, pad_right = 0, 0, 0, 0

    # 缩放
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    # 填充背景色
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, 
                                   borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img

def run_debug_test(image_path):
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        return

    # 1. 加载模型
    model = YOLO(MODEL_PATH)
    
    # 2. 读取并预处理图片
    raw_img = cv2.imread(image_path)
    if raw_img is None:
        print(f"❌ 错误: 无法读取图片 {image_path}")
        return

    # 关键步骤：补齐为正方形
    processed_img = resize_and_pad(raw_img, size=TARGET_SIZE)
    
    # 3. 推理
    results = model.predict(processed_img, conf=CONF_THRESHOLD)
    res = results[0]

    # 4. 可视化保存（非常重要，一眼看出模型在看哪）
    debug_img = res.plot()
    cv2.imwrite("debug_inference_view_2.jpg", debug_img)
    print("📂 调试图已保存至: debug_inference_view_2.jpg")

    # 5. 解析并排序
    detections = []
    for box in res.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        bbox = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
        x_center = (bbox[0] + bbox[2]) / 2
        
        detections.append({
            'name': model.names[cls_id],
            'x_center': x_center,
            'conf': conf
        })

    if not detections:
        print("⚠️ 结果: 未识别到任何字符。请检查调试图，可能是模型置信度太低或背景色反转。")
        return

    # 按 X 轴坐标从左到右排序
    detections.sort(key=lambda x: x['x_center'])
    
    # 映射符号并拼接
    raw_eq = "".join([d['name'] for d in detections])
    calc_eq = raw_eq.replace('times', '*').replace('div', '/').replace('add', '+').replace('sub', '-')
    
    print("-" * 30)
    print(f"📌 识别序列: {raw_eq}")
    print(f"📌 待算算式: {calc_eq}")
    
    # 6. 安全计算
    try:
        # 只允许数字和运算符进入 eval
        # 注意：这里直接用 eval 仅限本地测试，Streamlit 建议用更安全的库
        result = eval(calc_eq)
        print(f"✅ 计算答案: {result}")
    except Exception as e:
        print(f"❌ 计算失败: {e} (请检查算式逻辑是否完整，如括号是否闭合)")

if __name__ == "__main__":
    # 填入你想要测试的图片路径
    test_image_path = "test_handwriting_2.jpg" 
    run_debug_test(test_image_path)