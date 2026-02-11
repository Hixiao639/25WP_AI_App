from ultralytics import YOLO
import cv2

# 加载训练好的模型
model = YOLO('runs/detect/math_yolo_run/weights/best.pt')

def recognize_equation(image_path_or_array):
    # 1. 预测
    results = model(image_path_or_array)
    
    # 2. 解析结果
    detections = []
    for r in results:
        boxes = r.boxes.data.cpu().numpy() # 获取坐标数据 [x1, y1, x2, y2, conf, cls]
        
        for box in boxes:
            x1, y1, x2, y2, conf, cls_id = box
            name = model.names[int(cls_id)]
            detections.append({
                'name': name,
                'x_center': (x1 + x2) / 2, # 使用中心点作为排序依据
                'conf': conf,
                'bbox': [x1, y1, x2, y2]
            })
    
    # 3. 核心逻辑：按 x_center 从左到右排序
    if not detections:
        return "未识别到算式"
        
    detections.sort(key=lambda x: x['x_center'])
    
    # 4. 拼接字符串
    equation_str = ""
    for d in detections:
        symbol = d['name']
        # 简单转换：将 times 转为 *, div 转为 /
        if symbol == 'times': symbol = '*'
        elif symbol == 'div': symbol = '/'
        
        equation_str += symbol
        
    return equation_str

# 测试
# img = cv2.imread("test_handwriting.jpg")
# eq = recognize_equation(img)
# print(f"识别出的算式: {eq}")
# print(f"计算结果: {eval(eq)}")