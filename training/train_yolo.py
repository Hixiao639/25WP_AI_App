from ultralytics import YOLO
import os

def train_math_model():
    # 1. 加载模型 (n 为最轻量级，适合 CPU)
    model = YOLO('yolov8n.pt') 

    # 2. 开始训练
    # 注意：workers=0 是单核/单进程模式的关键
    results = model.train(
        data='math_data.yaml', 
        epochs=50, 
        imgsz=416,           # 提高速度：从 640 降到 416，显著减少 CPU 计算量
        batch=8,             # CPU 训练建议小 batch，防止内存溢出
        device='cpu',        # 强制使用 CPU
        workers=0,           # 设置为 0 或 1 以减少多线程切换开销，适合单核环境
        patience=0,          # 禁用早停
        plots=True,          # 自动生成训练过程的曲线图 (results.png)
        save=True,           # 自动保存模型快照
        val=True,            # 开启训练后的验证
        name='math_v2_cpu',  # 文件夹名称
        exist_ok=True        # 覆盖同名文件夹
    )

    # 3. 验证并保存【带标注】的图像
    # YOLO 会自动在 runs/detect/math_v2_cpu/ 文件夹下生成 val_batchX_labels.jpg (真值) 
    # 和 val_batchX_pred.jpg (预测值)，上面自带框、类别和置信度。
    metrics = model.val() 
    print(f"验证完成，结果保存在 runs/detect/math_v2_cpu/")

if __name__ == '__main__':
    train_math_model()