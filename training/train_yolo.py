from ultralytics import YOLO

def train():
    # 1. 加载模型 (使用 nano 版本，速度最快，适合 CPU/移动端)
    model = YOLO('yolov8n.pt') 

    # 2. 训练
    # data: 指向刚才生成的 yaml 文件
    # epochs: 50 轮通常够用了
    # imgsz: 对应生成时的图片大小
    results = model.train(
        data='math_data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        name='math_yolo_run' # 训练结果保存的文件夹名
    )
    
    # 3. 验证 (可选)
    metrics = model.val()
    print(metrics.box.map)

if __name__ == '__main__':
    train()