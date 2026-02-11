from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt') 
    model.train(
        data='math_data.yaml', 
        epochs=50, 
        imgsz=640, 
        batch=16,
        patience=0,      # 设置为0，禁用早停，强制跑完50轮
        cache=False,     # 不使用缓存，确保读取的是你新生成的图片
        name='math_v2'
    )

if __name__ == '__main__':
    train()