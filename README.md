# SnapCalc-AI

一个手写算式识别与计算的应用。

## 目录结构
- `app.py`: Streamlit 主程序入口
- `packages/`: 核心功能模块
  - `image_processing.py`: 图像预处理
  - `model_loader.py`: 模型加载与推理
  - `solver.py`: 算式逻辑计算
- `training/`: 模型训练代码
- `models/`: 存放训练好的模型
- `assets/`: 静态资源

## 安装步骤
1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
2. 运行应用:
   ```bash
   streamlit run app.py
   ```

## 模型训练过程详解
training/
1. 下载(raw_data/.)：单个数字(Github私人库)或符号(Kaggle)的PNG集
2. 数据集准备(```generate_data.py```)：将数字符号粘贴到白板上
3. 训练模型(```train_yolo.py```): 得到```best.pt```模型
4. 检验模型效果(```predict_logic.py```)