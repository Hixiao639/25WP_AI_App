import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd

# 设置页面
st.set_page_config(page_title="手写数字识别器", layout="centered")
st.title("🔢 手写数字识别画板")
st.write("在下方画板中写一个 0-9 之间的数字，模型会实时预测！")

# 1. 加载模型 (增加缓存以提高性能)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mnist_model.h5')

model = load_my_model()

# 2. 创建画板
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("画板")
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",
        stroke_width=20,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# 3. 处理图像并预测
if canvas_result.image_data is not None:
    # 将画板数据转换为灰度图并缩放至 28x28
    img = Image.fromarray(canvas_result.image_data.astype('uint8'))
    img = img.convert('L') # 转为灰度
    img = img.resize((28, 28)) # 缩放到模型要求的尺寸
    
    # 转换为模型输入的数组格式
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # 预测
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction)
    confidence = np.max(prediction)

    with col2:
        st.subheader("识别结果")
        st.metric(label="预测数字", value=pred_label)
        st.write(f"置信度: {confidence:.2%}")
        
        # 绘制概率条形图
        chart_data = pd.DataFrame(
            prediction[0], 
            index=[str(i) for i in range(10)], 
            columns=["概率"]
        )
        st.bar_chart(chart_data)

# 展示预处理后的微缩图（调试用）
if st.checkbox("显示模型看到的图像 (28x28)"):
    st.image(img, width=100)