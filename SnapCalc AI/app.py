import streamlit as st
import os
from packages import image_processing, model_loader, solver

def main():
    st.set_page_config(page_title="SnapCalc AI", page_icon="🧮")
    
    # 侧边栏/Banner
    if os.path.exists("assets/banner.png"):
        st.image("assets/banner.png", use_column_width=True)
    
    st.title("SnapCalc AI - 手写算式识别器")
    st.markdown("上传一张包含手写算式的图片，AI 将自动识别并计算结果。")

    # 文件上传
    uploaded_file = st.file_uploader("上传图片", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # 显示原图
        st.image(uploaded_file, caption="原始图片", use_column_width=True)
        
        if st.button("开始识别与计算"):
            with st.spinner("正在处理图像..."):
                # TODO: 读取图片并转换为 OpenCV 格式
                # image = ... 
                pass
                
            with st.spinner("正在识别数字与符号..."):
                # TODO: 调用模型进行推理
                # results = model_loader.predict(image)
                pass
            
            with st.spinner("正在计算结果..."):
                # TODO: 组合公式并计算
                # equation_str = "1 + 2" # 示例
                # result = solver.calculate(equation_str)
                # st.success(f"识别结果: {equation_str} = {result}")
                st.info("功能开发中...")

if __name__ == "__main__":
    main()
