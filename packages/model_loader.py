import tensorflow as tf
import numpy as np

class ModelLoader:
    def __init__(self, model_path='models/digit_model.h5'):
        self.model_path = model_path
        self.model = None
        
    def load_model(self):
        """加载训练好的模型"""
        # try:
        #     self.model = tf.keras.models.load_model(self.model_path)
        #     print("模型加载成功")
        # except Exception as e:
        #     print(f"模型加载失败: {e}")
        pass

    def predict(self, image_segment):
        """
        对单个切片（数字/符号）进行预测
        """
        if self.model is None:
            return None
        # TODO: preprocess segment for model input
        # return prediction
        pass
