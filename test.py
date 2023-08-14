import numpy as np
from keras.preprocessing import image
from keras.models import load_model

model = load_model('mnist_model.keras')

# 载入要预测的图像
image_path = '3.png'  # 替换为实际的图像文件路径
img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")  # 调整图像大小并转为灰度图
img_array = image.img_to_array(img)
img_array = img_array.reshape(1, 784).astype('float32') / 255.0  # 预处理图像

# 进行预测
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)  # 获取预测结果中概率最高的类别索引

print("Predicted Label:", predicted_label)