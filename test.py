import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

model = load_model('mnist_model.keras')

# 载入要预测的图像
image_path = '7.png'  # 替换为实际的图像文件路径
img = image.load_img(image_path, target_size=(28, 28), color_mode="grayscale")  # 调整图像大小并转为灰度图
img_array = image.img_to_array(img)
img_array = img_array.reshape(1, 784).astype('float32') / 255.0  # 预处理图像

# 进行预测
predictions = model.predict(img_array)
predicted_label = np.argmax(predictions)  # 获取预测结果中概率最高的类别索引

print("Predicted Label:", predicted_label)

# 显示预测结果和输入图像
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title(f'Input Image (Label: {predicted_label})')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), predictions[0])
plt.title('Predicted Probabilities')
plt.xticks(range(10))
plt.xlabel('Class')
plt.ylabel('Probability')

plt.tight_layout()
plt.show()