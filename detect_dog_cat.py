import torch
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# Tải mô hình YOLOv5 (sử dụng mô hình được huấn luyện sẵn)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Tải ảnh chứa chó mèo
image_path = 'home1.webp'  # Thay bằng đường dẫn tới ảnh của bạn
image = Image.open(image_path)

# Thực hiện phát hiện đối tượng
results = model(image)

# In các thông tin phát hiện
results.print()  # In thông tin phát hiện

# Lấy kết quả phát hiện dưới dạng DataFrame
df = results.pandas().xyxy[0]  # Lấy bounding box và nhãn của các đối tượng

# Lọc các đối tượng chỉ có nhãn là "dog" và "cat"
detected_animals = df[(df['name'] == 'dog') | (df['name'] == 'cat')]

# Hiển thị kết quả phát hiện với nhãn dán
results.show()  # Hiển thị ảnh với các bounding box và nhãn

# Mở lại ảnh gốc để dán nhãn
image_with_labels = cv2.imread(image_path)

# Duyệt qua các đối tượng đã được phát hiện
for index, row in detected_animals.iterrows():
    x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
    label = row['name']  # Nhãn của đối tượng, ví dụ: 'dog' hoặc 'cat'
    
    # Vẽ bounding box
    cv2.rectangle(image_with_labels, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Dán nhãn lên ảnh
    cv2.putText(image_with_labels, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# Hiển thị ảnh đã dán nhãn
plt.imshow(cv2.cvtColor(image_with_labels, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

# Lưu ảnh đã dán nhãn xuống ổ cứng
output_image_path = 'labeled_image.jpg'
cv2.imwrite(output_image_path, image_with_labels)
