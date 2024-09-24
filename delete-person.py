import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Bước 1: Sử dụng YOLO để phát hiện đối tượng
def detect_objects(image_path, model_name='yolov5s'):
    # Tải mô hình YOLOv5
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)

    # Đọc ảnh
    image = cv2.imread(image_path)

    # Phát hiện đối tượng trong ảnh
    results = model(image)

    # Trả về bounding box và nhãn đối tượng
    return results.pandas().xyxy[0], image

# Bước 2: Tìm đường viền đối tượng trong bounding box
def find_object_contour(image, bounding_box):
    # Cắt ảnh trong vùng bounding box
    x_min, y_min, x_max, y_max = int(bounding_box['xmin']), int(bounding_box['ymin']), int(bounding_box['xmax']), int(bounding_box['ymax'])
    object_roi = image[y_min:y_max, x_min:x_max]

    # Chuyển sang ảnh grayscale
    gray_roi = cv2.cvtColor(object_roi, cv2.COLOR_BGR2GRAY)

    # Áp dụng bộ lọc Gaussian để giảm nhiễu
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Áp dụng Canny Edge Detection để tìm biên của đối tượng
    edges = cv2.Canny(blurred_roi, 50, 150)

    # Tìm các contours (đường viền)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trả về contours và vùng đối tượng
    return contours, (x_min, y_min, x_max, y_max)

# Bước 3: Xóa đối tượng theo đường viền với inpainting
def remove_object_with_contour_inpainting(image, contours, bbox_coords):
    # Tạo mask cho vùng đối tượng
    mask = np.zeros(image.shape[:2], dtype="uint8")

    x_min, y_min, x_max, y_max = bbox_coords

    # Đặt contours vào vùng của bounding box trên mask (vì contours nằm trong bounding box)
    for contour in contours:
        # Đường viền contour cần được dịch về vị trí của toàn ảnh
        contour_shifted = contour + [x_min, y_min]
        cv2.drawContours(mask, [contour_shifted], -1, 255, thickness=cv2.FILLED)

    # Áp dụng inpainting để xóa đối tượng dựa trên đường viền chính xác
    inpainted_image = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

    return inpainted_image

# Bước 4: Chọn đối tượng cần xóa và áp dụng xóa đối tượng
def remove_objects(image_path, target_class='person', model_name='yolov5s'):
    # Phát hiện đối tượng
    detections, image = detect_objects(image_path, model_name)    

    # Lặp qua các bounding box để xóa các đối tượng có nhãn target_class
    for index, row in detections.iterrows():
        if row['name'] == target_class:  # Chỉ xóa đối tượng có nhãn 'person' (hoặc lớp bạn chọn)
            print(f"Removing object: {row['name']} at {row['xmin']}, {row['ymin']}, {row['xmax']}, {row['ymax']}")

            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            # Vẽ bounding box
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # Hiển thị ảnh đã xóa đối tượng
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title(f'Image found {target_class}')
            plt.show()
                    
            # Bước 1: Tìm đường viền của đối tượng
            contours, bbox_coords = find_object_contour(image, row)
            
            # Bước 2: Xóa đối tượng theo đường viền với inpainting
            image = remove_object_with_contour_inpainting(image, contours, bbox_coords)

    # Hiển thị ảnh đã xóa đối tượng
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Image after removing {target_class}')
    plt.show()

    # Lưu ảnh đã dán nhãn xuống ổ cứng
    output_image_path = 'susubg.jpg'
    cv2.imwrite(output_image_path, image)

# Thay thế đường dẫn tới ảnh của bạn
image_path = 'Susu.jpg'

# Xóa đối tượng 'person' khỏi ảnh
remove_objects(image_path, target_class='person')