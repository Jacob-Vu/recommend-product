import cv2
import numpy as np
import matplotlib.pyplot as plt

def non_max_suppression(boxes, overlap_thresh):
    """
    Thực hiện Non-Maximum Suppression (NMS) để loại bỏ các bounding box chồng lấn.
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,0] + boxes[:,2]
    y2 = boxes[:,1] + boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    return boxes[pick].astype(int)

def main():
    # Bước 1: Đọc ảnh và chuyển sang ảnh xám
    image_path = 'pets.jpg'  # Thay bằng đường dẫn tới ảnh của bạn
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Bước 2: Áp dụng Canny Edge Detection
    edges = cv2.Canny(image, threshold1=100, threshold2=200)

    # Hiển thị ảnh sau khi tìm biên với Canny
    plt.imshow(edges, cmap='gray')
    plt.title('Edges detected by Canny')
    plt.show()

    # Bước 3: Tìm các contours từ kết quả Canny
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Đọc lại ảnh gốc để hiển thị các contours lên ảnh màu
    image_color = cv2.imread(image_path)

    # Tạo danh sách các bounding box cho các contours
    bounding_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Bước 4: Áp dụng Non-Maximum Suppression (NMS)
    nms_boxes = non_max_suppression(np.array(bounding_boxes), overlap_thresh=0.3)

    # Bước 5: Vẽ bounding box đã qua NMS lên ảnh gốc
    for (x, y, w, h) in nms_boxes:
        cv2.rectangle(image_color, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Hiển thị kết quả ảnh với bounding boxes sau NMS
    plt.imshow(cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB))
    plt.title('Bounding boxes after NMS')
    plt.show()

if __name__ == "__main__":
    main()
