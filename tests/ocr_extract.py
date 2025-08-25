import cv2
import pytesseract
import numpy as np
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"
image_path = "data/floorplan/IMG_5455.JPG"

img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# 检测直线
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=10)

if lines is not None:
    img_with_lines = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("Detected Lines", img_with_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

orc_data = pytesseract.image_to_data(gray, output_type=Output.DICT)

numbers = []
scales = []
img_with_labels = img.copy()
for i, text in enumerate(orc_data["text"]):
    if text.strip().isdigit():
        x, y, w, h = (
            orc_data["left"][i],
            orc_data["top"][i],
            orc_data["width"][i],
            orc_data["height"][i],
        )
        center = (x + w // 2, y + h // 2)
        value = int(text)
        numbers.append((value, center, (x, y, w, h)))
        # vertical line
        if w < h:
            scale = value / h
        else:
            scale = value / w
        scales.append((value, w, scale, (x, y, w, h)))
        # Draw a rectangle around the detected number (the whole bounding box)
        cv2.rectangle(img_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Put the number label at the center
        cv2.putText(
            img_with_labels,
            text,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
cv2.imshow("Detected Number Labels", img_with_labels)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(scales)
# # 5. 匹配最近直线
# def line_distance(pt, line):
#     x0, y0 = pt
#     x1, y1, x2, y2 = line
#     return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
#         (y2 - y1) ** 2 + (x2 - x1) ** 2
#     )
# results = []
# for num, center in numbers:
#     best_line = min(lines[:,0], key=lambda l: line_distance(center, l))
#     x1,y1,x2,y2 = best_line
#     P = np.sqrt((x1-x2)**2 + (y1-y2)**2)
#     scale = num / P
#     results.append((num, best_line, scale))

# for ratio in results:
#     print(ratio)