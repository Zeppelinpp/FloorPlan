import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import re
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# 设置tesseract路径
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

class ScaleDetector:
    def __init__(self, image_path: str):
        """
        初始化比例尺检测器
        
        Args:
            image_path: 图像文件路径
        """
        self.image_path = image_path
        self.img = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.img.shape[:2]
        
    def detect_arrows_and_numbers(self) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        检测图像中的数字和其边界框
        
        Returns:
            List of (number_value, (x, y, w, h)) tuples
        """
        # 使用OCR检测文本
        ocr_data = pytesseract.image_to_data(self.gray, output_type=Output.DICT, config='--psm 6')
        
        detected_numbers = []
        
        for i, text in enumerate(ocr_data["text"]):
            if text.strip():
                # 检查是否为数字（可能包含空格或其他字符）
                cleaned_text = re.sub(r'[^\d]', '', text.strip())
                if cleaned_text.isdigit() and len(cleaned_text) >= 3:  # 至少3位数，过滤噪声
                    x = ocr_data["left"][i]
                    y = ocr_data["top"][i]
                    w = ocr_data["width"][i]
                    h = ocr_data["height"][i]
                    
                    # 过滤太小的检测结果
                    if w > 20 and h > 10:
                        number_value = int(cleaned_text)
                        detected_numbers.append((number_value, (x, y, w, h)))
        
        return detected_numbers
    
    def find_scale_numbers_in_top_region(self, top_ratio: float = 0.15) -> List[Tuple[int, Tuple[int, int, int, int]]]:
        """
        在图像顶部区域寻找比例尺数字（通常标注在顶部）
        
        Args:
            top_ratio: 顶部区域占总高度的比例
            
        Returns:
            List of (number_value, (x, y, w, h)) tuples in top region
        """
        all_numbers = self.detect_arrows_and_numbers()
        top_boundary = int(self.height * top_ratio)
        
        # 筛选顶部区域的数字
        top_numbers = []
        for number, (x, y, w, h) in all_numbers:
            if y < top_boundary:
                top_numbers.append((number, (x, y, w, h)))
        
        # 按x坐标排序（从左到右）
        top_numbers.sort(key=lambda item: item[1][0])
        
        return top_numbers
    
    def detect_horizontal_lines_near_numbers(self, numbers: List[Tuple[int, Tuple[int, int, int, int]]]) -> List[Tuple]:
        """
        检测数字附近的水平线（可能是比例尺的标注线）
        
        Args:
            numbers: 检测到的数字列表
            
        Returns:
            检测到的线段信息
        """
        # 边缘检测
        edges = cv2.Canny(self.gray, 50, 150, apertureSize=3)
        
        # 检测直线
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, 
                               minLineLength=100, maxLineGap=10)
        
        horizontal_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 检查是否为水平线（角度接近0度）
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # 水平线
                    horizontal_lines.append((x1, y1, x2, y2))
        
        return horizontal_lines
    
    def calculate_scale_ratio(self, target_number: Optional[int] = None) -> Optional[float]:
        """
        计算比例尺比例
        
        Args:
            target_number: 目标数字，如果为None则自动选择最大的数字
            
        Returns:
            比例尺比例值 (实际距离/像素距离)
        """
        top_numbers = self.find_scale_numbers_in_top_region()
        
        if not top_numbers:
            print("未在顶部区域检测到数字")
            return None
        
        # 如果未指定目标数字，选择最大的数字（通常是主要的比例尺标注）
        if target_number is None:
            target_number = max(top_numbers, key=lambda x: x[0])[0]
        
        # 找到目标数字的位置信息
        target_box = None
        for number, box in top_numbers:
            if number == target_number:
                target_box = box
                break
        
        if target_box is None:
            print(f"未找到目标数字 {target_number}")
            return None
        
        x, y, w, h = target_box
        
        # 计算比例尺：数字值 / max(w, h)
        scale_ratio = target_number / max(w, h)
        
        print(f"检测到的数字: {target_number}")
        print(f"数字边界框: x={x}, y={y}, w={w}, h={h}")
        print(f"计算的比例尺: {target_number} / {max(w, h)} = {scale_ratio:.2f}")
        
        return scale_ratio
    
    def visualize_detection(self, save_path: Optional[str] = None):
        """
        可视化检测结果
        
        Args:
            save_path: 保存路径，如果为None则显示图像
        """
        img_vis = self.img.copy()
        top_numbers = self.find_scale_numbers_in_top_region()
        
        # 绘制检测到的数字框
        for number, (x, y, w, h) in top_numbers:
            # 绘制边界框
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 添加数字标签
            cv2.putText(img_vis, str(number), (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 检测并绘制水平线
        horizontal_lines = self.detect_horizontal_lines_near_numbers(top_numbers)
        for x1, y1, x2, y2 in horizontal_lines:
            cv2.line(img_vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, img_vis)
            print(f"可视化结果已保存到: {save_path}")
        else:
            cv2.imshow("Scale Detection", img_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return img_vis

def main():
    # 测试示例
    image_path = "data/floorplan/IMG_5455.JPG"
    
    detector = ScaleDetector(image_path)
    
    # 检测顶部区域的数字
    top_numbers = detector.find_scale_numbers_in_top_region()
    print("检测到的顶部数字:")
    for number, (x, y, w, h) in top_numbers:
        print(f"数字: {number}, 位置: ({x}, {y}), 尺寸: {w}x{h}")
    
    # 计算比例尺
    scale_ratio = detector.calculate_scale_ratio()
    if scale_ratio:
        print(f"\n最终比例尺: {scale_ratio:.4f} (实际单位/像素)")
    
    # 可视化结果
    detector.visualize_detection()

if __name__ == "__main__":
    main()
