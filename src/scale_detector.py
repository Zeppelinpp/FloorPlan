import cv2
import pytesseract
import numpy as np
from pytesseract import Output
import re
from typing import List, Tuple, Optional, Dict
import os

# 设置tesseract路径
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


class FloorPlanScaleDetector:
    """
    户型图比例尺检测器
    专门用于识别户型图中双箭头之间的数值并计算比例尺
    """

    def __init__(self, image_path: str):
        """
        初始化比例尺检测器

        Args:
            image_path: 图像文件路径
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        self.image_path = image_path
        self.img = cv2.imread(image_path)
        if self.img is None:
            raise ValueError(f"无法读取图像文件: {image_path}")

        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.img.shape[:2]

    def detect_scale_numbers(
        self, region: str = "top", region_ratio: float = 0.15
    ) -> List[Dict]:
        """
        检测指定区域的比例尺数字

        Args:
            region: 检测区域 ("top", "bottom", "left", "right", "all")
            region_ratio: 区域占总尺寸的比例

        Returns:
            检测结果列表，每个元素包含 {"number": int, "bbox": (x,y,w,h), "confidence": float}
        """
        # 根据区域选择ROI
        if region == "top":
            roi = self.gray[: int(self.height * region_ratio), :]
            y_offset = 0
        elif region == "bottom":
            roi = self.gray[int(self.height * (1 - region_ratio)) :, :]
            y_offset = int(self.height * (1 - region_ratio))
        elif region == "left":
            roi = self.gray[:, : int(self.width * region_ratio)]
            y_offset = 0
        elif region == "right":
            roi = self.gray[:, int(self.width * (1 - region_ratio)) :]
            y_offset = 0
        else:  # all
            roi = self.gray
            y_offset = 0

        # OCR检测
        ocr_data = pytesseract.image_to_data(
            roi,
            output_type=Output.DICT,
            config="--psm 6 -c tessedit_char_whitelist=0123456789",
        )

        detected_numbers = []

        for i in range(len(ocr_data["text"])):
            text = ocr_data["text"][i].strip()
            confidence = int(ocr_data["conf"][i])

            if text and confidence > 30:  # 置信度阈值
                # 清理文本，只保留数字
                cleaned_text = re.sub(r"[^\d]", "", text)
                if cleaned_text.isdigit() and len(cleaned_text) >= 3:  # 至少3位数
                    x = ocr_data["left"][i]
                    y = ocr_data["top"][i] + y_offset
                    w = ocr_data["width"][i]
                    h = ocr_data["height"][i]

                    # 过滤太小的检测结果
                    if w > 15 and h > 8:
                        number_value = int(cleaned_text)
                        detected_numbers.append(
                            {
                                "number": number_value,
                                "bbox": (x, y, w, h),
                                "confidence": confidence,
                            }
                        )

        # 按置信度排序
        detected_numbers.sort(key=lambda x: x["confidence"], reverse=True)
        return detected_numbers

    def detect_arrow_lines_near_number(
        self, number_bbox: Tuple[int, int, int, int], search_radius: int = 50
    ) -> List[Tuple[int, int, int, int]]:
        """
        检测数字附近的箭头线段

        Args:
            number_bbox: 数字的边界框 (x, y, w, h)
            search_radius: 搜索半径

        Returns:
            检测到的线段列表 [(x1, y1, x2, y2), ...]
        """
        x, y, w, h = number_bbox
        center_x, center_y = x + w // 2, y + h // 2

        # 创建搜索区域
        search_x1 = max(0, x - search_radius)
        search_y1 = max(0, y - search_radius)
        search_x2 = min(self.width, x + w + search_radius)
        search_y2 = min(self.height, y + h + search_radius)

        # 在搜索区域进行边缘检测
        roi = self.gray[search_y1:search_y2, search_x1:search_x2]
        edges = cv2.Canny(roi, 50, 150, apertureSize=3)

        # 检测直线
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=30, minLineLength=40, maxLineGap=5
        )

        detected_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # 转换回原图坐标
                x1 += search_x1
                y1 += search_y1
                x2 += search_x1
                y2 += search_y1

                # 检查是否为水平线（双箭头通常是水平的）
                angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 10 or angle > 170:  # 水平线
                    detected_lines.append((x1, y1, x2, y2))

        return detected_lines

    def find_arrow_line_for_number(
        self, number_bbox: Tuple[int, int, int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        为指定数字找到对应的双箭头线段

        Args:
            number_bbox: 数字的边界框

        Returns:
            最匹配的线段 (x1, y1, x2, y2) 或 None
        """
        x, y, w, h = number_bbox
        center_x, center_y = x + w // 2, y + h // 2

        # 检测附近的线段
        lines = self.detect_arrow_lines_near_number(number_bbox)

        if not lines:
            return None

        # 找到最接近数字中心且方向合适的线段
        best_line = None
        min_distance = float("inf")

        for line in lines:
            x1, y1, x2, y2 = line
            line_center_x = (x1 + x2) / 2
            line_center_y = (y1 + y2) / 2

            # 计算线段中心到数字中心的距离
            distance = np.sqrt(
                (line_center_x - center_x) ** 2 + (line_center_y - center_y) ** 2
            )

            # 检查线段长度是否合理（不能太短）
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if line_length < 30:  # 过滤太短的线段
                continue

            if distance < min_distance:
                min_distance = distance
                best_line = line

        return best_line

    def calculate_scale_ratio(
        self, target_number: Optional[int] = None, region: str = "top"
    ) -> Optional[Dict]:
        """
        计算比例尺比例

        Args:
            target_number: 目标数字，如果为None则自动选择最合适的数字
            region: 检测区域

        Returns:
            包含比例尺信息的字典: {
                "scale_ratio": float,
                "number": int,
                "bbox": tuple,
                "pixel_length": float,
                "real_length": int,
                "arrow_line": tuple or None
            }
        """
        numbers = self.detect_scale_numbers(region=region)

        if not numbers:
            return None

        # 选择目标数字
        if target_number is None:
            # 自动选择：优先选择较大的数字（通常是主要标注）
            target_result = max(numbers, key=lambda x: x["number"])
        else:
            # 查找指定数字
            target_result = None
            for num_info in numbers:
                if num_info["number"] == target_number:
                    target_result = num_info
                    break

            if target_result is None:
                return None

        number = target_result["number"]
        number_bbox = target_result["bbox"]
        x, y, w, h = number_bbox

        # 寻找对应的双箭头线段
        arrow_line = self.find_arrow_line_for_number(number_bbox)

        if arrow_line:
            # 使用箭头线段的长度
            x1, y1, x2, y2 = arrow_line
            pixel_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        else:
            # 如果找不到线段，回退到使用文本框尺寸
            pixel_length = max(w, h)
            print(f"警告: 未找到数字 {number} 对应的箭头线段，使用文本框尺寸")

        # 计算比例尺：实际长度(mm) / 像素长度
        scale_ratio = number / pixel_length

        return {
            "scale_ratio": scale_ratio,
            "number": number,
            "bbox": number_bbox,
            "pixel_length": pixel_length,
            "real_length": number,
            "confidence": target_result["confidence"],
            "arrow_line": arrow_line,
        }

    def get_all_scale_candidates(self) -> List[Dict]:
        """
        获取所有可能的比例尺候选

        Returns:
            所有检测到的数字及其比例尺信息
        """
        all_numbers = self.detect_scale_numbers(region="all")
        candidates = []

        for num_info in all_numbers:
            number = num_info["number"]
            x, y, w, h = num_info["bbox"]
            pixel_length = max(w, h)
            scale_ratio = number / pixel_length

            candidates.append(
                {
                    "scale_ratio": scale_ratio,
                    "number": number,
                    "bbox": (x, y, w, h),
                    "pixel_length": pixel_length,
                    "real_length": number,
                    "confidence": num_info["confidence"],
                    "position_type": self._classify_position(x, y),
                }
            )

        return candidates

    def _classify_position(self, x: int, y: int) -> str:
        """分类数字的位置（顶部、底部等）"""
        if y < self.height * 0.2:
            return "top"
        elif y > self.height * 0.8:
            return "bottom"
        elif x < self.width * 0.2:
            return "left"
        elif x > self.width * 0.8:
            return "right"
        else:
            return "center"

    def visualize_detections(
        self,
        save_path: Optional[str] = None,
        show_all: bool = False,
        show_arrows: bool = True,
    ) -> np.ndarray:
        """
        可视化检测结果

        Args:
            save_path: 保存路径
            show_all: 是否显示所有检测结果，否则只显示顶部区域
            show_arrows: 是否显示检测到的箭头线段

        Returns:
            可视化图像
        """
        img_vis = self.img.copy()

        if show_all:
            numbers = self.detect_scale_numbers(region="all")
        else:
            numbers = self.detect_scale_numbers(region="top")

        # 绘制检测结果
        for i, num_info in enumerate(numbers):
            number = num_info["number"]
            x, y, w, h = num_info["bbox"]
            confidence = num_info["confidence"]

            # 不同颜色表示不同置信度
            if confidence > 80:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif confidence > 50:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 0, 255)  # 红色 - 低置信度

            # 绘制边界框
            cv2.rectangle(img_vis, (x, y), (x + w, y + h), color, 2)

            # 添加标签
            label = f"{number} ({confidence}%)"
            cv2.putText(
                img_vis, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )

            # 绘制对应的箭头线段
            if show_arrows:
                arrow_line = self.find_arrow_line_for_number((x, y, w, h))
                if arrow_line:
                    x1, y1, x2, y2 = arrow_line
                    # 用蓝色绘制箭头线段
                    cv2.line(
                        img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3
                    )
                    # 添加线段长度标注
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    cv2.putText(
                        img_vis,
                        f"{line_length:.1f}px",
                        (mid_x, mid_y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )

        if save_path:
            cv2.imwrite(save_path, img_vis)

        return img_vis


def demo_scale_detection(image_path: str):
    """
    演示比例尺检测功能
    """
    print(f"分析图像: {image_path}")
    print("=" * 50)

    try:
        detector = FloorPlanScaleDetector(image_path)

        # 1. 获取主要比例尺
        main_scale = detector.calculate_scale_ratio()
        if main_scale:
            print("主要比例尺信息:")
            print(f"  数字: {main_scale['number']}")
            print(f"  文本位置: ({main_scale['bbox'][0]}, {main_scale['bbox'][1]})")
            print(f"  文本尺寸: {main_scale['bbox'][2]} x {main_scale['bbox'][3]} 像素")
            if main_scale["arrow_line"]:
                x1, y1, x2, y2 = main_scale["arrow_line"]
                print(f"  箭头线段: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
                print(f"  箭头长度: {main_scale['pixel_length']:.1f} 像素")
            else:
                print(
                    f"  未找到箭头线段，使用文本框尺寸: {main_scale['pixel_length']:.1f} 像素"
                )
            print(f"  比例尺: {main_scale['scale_ratio']:.4f} (mm/像素)")
            print(f"  置信度: {main_scale['confidence']}%")
        else:
            print("未检测到有效的比例尺")
            return

        print("\n" + "=" * 50)

        # 2. 获取所有候选
        candidates = detector.get_all_scale_candidates()
        print(f"检测到 {len(candidates)} 个数字:")
        for i, candidate in enumerate(candidates[:10]):  # 显示前10个
            print(
                f"  {i + 1}. 数字:{candidate['number']}, "
                f"比例尺:{candidate['scale_ratio']:.2f}, "
                f"位置:{candidate['position_type']}, "
                f"置信度:{candidate['confidence']}%"
            )

        # 3. 可视化
        img_vis = detector.visualize_detections()
        cv2.imshow("Scale Detection Results", img_vis)
        print("\n按任意键关闭图像窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"处理过程中出现错误: {e}")


if __name__ == "__main__":
    # 测试用例
    image_path = "data/floorplan/IMG_5455.JPG"
    demo_scale_detection(image_path)
