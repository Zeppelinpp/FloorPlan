import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("QWEN_BASE_URL"),
    api_key=os.getenv("QWEN_API_KEY"),
)

with open("D:\\projects\\FloorPlan\\data\\floorplan\\IMG_5455.JPG", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# 视觉引导版提示词
prompt = """
你是一个精通建筑图纸的AI助手。请仔细观察这张户型图，完成卫生间尺寸提取任务。

【视觉识别要点】
1. 卫生间特征：
   - 通常较小的房间
   - 内部有马桶、洗手池等卫浴设施图标
   - 标注含"卫生间"字样

2. 面积标注位置：
   - 通常在房间内部
   - 格式：数字+㎡（如4.1㎡）

3. 标尺识别：
   - 图纸四周的双向箭头线段
   - 每段都有毫米数标注
   - 垂直标尺测量水平边，水平标尺测量垂直边

【计算方法】
已知：面积A（㎡）、一边长度L1（通过标尺读取，mm→m）
求解：另一边L2 = A ÷ L1

【视觉对齐技巧】
想象从卫生间的边延伸出虚拟直线，看这条线与哪些标尺段相交：
- 单段对齐：直接读数
- 多段对齐：累加所有相交段的数值

请逐个分析图中所有卫生间，输出其面积和长宽尺寸。
"""

# 交互式引导
interactive_prompt = """
现在，请你：
1. 先告诉我你在图中看到了几个卫生间？它们分别在什么位置？
2. 然后逐个分析每个卫生间的尺寸信息
3. 展示你的测量和计算过程
"""

response = client.chat.completions.create(
    model="qvq-max",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
            {"type": "text", "text": interactive_prompt}
        ]},
    ],
    extra_body={"enable_thinking": True},
)

print("模型回复：")
print(response.choices[0].message.content)
