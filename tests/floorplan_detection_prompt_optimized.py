import os
import base64
import io
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("QWEN_BASE_URL"),
    api_key=os.getenv("QWEN_API_KEY"),
)


with open("data/floorplan/IMG_5454.JPG", "rb") as example_image_file:
    example_encoded_string = base64.b64encode(example_image_file.read()).decode("utf-8")

with open("data/floorplan/IMG_5455.JPG", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# 优化后的提示词
prompt = """
你是一个专业的户型图信息提取专家。你的任务是精确提取户型图中所有卫生间的尺寸信息。

## 任务目标
从户型图中提取每个卫生间的：
1. 面积（平方米）
2. 长度（米）
3. 宽度（米）

## 详细步骤

### 第一步：识别所有卫生间
- 在户型图中查找标注为"卫生间"的区域（可能标注为：卫生间01、卫生间02、卫生间A、卫生间B等）
- 注意：一张户型图可能包含多个卫生间
- 每个卫生间通常有明确的边界线和房间名称标注

### 第二步：提取面积信息
- 查看每个卫生间内部或附近的面积标注
- 面积通常以"X.X㎡"或"X.Xm²"的格式显示
- 记录精确的数值，包括小数点后的数字

### 第三步：确定一条边的长度
- 观察户型图四周的标尺（通常在图的上下左右边缘）
- 标尺由多段双向箭头组成，每段都有对应的毫米数标注
- 找到与卫生间某条边对齐的标尺段：
  a. 如果卫生间的一条边正好对应标尺的某一段，直接读取该段数值
  b. 如果卫生间的边跨越多段标尺，需要将对应的多段数值相加
  c. 将毫米转换为米（除以1000）

### 第四步：计算另一条边
- 使用公式：另一边长度 = 面积 ÷ 已知边长度
- 保留合理的小数位数（通常3位）

### 第五步：验证计算
- 验证：长 × 宽 ≈ 面积（允许0.1平方米内的误差）
- 确定哪个是长边，哪个是宽边（通常长边 > 宽边）

## 特殊情况处理
1. 如果无法从标尺直接确定任何一条边：
   - 参考相邻房间的尺寸进行估算
   - 使用房间的形状比例估算长宽比
   
2. 如果卫生间形状不规则：
   - 将其近似为矩形处理
   - 使用最接近的矩形边界计算

## 输出格式
对于每个卫生间，请按以下格式输出：
```
卫生间[编号/名称]：
- 面积：X.X平方米
- 长：X.XXXm
- 宽：X.XXXm
- 验证：X.XXX × X.XXX = X.X ✓
```

## 重要提醒
- 所有测量都基于图中的标尺，不要凭视觉估计
- 注意单位转换：标尺通常是毫米，需转换为米
- 如果有多个卫生间，请全部提取
"""

user_prompt = """
<分析示例>
以第一张图(IMG_5454.JPG)为例：

我观察到这张户型图中有两个卫生间：
1. 卫生间01（左侧）- 标注面积4.1㎡
2. 卫生间02（中间）- 标注面积4.0㎡

对于卫生间01：
- 查看左侧纵向标尺，找到与卫生间01水平边对齐的段
- 该段标注为2075mm，即2.075m
- 计算另一边：4.1 ÷ 2.075 = 1.976m
- 验证：2.075 × 1.976 = 4.100 ✓

结果：
卫生间01：
- 面积：4.1平方米
- 长：2.075m
- 宽：1.976m
- 验证：2.075 × 1.976 = 4.1 ✓
</分析示例>

请按照这个方法分析图中的所有卫生间。
"""

response = client.chat.completions.create(
    model="qvq-max",
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{example_encoded_string}"}},
            {"type": "text", "text": user_prompt}
        ]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_string}"}},
            {"type": "text", "text": "现在请分析第二张图片(IMG_5455.JPG)中所有卫生间的尺寸信息。"}
        ]},
    ],
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={
        "include_usage": True
    },
)

reasoning_content = ""  # 完整思考过程
answer_content = ""  # 完整回复
is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in response:
    if not chunk.choices:
        print("\n" + "="*20+"Usage"+"="*20)
        print(chunk.usage)
        continue
    delta = chunk.choices[0].delta

    # 只收集思考内容
    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
        if not is_answering:
            print(delta.reasoning_content, end="", flush=True)
        reasoning_content += delta.reasoning_content

    # 收到content，开始进行回复
    if hasattr(delta, "content") and delta.content:
        if not is_answering:
            print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
            is_answering = True
        print(delta.content, end="", flush=True)
        answer_content += delta.content
