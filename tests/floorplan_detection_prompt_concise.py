import os
import base64
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

# 简洁版提示词
prompt = """
你是户型图尺寸提取专家。请提取图中所有卫生间的精确尺寸。

核心方法：
1. 从房间标注中读取面积（如4.1㎡）
2. 从边缘标尺找到一条边的长度（毫米转米）
3. 计算另一边：面积÷已知边
4. 验证：长×宽=面积

标尺读取技巧：
- 找到与卫生间边对齐的标尺段
- 如跨多段，将数值相加
- 记住：标尺单位是毫米，需÷1000转为米

输出格式：
卫生间[名称]：面积X.X㎡，长X.XXm，宽X.XXm
"""

# 使用CoT（Chain of Thought）引导
cot_prompt = """
让我们一步步分析：

第一步：识别卫生间
- 图中有哪些卫生间？列出它们的名称和位置

第二步：读取面积
- 每个卫生间标注的面积是多少？

第三步：测量边长
- 观察标尺，哪条边可以直接测量？
- 对应的标尺数值是多少毫米？

第四步：计算和验证
- 根据面积和已知边，计算另一边
- 验证计算是否正确
"""

response = client.chat.completions.create(
    model="qvq-max",
    messages=[
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{encoded_string}"},
                },
                {"type": "text", "text": cot_prompt},
            ],
        },
    ],
    extra_body={"enable_thinking": True},
    stream=True,
    stream_options={"include_usage": True},
)

reasoning_content = ""  # 完整思考过程
answer_content = ""  # 完整回复
is_answering = False  # 是否进入回复阶段
print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in response:
    if not chunk.choices:
        print("\n" + "=" * 20 + "Usage" + "=" * 20)
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
