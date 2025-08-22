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


with open("D:\\projects\\FloorPlan\\data\\floorplan\\IMG_5454.JPG", "rb") as example_image_file:
    example_encoded_string = base64.b64encode(example_image_file.read()).decode("utf-8")

with open("D:\\projects\\FloorPlan\\data\\floorplan\\IMG_5455.JPG", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

prompt = """
你是一个户型图信息提取专家，请根据上传的户型图定位到卫生间的面积和长宽信息并结构化返回结果
思路：不考虑长宽，先提取面积，然后确定任意一条边的长度，另一条边的长度即为面积除以已知边长

请按照以下步骤:
1. 定位到户型图中的卫生间
2. 提取卫生间下边标注的面积信息
3. 在定位的框选区域内横纵绘制辅助线，辅助线与标尺的箭头对齐的边即为可以确定的一条边的长度
4. 根据面积信息和能确定的标尺信息计算另一个边的长度
5. 验证长宽乘积是否等于先前提取的面积信息

注意:
辅助线与标尺的交点之间可能包含多段双向箭头之间的线段，这种情况边长度为多段数值之和
如果没有可以确定的边，可以通过别紧挨着的房间近似估计边长
"""
user_prompt="""
<例子>
比如这个图的卫生间01的面积是4.1平方米，横向可以看到左侧的标尺的两个箭头能对齐卫生间01的其中一条边，长度是2075mm，即2.075m，所以另一条边是4.1/2.075=1.976m
其中2.075m是长，1.976m是宽
2.075 * 1.976 = 4.1
所以卫生间01:
面积: 4.1平方米
长: 2.075m
宽: 1.976m
</例子>
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
            {"type": "text", "text": "按照先前的例子，分析提取这张图片中卫生间相对应的信息"}
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
