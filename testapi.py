import os
import base64
from dotenv import load_dotenv
from zai import ZhipuAiClient

# 1. 加载 .env
load_dotenv()

api_key = os.getenv("ZHIPUAI_API_KEY")
if not api_key:
    raise ValueError("❌ 没读取到 API KEY")

client = ZhipuAiClient(api_key=api_key)

# 2. 读取 PNG 图片
img_path = "test.png"

if not os.path.exists(img_path):
    raise FileNotFoundError("❌ 找不到 test.png")

with open(img_path, "rb") as f:
    img_bytes = f.read()

# ⚠️ 关键：base64编码
img_base = base64.b64encode(img_bytes).decode("utf-8")

# ⚠️ 关键：必须是 image/png
data_url = f"data:image/png;base64,{img_base}"

# 调试用（确认格式）
print("PREFIX CHECK:", data_url[:30])  
# 应该看到：data:image/png;base64,...

# 3. 调用模型
response = client.chat.completions.create(
    model="glm-4.6v",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                },
                {
                    "type": "text",
                    "text": "这张图片里有什么？"
                }
            ]
        }
    ],
    thinking={"type": "enabled"}
)

print("\n=== MODEL OUTPUT ===")
print(response.choices[0].message)
print("MODEL USED:", getattr(response, "model", "NO MODEL FIELD"))