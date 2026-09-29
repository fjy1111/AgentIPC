
from agentipc.providers.env import create_provider_bundle_from_env
from dotenv import load_dotenv
load_dotenv()  # 这会读取 .env 文件到环境变量

from agentipc.providers.env import create_provider_bundle_from_env

# 其余代码保持不变...
# 从环境变量创建 provider
resolved = create_provider_bundle_from_env()

print(f"LLM Provider: {resolved.config.llm_provider}")
print(f"Embedding Provider: {resolved.config.embedding_provider}")
print(f"Embedding Dimension: {resolved.bundle.embedding.dim}")

# 测试 LLM
print("\n测试 LLM...")
response = resolved.bundle.llm.complete([
    {"role": "user", "content": "你好，请用一句话介绍自己"}
])
print(f"LLM 响应: {response.text}")

# 测试 Embedding
print("\n测试 Embedding...")
vectors = resolved.bundle.embedding.embed(["测试文本"])
print(f"Embedding 维度: {vectors.shape}")
print(f"Embedding 前5个值: {vectors[0][:5]}")