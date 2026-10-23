from pathlib import Path

# 项目根目录：不管在哪个目录执行 streamlit run，路径都指向这个文件所在的文件夹
BASE_DIR = Path(__file__).resolve().parent

#chroma
collection_name = "rag"
persist_directory = str(BASE_DIR / "chroma_db")

#对话历史
chat_history_path = str(BASE_DIR / "chat_history")
max_history_messages = 10          # 只带最近 5 轮给模型（一问一答算 2 条）

#spliter
chunk_size = 400                   # 一张尺码表大约 200~400 字，1000 会把几张表混在一块
chunk_overlap = 50
separators = ["\n\n", "\n", "。", "！", "？", "；", "!", "?", ";", " ", ""]   # 去掉了 "."，避免把 72.5kg 切断

#retriever
retriever_top_k = 4                # 检索最多返回几条（原来的 similarity_threshold = 2）
score_threshold = None             # 相关度阈值，None 表示不过滤；调好后可设成 0.3 之类的数

#model
# 本地向量模型（第一次运行会自动下载约 100MB，之后走本地缓存，不需要 API Key）
embedding_model_name = "BAAI/bge-small-zh-v1.5"
embedding_device = "cpu"           # 有 N 卡且装了 CUDA 版 torch 可改成 "cuda"
hf_endpoint = "https://hf-mirror.com"   # 国内镜像，下载快；想用官方源就改成 None
chat_model_name = "qwen3.8-max"
# 百炼 Token Plan 个人版的 OpenAI 兼容地址（以控制台上显示的为准）
# Key 以 sk-sp- 开头，从环境变量 DASHSCOPE_API_KEY 读取，不要写进代码
chat_base_url = "https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1"
chat_api_key_env = "DASHSCOPE_API_KEY"
