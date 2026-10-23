"""
基于 Streamlit 完成 WEB 网页上传服务
Streamlit：当 WEB 页面元素发生变化时，代码会从头重新执行一遍
"""
import traceback
import streamlit as st
from knowledge_base import KnowledgeBaseService

# 添加网页标题
st.title("知识库更新服务")


@st.cache_resource
def load_service():
    # 整个应用只创建一次，所有用户共用一个 Chroma 连接
    return KnowledgeBaseService()


def decode_text(raw: bytes) -> str:
    # utf-8-sig 能顺带去掉记事本加的 BOM；失败再按 GBK 试（Windows 记事本常见）
    for encoding in ("utf-8-sig", "gbk"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法识别文件编码，请另存为 UTF-8 后再上传")


# file_uploader
uploader_file = st.file_uploader(
    label="请上传TXT文件",
    type=["txt"],
    accept_multiple_files=False,    # False 表示仅接受一个文件的上传
)

# 记录本次会话已经处理过的文件，避免页面刷新时重复入库（第一次显示成功，刷新后变成跳过）
if "processed" not in st.session_state:
    st.session_state["processed"] = {}

if uploader_file is not None:
    st.subheader(f"文件名：{uploader_file.name}")
    st.write(f"格式：{uploader_file.type} | 大小：{uploader_file.size / 1024:.2f} KB")

    file_key = f"{uploader_file.name}-{uploader_file.size}"
    if file_key not in st.session_state["processed"]:
        try:
            text = decode_text(uploader_file.getvalue())
            with st.spinner("载入知识库中..."):     # 在 spinner 内的代码执行过程中，会有一个转圈动画
                result = load_service().upload_by_str(text, uploader_file.name)
        except ValueError as e:
            result = f"[失败] {e}"
        except Exception as e:
            traceback.print_exc()   # 完整报错打印到终端
            result = f"[失败] {type(e).__name__}: {e}"
        st.session_state["processed"][file_key] = result

    st.write(st.session_state["processed"][file_key])
