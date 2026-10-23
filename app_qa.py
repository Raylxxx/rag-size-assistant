import logging
import traceback
import uuid

import streamlit as st

from rag import RagService

logger = logging.getLogger(__name__)

#标题
st.title("智能尺码助手")
st.divider()   #分隔符


@st.cache_resource
def load_rag():
    # 整个应用只创建一次；不同用户靠 session_id 区分记忆
    return RagService()


# 每个浏览器会话一个独立 id，记忆互不干扰（原来所有人共用 user_001）
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "你好，告诉我你的身高、体重和想买的品类，我来帮你推荐尺码。"}
    ]

for message in st.session_state["messages"]:
    st.chat_message(message["role"]).write(message["content"])

#在页面最下方提供用户输入栏
prompt = st.chat_input("例如：身高175，体重140斤，T恤穿什么码？")

if prompt:
    st.chat_message("user").write(prompt)
    st.session_state["messages"].append({"role": "user", "content": prompt})

    session_config = {"configurable": {"session_id": st.session_state["session_id"]}}

    with st.chat_message("assistant"):
        try:
            stream = load_rag().chain.stream({"input": prompt}, session_config)
            # write_stream 边显示边拼接，结束后返回完整字符串
            answer = st.write_stream(stream)
        except Exception as e:
            traceback.print_exc()   # 完整报错打印到终端
            # 页面上直接显示真实异常类型和信息，方便定位（如 401 Key 无效、模型名不存在、模型下载失败等）
            answer = f"抱歉，服务暂时不可用。\n\n{type(e).__name__}: {e}"
            st.error(answer)

    st.session_state["messages"].append({"role": "assistant", "content": answer})
