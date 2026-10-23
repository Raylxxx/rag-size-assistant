import logging
import os

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

import config_data as config
from file_history_store import get_history
from vector_stores import VectorStoreService, get_embeddings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一名服装尺码顾问，根据【参考资料】回答用户的尺码问题。

规则：
1. 只使用参考资料里的尺码数据，资料里没有的品牌或品类，直接说明无法确定，不要编造数值。
2. 用户缺少关键信息（身高、体重、品类）时，先追问，不要猜。
3. 用户身材处在两个尺码之间时，给出两个选项，并说明哪个偏紧、哪个偏松。
4. 回答时注明依据来自哪份资料。
5. 回答简洁专业。

【参考资料】
{context}"""

REWRITE_PROMPT = """根据对话历史，把用户最新的问题改写成一个不依赖上下文、可以单独拿去检索的完整问题。
必须保留身高、体重、性别、品牌、品类等关键信息。只输出改写后的问题，不要回答它。"""


# 调试用：设置 logging 级别为 DEBUG 时才打印最终 Prompt，平时不刷屏
def log_prompt(prompt):
    logger.debug("最终 Prompt：\n%s", prompt.to_string())
    return prompt


def format_document(docs: list[Document]) -> str:
    if not docs:
        return "无相关参考资料"
    return "\n\n".join(
        f"[来源：{doc.metadata.get('source', '未知')}]\n{doc.page_content}" for doc in docs
    )


class RagService:

    def __init__(self):
        self.vector_service = VectorStoreService(
            embedding=get_embeddings()
        )
        # Token Plan 的 Key 只能走它专用的 OpenAI 兼容地址，ChatTongyi 走的默认地址会报 401
        self.chat_model = ChatOpenAI(
            model=config.chat_model_name,
            base_url=config.chat_base_url,
            api_key=os.environ.get(config.chat_api_key_env),
        )

        # 多轮对话时，先把"那裤子呢？"这种问题改写成完整问题再去检索
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", REWRITE_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{input}"),
        ])
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("history"),
            ("user", "{input}"),
        ])

        self.chain = self._build_chain()

    def _build_chain(self):
        retriever = self.vector_service.get_retriever()
        rewrite_chain = self.rewrite_prompt | self.chat_model | StrOutputParser()

        def build_search_query(value):
            # 第一轮没有历史，不用改写，省一次模型调用
            if not value["history"]:
                return value["input"]
            query = rewrite_chain.invoke(value)
            logger.debug("检索问题改写：%s -> %s", value["input"], query)
            return query

        # 输入 {"input", "history"}，assign 保留原字段的同时新增 "context"
        # 等价于原来的 {"input": RunnablePassthrough(), "context": ...} + format_for_prompt_template
        chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(build_search_query) | retriever | format_document
            )
            | self.answer_prompt
            | RunnableLambda(log_prompt)
            | self.chat_model
            | StrOutputParser()
        )

        # 增强链：自动读取/保存对话历史
        return RunnableWithMessageHistory(
            chain,
            get_history,
            input_messages_key="input",
            history_messages_key="history",
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rag = RagService()
    session_config = {"configurable": {"session_id": "cli_test"}}
    print(rag.chain.invoke({"input": "我身高175，体重140斤，T恤穿什么码？"}, session_config))
    print(rag.chain.invoke({"input": "那裤子呢？"}, session_config))   # 看日志里改写后的检索问题
