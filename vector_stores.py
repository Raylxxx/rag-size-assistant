"""
负责查，提供检索器，按照问题去找相似内容。
"""
import os

from langchain_chroma import Chroma

import config_data as config


def get_embeddings():
    """创建本地向量模型。入库（knowledge_base）和检索（rag）都调用这里，保证两边用的是同一个模型"""
    # 必须在 import huggingface 相关库之前设置；用户自己设置过环境变量时不覆盖
    if config.hf_endpoint:
        os.environ.setdefault("HF_ENDPOINT", config.hf_endpoint)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=config.embedding_model_name,
        model_kwargs={"device": config.embedding_device},
        encode_kwargs={"normalize_embeddings": True},   # 归一化，bge 官方推荐
    )


class VectorStoreService:
    def __init__(self, embedding):
        # embedding 由外部传入，保证和入库时用的是同一个模型
        self.embedding = embedding
        #属性里放仓库
        self.vector_store = Chroma(
            collection_name=config.collection_name,
            embedding_function=self.embedding,
            persist_directory=config.persist_directory,
        )

    def get_retriever(self):
        # 没配阈值：普通 top-k 检索
        if config.score_threshold is None:
            return self.vector_store.as_retriever(search_kwargs={"k": config.retriever_top_k})

        # 配了阈值：低于阈值的结果直接丢掉，一条都不相关时返回空列表
        return self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": config.retriever_top_k,
                "score_threshold": config.score_threshold,
            },
        )


# 直接运行时才会运行，被别的文件 import 时不会运行，是一段"自测代码"
if __name__ == "__main__":
    service = VectorStoreService(get_embeddings())

    # 调阈值时先看原始分数：相关问题和无关问题（如"你好"）各跑一次，阈值取两者中间
    question = "我的体重180斤，尺码推荐？"
    for doc, score in service.vector_store.similarity_search_with_relevance_scores(question, k=4):
        print(f"{score:.3f}  {doc.metadata.get('source')}  {doc.page_content[:40]}")
