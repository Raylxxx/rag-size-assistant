import hashlib
from datetime import datetime

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config_data as config
from vector_stores import get_embeddings


#把一段文本算成指纹（32 位十六进制字符串）
def get_string_md5(input_str: str, encoding: str = "utf-8") -> str:
    return hashlib.md5(input_str.encode(encoding)).hexdigest()


class KnowledgeBaseService:

    def __init__(self, operator: str = "admin"):
        self.operator = operator

        # Chroma 会自己创建持久化目录
        self.chroma = Chroma(
            collection_name=config.collection_name,            #数据库的表名
            embedding_function=get_embeddings(),
            persist_directory=config.persist_directory,          #数据库本地存储文件夹
        )

        self.spliter = RecursiveCharacterTextSplitter(     #文本分割器
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    def is_uploaded(self, md5_hex: str) -> bool:
        """指纹存在每个块的 metadata 里，直接查向量库，库和指纹永远同步（不再需要 md5.text）"""
        result = self.chroma.get(where={"md5": md5_hex}, limit=1)
        return len(result["ids"]) > 0

    def delete_by_source(self, filename: str) -> int:
        """删掉同名文件的所有旧块，返回删了几块"""
        old = self.chroma.get(where={"source": filename})
        if old["ids"]:
            self.chroma.delete(ids=old["ids"])
        return len(old["ids"])

    def upload_by_str(self, data: str, filename: str) -> str:
        """
        内容完全没变：跳过
        同名文件内容变了：删旧块，写新块
        新文件：切分 + 向量化 + 入库
        """
        data = data.strip()
        if not data:
            return "[失败] 文件内容为空"

        md5_hex = get_string_md5(data)
        if self.is_uploaded(md5_hex):
            return "[跳过] 内容已经存在知识库中"

        deleted = self.delete_by_source(filename)

        # 短文本切分器会原样返回 [data]，不需要自己判断长度
        chunks = self.spliter.split_text(data)

        metadata = {
            "source": filename,
            "md5": md5_hex,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": self.operator,
        }
        self.chroma.add_texts(chunks, metadatas=[dict(metadata) for _ in chunks])

        if deleted:
            return f"[更新] 已替换旧版本（删除 {deleted} 块），写入 {len(chunks)} 块"
        return f"[成功] 已写入 {len(chunks)} 块"


if __name__ == "__main__":
    service = KnowledgeBaseService()
    print(service.upload_by_str("测试：M码 身高170-175cm 体重60-65kg", "test.txt"))
    print(service.upload_by_str("测试：M码 身高170-175cm 体重60-65kg", "test.txt"))  # 第二次应该显示跳过
