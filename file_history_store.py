import json
import os
from typing import Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict

import config_data as config


def get_history(session_id):
    return FileChatMessageHistory(session_id, config.chat_history_path)


class FileChatMessageHistory(BaseChatMessageHistory):

    def __init__(self, session_id, storage_path, max_messages=config.max_history_messages):
        self.session_id = session_id
        self.storage_path = storage_path
        self.max_messages = max_messages
        self.file_path = os.path.join(self.storage_path, f"{self.session_id}.json")
        os.makedirs(self.storage_path, exist_ok=True)

    def _load_all(self) -> list[BaseMessage]:
        """硬盘上的完整记录；文件不存在或写坏了都当作没有历史"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return messages_from_dict(json.load(f))
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        # 用完整记录追加，避免把截断后的结果写回硬盘
        all_messages = self._load_all()
        all_messages.extend(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([message_to_dict(m) for m in all_messages], f, ensure_ascii=False, indent=2)

    @property
    def messages(self) -> list[BaseMessage]:
        """给模型看的只有最近 max_messages 条"""
        return self._load_all()[-self.max_messages:]

    def clear(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
