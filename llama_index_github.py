from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.response_synthesizers import ResponseMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from llama_index.core.readers.base import BaseReader
from llama_index.readers.github import GithubRepositoryReader, GithubClient

from pathlib import Path
import os
import logging
import pickle
from llm_settings import set_llm, set_embed_model

logging.basicConfig(level=logging.DEBUG, force=True)

set_llm()
set_embed_model()


# -----------------------------
# Documentsオブジェクト生成
# -----------------------------
documents = None
if os.path.exists("docs.pkl"):
    with open("docs.pkl", "rb") as f:
        documents = pickle.load(f)
github_token = os.environ.get("GITHUB_TOKEN")
github_client = GithubClient(github_token=github_token, verbose=True)
if documents is None:
    documents = GithubRepositoryReader(
        github_client,
        owner="nak4mura",
        repo="boost_honeypot",
        filter_directories=(
            ["docs"],
            GithubRepositoryReader.FilterType.INCLUDE,
        ),
        filter_file_extensions=(
            [
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".ico",
                "json",
                ".ipynb",
            ],
            GithubRepositoryReader.FilterType.EXCLUDE,
        ),
        verbose = True,
    ).load_data(branch="main")

    with open("docs.pkl", "wb") as f:
        pickle.dump(documents, f)

print(f"#####################################")
print(f"Loaded {len(documents)} docs")
print(f"#####################################")

# Indexオブジェクト生成
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# Retriever + SimpleChatEngine 構築
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
response_synthesizer = get_response_synthesizer(
    response_mode=ResponseMode.COMPACT
)
chat_memory = ChatMemoryBuffer.from_defaults(
    token_limit=10000,
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

# チャット実行
while True:
    user_input = input("👤 あなた: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = query_engine.query(user_input)
    print(chat_memory.get_all())
    print(f"🤖 AI: {response.response}")
