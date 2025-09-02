from llama_index.core import (
    VectorStoreIndex,
    get_response_synthesizer,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llm_settings import set_llm, get_embed_model

from qdrant_client import QdrantClient

import logging
import sys

# ログ設定
logging.basicConfig(
    level=logging.DEBUG,
    force=True,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("llama-index.log")
    ]
)

class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

log_file = open("output.log", "w")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

set_llm()
embed_model = get_embed_model()

Settings.chunk_size = 512

# -----------------------------
# Qdrant クライアントとベクトルストアの設定
# -----------------------------
qdrant_client = QdrantClient(
    host="localhost",  # or your remote host
    port=6333,
)

collection_name = "llama_index_qdrant"

qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
)

storage_context = StorageContext.from_defaults(vector_store=qdrant_store)

# -----------------------------
# インデックス構築（Qdrant）
# -----------------------------
index = VectorStoreIndex.from_vector_store(
    vector_store=qdrant_store
)

# -----------------------------
# Retriever + QueryEngine
# -----------------------------
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    embed_model=embed_model
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
