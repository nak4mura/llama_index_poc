from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llm_settings import set_llm, set_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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
set_embed_model()

Settings.chunk_size = 512

# -----------------------------
# Qdrant クライアントとベクトルストアの設定
# -----------------------------
qdrant_client = QdrantClient(
    host="localhost",  # or your remote host
    port=6333,
)

collection_name = "llama_index_qdrant"
embedding_dim = 768  # cl-nagoya/sup-simcse-ja-base の次元数
qdrant_client.recreate_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(
        size=embedding_dim,
        distance=Distance.COSINE
    )
)

qdrant_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=collection_name,
)

storage_context = StorageContext.from_defaults(vector_store=qdrant_store)

# -----------------------------
# ドキュメント読み込み & インデックス構築（Qdrant保存）
# -----------------------------
documents = SimpleDirectoryReader(input_dir="./input/txt/").load_data()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)
# http://localhost:6333/dashboard#/collections