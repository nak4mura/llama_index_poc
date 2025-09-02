from llama_index.readers.file import PyMuPDFReader
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path
import glob
from llm_settings import set_llm, set_embed_model

set_llm()
set_embed_model()

# ドキュメント読み込み
reader = PyMuPDFReader()
pdf_paths = glob.glob("./input/pdf/*")
documents = []
for path in pdf_paths:
    docs = reader.load_data(file_path=path)
    documents.extend(docs)

# インデックス作成
index = VectorStoreIndex.from_documents(documents)

# 質問
query_engine = index.as_query_engine()
response = query_engine.query("PDFの内容を要約してください。")

print("\n=== 回答 ===")
print(response)
