from llama_index.core import (
    SimpleDirectoryReader,
    get_response_synthesizer,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llm_settings import set_llm, set_embed_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler, CBEventType
import logging
import sys
logging.basicConfig(
    level=logging.DEBUG,
    force=True,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("tmp.log")
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

import json

import argparse

from llama_index.core import SummaryIndex, VectorStoreIndex, TreeIndex, KeywordTableIndex, StorageContext
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.graph_stores import SimpleGraphStore

def run_summary_index(documents):
    sum_index = SummaryIndex.from_documents(documents, show_progress=True)
    print("SummaryIndex の Nodes:")
    print(sum_index.index_struct)
    for node in sum_index.docstore.docs.values():
        print(json.dumps(node.to_dict(), indent=2, ensure_ascii=False))

def run_vector_index(documents):
    vec_index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("VectorStoreIndex の Nodes:")
    print(vec_index.index_struct)
    for node in vec_index.docstore.docs.values():
        print(json.dumps(node.to_dict(), indent=2, ensure_ascii=False))

def run_tree_index(documents):
    tree_index = TreeIndex.from_documents(documents, show_progress=True)
    print("TreeIndex の Nodes:")
    print(tree_index.index_struct)
    for node in tree_index.docstore.docs.values():
        print(json.dumps(node.to_dict(), indent=2, ensure_ascii=False))

def run_keyword_index(documents):
    kwrd_index = KeywordTableIndex.from_documents(documents, show_progress=True)
    print("KeywordTableIndex の Nodes:")
    print(kwrd_index.index_struct)
    for node in kwrd_index.docstore.docs.values():
        print(json.dumps(node.to_dict(), indent=2, ensure_ascii=False))

def run_knowledge_index(documents):
    graph_store = SimpleGraphStore()
    storage_context = StorageContext.from_defaults(graph_store=graph_store)
    know_index = KnowledgeGraphIndex.from_documents(documents, storage_context=storage_context)
    print("KnowledgeGraphIndex に関連付けられた Nodes:")
    print(know_index.index_struct)
    for node in know_index.docstore.docs.values():
        print(json.dumps(node.to_dict(), indent=2, ensure_ascii=False))

INDEX_MAP = {
    "summary": run_summary_index,
    "vector": run_vector_index,
    "tree": run_tree_index,
    "keyword": run_keyword_index,
    "knowledge": run_knowledge_index,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Choose which index to run")
    parser.add_argument("--index", choices=INDEX_MAP.keys(), required=True, help="Index type to run")
    args = parser.parse_args()
    set_llm()
    set_embed_model()

    # 128, 256, 512, 1024, 2048
    Settings.chunk_size=512

    # デバッグハンドラーとコールバックマネージャーの設定
    debug_handler = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([debug_handler])
    # Settings に LLM とコールバックマネージャーを設定
    Settings.callback_manager = callback_manager

    # Documentsオブジェクト生成
    documents = SimpleDirectoryReader(
        input_dir="./input/txt/",
    ).load_data()

    INDEX_MAP[args.index](documents)
