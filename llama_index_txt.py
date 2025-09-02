from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    get_response_synthesizer,
    Settings
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.query_engine import RetrieverQueryEngine
from llm_settings import set_llm, get_embed_model
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

set_llm()
embed_model = get_embed_model()


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

# Indexオブジェクト生成
index = VectorStoreIndex.from_documents(
    documents,
    show_progress=True
)

# Retriever + SimpleChatEngine 構築
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
    embed_model=embed_model
)

response_synthesizer = get_response_synthesizer(
    # Response Mode	概要
    # ResponseMode.REFINE	各ノードを順次調べて答えを作成しながら洗練させていく
    # ResponseMode.COMPACT	実際はCompactAndRefineという処理であり、ほぼRefineと同じだが、チャンクをLLMのコンテキスト長を最大限利用するようにrepackするためより高速
    # ResponseMode.TREE_SUMMARIZE	抽出されたノードを使いサマライズを繰り返すことでチャンクを圧縮した後にクエリを行う。バージョンにより多少動作が変わる
    # ResponseMode.SIMPLE_SUMMARIZE	ノードを単純に結合してひとつのチャンクとしてクエリする。LLMコンテキスト長を超えるとエラーとなる
    # ResponseMode.GENERATION	ノード情報を使わず単にLLMに問い合わせる
    # ResponseMode.ACCUMULATE	各ノードに対するLLMの結果を計算して結果を連結する
    # ResponseMode.COMPACT_ACCUMULATE	ACCUMULATEとほぼ同じだが、チャンクをLLMのコンテキスト長を最大限利用するようにrepackするためより高速
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
