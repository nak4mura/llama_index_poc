from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def set_llm():
    """
    共通の LlamaCPP インスタンスを生成し、Settings.llm に設定する。
    """
    Settings.llm = LlamaCPP(
        model_path="./models/Llama-3-ELYZA-JP-8B-q4_k_m.gguf",
        temperature=0.7,
        max_new_tokens=512,
    )

def set_embed_model():
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="cl-nagoya/sup-simcse-ja-base"
    )

def get_embed_model():
    embed_model = HuggingFaceEmbedding(
        model_name="cl-nagoya/sup-simcse-ja-base"
    )
    Settings.embed_model = embed_model
    return embed_model
