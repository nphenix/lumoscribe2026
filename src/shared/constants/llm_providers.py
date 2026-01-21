"""LLM Provider 相关常量。

用于种子化数据库中的 LLM Provider 配置。
"""

# Provider 类型的“推荐配置项”（供中台/文档提示使用）
# 说明：实际存储时建议使用下方的 `key`（小写）写入 provider.config_json
PROVIDER_TYPE_CONFIG_KEYS = {
    "openai_compatible": [
        {"key": "model", "display": "LLM_MODEL"},
        {"key": "temperature", "display": "LLM_TEMPERATURE"},
        {"key": "max_tokens", "display": "LLM_MAX_TOKENS"},
        {"key": "timeout_seconds", "display": "LLM_TIMEOUT_SECONDS"},
        {"key": "stream", "display": "LLM_STREAM"},
    ],
    "ollama": [
        {"key": "ollama_model", "display": "OLLAMA_MODEL"},
        {"key": "max_tokens", "display": "LLM_MAX_TOKENS"},
    ],
    "flagembedding": [
        # 本地模式（默认）
        {"key": "device", "display": "FLAGEMBEDDING_DEVICE"},
        {"key": "use_fp16", "display": "FLAGEMBEDDING_USE_FP16"},
        {"key": "trust_remote_code", "display": "FLAGEMBEDDING_TRUST_REMOTE_CODE"},
        {"key": "batch_size", "display": "FLAGEMBEDDING_BATCH_SIZE"},
        {"key": "max_tokens", "display": "LLM_MAX_TOKENS"},
        # Embedding 模型配置
        {"key": "embedding_model_path", "display": "EMBEDDING_MODEL_PATH"},
        {"key": "embedding_dimension", "display": "EMBEDDING_DIMENSION"},
        # Rerank 模型配置
        {"key": "rerank_model_path", "display": "RERANK_MODEL_PATH"},
        {"key": "rerank_top_k", "display": "RERANK_TOP_K"},
        # 远程模式（可选，默认不使用）
        {"key": "remote", "display": "FLAGEMBEDDING_REMOTE"},
        {"key": "host", "display": "FLAGEMBEDDING_HOST"},
    ],
}

# Provider 定义（key 必须是唯一标识符）
DEFAULT_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "provider_type": "openai_compatible",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "enabled": True,
        "description": "OpenAI 官方 API（支持 chat、multimodal）",
    },
    "ollama": {
        "name": "Ollama",
        "provider_type": "ollama",
        "api_key_env": None,
        "base_url": "http://localhost:11434",
        "enabled": True,
        "description": "本地 Ollama 服务（支持 chat / embedding / rerank，按模型能力而定）",
    },
    "qwen": {
        "name": "通义千问",
        "provider_type": "openai_compatible",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "enabled": True,
        "description": "阿里云通义千问（支持 multimodal）",
    },
    "mineru": {
        "name": "MinerU",
        "provider_type": "mineru",
        "api_key_env": None,
        "base_url": None,
        "enabled": True,
        "description": "MinerU PDF 解析服务",
    },
}
