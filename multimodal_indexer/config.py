"""
配置管理模块
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class MilvusConfig:
    """Milvus 配置"""
    host: str = "localhost"
    port: int = 19530
    user: str = ""
    password: str = ""
    database: str = "default"
    collection_name: str = "multimodal_files"
    vector_dim: int = 1024
    index_type: str = "HNSW"
    metric_type: str = "COSINE"
    index_params: Dict[str, Any] = field(default_factory=lambda: {"M": 16, "efConstruction": 200})


@dataclass
class EmbeddingConfig:
    """嵌入模型配置"""
    text_model: str = "BAAI/bge-m3"
    image_model: str = "BAAI/bge-m3"
    audio_model: str = "BAAI/bge-m3"
    multimodal_model: str = "BAAI/bge-m3"
    batch_size: int = 12
    max_length: int = 8192
    use_openai_api: bool = False
    openai_model: str = "text-embedding-3-small"
    use_fp16: bool = True
    normalize_embeddings: bool = True


@dataclass
class ProcessingConfig:
    """处理配置"""
    max_concurrent: int = 10
    chunk_size: int = 1000
    supported_extensions: list = field(default_factory=lambda: [
        '.pdf', '.txt', '.md', '.doc', '.docx',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp',
        '.mp3', '.wav', '.m4a',
        '.mp4', '.avi', '.mov'
    ])
    skip_existing: bool = True
    enable_ocr: bool = True
    enable_speech_recognition: bool = True


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class Config:
    """主配置类"""
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> "Config":
        """从配置文件加载"""
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        return cls.from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> "Config":
        """从字典创建配置"""
        config = cls()
        
        # 更新 Milvus 配置
        if 'milvus' in config_data:
            milvus_data = config_data['milvus']
            for key, value in milvus_data.items():
                if hasattr(config.milvus, key):
                    setattr(config.milvus, key, value)
        
        # 更新嵌入配置
        if 'embedding' in config_data:
            embedding_data = config_data['embedding']
            for key, value in embedding_data.items():
                if hasattr(config.embedding, key):
                    setattr(config.embedding, key, value)
        
        # 更新处理配置
        if 'processing' in config_data:
            processing_data = config_data['processing']
            for key, value in processing_data.items():
                if hasattr(config.processing, key):
                    setattr(config.processing, key, value)
        
        # 更新日志配置
        if 'logging' in config_data:
            logging_data = config_data['logging']
            for key, value in logging_data.items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)
        
        return config
    
    def apply_env_overrides(self):
        """应用环境变量覆盖"""
        # Milvus 配置
        if os.getenv('MILVUS_HOST'):
            self.milvus.host = os.getenv('MILVUS_HOST')
        if os.getenv('MILVUS_PORT'):
            self.milvus.port = int(os.getenv('MILVUS_PORT'))
        if os.getenv('MILVUS_USER'):
            self.milvus.user = os.getenv('MILVUS_USER')
        if os.getenv('MILVUS_PASSWORD'):
            self.milvus.password = os.getenv('MILVUS_PASSWORD')
        if os.getenv('MILVUS_DATABASE'):
            self.milvus.database = os.getenv('MILVUS_DATABASE')
        if os.getenv('MILVUS_COLLECTION'):
            self.milvus.collection_name = os.getenv('MILVUS_COLLECTION')
        
        # 处理配置
        if os.getenv('MAX_CONCURRENT'):
            self.processing.max_concurrent = int(os.getenv('MAX_CONCURRENT'))
        if os.getenv('BATCH_SIZE'):
            self.embedding.batch_size = int(os.getenv('BATCH_SIZE'))
        
        # 日志配置
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.logging.file_path = os.getenv('LOG_FILE')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'milvus': {
                'host': self.milvus.host,
                'port': self.milvus.port,
                'user': self.milvus.user,
                'password': self.milvus.password,
                'database': self.milvus.database,
                'collection_name': self.milvus.collection_name,
                'vector_dim': self.milvus.vector_dim,
                'index_type': self.milvus.index_type,
                'metric_type': self.milvus.metric_type,
                'index_params': self.milvus.index_params,
            },
            'embedding': {
                'text_model': self.embedding.text_model,
                'image_model': self.embedding.image_model,
                'audio_model': self.embedding.audio_model,
                'multimodal_model': self.embedding.multimodal_model,
                'batch_size': self.embedding.batch_size,
                'max_length': self.embedding.max_length,
            },
            'processing': {
                'max_concurrent': self.processing.max_concurrent,
                'chunk_size': self.processing.chunk_size,
                'supported_extensions': self.processing.supported_extensions,
                'skip_existing': self.processing.skip_existing,
                'enable_ocr': self.processing.enable_ocr,
                'enable_speech_recognition': self.processing.enable_speech_recognition,
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'file_path': self.logging.file_path,
                'max_file_size': self.logging.max_file_size,
                'backup_count': self.logging.backup_count,
            }
        }
    
    def save_to_file(self, config_path: str):
        """保存到配置文件"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


def setup_logging(config: LoggingConfig):
    """设置日志系统"""
    level = getattr(logging, config.level.upper())
    
    # 创建根日志记录器
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建格式化器
    formatter = logging.Formatter(config.format)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果配置了文件路径）
    if config.file_path:
        from logging.handlers import RotatingFileHandler
        os.makedirs(os.path.dirname(config.file_path), exist_ok=True)
        file_handler = RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)


def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置"""
    if config_path is None:
        config_path = os.getenv('CONFIG_PATH', 'config/config.json')
    
    config = Config.from_file(config_path)
    config.apply_env_overrides()
    
    # 设置日志
    setup_logging(config.logging)
    
    return config