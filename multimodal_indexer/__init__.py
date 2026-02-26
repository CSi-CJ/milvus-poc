"""
多模态文件索引器

一个基于 Milvus 的多文件类型解析和向量索引系统。
"""

__version__ = "0.1.0"
__author__ = "Multimodal Indexer Team"

# 核心模型
from .models import ParsedContent, FileMetadata, ContentMetadata, IndexRecord

# 解析器
from .parsers import BaseFileParser, FileParserRegistry
from .parsers.factory import create_default_registry, get_parser_info

# 嵌入器
from .embedder import VectorEmbedder

# 索引管理器
from .index_manager import IndexManager

# 文件处理器
from .file_processor import FileProcessor, BatchProcessor

# 配置
from .config import Config, load_config

# 系统管理器
from .system import MultimodalIndexerSystem, create_system, create_system_sync

__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    
    # 核心模型
    "ParsedContent",
    "FileMetadata", 
    "ContentMetadata",
    "IndexRecord",
    
    # 解析器
    "BaseFileParser",
    "FileParserRegistry",
    "create_default_registry",
    "get_parser_info",
    
    # 嵌入器
    "VectorEmbedder",
    
    # 索引管理器
    "IndexManager",
    
    # 文件处理器
    "FileProcessor",
    "BatchProcessor",
    
    # 配置
    "Config",
    "load_config",
    
    # 系统管理器
    "MultimodalIndexerSystem",
    "create_system",
    "create_system_sync",
]