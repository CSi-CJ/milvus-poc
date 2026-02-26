"""
系统工厂和管理器
"""

import logging
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from .config import Config, load_config
from .parsers.factory import create_default_registry
from .embedder import VectorEmbedder
from .index_manager import IndexManager
from .file_processor import FileProcessor, BatchProcessor


class MultimodalIndexerSystem:
    """多模态索引器系统管理器"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or load_config()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 组件
        self.parser_registry = None
        self.embedder = None
        self.index_manager = None
        self.file_processor = None
        self.batch_processor = None
        
        # 初始化状态
        self.initialized = False
    
    def initialize(self):
        """初始化系统"""
        if self.initialized:
            return
        
        self.logger.info("Initializing multimodal indexer system...")
        
        try:
            # 1. 创建解析器注册表
            self.logger.info("Creating parser registry...")
            self.parser_registry = create_default_registry(self.config.processing.__dict__)
            self.logger.info(f"Registered {len(self.parser_registry.parsers)} parsers")
            
            # 2. 创建向量嵌入器
            self.logger.info("Initializing vector embedder...")
            self.embedder = VectorEmbedder(self.config.embedding)
            if not self.embedder.is_ready():
                self.logger.warning("Vector embedder not ready - some models may be missing")
            
            # 3. 创建索引管理器
            self.logger.info("Connecting to Milvus...")
            self.index_manager = IndexManager(self.config.milvus)
            
            # 4. 创建文件处理器
            self.logger.info("Creating file processors...")
            self.file_processor = FileProcessor(
                self.parser_registry, 
                self.embedder, 
                self.index_manager, 
                self.config
            )
            
            self.batch_processor = BatchProcessor(
                self.file_processor, 
                self.config.processing.max_concurrent
            )
            
            self.initialized = True
            self.logger.info("System initialization completed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            self.cleanup()
            raise
    
    def cleanup(self):
        """清理资源"""
        if self.index_manager:
            try:
                self.index_manager.close()
            except Exception as e:
                self.logger.warning(f"Error closing index manager: {e}")
        
        self.initialized = False
        self.logger.info("System cleanup completed")
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'initialized': self.initialized,
            'config': {
                'milvus_host': self.config.milvus.host,
                'milvus_port': self.config.milvus.port,
                'collection_name': self.config.milvus.collection_name,
                'vector_dim': self.config.milvus.vector_dim,
                'max_concurrent': self.config.processing.max_concurrent,
            }
        }
        
        if self.initialized:
            # 解析器信息
            info['parsers'] = {
                'count': len(self.parser_registry.parsers),
                'types': self.parser_registry.list_parsers(),
                'supported_extensions': self.parser_registry.get_supported_extensions()
            }
            
            # 嵌入器信息
            info['embedder'] = self.embedder.get_model_info()
            
            # 索引管理器信息
            try:
                health = self.index_manager.health_check()
                info['milvus'] = health
            except Exception as e:
                info['milvus'] = {'error': str(e)}
        
        return info
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


@asynccontextmanager
async def create_system(config: Optional[Config] = None):
    """异步上下文管理器创建系统
    
    Args:
        config: 配置对象
        
    Yields:
        MultimodalIndexerSystem: 初始化的系统实例
    """
    system = MultimodalIndexerSystem(config)
    try:
        system.initialize()
        yield system
    finally:
        system.cleanup()


def create_system_sync(config: Optional[Config] = None) -> MultimodalIndexerSystem:
    """同步创建系统
    
    Args:
        config: 配置对象
        
    Returns:
        MultimodalIndexerSystem: 系统实例（需要手动调用 initialize）
    """
    return MultimodalIndexerSystem(config)