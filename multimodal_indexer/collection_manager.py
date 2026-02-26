"""
Milvus Collection 管理器 - 专门负责集合的创建、更新和删除
"""

import logging
from typing import Dict, Any, Optional, List, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from .embedder import VectorEmbedder

try:
    from pymilvus import (
        connections, Collection, CollectionSchema, FieldSchema, DataType,
        utility, MilvusException
    )
except ImportError:
    connections = None
    Collection = None
    CollectionSchema = None
    FieldSchema = None
    DataType = None
    utility = None
    MilvusException = Exception

from .config import MilvusConfig


class CollectionManager:
    """Milvus Collection 管理器"""
    
    def __init__(self, config: MilvusConfig, connection_alias: str = "default"):
        self.config = config
        self.connection_alias = connection_alias
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if connections is None:
            raise ImportError("pymilvus not installed. Please install with: pip install pymilvus")
    
    def create_collection_schema(self, vector_dim: int) -> CollectionSchema:
        """创建增强的集合架构，包含chunk内容存储、图像数据和音频转录
        
        Args:
            vector_dim: 向量维度
            
        Returns:
            CollectionSchema: 集合架构
        """
        fields = [
            # 主键和基本信息
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=100, is_primary=True),
            FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),
            FieldSchema(name="file_type", dtype=DataType.VARCHAR, max_length=50),
            
            # 内容信息
            FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_index", dtype=DataType.INT32),
            
            # chunk内容存储
            FieldSchema(name="chunk_content", dtype=DataType.VARCHAR, max_length=65535),  # 存储实际的chunk内容
            FieldSchema(name="chunk_summary", dtype=DataType.VARCHAR, max_length=1000),   # 内容摘要
            FieldSchema(name="content_length", dtype=DataType.INT32),                     # 内容长度
            
            # 图像数据存储
            FieldSchema(name="image_data", dtype=DataType.VARCHAR, max_length=65535),    # Base64编码的图像数据（65KB限制，Milvus最大限制）
            FieldSchema(name="image_format", dtype=DataType.VARCHAR, max_length=20),      # 图像格式（PNG, JPG等）
            FieldSchema(name="image_size", dtype=DataType.VARCHAR, max_length=50),        # 图像尺寸（如"800x600"）
            FieldSchema(name="ocr_text", dtype=DataType.VARCHAR, max_length=10000),       # OCR提取的文本
            
            # 音频数据存储
            FieldSchema(name="audio_transcript", dtype=DataType.VARCHAR, max_length=10000),  # 语音识别转录文本
            FieldSchema(name="audio_language", dtype=DataType.VARCHAR, max_length=20),       # 识别的语言
            FieldSchema(name="audio_confidence", dtype=DataType.FLOAT),                      # 识别置信度
            
            # 向量嵌入
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            
            # 增强的元数据
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="file_metadata", dtype=DataType.JSON),  # 文件级元数据
            FieldSchema(name="content_metadata", dtype=DataType.JSON),  # 内容级元数据
            
            # 时间戳
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64)
        ]
        
        return CollectionSchema(fields, "Enhanced multimodal file index collection with image and audio support (v5)")
    
    def create_collection(self, collection_name: str, vector_dim: int,
                         drop_if_exists: bool = False) -> Collection:
        """创建集合
        
        Args:
            collection_name: 集合名称
            vector_dim: 向量维度
            drop_if_exists: 如果存在是否删除重建
            
        Returns:
            Collection: 创建的集合对象
        """
        try:
            # 检查集合是否已存在
            if utility.has_collection(collection_name, using=self.connection_alias):
                if drop_if_exists:
                    self.logger.info(f"Dropping existing collection: {collection_name}")
                    self.drop_collection(collection_name)
                else:
                    self.logger.info(f"Collection {collection_name} already exists")
                    collection = Collection(collection_name, using=self.connection_alias)
                    return collection
            
            # 创建集合架构
            schema = self.create_collection_schema(vector_dim)
            
            # 创建集合
            collection = Collection(
                name=collection_name,
                schema=schema,
                using=self.connection_alias
            )
            
            # 创建索引
            self.create_indexes(collection)
            
            # 加载集合
            collection.load()
            
            self.logger.info(f"Successfully created collection: {collection_name}")
            return collection
            
        except Exception as e:
            self.logger.error(f"Failed to create collection {collection_name}: {e}")
            raise
    
    def create_indexes(self, collection: Collection):
        """创建索引
        
        Args:
            collection: 集合对象
        """
        try:
            # 创建向量索引
            vector_index_params = {
                "metric_type": self.config.metric_type,
                "index_type": self.config.index_type,
                "params": self.config.index_params
            }
            
            collection.create_index(
                field_name="vector",
                index_params=vector_index_params
            )
            
            # 不创建标量字段索引，避免索引名称冲突
            # Milvus会自动为主键创建索引
            
            self.logger.info(f"Created vector index for collection {collection.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create indexes: {e}")
            raise
    
    def drop_collection(self, collection_name: str) -> bool:
        """删除集合（先卸载再删除）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            bool: 是否成功删除
        """
        try:
            if not utility.has_collection(collection_name, using=self.connection_alias):
                self.logger.info(f"Collection {collection_name} does not exist")
                return True
            
            # 获取集合对象
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 强制卸载集合
            self._ensure_collection_unloaded(collection)
            
            # 删除集合
            utility.drop_collection(collection_name, using=self.connection_alias)
            self.logger.info(f"Successfully dropped collection: {collection_name}")
            return True
                
        except Exception as e:
            self.logger.error(f"Failed to drop collection {collection_name}: {e}")
            return False
    
    def _ensure_collection_unloaded(self, collection: Collection) -> bool:
        """确保集合已卸载
        
        Args:
            collection: 集合对象
            
        Returns:
            bool: 是否成功卸载
        """
        try:
            collection_name = collection.name
            
            # 检查加载状态
            load_state = utility.load_state(collection_name, using=self.connection_alias)
            self.logger.debug(f"Collection {collection_name} load state: {load_state}")
            
            # 多种方式检查是否已加载
            is_loaded = False
            if hasattr(load_state, 'state'):
                is_loaded = load_state.state.name == "Loaded"
            elif hasattr(load_state, 'name'):
                is_loaded = load_state.name == "Loaded"
            else:
                # 字符串形式检查
                is_loaded = str(load_state) == "Loaded"
            
            if is_loaded:
                self.logger.info(f"Unloading collection: {collection_name}")
                collection.release()
                
                # 等待卸载完成
                import time
                max_wait = 10  # 最多等待10秒
                wait_time = 0
                while wait_time < max_wait:
                    time.sleep(0.5)
                    wait_time += 0.5
                    
                    current_state = utility.load_state(collection_name, using=self.connection_alias)
                    if str(current_state) != "Loaded":
                        self.logger.info(f"Collection {collection_name} successfully unloaded")
                        return True
                
                self.logger.warning(f"Collection {collection_name} unload timeout after {max_wait}s")
                return False
            else:
                self.logger.debug(f"Collection {collection_name} is not loaded, no need to unload")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to unload collection {collection.name}: {e}")
            return False
    
    def recreate_collection(self, collection_name: str, vector_dim: int) -> Collection:
        """重新创建集合（删除现有集合并创建新的）
        
        Args:
            collection_name: 集合名称
            vector_dim: 向量维度
            
        Returns:
            Collection: 创建的集合对象
        """
        return self.create_collection(collection_name, vector_dim, drop_if_exists=True)
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """获取集合详细信息
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict: 集合信息
        """
        try:
            if not utility.has_collection(collection_name, using=self.connection_alias):
                return {"exists": False}
            
            collection = Collection(collection_name, using=self.connection_alias)
            
            # 获取集合统计信息
            stats = {
                "exists": True,
                "name": collection.name,
                "num_entities": collection.num_entities,
                "has_index": collection.has_index(),
                "is_loaded": utility.load_state(collection.name, using=self.connection_alias),
                "description": collection.description,
                "schema": {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dtype),
                            "is_primary": field.is_primary,
                            "max_length": getattr(field, 'max_length', None),
                            "dim": getattr(field, 'dim', None)
                        }
                        for field in collection.schema.fields
                    ]
                },
                "indexes": []
            }
            
            # 获取索引信息
            try:
                indexes = collection.indexes
                for index in indexes:
                    stats["indexes"].append({
                        "field_name": index.field_name,
                        "index_name": index.index_name,
                        "params": index.params
                    })
            except Exception as e:
                self.logger.debug(f"Could not get index info: {e}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return {"exists": False, "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """列出所有集合
        
        Returns:
            List[str]: 集合名称列表
        """
        try:
            return utility.list_collections(using=self.connection_alias)
        except Exception as e:
            self.logger.error(f"Failed to list collections: {e}")
            return []
    
    def collection_exists(self, collection_name: str) -> bool:
        """检查集合是否存在
        
        Args:
            collection_name: 集合名称
            
        Returns:
            bool: 是否存在
        """
        try:
            return utility.has_collection(collection_name, using=self.connection_alias)
        except Exception as e:
            self.logger.error(f"Failed to check collection existence: {e}")
            return False
    
    def get_collection_schema_version(self, collection_name: str) -> str:
        """获取集合架构版本（用于检查是否需要升级）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            str: 架构版本标识
        """
        try:
            if not self.collection_exists(collection_name):
                return "none"
            
            collection = Collection(collection_name, using=self.connection_alias)
            field_names = [field.name for field in collection.schema.fields]
            
            # 检查是否包含音频转录字段（v5版本 - 音频支持）
            audio_fields = ["audio_transcript", "audio_language", "audio_confidence"]
            if all(field in field_names for field in audio_fields):
                return "v5_audio_support"
            
            # 检查是否包含图像存储字段（v4版本 - 高质量）
            image_fields = ["image_data", "image_format", "image_size", "ocr_text"]
            if all(field in field_names for field in image_fields):
                return "v4_high_quality"
            
            # 检查是否包含chunk内容字段（v2版本）
            chunk_fields = ["chunk_content", "chunk_summary", "content_length", 
                           "file_metadata", "content_metadata"]
            if all(field in field_names for field in chunk_fields):
                return "v2_enhanced"
            
            # 基础版本
            return "v1_basic"
                
        except Exception as e:
            self.logger.error(f"Failed to get schema version: {e}")
            return "unknown"
    
    def upgrade_collection_schema(self, collection_name: str, vector_dim: int,
                                backup_data: bool = True) -> bool:
        """升级集合架构到最新版本
        
        Args:
            collection_name: 集合名称
            vector_dim: 向量维度
            backup_data: 是否备份数据
            
        Returns:
            bool: 是否成功升级
        """
        try:
            current_version = self.get_collection_schema_version(collection_name)
            
            if current_version == "v5_audio_support":
                self.logger.info(f"Collection {collection_name} already has latest schema")
                return True
            
            if current_version == "none":
                self.logger.info(f"Collection {collection_name} does not exist, creating new one")
                self.create_collection(collection_name, vector_dim)
                return True
            
            # 需要升级架构
            self.logger.info(f"Upgrading collection {collection_name} from {current_version} to v5_audio_support")
            
            # TODO: 实现数据迁移逻辑
            # 目前简单地重新创建集合
            self.logger.warning("Schema upgrade requires recreating collection - existing data will be lost")
            self.recreate_collection(collection_name, vector_dim)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upgrade collection schema: {e}")
            return False