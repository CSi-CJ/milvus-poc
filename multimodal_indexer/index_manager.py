"""
Milvus 索引管理器 - 重构版本，使用独立的CollectionManager
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
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
from .models import IndexRecord
from .collection_manager import CollectionManager


class IndexManager:
    """Milvus 索引管理器 - 重构版本"""
    
    def __init__(self, config: MilvusConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.connection_alias = "default"
        self.collections: Dict[str, Collection] = {}
        self.connected = False
        
        if connections is None:
            raise ImportError("pymilvus not installed. Please install with: pip install pymilvus")
        
        # 初始化collection管理器
        self.collection_manager = CollectionManager(config, self.connection_alias)
        
        # 连接到 Milvus
        self._connect()
    
    def _connect(self):
        """连接到 Milvus 集群"""
        try:
            self.logger.info(f"Connecting to Milvus at {self.config.host}:{self.config.port}")
            
            # 建立连接
            connections.connect(
                alias=self.connection_alias,
                host=self.config.host,
                port=self.config.port,
                user=self.config.user,
                password=self.config.password,
                db_name=self.config.database
            )
            
            # 测试连接
            if utility.has_collection(self.config.collection_name, using=self.connection_alias):
                self.logger.info("Connection test successful")
            
            self.connected = True
            self.logger.info("Successfully connected to Milvus")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Milvus: {e}")
            self.connected = False
            raise
    
    def _ensure_connected(self):
        """确保连接可用"""
        if not self.connected:
            self.logger.warning("Not connected to Milvus, attempting to reconnect...")
            self._connect()
    
    def create_collection(self, collection_name: Optional[str] = None, 
                         vector_dim: Optional[int] = None,
                         embedder: Optional['VectorEmbedder'] = None) -> Collection:
        """创建集合（使用CollectionManager）"""
        self._ensure_connected()
        
        if collection_name is None:
            collection_name = self.config.collection_name
        
        if vector_dim is None:
            if embedder is not None:
                vector_dim = embedder.get_vector_dimension()
                self.logger.info(f"Using actual vector dimension from embedder: {vector_dim}")
            else:
                vector_dim = self.config.vector_dim
                self.logger.info(f"Using configured vector dimension: {vector_dim}")
        
        # 检查是否需要升级架构
        schema_version = self.collection_manager.get_collection_schema_version(collection_name)
        if schema_version not in ["v2_enhanced", "none"]:
            self.logger.info(f"Upgrading collection schema from {schema_version} to v2_enhanced")
            self.collection_manager.upgrade_collection_schema(collection_name, vector_dim)
        
        collection = self.collection_manager.create_collection(collection_name, vector_dim)
        self.collections[collection_name] = collection
        return collection
    
    def drop_collection(self, collection_name: Optional[str] = None) -> bool:
        """删除集合（使用CollectionManager）"""
        self._ensure_connected()
        
        if collection_name is None:
            collection_name = self.config.collection_name
        
        # 从缓存中移除
        if collection_name in self.collections:
            del self.collections[collection_name]
        
        return self.collection_manager.drop_collection(collection_name)
    
    def recreate_collection(self, collection_name: Optional[str] = None,
                           vector_dim: Optional[int] = None,
                           embedder: Optional['VectorEmbedder'] = None) -> Collection:
        """重新创建集合（使用CollectionManager）"""
        if collection_name is None:
            collection_name = self.config.collection_name
        
        if vector_dim is None:
            if embedder is not None:
                vector_dim = embedder.get_vector_dimension()
            else:
                vector_dim = self.config.vector_dim
        
        # 从缓存中移除
        if collection_name in self.collections:
            del self.collections[collection_name]
        
        collection = self.collection_manager.recreate_collection(collection_name, vector_dim)
        self.collections[collection_name] = collection
        return collection

    def get_collection(self, collection_name: Optional[str] = None,
                      embedder: Optional['VectorEmbedder'] = None) -> Collection:
        """获取集合"""
        if collection_name is None:
            collection_name = self.config.collection_name
        
        if collection_name in self.collections:
            return self.collections[collection_name]
        
        # 尝试加载现有集合
        if utility.has_collection(collection_name, using=self.connection_alias):
            collection = Collection(collection_name, using=self.connection_alias)
            self.collections[collection_name] = collection
            return collection
        
        # 创建新集合
        return self.create_collection(collection_name, embedder=embedder)
    
    def search_vectors(self, query_vectors: List[List[float]], 
                      collection_name: Optional[str] = None,
                      top_k: int = 10, 
                      search_params: Optional[Dict] = None,
                      output_fields: Optional[List[str]] = None,
                      expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """搜索向量（增强版本，包含chunk内容）
        
        Args:
            query_vectors: 查询向量列表
            collection_name: 集合名称
            top_k: 返回结果数量
            search_params: 搜索参数
            output_fields: 输出字段
            expr: 过滤表达式
            
        Returns:
            List[Dict]: 搜索结果
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        # 确保集合已加载
        if not collection.has_index():
            self.logger.warning(f"Collection {collection.name} has no index, creating...")
            self.collection_manager.create_indexes(collection)
        
        collection.load()
        
        try:
            # 默认搜索参数
            if search_params is None:
                search_params = {
                    "metric_type": self.config.metric_type,
                    "params": {"ef": 200}  # HNSW 参数
                }
            
            # 增强的输出字段（包含chunk内容、图像数据和音频转录）
            if output_fields is None:
                output_fields = [
                    "file_path", "file_name", "file_type", 
                    "content_type", "chunk_index", 
                    "chunk_content", "chunk_summary", "content_length",
                    "image_data", "image_format", "image_size", "ocr_text",  # 图像字段
                    "audio_transcript", "audio_language", "audio_confidence",  # 音频转录字段
                    "metadata", "file_metadata", "content_metadata"
                ]
            
            # 执行搜索
            results = collection.search(
                data=query_vectors,
                anns_field="vector",
                param=search_params,
                limit=top_k,
                expr=expr,
                output_fields=output_fields
            )
            
            # 格式化结果
            formatted_results = self._format_search_results(results)
            
            self.logger.debug(f"Search returned {len(formatted_results)} results")
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Failed to search vectors: {e}")
            raise
    
    def _format_search_results(self, results) -> List[Dict[str, Any]]:
        """格式化搜索结果（增强版本）
        
        Args:
            results: Milvus 搜索结果
            
        Returns:
            List[Dict]: 格式化的结果
        """
        formatted_results = []
        
        for hits in results:
            for hit in hits:
                result = {
                    'id': hit.id,
                    'score': float(hit.score),
                    'distance': float(hit.distance) if hasattr(hit, 'distance') else float(hit.score)
                }
                
                # 添加实体字段
                if hasattr(hit, 'entity'):
                    entity = hit.entity
                    for field_name in entity.fields:
                        value = entity.get(field_name)
                        result[field_name] = value
                
                # 处理chunk内容显示
                if 'chunk_content' in result:
                    # 为长内容添加截断显示
                    content = result['chunk_content']
                    if isinstance(content, str) and len(content) > 500:
                        result['chunk_content_preview'] = content[:500] + "..."
                    else:
                        result['chunk_content_preview'] = content
                
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """获取集合统计信息（使用CollectionManager）
        
        Args:
            collection_name: 集合名称
            
        Returns:
            Dict: 统计信息
        """
        self._ensure_connected()
        
        if collection_name is None:
            collection_name = self.config.collection_name
        
        try:
            # 使用CollectionManager获取详细信息
            collection_info = self.collection_manager.get_collection_info(collection_name)
            
            if not collection_info.get("exists", False):
                return {"error": "Collection does not exist"}
            
            return collection_info
            
        except Exception as e:
            self.logger.error(f"Failed to get collection stats: {e}")
            return {"error": str(e)}
    
    def insert_vectors(self, data: List[Dict[str, Any]], 
                      collection_name: Optional[str] = None,
                      embedder: Optional['VectorEmbedder'] = None) -> List[str]:
        """插入向量数据
        
        Args:
            data: 要插入的数据列表
            collection_name: 集合名称
            embedder: 嵌入器实例，用于获取实际向量维度
            
        Returns:
            List[str]: 插入的记录 ID 列表
        """
        self._ensure_connected()
        
        if not data:
            return []
        
        collection = self.get_collection(collection_name, embedder)
        
        try:
            # 插入数据
            insert_result = collection.insert(data)
            
            # 刷新数据
            collection.flush()
            
            self.logger.info(f"Inserted {len(data)} records into {collection.name}")
            
            return insert_result.primary_keys
            
        except Exception as e:
            self.logger.error(f"Failed to insert vectors: {e}")
            raise
    
    def delete_by_ids(self, ids: List[str], 
                     collection_name: Optional[str] = None) -> int:
        """根据 ID 删除记录
        
        Args:
            ids: 要删除的 ID 列表
            collection_name: 集合名称
            
        Returns:
            int: 删除的记录数量
        """
        self._ensure_connected()
        
        if not ids:
            return 0
        
        collection = self.get_collection(collection_name)
        
        try:
            # 构建删除表达式
            id_list = "', '".join(ids)
            expr = f"id in ['{id_list}']"
            
            # 执行删除
            delete_result = collection.delete(expr)
            
            # 刷新数据
            collection.flush()
            
            deleted_count = delete_result.delete_count
            self.logger.info(f"Deleted {deleted_count} records from {collection.name}")
            
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete records: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查
        
        Returns:
            Dict: 健康状态信息
        """
        try:
            self._ensure_connected()
            
            # 检查连接状态
            collections_list = self.collection_manager.list_collections()
            
            return {
                'connected': True,
                'host': self.config.host,
                'port': self.config.port,
                'database': self.config.database,
                'collections': collections_list,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'connected': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def close(self):
        """关闭连接"""
        try:
            if self.connected:
                connections.disconnect(self.connection_alias)
                self.connected = False
                self.logger.info("Disconnected from Milvus")
        except Exception as e:
            self.logger.warning(f"Error closing connection: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()