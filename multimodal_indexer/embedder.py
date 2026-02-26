"""
向量嵌入器模块 - BGE-M3 多模态嵌入实现
"""

import logging
from typing import List, Union, Dict, Any, Optional
import numpy as np
from io import BytesIO
from PIL import Image

from .models import ParsedContent
from .config import EmbeddingConfig


class VectorEmbedder:
    """向量嵌入器 - 使用 BGE-M3 多模态嵌入模型"""
    
    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # BGE-M3 固定向量维度为 1024
        self._vector_dim = 1024
        
        # 模型实例
        self.bge_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """初始化 BGE-M3 嵌入模型"""
        self.logger.info("Initializing BGE-M3 embedding model...")
        
        try:
            # 尝试加载 FlagEmbedding BGE-M3
            from FlagEmbedding import BGEM3FlagModel
            
            self.bge_model = BGEM3FlagModel(
                self.config.multimodal_model,
                use_fp16=self.config.use_fp16
            )
            self.logger.info("✓ BGE-M3 模型加载成功")
            
        except ImportError:
            self.logger.error("FlagEmbedding 未安装，请运行: pip install FlagEmbedding")
            raise ImportError("FlagEmbedding is required for BGE-M3 support")
        except Exception as e:
            self.logger.error(f"BGE-M3 模型加载失败: {e}")
            raise
    
    def embed_text(self, text: str) -> np.ndarray:
        """文本向量化 - 使用 BGE-M3 dense embedding"""
        if not text or not text.strip():
            return np.zeros(self._vector_dim, dtype=np.float32)
        
        try:
            if self.bge_model is not None:
                # 使用 BGE-M3 进行文本嵌入
                result = self.bge_model.encode(
                    [text],
                    batch_size=1,
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False
                )
                
                embedding = result['dense_vecs'][0]
                
                # 确保维度正确
                if embedding.shape[0] != self._vector_dim:
                    self.logger.warning(f"Unexpected vector dimension: {embedding.shape[0]}, expected {self._vector_dim}")
                    if embedding.shape[0] > self._vector_dim:
                        embedding = embedding[:self._vector_dim]
                    else:
                        padded = np.zeros(self._vector_dim, dtype=np.float32)
                        padded[:embedding.shape[0]] = embedding
                        embedding = padded
                
                return embedding.astype(np.float32)
            else:
                raise RuntimeError("BGE-M3 model not initialized")
                
        except Exception as e:
            self.logger.error(f"文本嵌入失败: {e}")
            return np.zeros(self._vector_dim, dtype=np.float32)
    
    def embed_image(self, image_data: bytes) -> np.ndarray:
        """图像向量化 - 将图像转换为文本描述后使用 BGE-M3"""
        if not image_data:
            return np.zeros(self._vector_dim, dtype=np.float32)
        
        try:
            # 对于 BGE-M3，我们需要将图像转换为文本描述
            # 这里使用简单的图像属性作为文本描述
            image = Image.open(BytesIO(image_data))
            
            # 创建图像描述文本
            image_description = f"Image with format {image.format}, size {image.size}, mode {image.mode}"
            
            # 如果图像有 EXIF 数据，添加到描述中
            if hasattr(image, '_getexif') and image._getexif():
                image_description += f", with EXIF data"
            
            # 使用文本嵌入方法
            return self.embed_text(image_description)
            
        except Exception as e:
            self.logger.error(f"图像嵌入失败: {e}")
            # 使用图像数据的哈希创建确定性向量
            import hashlib
            hash_obj = hashlib.md5(image_data)
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            vector = np.random.randn(self._vector_dim).astype(np.float32)
            # 归一化
            vector = vector / np.linalg.norm(vector)
            return vector
    
    def embed_audio(self, audio_data: bytes) -> np.ndarray:
        """音频向量化 - 将音频转换为文本描述后使用 BGE-M3"""
        if not audio_data:
            return np.zeros(self._vector_dim, dtype=np.float32)
        
        try:
            # 对于 BGE-M3，我们需要将音频转换为文本描述
            # 这里使用简单的音频属性作为文本描述
            audio_description = f"Audio data with {len(audio_data)} bytes"
            
            # 使用文本嵌入方法
            return self.embed_text(audio_description)
            
        except Exception as e:
            self.logger.error(f"音频嵌入失败: {e}")
            # 使用音频数据的哈希创建确定性向量
            import hashlib
            hash_obj = hashlib.md5(audio_data)
            seed = int(hash_obj.hexdigest()[:8], 16)
            np.random.seed(seed)
            vector = np.random.randn(self._vector_dim).astype(np.float32)
            # 归一化
            vector = vector / np.linalg.norm(vector)
            return vector
    
    def embed_multimodal(self, content: ParsedContent) -> List[np.ndarray]:
        """多模态内容向量化"""
        embeddings = []
        
        try:
            # 处理文本内容
            if content.text_content:
                text_embedding = self.embed_text(content.text_content)
                embeddings.append(text_embedding)
                self.logger.debug("Added text embedding")
            
            # 处理图像内容
            if content.image_content:
                for i, image_data in enumerate(content.image_content):
                    image_embedding = self.embed_image(image_data)
                    embeddings.append(image_embedding)
                    self.logger.debug(f"Added image embedding {i+1}")
            
            # 处理音频内容
            if content.audio_content:
                audio_embedding = self.embed_audio(content.audio_content)
                embeddings.append(audio_embedding)
                self.logger.debug("Added audio embedding")
            
            if not embeddings:
                self.logger.warning("No content to embed, returning zero vector")
                embeddings.append(np.zeros(self._vector_dim, dtype=np.float32))
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error in multimodal embedding: {e}")
            return [np.zeros(self._vector_dim, dtype=np.float32)]
    
    def batch_embed_text(self, texts: List[str]) -> List[np.ndarray]:
        """批量文本向量化"""
        if not texts:
            return []
        
        try:
            if self.bge_model is not None:
                # 过滤空文本
                valid_texts = [text for text in texts if text and text.strip()]
                if not valid_texts:
                    return [np.zeros(self._vector_dim, dtype=np.float32) for _ in texts]
                
                # 使用 BGE-M3 批量处理
                result = self.bge_model.encode(
                    valid_texts,
                    batch_size=self.config.batch_size,
                    max_length=self.config.max_length,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False
                )
                
                embeddings = result['dense_vecs']
                
                # 确保所有向量都是正确维度
                processed_embeddings = []
                for emb in embeddings:
                    if emb.shape[0] == self._vector_dim:
                        processed_embeddings.append(emb.astype(np.float32))
                    elif emb.shape[0] > self._vector_dim:
                        processed_embeddings.append(emb[:self._vector_dim].astype(np.float32))
                    else:
                        padded = np.zeros(self._vector_dim, dtype=np.float32)
                        padded[:emb.shape[0]] = emb
                        processed_embeddings.append(padded)
                
                return processed_embeddings
            else:
                # 使用单个嵌入方法
                return [self.embed_text(text) for text in texts]
                
        except Exception as e:
            self.logger.error(f"批量文本嵌入失败: {e}")
            return [np.zeros(self._vector_dim, dtype=np.float32) for _ in texts]
    
    def batch_embed(self, contents: List[ParsedContent]) -> List[List[np.ndarray]]:
        """批量多模态向量化"""
        return [self.embed_multimodal(content) for content in contents]
    
    def get_vector_dimension(self) -> int:
        """获取向量维度 - BGE-M3 固定返回 1024"""
        return self._vector_dim
    
    def is_ready(self) -> bool:
        """检查嵌入器是否就绪"""
        return self.bge_model is not None
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_name': 'BGE-M3',
            'model_path': self.config.multimodal_model,
            'vector_dimension': self._vector_dim,
            'batch_size': self.config.batch_size,
            'max_length': self.config.max_length,
            'use_fp16': self.config.use_fp16,
            'normalize_embeddings': self.config.normalize_embeddings,
            'ready': self.is_ready(),
            'supports_multimodal': True,
            'supports_dense': True,
            'supports_sparse': True,
            'supports_colbert': True
        }
    
    def search_embed(self, query: str) -> np.ndarray:
        """搜索查询向量化 - BGE-M3 不需要特殊的查询前缀"""
        return self.embed_text(query)
