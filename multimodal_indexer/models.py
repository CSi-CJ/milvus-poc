"""
核心数据模型定义
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib
import os


@dataclass
class ParsedContent:
    """解析后的内容结构"""
    text_content: Optional[str] = None
    image_content: Optional[List[bytes]] = None
    audio_content: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_type: str = ""
    
    def has_content(self) -> bool:
        """检查是否包含任何内容"""
        return bool(
            self.text_content or 
            self.image_content or 
            self.audio_content
        )


@dataclass
class FileMetadata:
    """文件元数据"""
    file_path: str
    file_name: str
    file_size: int
    file_type: str
    mime_type: str
    created_time: datetime
    modified_time: datetime
    checksum: str
    
    @classmethod
    def from_file_path(cls, file_path: str) -> "FileMetadata":
        """从文件路径创建元数据"""
        stat = os.stat(file_path)
        
        # 计算文件校验和
        with open(file_path, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # 获取文件类型
        file_ext = os.path.splitext(file_path)[1].lower()
        mime_type_map = {
            '.pdf': 'application/pdf',
            '.txt': 'text/plain',
            '.md': 'text/markdown',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.gif': 'image/gif',
            '.bmp': 'image/bmp',
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/mp4',
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
        }
        
        return cls(
            file_path=file_path,
            file_name=os.path.basename(file_path),
            file_size=stat.st_size,
            file_type=file_ext,
            mime_type=mime_type_map.get(file_ext, 'application/octet-stream'),
            created_time=datetime.fromtimestamp(stat.st_ctime),
            modified_time=datetime.fromtimestamp(stat.st_mtime),
            checksum=checksum
        )


@dataclass
class ContentMetadata:
    """内容元数据"""
    content_type: str  # text, image, audio, video
    content_length: int
    language: Optional[str] = None
    encoding: Optional[str] = None
    dimensions: Optional[Tuple[int, int]] = None  # for images/videos
    duration: Optional[float] = None  # for audio/videos


@dataclass
class ChunkContent:
    """Chunk内容数据（增强版，支持图像和音频）"""
    content: str  # 实际的chunk内容
    content_type: str  # text, image_description, audio_transcript等
    summary: str = ""  # 内容摘要
    
    # 图像相关字段
    image_data: Optional[str] = None  # Base64编码的图像数据
    image_format: Optional[str] = None  # 图像格式（PNG, JPG等）
    image_size: Optional[str] = None  # 图像尺寸（如"800x600"）
    ocr_text: Optional[str] = None  # OCR提取的文本
    
    # 音频相关字段
    audio_transcript: Optional[str] = None  # 语音识别转录文本
    audio_language: Optional[str] = None  # 识别的语言
    audio_confidence: Optional[float] = None  # 识别置信度
    
    def get_display_content(self, max_length: int = 500) -> str:
        """获取用于显示的内容（截断长内容）"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."
    
    def get_summary_or_content(self, max_length: int = 200) -> str:
        """获取摘要或截断的内容"""
        if self.summary:
            return self.summary[:max_length]
        return self.get_display_content(max_length)
    
    def has_image(self) -> bool:
        """检查是否包含图像数据"""
        return bool(self.image_data)
    
    def get_image_info(self) -> Dict[str, Any]:
        """获取图像信息"""
        return {
            'has_image': self.has_image(),
            'format': self.image_format,
            'size': self.image_size,
            'ocr_text': self.ocr_text,
            'data_size': len(self.image_data) if self.image_data else 0
        }
    
    def has_audio_transcript(self) -> bool:
        """检查是否包含音频转录文本"""
        return bool(self.audio_transcript)
    
    def get_audio_info(self) -> Dict[str, Any]:
        """获取音频信息"""
        return {
            'has_transcript': self.has_audio_transcript(),
            'transcript': self.audio_transcript,
            'language': self.audio_language,
            'confidence': self.audio_confidence
        }


@dataclass
class IndexRecord:
    """增强的索引记录"""
    id: str
    file_metadata: FileMetadata
    content_metadata: ContentMetadata
    vector_embedding: List[float]
    chunk_content: ChunkContent  # 新增：chunk内容
    chunk_index: int = 0
    parent_id: Optional[str] = None
    
    def to_milvus_data(self) -> Dict[str, Any]:
        """转换为增强的 Milvus 插入格式（支持图像和音频数据）"""
        return {
            'id': self.id,
            'file_path': self.file_metadata.file_path,
            'file_name': self.file_metadata.file_name,
            'file_type': self.file_metadata.file_type,
            'content_type': self.content_metadata.content_type,
            'chunk_index': self.chunk_index,
            
            # chunk内容字段
            'chunk_content': self.chunk_content.content,
            'chunk_summary': self.chunk_content.summary,
            'content_length': len(self.chunk_content.content),
            
            # 图像数据字段
            'image_data': self.chunk_content.image_data or "",
            'image_format': self.chunk_content.image_format or "",
            'image_size': self.chunk_content.image_size or "",
            'ocr_text': self.chunk_content.ocr_text or "",
            
            # 音频数据字段
            'audio_transcript': self.chunk_content.audio_transcript or "",
            'audio_language': self.chunk_content.audio_language or "",
            'audio_confidence': self.chunk_content.audio_confidence or 0.0,
            
            'vector': self.vector_embedding,
            
            # 分离的元数据结构
            'metadata': {
                'parent_id': self.parent_id,
                'content_type_detail': self.chunk_content.content_type,
                'has_image': self.chunk_content.has_image(),
                'has_audio_transcript': self.chunk_content.has_audio_transcript(),
            },
            'file_metadata': {
                'file_size': self.file_metadata.file_size,
                'mime_type': self.file_metadata.mime_type,
                'checksum': self.file_metadata.checksum,
                'created_time': self.file_metadata.created_time.isoformat(),
                'modified_time': self.file_metadata.modified_time.isoformat(),
            },
            'content_metadata': {
                'content_length': self.content_metadata.content_length,
                'language': self.content_metadata.language,
                'encoding': self.content_metadata.encoding,
                'dimensions': self.content_metadata.dimensions,
                'duration': self.content_metadata.duration,
            },
            
            'created_at': int(datetime.now().timestamp() * 1000),
            'updated_at': int(datetime.now().timestamp() * 1000)
        }