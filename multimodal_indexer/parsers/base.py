"""
文件解析器基类
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from ..models import ParsedContent


class BaseFileParser(ABC):
    """文件解析器基类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """检查是否能解析指定文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 是否支持解析该文件
        """
        pass
    
    @abstractmethod
    def parse(self, file_path: str) -> ParsedContent:
        """解析文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            ParsedContent: 解析后的内容
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 文件格式不支持或损坏
            Exception: 其他解析错误
        """
        pass
    
    def _validate_file(self, file_path: str) -> None:
        """验证文件是否存在且可读
        
        Args:
            file_path: 文件路径
            
        Raises:
            FileNotFoundError: 文件不存在
            PermissionError: 文件无法读取
        """
        import os
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise PermissionError(f"File is not readable: {file_path}")
    
    def _get_file_extension(self, file_path: str) -> str:
        """获取文件扩展名
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 小写的文件扩展名
        """
        import os
        return os.path.splitext(file_path)[1].lower()
    
    def _create_error_content(self, file_path: str, error_message: str) -> ParsedContent:
        """创建错误内容对象
        
        Args:
            file_path: 文件路径
            error_message: 错误信息
            
        Returns:
            ParsedContent: 包含错误信息的内容对象
        """
        return ParsedContent(
            text_content=None,
            image_content=None,
            audio_content=None,
            metadata={
                'error': error_message,
                'file_path': file_path,
                'parser': self.__class__.__name__
            },
            file_type=self._get_file_extension(file_path)
        )