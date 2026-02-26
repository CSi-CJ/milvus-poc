"""
文件解析器注册表
"""

from typing import List, Optional
import logging

from .base import BaseFileParser


class FileParserRegistry:
    """文件解析器注册表"""
    
    def __init__(self):
        self.parsers: List[BaseFileParser] = []
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def register(self, parser: BaseFileParser) -> None:
        """注册解析器
        
        Args:
            parser: 文件解析器实例
        """
        if not isinstance(parser, BaseFileParser):
            raise TypeError("Parser must be an instance of BaseFileParser")
        
        self.parsers.append(parser)
        self.logger.info(f"Registered parser: {parser.__class__.__name__}")
    
    def unregister(self, parser_class: type) -> bool:
        """注销解析器
        
        Args:
            parser_class: 解析器类
            
        Returns:
            bool: 是否成功注销
        """
        for i, parser in enumerate(self.parsers):
            if isinstance(parser, parser_class):
                removed_parser = self.parsers.pop(i)
                self.logger.info(f"Unregistered parser: {removed_parser.__class__.__name__}")
                return True
        return False
    
    def get_parser(self, file_path: str) -> Optional[BaseFileParser]:
        """获取适合的解析器
        
        Args:
            file_path: 文件路径
            
        Returns:
            Optional[BaseFileParser]: 适合的解析器，如果没有则返回 None
        """
        for parser in self.parsers:
            try:
                if parser.can_parse(file_path):
                    self.logger.debug(f"Found parser for {file_path}: {parser.__class__.__name__}")
                    return parser
            except Exception as e:
                self.logger.warning(f"Error checking parser {parser.__class__.__name__} for {file_path}: {e}")
                continue
        
        self.logger.warning(f"No parser found for file: {file_path}")
        return None
    
    def get_supported_extensions(self) -> List[str]:
        """获取所有支持的文件扩展名
        
        Returns:
            List[str]: 支持的文件扩展名列表
        """
        extensions = set()
        
        # 这里我们使用一些常见的测试文件来检查支持的扩展名
        test_files = [
            'test.pdf', 'test.txt', 'test.md', 'test.doc', 'test.docx',
            'test.png', 'test.jpg', 'test.jpeg', 'test.gif', 'test.bmp',
            'test.mp3', 'test.wav', 'test.m4a',
            'test.mp4', 'test.avi', 'test.mov'
        ]
        
        for test_file in test_files:
            for parser in self.parsers:
                try:
                    if parser.can_parse(test_file):
                        ext = test_file.split('.')[-1].lower()
                        extensions.add(f'.{ext}')
                        break
                except:
                    continue
        
        return sorted(list(extensions))
    
    def list_parsers(self) -> List[str]:
        """列出所有注册的解析器
        
        Returns:
            List[str]: 解析器类名列表
        """
        return [parser.__class__.__name__ for parser in self.parsers]
    
    def clear(self) -> None:
        """清空所有注册的解析器"""
        count = len(self.parsers)
        self.parsers.clear()
        self.logger.info(f"Cleared {count} parsers from registry")


# 创建全局注册表实例
default_registry = FileParserRegistry()