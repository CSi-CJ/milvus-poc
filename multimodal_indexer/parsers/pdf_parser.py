"""
PDF 文件解析器
"""

import os
from typing import List, Dict, Any
import logging

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from ..models import ParsedContent
from .base import BaseFileParser


class PDFParser(BaseFileParser):
    """PDF 文件解析器"""
    
    def __init__(self):
        super().__init__()
        if fitz is None:
            self.logger.warning("PyMuPDF not installed. PDF parsing will be limited.")
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否能解析 PDF 文件"""
        return file_path.lower().endswith('.pdf')
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析 PDF 文件"""
        self._validate_file(file_path)
        
        if fitz is None:
            return self._create_error_content(
                file_path, 
                "PyMuPDF not installed. Please install with: pip install PyMuPDF"
            )
        
        try:
            # 打开 PDF 文档
            doc = fitz.open(file_path)
            
            # 提取文本内容
            text_content = ""
            images = []
            metadata = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
            }
            
            # 逐页处理
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # 提取文本
                page_text = page.get_text()
                if page_text.strip():
                    text_content += f"\n--- Page {page_num + 1} ---\n"
                    text_content += page_text
                    
                    # 为包含文本的页面生成截图
                    try:
                        # 生成高质量页面截图 (2x缩放以提高清晰度)
                        mat = fitz.Matrix(2, 2)
                        page_pix = page.get_pixmap(matrix=mat)
                        page_screenshot = page_pix.tobytes("png")
                        images.append(page_screenshot)
                        
                        self.logger.debug(f"Generated page screenshot for page {page_num + 1}: {page_pix.width}x{page_pix.height}")
                        page_pix = None
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to generate page screenshot for page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
            # 清理文本内容
            text_content = text_content.strip()
            if not text_content:
                text_content = None
            
            return ParsedContent(
                text_content=text_content,
                image_content=images if images else None,
                audio_content=None,
                metadata=metadata,
                file_type='.pdf'
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _is_meaningful_image(self, pix) -> bool:
        """检查图像是否有意义（不是纯黑色或装饰性图形）
        
        Args:
            pix: PyMuPDF Pixmap对象
            
        Returns:
            bool: 是否是有意义的图像
        """
        try:
            # 获取图像尺寸
            width, height = pix.width, pix.height
            
            # 过滤太小的图像（装饰性图标）
            if width < 100 or height < 100:
                return False
            
            # 过滤长宽比异常的图像（可能是分隔线）
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 10:  # 长宽比超过10:1的图像
                return False
            
            # 采样检查像素颜色多样性
            sample_size = min(100, width * height // 10)  # 采样10%的像素，最多100个
            
            if sample_size < 10:
                return False
            
            # 获取像素数据进行分析
            try:
                # 转换为RGB模式进行分析
                if pix.n == 1:  # 灰度图
                    # 对于灰度图，检查是否有足够的灰度变化
                    samples = pix.samples
                    if len(samples) < sample_size:
                        return False
                    
                    # 检查前sample_size个像素的灰度值
                    gray_values = set()
                    for i in range(0, min(sample_size, len(samples))):
                        gray_values.add(samples[i])
                    
                    # 如果灰度值种类太少，可能是单色图像
                    if len(gray_values) < 3:
                        return False
                        
                elif pix.n >= 3:  # RGB或RGBA
                    samples = pix.samples
                    if len(samples) < sample_size * 3:
                        return False
                    
                    # 检查颜色多样性
                    colors = set()
                    for i in range(0, min(sample_size * 3, len(samples)), 3):
                        if i + 2 < len(samples):
                            r, g, b = samples[i], samples[i+1], samples[i+2]
                            colors.add((r, g, b))
                    
                    # 如果颜色种类太少，可能是单色图像
                    if len(colors) < 5:
                        return False
                    
                    # 检查是否主要是黑色
                    black_pixels = sum(1 for color in colors if color == (0, 0, 0))
                    if black_pixels > len(colors) * 0.8:  # 80%以上是黑色
                        return False
                
                return True
                
            except Exception as e:
                self.logger.debug(f"Error analyzing image pixels: {e}")
                # 如果无法分析像素，基于尺寸判断
                return width >= 200 and height >= 200
                
        except Exception as e:
            self.logger.debug(f"Error checking image meaningfulness: {e}")
            return True  # 默认保留图像