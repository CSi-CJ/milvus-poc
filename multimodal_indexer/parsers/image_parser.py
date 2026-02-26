"""
图像文件解析器
"""

import os
from typing import Dict, Any, Optional
import logging

try:
    from PIL import Image, ExifTags
except ImportError:
    Image = None
    ExifTags = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from ..models import ParsedContent
from .base import BaseFileParser


class ImageParser(BaseFileParser):
    """图像文件解析器"""
    
    def __init__(self, enable_ocr: bool = True):
        super().__init__()
        self.enable_ocr = enable_ocr
        self.supported_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
        
        if Image is None:
            self.logger.warning("Pillow not installed. Image parsing will be disabled.")
        
        if enable_ocr and pytesseract is None:
            self.logger.warning("pytesseract not installed. OCR will be disabled.")
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否能解析图像文件"""
        ext = self._get_file_extension(file_path)
        return ext in self.supported_extensions
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析图像文件"""
        self._validate_file(file_path)
        
        if Image is None:
            return self._create_error_content(
                file_path,
                "Pillow not installed. Please install with: pip install Pillow"
            )
        
        try:
            # 打开图像
            with Image.open(file_path) as img:
                # 获取基本信息
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                }
                
                # 提取 EXIF 数据
                exif_data = self._extract_exif(img)
                if exif_data:
                    metadata['exif'] = exif_data
                
                # 转换为字节数据
                img_bytes = self._image_to_bytes(img)
                
                # OCR 文本提取
                text_content = None
                if self.enable_ocr and pytesseract is not None:
                    text_content = self._extract_text_ocr(img)
                
                return ParsedContent(
                    text_content=text_content,
                    image_content=[img_bytes] if img_bytes else None,
                    audio_content=None,
                    metadata=metadata,
                    file_type=self._get_file_extension(file_path)
                )
                
        except Exception as e:
            self.logger.error(f"Error parsing image {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_exif(self, img) -> Optional[Dict[str, Any]]:
        """提取 EXIF 数据"""
        if Image is None:
            return None
            
        try:
            exif_dict = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                exif = img._getexif()
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
            
            return exif_dict if exif_dict else None
            
        except Exception as e:
            self.logger.debug(f"Failed to extract EXIF data: {e}")
            return None
    
    def _image_to_bytes(self, img) -> Optional[bytes]:
        """将图像转换为字节数据"""
        if Image is None:
            return None
            
        try:
            import io
            
            # 如果是 RGBA 模式，转换为 RGB
            if img.mode == 'RGBA':
                # 创建白色背景
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[-1])  # 使用 alpha 通道作为 mask
                img = background
            elif img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            
            # 保存为字节数据
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            return img_bytes.getvalue()
            
        except Exception as e:
            self.logger.error(f"Failed to convert image to bytes: {e}")
            return None
    
    def _extract_text_ocr(self, img) -> Optional[str]:
        """使用 OCR 提取文本"""
        if pytesseract is None:
            return None
        
        try:
            # 使用 Tesseract 进行 OCR
            text = pytesseract.image_to_string(img, lang='eng+chi_sim')
            text = text.strip()
            
            if text:
                self.logger.debug(f"OCR extracted {len(text)} characters")
                return text
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"OCR failed: {e}")
            return None