#!/usr/bin/env python3
"""
改进的OCR引擎，支持多种OCR方案
"""

import os
import logging
from typing import Optional, List, Dict, Any
from PIL import Image
import io
import base64

class ImprovedOCREngine:
    """改进的OCR引擎，支持多种OCR方案"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_engines = self._detect_available_engines()
        self.primary_engine = self._select_primary_engine()
        
        # 初始化选定的引擎
        self._init_engines()
    
    def _detect_available_engines(self) -> List[str]:
        """检测可用的OCR引擎"""
        engines = []
        
        # 检测EasyOCR
        try:
            import easyocr
            engines.append('easyocr')
            self.logger.info("✅ EasyOCR可用")
        except ImportError:
            self.logger.debug("EasyOCR不可用")
        
        # 检测PaddleOCR
        try:
            from paddleocr import PaddleOCR
            engines.append('paddleocr')
            self.logger.info("✅ PaddleOCR可用")
        except ImportError:
            self.logger.debug("PaddleOCR不可用")
        except Exception as e:
            self.logger.debug(f"PaddleOCR初始化失败: {e}")
        
        # 检测Tesseract
        try:
            import pytesseract
            import subprocess
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                engines.append('tesseract')
                self.logger.info("✅ Tesseract可用")
        except Exception:
            self.logger.debug("Tesseract不可用")
        
        return engines
    
    def _select_primary_engine(self) -> Optional[str]:
        """选择主要OCR引擎"""
        # 优先级：PaddleOCR > EasyOCR > Tesseract
        # PaddleOCR对中文支持最佳，优先使用
        priority = ['paddleocr', 'easyocr', 'tesseract']
        
        for engine in priority:
            if engine in self.available_engines:
                self.logger.info(f"选择主要OCR引擎: {engine}")
                return engine
        
        self.logger.warning("没有可用的OCR引擎")
        return None
    
    def _init_engines(self):
        """初始化OCR引擎"""
        self.engines = {}
        
        # 初始化EasyOCR
        if 'easyocr' in self.available_engines:
            try:
                import easyocr
                self.engines['easyocr'] = easyocr.Reader(['ch_sim', 'en'], verbose=False)
                self.logger.info("EasyOCR初始化成功")
            except Exception as e:
                self.logger.error(f"EasyOCR初始化失败: {e}")
                self.available_engines.remove('easyocr')
        
        # 初始化PaddleOCR
        if 'paddleocr' in self.available_engines:
            try:
                # 设置环境变量以避免连接检查
                os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True'
                from paddleocr import PaddleOCR
                self.engines['paddleocr'] = PaddleOCR(use_angle_cls=True, lang='ch', show_log=False)
                self.logger.info("PaddleOCR初始化成功")
            except Exception as e:
                self.logger.error(f"PaddleOCR初始化失败: {e}")
                self.available_engines.remove('paddleocr')
        
        # Tesseract不需要特殊初始化
        if 'tesseract' in self.available_engines:
            self.logger.info("Tesseract准备就绪")
    
    def extract_text(self, image_data: bytes) -> str:
        """从图像中提取文本
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            str: 提取的文本内容
        """
        if not self.primary_engine:
            self.logger.warning("没有可用的OCR引擎")
            return ""
        
        # 尝试使用主要引擎
        try:
            if self.primary_engine == 'easyocr':
                return self._extract_with_easyocr(image_data)
            elif self.primary_engine == 'paddleocr':
                return self._extract_with_paddleocr(image_data)
            elif self.primary_engine == 'tesseract':
                return self._extract_with_tesseract(image_data)
        except Exception as e:
            self.logger.warning(f"主要OCR引擎 {self.primary_engine} 失败: {e}")
        
        # 尝试备用引擎
        for engine in self.available_engines:
            if engine != self.primary_engine:
                try:
                    self.logger.info(f"尝试备用OCR引擎: {engine}")
                    if engine == 'easyocr':
                        return self._extract_with_easyocr(image_data)
                    elif engine == 'paddleocr':
                        return self._extract_with_paddleocr(image_data)
                    elif engine == 'tesseract':
                        return self._extract_with_tesseract(image_data)
                except Exception as e:
                    self.logger.warning(f"备用OCR引擎 {engine} 失败: {e}")
                    continue
        
        self.logger.error("所有OCR引擎都失败了")
        return ""
    
    def _extract_with_easyocr(self, image_data: bytes) -> str:
        """使用EasyOCR提取文本"""
        if 'easyocr' not in self.engines:
            raise Exception("EasyOCR未初始化")
        
        # 将字节数据转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为numpy数组
        import numpy as np
        image_array = np.array(image)
        
        # 执行OCR
        results = self.engines['easyocr'].readtext(image_array)
        
        # 提取文本
        texts = [result[1] for result in results if result[2] > 0.5]  # 置信度阈值0.5
        
        return self._post_process_text('\n'.join(texts))
    
    def _extract_with_paddleocr(self, image_data: bytes) -> str:
        """使用PaddleOCR提取文本"""
        if 'paddleocr' not in self.engines:
            raise Exception("PaddleOCR未初始化")
        
        # 将字节数据转换为PIL图像
        image = Image.open(io.BytesIO(image_data))
        
        # 转换为numpy数组
        import numpy as np
        image_array = np.array(image)
        
        # 执行OCR
        results = self.engines['paddleocr'].ocr(image_array, cls=True)
        
        # 提取文本
        texts = []
        if results and results[0]:
            for line in results[0]:
                if line and len(line) > 1 and line[1][1] > 0.5:  # 置信度阈值0.5
                    texts.append(line[1][0])
        
        return self._post_process_text('\n'.join(texts))
    
    def _extract_with_tesseract(self, image_data: bytes) -> str:
        """使用Tesseract提取文本"""
        try:
            import pytesseract
            from PIL import Image
            
            # 将字节数据转换为PIL图像
            image = Image.open(io.BytesIO(image_data))
            
            # 执行OCR
            config = '--psm 6 --oem 1'
            text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=config)
            
            return self._post_process_text(text)
            
        except ImportError:
            raise Exception("pytesseract未安装")
        except Exception as e:
            raise Exception(f"Tesseract OCR失败: {e}")
    
    def _post_process_text(self, text: str) -> str:
        """后处理OCR文本"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = text.strip()
        
        # 处理中文字符间的空格
        import re
        
        # 移除中文字符之间的单个空格
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 移除中文字符之间的单个空格
            line = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', line)
            
            # 移除中文字符和标点符号之间的空格
            line = re.sub(r'([\u4e00-\u9fff])\s+([，。！？；：、])', r'\1\2', line)
            line = re.sub(r'([，。！？；：、])\s+([\u4e00-\u9fff])', r'\1\2', line)
            
            # 清理多余空格，但保留英文单词间的空格
            line = re.sub(r'\s+', ' ', line)
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """获取OCR引擎信息"""
        return {
            'available_engines': self.available_engines,
            'primary_engine': self.primary_engine,
            'initialized_engines': list(self.engines.keys())
        }

def test_improved_ocr():
    """测试改进的OCR引擎"""
    print("🧪 测试改进的OCR引擎")
    print("="*50)
    
    # 初始化OCR引擎
    ocr_engine = ImprovedOCREngine()
    
    # 显示引擎信息
    info = ocr_engine.get_engine_info()
    print(f"可用引擎: {info['available_engines']}")
    print(f"主要引擎: {info['primary_engine']}")
    print(f"已初始化: {info['initialized_engines']}")
    
    if not info['primary_engine']:
        print("❌ 没有可用的OCR引擎")
        return
    
    # 创建测试图像
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.new('RGB', (500, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype("simsun.ttc", 24)
        except:
            font = ImageFont.load_default()
        
        # 测试文本
        draw.text((20, 30), "Hello World", fill='black', font=font)
        draw.text((20, 70), "你好世界", fill='black', font=font)
        draw.text((20, 110), "自办会议飞检规则", fill='black', font=font)
        draw.text((20, 150), "AstraZeneca 阿斯利康", fill='black', font=font)
        
        # 保存图像
        img.save("improved_ocr_test.png")
        print("✅ 测试图像创建成功")
        
        # 转换为字节数据
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        image_data = img_bytes.getvalue()
        
        # 执行OCR
        print("🔄 执行OCR识别...")
        extracted_text = ocr_engine.extract_text(image_data)
        
        if extracted_text:
            print("✅ OCR识别成功:")
            print(f"识别结果:\n{extracted_text}")
        else:
            print("⚠️  OCR未识别到文本")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")

if __name__ == "__main__":
    test_improved_ocr()