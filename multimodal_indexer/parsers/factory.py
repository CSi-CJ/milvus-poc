"""
解析器工厂函数
"""

import logging
from typing import Dict, Any

from .registry import FileParserRegistry
from .pdf_parser import PDFParser
from .text_parser import TextParser
from .image_parser import ImageParser
from .audio_parser import AudioParser
from .video_parser import VideoParser


def create_default_registry(config: Dict[str, Any] = None) -> FileParserRegistry:
    """创建默认的解析器注册表
    
    Args:
        config: 配置字典，包含解析器的配置选项
        
    Returns:
        FileParserRegistry: 配置好的解析器注册表
    """
    logger = logging.getLogger(__name__)
    
    if config is None:
        config = {}
    
    # 获取配置选项
    enable_ocr = config.get('enable_ocr', True)
    enable_speech_recognition = config.get('enable_speech_recognition', True)
    max_video_frames = config.get('max_video_frames', 10)
    
    # 创建注册表
    registry = FileParserRegistry()
    
    # 注册解析器
    try:
        # PDF 解析器
        pdf_parser = PDFParser()
        registry.register(pdf_parser)
        logger.info("Registered PDF parser")
    except Exception as e:
        logger.warning(f"Failed to register PDF parser: {e}")
    
    try:
        # 文本解析器
        text_parser = TextParser()
        registry.register(text_parser)
        logger.info("Registered text parser")
    except Exception as e:
        logger.warning(f"Failed to register text parser: {e}")
    
    try:
        # 图像解析器
        image_parser = ImageParser(enable_ocr=enable_ocr)
        registry.register(image_parser)
        logger.info(f"Registered image parser (OCR: {enable_ocr})")
    except Exception as e:
        logger.warning(f"Failed to register image parser: {e}")
    
    try:
        # 音频解析器
        audio_parser = AudioParser(enable_transcription=enable_speech_recognition)
        registry.register(audio_parser)
        logger.info(f"Registered audio parser (transcription: {enable_speech_recognition})")
    except Exception as e:
        logger.warning(f"Failed to register audio parser: {e}")
    
    try:
        # 视频解析器
        video_parser = VideoParser(
            enable_audio_extraction=enable_speech_recognition,
            max_frames=max_video_frames
        )
        registry.register(video_parser)
        logger.info(f"Registered video parser (audio extraction: {enable_speech_recognition})")
    except Exception as e:
        logger.warning(f"Failed to register video parser: {e}")
    
    logger.info(f"Created parser registry with {len(registry.parsers)} parsers")
    logger.info(f"Supported extensions: {registry.get_supported_extensions()}")
    
    return registry


def get_parser_info() -> Dict[str, Any]:
    """获取解析器信息
    
    Returns:
        Dict: 包含解析器信息的字典
    """
    info = {
        'parsers': [],
        'dependencies': {
            'required': ['Pillow'],
            'optional': {
                'PyMuPDF': 'PDF parsing',
                'python-docx': 'Word document parsing',
                'pytesseract': 'OCR text extraction',
                'librosa': 'Audio analysis',
                'openai-whisper': 'Speech recognition',
                'opencv-python': 'Video processing',
                'ffmpeg-python': 'Video audio extraction',
                'chardet': 'Text encoding detection',
            }
        }
    }
    
    # 检查各个解析器的可用性
    parsers_status = []
    
    # PDF 解析器
    try:
        import fitz
        parsers_status.append({
            'name': 'PDFParser',
            'status': 'available',
            'formats': ['.pdf'],
            'dependencies': ['PyMuPDF']
        })
    except ImportError:
        parsers_status.append({
            'name': 'PDFParser',
            'status': 'missing_dependencies',
            'formats': ['.pdf'],
            'dependencies': ['PyMuPDF'],
            'install_command': 'pip install PyMuPDF'
        })
    
    # 文本解析器
    parsers_status.append({
        'name': 'TextParser',
        'status': 'available',
        'formats': ['.txt', '.md', '.docx'],
        'dependencies': ['python-docx (optional)', 'chardet (optional)']
    })
    
    # 图像解析器
    try:
        from PIL import Image
        status = 'available'
        deps = ['Pillow']
        try:
            import pytesseract
            deps.append('pytesseract')
        except ImportError:
            deps.append('pytesseract (optional)')
    except ImportError:
        status = 'missing_dependencies'
        deps = ['Pillow']
    
    parsers_status.append({
        'name': 'ImageParser',
        'status': status,
        'formats': ['.png', '.jpg', '.jpeg', '.gif', '.bmp'],
        'dependencies': deps
    })
    
    # 音频解析器
    try:
        import librosa
        status = 'available'
        deps = ['librosa']
        try:
            import whisper
            deps.append('openai-whisper')
        except ImportError:
            deps.append('openai-whisper (optional)')
    except ImportError:
        status = 'missing_dependencies'
        deps = ['librosa']
    
    parsers_status.append({
        'name': 'AudioParser',
        'status': status,
        'formats': ['.mp3', '.wav', '.m4a'],
        'dependencies': deps
    })
    
    # 视频解析器
    try:
        import cv2
        status = 'available'
        deps = ['opencv-python']
        try:
            import ffmpeg
            deps.append('ffmpeg-python')
        except ImportError:
            deps.append('ffmpeg-python (optional)')
    except ImportError:
        status = 'missing_dependencies'
        deps = ['opencv-python']
    
    parsers_status.append({
        'name': 'VideoParser',
        'status': status,
        'formats': ['.mp4', '.avi', '.mov'],
        'dependencies': deps
    })
    
    info['parsers'] = parsers_status
    return info