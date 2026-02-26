"""
视频文件解析器 - 增强版，支持完整的视频帧OCR文本提取
"""

import os
import tempfile
from typing import Dict, Any, Optional, List
import logging

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import ffmpeg
except ImportError:
    ffmpeg = None

from ..models import ParsedContent
from .base import BaseFileParser
from .audio_parser import AudioParser


class VideoParser(BaseFileParser):
    """视频文件解析器 - 增强版"""
    
    def __init__(self, enable_audio_extraction: bool = True, max_frames: int = 15, 
                 enable_enhanced_ocr: bool = True):
        super().__init__()
        self.enable_audio_extraction = enable_audio_extraction
        self.max_frames = max_frames
        self.enable_enhanced_ocr = enable_enhanced_ocr
        self.supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
        self.audio_parser = AudioParser() if enable_audio_extraction else None
        
        # 初始化增强OCR处理器 - 使用ChatGPT帧策略方案
        if enable_enhanced_ocr:
            try:
                from ..chatgpt_frame_strategy_processor import ChatGPTFrameStrategyProcessor
                self.enhanced_ocr = ChatGPTFrameStrategyProcessor()
                self.enhanced_ocr_type = 'chatgpt_frame_strategy'
                self.logger.info("✅ ChatGPT帧策略视频OCR处理器初始化成功")
            except ImportError as e:
                self.logger.warning(f"ChatGPT帧策略OCR处理器不可用: {e}")
                self.enhanced_ocr = None
                self.enhanced_ocr_type = None
        else:
            self.enhanced_ocr = None
            self.enhanced_ocr_type = None
        
        if cv2 is None:
            self.logger.warning("OpenCV not installed. Video parsing will be limited.")
        
        if enable_audio_extraction and ffmpeg is None:
            self.logger.warning("ffmpeg-python not installed. Audio extraction will be disabled.")
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否能解析视频文件"""
        ext = self._get_file_extension(file_path)
        return ext in self.supported_extensions
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析视频文件"""
        self._validate_file(file_path)
        
        if cv2 is None:
            return self._create_error_content(
                file_path,
                "OpenCV not installed. Please install with: pip install opencv-python"
            )
        
        try:
            # 打开视频文件
            cap = cv2.VideoCapture(file_path)
            
            if not cap.isOpened():
                return self._create_error_content(file_path, "Failed to open video file")
            
            # 获取视频信息
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            metadata = {
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height,
                'resolution': f"{width}x{height}",
                'enhanced_ocr_enabled': self.enhanced_ocr is not None
            }
            
            # 提取关键帧
            key_frames = self._extract_key_frames(cap, frame_count)
            
            cap.release()
            
            # 提取音频
            audio_content = None
            text_content = None
            if self.enable_audio_extraction and self.audio_parser:
                audio_result = self._extract_audio(file_path)
                if audio_result:
                    audio_content = audio_result.get('audio_data')
                    text_content = audio_result.get('transcription')
                    if audio_result.get('metadata'):
                        metadata['audio'] = audio_result['metadata']
            
            # 如果没有音频转录内容，或者音频转录内容很少，使用增强OCR提取视频帧文本
            if self.enhanced_ocr and key_frames:
                if not text_content or len(text_content.strip()) < 50:
                    self.logger.info("🔄 音频转录内容不足，启用增强视频OCR提取...")
                    
                    # 根据不同的OCR处理器类型使用不同的策略
                    if hasattr(self, 'enhanced_ocr_type') and self.enhanced_ocr_type == 'chatgpt_frame_strategy':
                        self.logger.info("🚀 使用ChatGPT完整帧策略处理视频...")
                        ocr_results = self.enhanced_ocr.extract_frames_with_chatgpt_strategy(file_path)
                    elif hasattr(self, 'enhanced_ocr_type') and self.enhanced_ocr_type == 'pure_chatgpt':
                        self.logger.info("🚀 使用纯ChatGPT策略处理视频...")
                        ocr_results = self.enhanced_ocr.process_video_with_pure_chatgpt_strategy(file_path)
                    elif hasattr(self, 'enhanced_ocr_type') and self.enhanced_ocr_type == 'enhanced_chatgpt':
                        self.logger.info("🚀 使用增强ChatGPT策略处理视频...")
                        ocr_results = self.enhanced_ocr.process_video_with_enhanced_strategy(file_path)
                    else:
                        # 使用传统的帧处理方式
                        ocr_results = self.enhanced_ocr.extract_comprehensive_text_from_frames(key_frames)
                    
                    # 合并所有帧的OCR文本
                    frame_texts = []
                    total_confidence = 0
                    successful_frames = 0
                    
                    for result in ocr_results:
                        if result['text'].strip():
                            # ChatGPT帧策略包含时间戳和优先级信息
                            if 'timestamp' in result and 'priority_score' in result:
                                frame_texts.append(f"[帧 {result['frame_number']} - {result['timestamp']:.1f}s - 优先级:{result['priority_score']:.1f}] {result['text']}")
                            else:
                                frame_texts.append(f"[帧 {result['frame_number']}] {result['text']}")
                            total_confidence += result['confidence']
                            successful_frames += 1
                    
                    if frame_texts:
                        ocr_text = '\n\n'.join(frame_texts)
                        avg_confidence = total_confidence / successful_frames if successful_frames > 0 else 0
                        
                        # 如果OCR文本比音频转录更丰富，使用OCR文本
                        if not text_content or len(ocr_text) > len(text_content) * 2:
                            text_content = ocr_text
                            extraction_method = getattr(self, 'enhanced_ocr_type', 'enhanced_multi_engine')
                            metadata['ocr_extraction'] = {
                                'successful_frames': successful_frames,
                                'total_frames': len(ocr_results),
                                'average_confidence': avg_confidence,
                                'extraction_method': extraction_method
                            }
                            
                            # 添加ChatGPT帧策略的额外信息
                            if self.enhanced_ocr_type == 'chatgpt_frame_strategy':
                                metadata['frame_strategy'] = {
                                    'similarity_filtering': True,
                                    'priority_ranking': True,
                                    'fps_extraction': '1_fps'
                                }
                            
                            self.logger.info(f"✅ 增强OCR提取成功: {successful_frames}/{len(ocr_results)} 帧, 平均置信度: {avg_confidence:.3f}")
                        else:
                            # 将OCR文本作为补充信息添加到元数据
                            metadata['supplementary_ocr'] = ocr_text
                            self.logger.info("📝 OCR文本作为补充信息保存")
                    else:
                        self.logger.warning("⚠️  增强OCR未能提取到文本内容")
            
            return ParsedContent(
                text_content=text_content,
                image_content=key_frames if key_frames else None,
                audio_content=audio_content,
                metadata=metadata,
                file_type=self._get_file_extension(file_path)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing video {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_key_frames(self, cap, frame_count: int) -> Optional[List[bytes]]:
        """智能提取关键帧 - 基于场景变化和质量优化"""
        try:
            frames = []
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 启用场景检测和质量优化
            scene_threshold = 0.3
            min_frame_interval = 30
            high_quality_frames = True
            
            # 如果启用场景检测
            if scene_threshold > 0 and frame_count > min_frame_interval:
                selected_frames = self._detect_scene_changes(cap, frame_count, fps, scene_threshold, min_frame_interval)
            else:
                # 回退到均匀采样
                selected_frames = self._uniform_sampling(frame_count)
            
            self.logger.info(f"Selected {len(selected_frames)} frames for extraction")
            
            # 提取选定的帧
            for frame_idx in selected_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 高质量处理
                    if high_quality_frames:
                        frame = self._enhance_frame_quality(frame)
                    
                    # 转换为高质量PNG
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # 较低压缩，保持质量
                    success, buffer = cv2.imencode('.png', frame, encode_params)
                    
                    if success:
                        frame_bytes = buffer.tobytes()
                        frames.append(frame_bytes)
                        
                        # 记录帧信息
                        timestamp = frame_idx / fps if fps > 0 else 0
                        self.logger.debug(f"Extracted frame {frame_idx} at {timestamp:.2f}s, size: {len(frame_bytes)} bytes")
                    else:
                        self.logger.warning(f"Failed to encode frame {frame_idx}")
                else:
                    self.logger.warning(f"Failed to read frame {frame_idx}")
            
            self.logger.info(f"Successfully extracted {len(frames)} high-quality key frames")
            return frames if frames else None
            
        except Exception as e:
            self.logger.warning(f"Failed to extract key frames: {e}")
            return None
    
    def _detect_scene_changes(self, cap, frame_count: int, fps: float, scene_threshold: float, min_frame_interval: int) -> List[int]:
        """检测场景变化点"""
        try:
            import numpy as np
            
            scene_frames = [0]  # 总是包含第一帧
            prev_hist = None
            last_selected = 0
            
            # 计算检查间隔（避免过于频繁的检查）
            check_interval = max(1, int(fps * 0.5))  # 每0.5秒检查一次
            
            for frame_idx in range(0, frame_count, check_interval):
                if len(scene_frames) >= self.max_frames:
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    continue
                
                # 计算直方图
                hist = self._calculate_histogram(frame)
                
                if prev_hist is not None:
                    # 计算直方图差异
                    similarity = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    
                    # 如果相似度低于阈值，认为是场景变化
                    if similarity < (1 - scene_threshold):
                        # 确保与上一个选中帧有足够间隔
                        if frame_idx - last_selected >= min_frame_interval:
                            scene_frames.append(frame_idx)
                            last_selected = frame_idx
                            self.logger.debug(f"Scene change detected at frame {frame_idx}, similarity: {similarity:.3f}")
                
                prev_hist = hist
            
            # 如果场景变化点太少，补充一些均匀分布的帧
            if len(scene_frames) < self.max_frames // 2:
                uniform_frames = self._uniform_sampling(frame_count)
                for frame_idx in uniform_frames:
                    if frame_idx not in scene_frames and len(scene_frames) < self.max_frames:
                        scene_frames.append(frame_idx)
            
            # 确保包含最后一帧
            if frame_count - 1 not in scene_frames and len(scene_frames) < self.max_frames:
                scene_frames.append(frame_count - 1)
            
            scene_frames.sort()
            return scene_frames[:self.max_frames]
            
        except Exception as e:
            self.logger.warning(f"Scene detection failed: {e}, falling back to uniform sampling")
            return self._uniform_sampling(frame_count)
    
    def _calculate_histogram(self, frame):
        """计算帧的颜色直方图"""
        import numpy as np
        
        # 转换为HSV色彩空间，对光照变化更鲁棒
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # 计算H和S通道的直方图
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        
        # 归一化
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        return hist
    
    def _uniform_sampling(self, frame_count: int) -> List[int]:
        """均匀采样帧"""
        if frame_count <= self.max_frames:
            return list(range(frame_count))
        
        frame_interval = frame_count // self.max_frames
        return list(range(0, frame_count, frame_interval))[:self.max_frames]
    
    def _enhance_frame_quality(self, frame):
        """增强帧质量以提升OCR效果"""
        try:
            import numpy as np
            
            # 1. 去噪
            frame = cv2.bilateralFilter(frame, 9, 75, 75)
            
            # 2. 锐化
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            frame = cv2.filter2D(frame, -1, kernel)
            
            # 3. 对比度增强
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE (对比度限制自适应直方图均衡化)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
            
            return frame
            
        except Exception as e:
            self.logger.warning(f"Frame enhancement failed: {e}")
            return frame
    
    def _extract_audio(self, file_path: str) -> Optional[Dict[str, Any]]:
        """提取音频轨道"""
        if ffmpeg is None or self.audio_parser is None:
            return None
        
        temp_audio_file = None
        try:
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_audio_file = temp_file.name
            
            # 使用 ffmpeg 提取音频
            (
                ffmpeg
                .input(file_path)
                .output(temp_audio_file, acodec='pcm_s16le', ac=1, ar='16000')
                .overwrite_output()
                .run(quiet=True)
            )
            
            # 使用音频解析器处理提取的音频
            audio_result = self.audio_parser.parse(temp_audio_file)
            
            if audio_result.has_content():
                return {
                    'audio_data': audio_result.audio_content,
                    'transcription': audio_result.text_content,
                    'metadata': audio_result.metadata
                }
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Failed to extract audio from video: {e}")
            return None
        
        finally:
            # 清理临时文件
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.unlink(temp_audio_file)
                except:
                    pass