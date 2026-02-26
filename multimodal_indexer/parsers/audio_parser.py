"""
音频文件解析器
"""

import os
from typing import Dict, Any, Optional
import logging
import io

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

try:
    import whisper
except ImportError:
    whisper = None

try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from ..models import ParsedContent
from .base import BaseFileParser


class AudioParser(BaseFileParser):
    """音频文件解析器"""
    
    def __init__(self, enable_transcription: bool = True):
        super().__init__()
        self.enable_transcription = enable_transcription
        self.supported_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg'}
        self.whisper_model = None
        self._last_transcription_info = {}  # 存储最后一次转录的信息
        
        if librosa is None:
            self.logger.warning("librosa not installed. Audio parsing will be limited.")
        
        if enable_transcription and whisper is None:
            self.logger.warning("whisper not installed. Speech transcription will be disabled.")
    
    def can_parse(self, file_path: str) -> bool:
        """检查是否能解析音频文件"""
        ext = self._get_file_extension(file_path)
        return ext in self.supported_extensions
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析音频文件"""
        self._validate_file(file_path)
        
        if librosa is None:
            return self._create_error_content(
                file_path,
                "librosa not installed. Please install with: pip install librosa"
            )
        
        try:
            # 加载音频文件
            y, sr = librosa.load(file_path, sr=None)
            
            # 提取基本信息
            duration = librosa.get_duration(y=y, sr=sr)
            
            metadata = {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if y.ndim == 1 else y.shape[0],
                'samples': len(y),
            }
            
            # 提取音频特征
            features = self._extract_audio_features(y, sr)
            metadata.update(features)
            
            # 读取原始音频数据
            audio_data = self._load_audio_bytes(file_path)
            
            # 语音转录
            text_content = None
            if self.enable_transcription and whisper is not None:
                text_content = self._transcribe_audio(file_path)
                # 将转录信息添加到元数据中
                if hasattr(self, '_last_transcription_info'):
                    metadata.update(self._last_transcription_info)
            
            # 生成音频可视化截图
            images = []
            try:
                waveform_image = self._generate_waveform_screenshot(y, sr, file_path)
                if waveform_image:
                    images.append(waveform_image)
            except Exception as e:
                self.logger.warning(f"Failed to generate waveform screenshot for {file_path}: {e}")
            
            return ParsedContent(
                text_content=text_content,
                image_content=images if images else None,
                audio_content=audio_data,
                metadata=metadata,
                file_type=self._get_file_extension(file_path)
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing audio {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_audio_features(self, y, sr) -> Dict[str, Any]:
        """提取音频特征"""
        try:
            features = {}
            
            # 基本统计特征
            features['rms_energy'] = float(librosa.feature.rms(y=y).mean())
            features['zero_crossing_rate'] = float(librosa.feature.zero_crossing_rate(y).mean())
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid'] = float(spectral_centroids.mean())
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff'] = float(spectral_rolloff.mean())
            
            # MFCC 特征
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = mfccs.mean(axis=1).tolist()
            
            # 节拍和节奏
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Failed to extract audio features: {e}")
            return {}
    
    def _load_audio_bytes(self, file_path: str) -> Optional[bytes]:
        """加载音频文件的原始字节数据"""
        try:
            with open(file_path, 'rb') as f:
                return f.read()
        except Exception as e:
            self.logger.warning(f"Failed to load audio bytes: {e}")
            return None
    
    def _transcribe_audio(self, file_path: str) -> Optional[str]:
        """使用 Whisper 进行语音转录"""
        if whisper is None:
            return None
        
        try:
            # 懒加载 Whisper 模型
            if self.whisper_model is None:
                self.logger.info("Loading Whisper model...")
                self.whisper_model = whisper.load_model("base")
            
            # 转录音频
            result = self.whisper_model.transcribe(file_path)
            text = result["text"].strip()
            
            if text:
                self.logger.debug(f"Transcribed {len(text)} characters from audio")
                
                # 将转录相关信息添加到元数据中（这将在parse方法中使用）
                self._last_transcription_info = {
                    'detected_language': result.get("language", "unknown"),
                    'transcription_confidence': self._calculate_average_confidence(result.get("segments", [])),
                    'transcription_segments': len(result.get("segments", []))
                }
                
                return text
            else:
                return None
                
        except Exception as e:
            self.logger.warning(f"Audio transcription failed: {e}")
            return None
    
    def _calculate_average_confidence(self, segments) -> float:
        """计算转录的平均置信度"""
        if not segments:
            return 0.0
        
        try:
            confidences = []
            for segment in segments:
                if 'avg_logprob' in segment:
                    # 将log概率转换为0-1的置信度
                    confidence = min(1.0, max(0.0, (segment['avg_logprob'] + 1.0)))
                    confidences.append(confidence)
            
            return sum(confidences) / len(confidences) if confidences else 0.0
        except Exception:
            return 0.0
    
    def _generate_waveform_screenshot(self, y, sr, file_path: str) -> bytes:
        """生成音频波形图截图"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("matplotlib not available, cannot generate waveform screenshot")
            return None
        
        try:
            # 设置图像参数
            plt.style.use('default')
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            fig.suptitle(f'音频文件: {os.path.basename(file_path)}', fontsize=14, fontweight='bold')
            
            # 时间轴
            time = np.linspace(0, len(y) / sr, len(y))
            
            # 绘制波形图
            ax1.plot(time, y, color='blue', alpha=0.7)
            ax1.set_title('波形图 (Waveform)', fontsize=12)
            ax1.set_xlabel('时间 (秒)')
            ax1.set_ylabel('振幅')
            ax1.grid(True, alpha=0.3)
            
            # 限制显示时间（避免图像过于密集）
            if len(time) > 10000:
                step = len(time) // 5000
                ax1.plot(time[::step], y[::step], color='blue', alpha=0.7)
            
            # 绘制频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
            ax2.set_title('频谱图 (Spectrogram)', fontsize=12)
            ax2.set_xlabel('时间 (秒)')
            ax2.set_ylabel('频率 (Hz)')
            
            # 添加颜色条
            plt.colorbar(img, ax=ax2, format='%+2.0f dB')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存到字节流
            buffer = io.BytesIO()
            plt.savefig(buffer, format='PNG', dpi=100, bbox_inches='tight')
            plt.close(fig)  # 释放内存
            
            return buffer.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error generating waveform screenshot: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None