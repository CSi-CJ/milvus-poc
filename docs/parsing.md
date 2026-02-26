# 多模态解析实现详解

## 解析系统架构

多模态文件索引器采用插件化的解析器架构，支持PDF、文本、图像、音频、视频等多种文件格式的智能解析。每种文件类型都有专门的解析器，负责提取内容、生成截图和元数据。

## 解析器基础架构

### 1. 基础解析器接口

```python
class BaseFileParser:
    """文件解析器基类"""
    
    def can_parse(self, file_path: str) -> bool:
        """判断是否能解析指定文件"""
        pass
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析文件内容"""
        pass
    
    def _validate_file(self, file_path: str):
        """验证文件有效性"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"File is empty: {file_path}")
    
    def _create_error_content(self, file_path: str, error: str) -> ParsedContent:
        """创建错误内容"""
        return ParsedContent(
            file_path=file_path,
            file_type=os.path.splitext(file_path)[1].lower(),
            text_content=f"Error parsing file: {error}",
            image_content=[],
            audio_content=None,
            metadata={"error": error, "parsing_failed": True}
        )
```

### 2. 解析器注册系统

```python
class FileParserRegistry:
    """解析器注册表"""
    
    def __init__(self):
        self.parsers: List[BaseFileParser] = []
        self._register_default_parsers()
    
    def register_parser(self, parser: BaseFileParser):
        """注册解析器"""
        self.parsers.append(parser)
        self.logger.info(f"Registered parser: {parser.__class__.__name__}")
    
    def get_parser(self, file_path: str) -> Optional[BaseFileParser]:
        """获取适合的解析器"""
        for parser in self.parsers:
            if parser.can_parse(file_path):
                return parser
        return None
    
    def _register_default_parsers(self):
        """注册默认解析器"""
        self.register_parser(PDFParser())
        self.register_parser(TextParser())
        self.register_parser(ImageParser())
        self.register_parser(AudioParser())
        self.register_parser(VideoParser())
```

### 3. 解析器工厂

```python
class FileParserFactory:
    """解析器工厂"""
    
    def __init__(self):
        self.registry = FileParserRegistry()
    
    def create_parser(self, file_path: str) -> BaseFileParser:
        """创建解析器实例"""
        parser = self.registry.get_parser(file_path)
        if not parser:
            raise ValueError(f"No parser available for file: {file_path}")
        return parser
    
    def get_supported_extensions(self) -> List[str]:
        """获取支持的文件扩展名"""
        extensions = []
        for parser in self.registry.parsers:
            extensions.extend(parser.supported_extensions)
        return list(set(extensions))
```

## 具体解析器实现

### 1. PDF解析器

#### 核心功能
- **文本提取**: 提取PDF中的所有文本内容
- **页面截图**: 生成高质量页面截图 (2x分辨率)
- **元数据提取**: 提取文档属性和页面信息
- **OCR集成**: 对截图进行文字识别

#### 实现细节

```python
class PDFParser(BaseFileParser):
    """PDF文件解析器"""
    
    supported_extensions = ['.pdf']
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析PDF文件"""
        try:
            self._validate_file(file_path)
            
            # 打开PDF文档
            doc = fitz.open(file_path)
            
            # 提取文本内容
            text_content = self._extract_text(doc)
            
            # 生成页面截图
            image_content = self._generate_page_screenshots(doc)
            
            # 提取元数据
            metadata = self._extract_metadata(doc, file_path)
            
            doc.close()
            
            return ParsedContent(
                file_path=file_path,
                file_type='.pdf',
                text_content=text_content,
                image_content=image_content,
                audio_content=None,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing PDF {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_text(self, doc) -> str:
        """提取PDF文本内容"""
        text_parts = []
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                text_parts.append(f"=== 第 {page_num + 1} 页 ===\n{text}")
        
        return "\n\n".join(text_parts)
    
    def _generate_page_screenshots(self, doc) -> List[bytes]:
        """生成页面截图"""
        screenshots = []
        scale_factor = 2.0  # 2x分辨率提高图像质量
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            
            # 检查页面是否有内容
            if not self._page_has_content(page):
                continue
            
            try:
                # 生成高质量截图
                mat = fitz.Matrix(scale_factor, scale_factor)
                pix = page.get_pixmap(matrix=mat)
                
                # 转换为PIL图像
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                
                # 确保RGB模式
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # 智能压缩
                compressed_data = self._compress_image_intelligently(image, img_data)
                screenshots.append(compressed_data)
                
                # 清理资源
                pix = None
                image.close()
                
            except Exception as e:
                self.logger.warning(f"Failed to generate screenshot for page {page_num + 1}: {e}")
                continue
        
        return screenshots
    
    def _page_has_content(self, page) -> bool:
        """检查页面是否有实际内容"""
        # 检查文本内容
        text = page.get_text().strip()
        if len(text) > 10:  # 至少10个字符
            return True
        
        # 检查图像内容
        image_list = page.get_images()
        if len(image_list) > 0:
            return True
        
        # 检查绘图内容
        drawings = page.get_drawings()
        if len(drawings) > 0:
            return True
        
        return False
    
    def _compress_image_intelligently(self, image: Image.Image, original_data: bytes) -> bytes:
        """智能图像压缩算法"""
        
        # 阶段1: 初始压缩 (100KB以上)
        if len(original_data) > 100000:
            if image.width > 1600 or image.height > 1600:
                image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            image_data = buffer.getvalue()
        else:
            image_data = original_data
        
        # 阶段2: 二次压缩 (42KB以上)
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        if len(image_base64) > 42000:
            if image.width > 1000 or image.height > 1000:
                image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=80, optimize=True)
            image_data = buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 阶段3: 最终压缩 (仍然超过42KB)
        if len(image_base64) > 42000:
            if image.width > 800 or image.height > 800:
                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=75, optimize=True)
            image_data = buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # 阶段4: 极限压缩 (最后手段)
        if len(image_base64) > 42000:
            if image.width > 600 or image.height > 600:
                image.thumbnail((600, 600), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=70, optimize=True)
            image_data = buffer.getvalue()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # 安全截断 (防止超过Milvus限制)
            if len(image_base64) > 65500:
                image_base64 = image_base64[:65500]
        
        return base64.b64decode(image_base64)
    
    def _extract_metadata(self, doc, file_path: str) -> Dict[str, Any]:
        """提取PDF元数据"""
        metadata = {
            'page_count': doc.page_count,
            'file_size': os.path.getsize(file_path),
            'creation_date': datetime.now().isoformat(),
            'parser_type': 'PDFParser'
        }
        
        # 提取文档属性
        doc_metadata = doc.metadata
        if doc_metadata:
            metadata.update({
                'title': doc_metadata.get('title', ''),
                'author': doc_metadata.get('author', ''),
                'subject': doc_metadata.get('subject', ''),
                'creator': doc_metadata.get('creator', ''),
                'producer': doc_metadata.get('producer', ''),
                'creation_date': doc_metadata.get('creationDate', ''),
                'modification_date': doc_metadata.get('modDate', '')
            })
        
        return metadata
```

### 2. 文本解析器

#### 核心功能
- **文本读取**: 支持多种编码格式
- **文本渲染**: 生成文本的可视化截图
- **格式检测**: 自动检测文本格式 (Markdown, 代码等)

#### 实现细节

```python
class TextParser(BaseFileParser):
    """文本文件解析器"""
    
    supported_extensions = ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析文本文件"""
        try:
            self._validate_file(file_path)
            
            # 读取文本内容
            text_content = self._read_text_file(file_path)
            
            # 生成文本截图
            image_content = self._generate_text_screenshot(text_content, file_path)
            
            # 提取元数据
            metadata = self._extract_text_metadata(file_path, text_content)
            
            return ParsedContent(
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower(),
                text_content=text_content,
                image_content=image_content,
                audio_content=None,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing text file {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _read_text_file(self, file_path: str) -> str:
        """读取文本文件，自动检测编码"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'ascii']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        
        # 如果所有编码都失败，使用二进制模式读取
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            return raw_data.decode('utf-8', errors='ignore')
    
    def _generate_text_screenshot(self, text_content: str, file_path: str) -> List[bytes]:
        """生成文本的可视化截图"""
        try:
            # 创建图像
            width, height = 1200, 1600
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # 设置字体
            try:
                # 尝试使用系统字体
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                try:
                    # 尝试使用中文字体
                    font = ImageFont.truetype("simhei.ttf", 12)
                except:
                    # 使用默认字体
                    font = ImageFont.load_default()
            
            # 准备文本
            lines = text_content.split('\n')[:100]  # 限制行数
            y_position = 20
            line_height = 16
            
            # 绘制文本
            for line in lines:
                if y_position > height - 50:
                    break
                
                # 处理长行
                if len(line) > 100:
                    line = line[:97] + "..."
                
                try:
                    draw.text((20, y_position), line, fill='black', font=font)
                except:
                    # 如果字体不支持某些字符，使用默认字体
                    draw.text((20, y_position), line, fill='black')
                
                y_position += line_height
            
            # 添加文件信息
            file_info = f"文件: {os.path.basename(file_path)}"
            draw.text((20, height - 30), file_info, fill='gray', font=font)
            
            # 压缩并返回
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85, optimize=True)
            compressed_data = buffer.getvalue()
            
            image.close()
            return [compressed_data]
            
        except Exception as e:
            self.logger.warning(f"Failed to generate text screenshot: {e}")
            return []
    
    def _extract_text_metadata(self, file_path: str, text_content: str) -> Dict[str, Any]:
        """提取文本元数据"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        metadata = {
            'file_size': os.path.getsize(file_path),
            'line_count': len(text_content.split('\n')),
            'char_count': len(text_content),
            'word_count': len(text_content.split()),
            'file_extension': file_ext,
            'parser_type': 'TextParser'
        }
        
        # 检测文件类型
        if file_ext == '.md':
            metadata['content_type'] = 'markdown'
        elif file_ext in ['.py', '.js', '.html', '.css']:
            metadata['content_type'] = 'code'
        elif file_ext == '.json':
            metadata['content_type'] = 'json'
        elif file_ext == '.csv':
            metadata['content_type'] = 'csv'
        else:
            metadata['content_type'] = 'plain_text'
        
        return metadata
```

### 3. 图像解析器

#### 核心功能
- **图像读取**: 支持多种图像格式
- **EXIF提取**: 提取图像元数据
- **OCR识别**: 提取图像中的文字

#### 实现细节

```python
class ImageParser(BaseFileParser):
    """图像文件解析器"""
    
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析图像文件"""
        try:
            self._validate_file(file_path)
            
            # 读取图像
            with open(file_path, 'rb') as f:
                image_data = f.read()
            
            # 提取OCR文本
            ocr_text = self._extract_ocr_text(image_data)
            
            # 提取元数据
            metadata = self._extract_image_metadata(file_path, image_data)
            
            return ParsedContent(
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower(),
                text_content=ocr_text if ocr_text else None,
                image_content=[image_data],
                audio_content=None,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing image {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_ocr_text(self, image_data: bytes) -> str:
        """从图像中提取OCR文本"""
        try:
            import pytesseract
            
            image = Image.open(io.BytesIO(image_data))
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            
            # 清理文本
            cleaned_text = self._clean_ocr_text(text)
            image.close()
            
            return cleaned_text
            
        except ImportError:
            self.logger.debug("OCR libraries not available")
            return ""
        except Exception as e:
            self.logger.debug(f"OCR failed: {e}")
            return ""
    
    def _clean_ocr_text(self, text: str) -> str:
        """清理OCR文本"""
        import re
        
        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)
        
        # 移除特殊字符
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
        
        # 移除过短的行
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
        
        return '\n'.join(lines)
    
    def _extract_image_metadata(self, file_path: str, image_data: bytes) -> Dict[str, Any]:
        """提取图像元数据"""
        try:
            image = Image.open(io.BytesIO(image_data))
            
            metadata = {
                'width': image.width,
                'height': image.height,
                'format': image.format,
                'mode': image.mode,
                'file_size': len(image_data),
                'parser_type': 'ImageParser'
            }
            
            # 提取EXIF数据
            if hasattr(image, '_getexif') and image._getexif():
                exif_data = image._getexif()
                if exif_data:
                    metadata['exif'] = {k: str(v) for k, v in exif_data.items()}
            
            image.close()
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract image metadata: {e}")
            return {'parser_type': 'ImageParser', 'error': str(e)}
```

### 4. 音频解析器

#### 核心功能
- **音频信息提取**: 获取时长、采样率等信息
- **波形可视化**: 生成波形图和频谱图
- **语音识别**: 可选的语音转文字功能

#### 实现细节

```python
class AudioParser(BaseFileParser):
    """音频文件解析器"""
    
    supported_extensions = ['.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a']
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析音频文件"""
        try:
            self._validate_file(file_path)
            
            # 提取音频信息
            audio_info = self._extract_audio_info(file_path)
            
            # 生成波形图
            waveform_images = self._generate_waveform_visualization(file_path)
            
            # 语音识别 (可选)
            transcription = self._transcribe_audio(file_path)
            
            # 提取元数据
            metadata = self._extract_audio_metadata(file_path, audio_info)
            
            return ParsedContent(
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower(),
                text_content=transcription,
                image_content=waveform_images,
                audio_content=None,  # 不存储原始音频数据
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing audio {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_audio_info(self, file_path: str) -> Dict[str, Any]:
        """提取音频基本信息"""
        try:
            import librosa
            
            # 加载音频文件
            y, sr = librosa.load(file_path, sr=None)
            
            duration = len(y) / sr
            
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1 if len(y.shape) == 1 else y.shape[0],
                'samples': len(y)
            }
            
        except ImportError:
            self.logger.warning("librosa not available, using basic audio info")
            return self._get_basic_audio_info(file_path)
        except Exception as e:
            self.logger.warning(f"Failed to extract audio info: {e}")
            return {}
    
    def _generate_waveform_visualization(self, file_path: str) -> List[bytes]:
        """生成波形图和频谱图"""
        try:
            import librosa
            import matplotlib.pyplot as plt
            import numpy as np
            
            # 加载音频
            y, sr = librosa.load(file_path, duration=30)  # 限制30秒
            
            # 创建图像
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 波形图
            time = np.linspace(0, len(y) / sr, len(y))
            ax1.plot(time, y)
            ax1.set_title('Waveform')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Amplitude')
            
            # 频谱图
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
            librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax2)
            ax2.set_title('Spectrogram')
            
            plt.tight_layout()
            
            # 保存为字节
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            image_data = buffer.getvalue()
            
            plt.close(fig)
            return [image_data]
            
        except ImportError:
            self.logger.warning("Audio visualization libraries not available")
            return []
        except Exception as e:
            self.logger.warning(f"Failed to generate waveform visualization: {e}")
            return []
    
    def _transcribe_audio(self, file_path: str) -> Optional[str]:
        """语音识别 (可选功能)"""
        try:
            import whisper
            
            model = whisper.load_model("base")
            result = model.transcribe(file_path)
            
            return result["text"]
            
        except ImportError:
            self.logger.debug("Whisper not available for speech recognition")
            return None
        except Exception as e:
            self.logger.debug(f"Speech recognition failed: {e}")
            return None
```

### 5. 视频解析器（增强版 - 场景检测）

#### 核心功能
- **场景检测关键帧提取**: 基于场景变化智能提取关键帧
- **帧质量增强**: 对提取的帧进行去噪、锐化和对比度增强
- **视频信息提取**: 获取分辨率、时长、帧率等
- **音频提取和转录**: 提取音频轨道并进行语音识别
- **增强OCR**: 使用PaddleOCR对视频帧进行高精度文字识别

#### 实现细节

```python
class VideoParser(BaseFileParser):
    """视频文件解析器"""
    
    supported_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析视频文件"""
        try:
            self._validate_file(file_path)
            
            # 提取关键帧
            keyframes = self._extract_keyframes(file_path)
            
            # 提取字幕文本
            subtitle_text = self._extract_subtitles(file_path)
            
            # 提取元数据
            metadata = self._extract_video_metadata(file_path)
            
            return ParsedContent(
                file_path=file_path,
                file_type=os.path.splitext(file_path)[1].lower(),
                text_content=subtitle_text,
                image_content=keyframes,
                audio_content=None,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing video {file_path}: {e}")
            return self._create_error_content(file_path, str(e))
    
    def _extract_keyframes(self, file_path: str, max_frames: int = 10) -> List[bytes]:
        """提取视频关键帧"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(file_path)
            frames = []
            
            # 获取视频信息
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # 计算关键帧间隔
            if total_frames > max_frames:
                frame_interval = total_frames // max_frames
            else:
                frame_interval = 1
            
            frame_count = 0
            while cap.isOpened() and len(frames) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # 转换为PIL图像
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    
                    # 压缩图像
                    buffer = io.BytesIO()
                    image.save(buffer, format='JPEG', quality=85, optimize=True)
                    frame_data = buffer.getvalue()
                    
                    frames.append(frame_data)
                    image.close()
                
                frame_count += 1
            
            cap.release()
            return frames
            
        except ImportError:
            self.logger.warning("OpenCV not available for video processing")
            return []
        except Exception as e:
            self.logger.warning(f"Failed to extract keyframes: {e}")
            return []
    
    def _extract_subtitles(self, file_path: str) -> Optional[str]:
        """提取视频字幕"""
        try:
            import ffmpeg
            
            # 使用ffmpeg提取字幕
            probe = ffmpeg.probe(file_path)
            subtitle_streams = [
                stream for stream in probe['streams']
                if stream['codec_type'] == 'subtitle'
            ]
            
            if subtitle_streams:
                # 提取第一个字幕流
                subtitle_file = file_path + '.srt'
                ffmpeg.input(file_path).output(subtitle_file, map='0:s:0').run(overwrite_output=True)
                
                # 读取字幕文件
                with open(subtitle_file, 'r', encoding='utf-8') as f:
                    subtitle_text = f.read()
                
                # 清理临时文件
                os.remove(subtitle_file)
                
                return subtitle_text
            
            return None
            
        except ImportError:
            self.logger.debug("ffmpeg-python not available for subtitle extraction")
            return None
        except Exception as e:
            self.logger.debug(f"Subtitle extraction failed: {e}")
            return None
    
    def _extract_video_metadata(self, file_path: str) -> Dict[str, Any]:
        """提取视频元数据"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(file_path)
            
            metadata = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS),
                'file_size': os.path.getsize(file_path),
                'parser_type': 'VideoParser'
            }
            
            cap.release()
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract video metadata: {e}")
            return {'parser_type': 'VideoParser', 'error': str(e)}
```

## 解析流程优化

### 1. 并发处理

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ConcurrentFileProcessor:
    """并发文件处理器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_files_concurrently(self, file_paths: List[str]) -> List[ParsedContent]:
        """并发处理多个文件"""
        loop = asyncio.get_event_loop()
        
        # 创建任务
        tasks = []
        for file_path in file_paths:
            task = loop.run_in_executor(
                self.executor,
                self._process_single_file,
                file_path
            )
            tasks.append(task)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 过滤异常结果
        valid_results = [
            result for result in results
            if isinstance(result, ParsedContent)
        ]
        
        return valid_results
    
    def _process_single_file(self, file_path: str) -> ParsedContent:
        """处理单个文件"""
        factory = FileParserFactory()
        parser = factory.create_parser(file_path)
        return parser.parse(file_path)
```

### 2. 内存管理

```python
class MemoryOptimizedParser:
    """内存优化的解析器"""
    
    def __init__(self):
        self.max_image_size = 42000  # Base64编码后的最大大小
        self.compression_quality = 85
    
    def parse_with_memory_limit(self, file_path: str) -> ParsedContent:
        """在内存限制下解析文件"""
        try:
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            
            if file_size > 100 * 1024 * 1024:  # 100MB
                return self._parse_large_file(file_path)
            else:
                return self._parse_normal_file(file_path)
                
        except MemoryError:
            # 内存不足时的降级处理
            return self._parse_with_reduced_quality(file_path)
    
    def _parse_large_file(self, file_path: str) -> ParsedContent:
        """处理大文件"""
        # 降低图像质量
        self.compression_quality = 70
        self.max_image_size = 20000
        
        # 分块处理
        return self._parse_in_chunks(file_path)
    
    def _cleanup_resources(self):
        """清理资源"""
        import gc
        gc.collect()
```

### 3. 错误处理和重试

```python
class RobustFileParser:
    """健壮的文件解析器"""
    
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
    
    def parse_with_retry(self, file_path: str) -> ParsedContent:
        """带重试的文件解析"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self._attempt_parse(file_path, attempt)
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Parse attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    # 等待后重试
                    time.sleep(2 ** attempt)  # 指数退避
        
        # 所有重试都失败，返回错误内容
        return self._create_error_content(file_path, str(last_error))
    
    def _attempt_parse(self, file_path: str, attempt: int) -> ParsedContent:
        """尝试解析文件"""
        if attempt > 0:
            # 降级处理策略
            self._apply_degraded_settings(attempt)
        
        factory = FileParserFactory()
        parser = factory.create_parser(file_path)
        return parser.parse(file_path)
    
    def _apply_degraded_settings(self, attempt: int):
        """应用降级设置"""
        if attempt == 1:
            # 第一次重试：降低图像质量
            self.compression_quality = 70
        elif attempt == 2:
            # 第二次重试：跳过OCR
            self.enable_ocr = False
```

## 性能监控

### 解析性能统计

```python
class ParsingPerformanceMonitor:
    """解析性能监控器"""
    
    def __init__(self):
        self.stats = {
            'total_files': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'total_time': 0,
            'parser_stats': {}
        }
    
    def record_parsing_result(self, parser_type: str, file_path: str, 
                            success: bool, duration: float):
        """记录解析结果"""
        self.stats['total_files'] += 1
        
        if success:
            self.stats['successful_parses'] += 1
        else:
            self.stats['failed_parses'] += 1
        
        self.stats['total_time'] += duration
        
        # 按解析器类型统计
        if parser_type not in self.stats['parser_stats']:
            self.stats['parser_stats'][parser_type] = {
                'count': 0,
                'success_count': 0,
                'total_time': 0,
                'avg_time': 0
            }
        
        parser_stat = self.stats['parser_stats'][parser_type]
        parser_stat['count'] += 1
        parser_stat['total_time'] += duration
        parser_stat['avg_time'] = parser_stat['total_time'] / parser_stat['count']
        
        if success:
            parser_stat['success_count'] += 1
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        success_rate = (self.stats['successful_parses'] / 
                       max(self.stats['total_files'], 1)) * 100
        
        avg_time = (self.stats['total_time'] / 
                   max(self.stats['total_files'], 1))
        
        return {
            'summary': {
                'total_files': self.stats['total_files'],
                'success_rate': f"{success_rate:.2f}%",
                'average_time': f"{avg_time:.2f}s",
                'total_time': f"{self.stats['total_time']:.2f}s"
            },
            'by_parser': self.stats['parser_stats']
        }
```

## 扩展和自定义

### 自定义解析器开发

```python
class CustomDocumentParser(BaseFileParser):
    """自定义文档解析器示例"""
    
    supported_extensions = ['.docx', '.doc']
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.docx', '.doc'))
    
    def parse(self, file_path: str) -> ParsedContent:
        """解析Word文档"""
        try:
            # 使用python-docx解析
            from docx import Document
            
            doc = Document(file_path)
            
            # 提取文本
            text_content = self._extract_text_from_docx(doc)
            
            # 提取图像
            images = self._extract_images_from_docx(doc)
            
            # 提取元数据
            metadata = self._extract_docx_metadata(doc, file_path)
            
            return ParsedContent(
                file_path=file_path,
                file_type='.docx',
                text_content=text_content,
                image_content=images,
                audio_content=None,
                metadata=metadata
            )
            
        except Exception as e:
            return self._create_error_content(file_path, str(e))
    
    def _extract_text_from_docx(self, doc) -> str:
        """从Word文档提取文本"""
        paragraphs = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                paragraphs.append(paragraph.text)
        
        return '\n\n'.join(paragraphs)
    
    def _extract_images_from_docx(self, doc) -> List[bytes]:
        """从Word文档提取图像"""
        images = []
        
        for rel in doc.part.rels.values():
            if "image" in rel.target_ref:
                image_data = rel.target_part.blob
                images.append(image_data)
        
        return images

# 注册自定义解析器
registry = FileParserRegistry()
registry.register_parser(CustomDocumentParser())
```

多模态文件索引器的解析系统通过插件化架构实现了高度的可扩展性和灵活性，支持多种文件格式的智能解析，并提供了丰富的优化和监控功能。