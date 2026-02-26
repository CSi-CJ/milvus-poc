"""
文件处理器模块
"""

import os
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
import logging

from .models import ParsedContent, FileMetadata, ContentMetadata, IndexRecord, ChunkContent
from .parsers.registry import FileParserRegistry
from .embedder import VectorEmbedder
from .index_manager import IndexManager
from .config import Config


class FileProcessor:
    """文件处理器"""
    
    def __init__(self, parser_registry: FileParserRegistry, 
                 embedder: VectorEmbedder, index_manager: IndexManager,
                 config: Config):
        self.parser_registry = parser_registry
        self.embedder = embedder
        self.index_manager = index_manager
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_file(self, file_path: str, 
                          collection_name: Optional[str] = None) -> Dict[str, Any]:
        """处理单个文件
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            Dict: 处理结果
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"Processing file: {file_path}")
            
            # 1. 验证文件
            if not os.path.exists(file_path):
                return self._create_error_result(file_path, "File not found")
            
            # 2. 检查是否跳过已存在的文件
            if self.config.processing.skip_existing:
                if await self._file_already_processed(file_path, collection_name):
                    return self._create_skip_result(file_path, "File already processed")
            
            # 3. 获取文件解析器
            parser = self.parser_registry.get_parser(file_path)
            if not parser:
                return self._create_error_result(file_path, "No suitable parser found")
            
            # 4. 解析文件
            parsed_content = parser.parse(file_path)
            if not parsed_content.has_content():
                return self._create_error_result(file_path, "No content extracted from file")
            
            # 5. 生成向量嵌入
            embeddings = self.embedder.embed_multimodal(parsed_content)
            if not embeddings:
                return self._create_error_result(file_path, "Failed to generate embeddings")
            
            # 6. 创建文件元数据
            file_metadata = FileMetadata.from_file_path(file_path)
            
            # 7. 准备索引数据
            index_records = self._prepare_index_records(
                file_metadata, parsed_content, embeddings
            )
            
            # 调试：检查向量维度
            for i, record in enumerate(index_records):
                vector_len = len(record.vector_embedding)
                self.logger.debug(f"Record {i}: vector length = {vector_len}")
                if vector_len != self.embedder.get_vector_dimension():
                    self.logger.warning(f"Vector dimension mismatch: {vector_len} vs expected {self.embedder.get_vector_dimension()}")
            
            # 8. 检查向量维度并处理不匹配
            expected_dim = self.embedder.get_vector_dimension()
            actual_dim = len(embeddings[0]) if embeddings else 0
            
            if actual_dim != expected_dim:
                self.logger.warning(f"Vector dimension mismatch: actual={actual_dim}, expected={expected_dim}")
                # 尝试重新创建集合以匹配新的向量维度
                try:
                    self.logger.info(f"Recreating collection with correct dimension: {actual_dim}")
                    self.index_manager.recreate_collection(collection_name, actual_dim, self.embedder)
                except Exception as recreate_error:
                    self.logger.error(f"Failed to recreate collection: {recreate_error}")
                    return self._create_error_result(file_path, f"Vector dimension mismatch and failed to recreate collection: {recreate_error}")
            
            # 9. 插入到 Milvus
            milvus_data = [record.to_milvus_data() for record in index_records]
            
            # 调试：检查 Milvus 数据中的向量维度
            for i, data in enumerate(milvus_data):
                vector_len = len(data['vector'])
                self.logger.debug(f"Milvus data {i}: vector length = {vector_len}")
            
            inserted_ids = self.index_manager.insert_vectors(milvus_data, collection_name, self.embedder)
            
            # 10. 计算处理时间
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'status': 'success',
                'file_path': file_path,
                'file_size': file_metadata.file_size,
                'file_type': file_metadata.file_type,
                'embeddings_count': len(embeddings),
                'inserted_ids': inserted_ids,
                'processing_time': processing_time,
                'metadata': parsed_content.metadata
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'status': 'error',
                'file_path': file_path,
                'error': str(e),
                'processing_time': processing_time
            }
    
    def _prepare_index_records(self, file_metadata: FileMetadata, 
                              content: ParsedContent, 
                              embeddings: List) -> List[IndexRecord]:
        """准备索引记录（增强版本，包含chunk内容）
        
        Args:
            file_metadata: 文件元数据
            content: 解析的内容
            embeddings: 向量嵌入列表
            
        Returns:
            List[IndexRecord]: 索引记录列表
        """
        records = []
        
        for i, embedding in enumerate(embeddings):
            # 生成唯一 ID
            record_id = self._generate_record_id(file_metadata.file_path, i)
            
            # 确定内容类型和实际内容
            content_type, chunk_content = self._extract_chunk_content(content, i)
            
            # 创建内容元数据
            content_metadata = ContentMetadata(
                content_type=content_type,
                content_length=len(chunk_content.content) if chunk_content.content else 0,
                language=content.metadata.get('language'),
                encoding=content.metadata.get('encoding'),
                dimensions=content.metadata.get('dimensions'),
                duration=content.metadata.get('duration')
            )
            
            # 创建索引记录
            record = IndexRecord(
                id=record_id,
                file_metadata=file_metadata,
                content_metadata=content_metadata,
                vector_embedding=embedding.tolist(),
                chunk_content=chunk_content,  # 新增：chunk内容
                chunk_index=i
            )
            
            records.append(record)
        
        return records
    
    def _extract_chunk_content(self, content: ParsedContent, index: int) -> tuple[str, ChunkContent]:
        """提取chunk内容（增强版，支持图像数据存储）
        
        Args:
            content: 解析的内容
            index: 嵌入索引
            
        Returns:
            tuple: (content_type, ChunkContent)
        """
        # 优先处理音频内容（即使有text_content也应该标记为audio类型）
        if content.audio_content:
            # 从解析内容中获取转录文本
            audio_transcript = content.text_content  # 音频解析器会将转录文本放在text_content中
            
            audio_description = f"Audio content from {content.file_type} file"
            
            # 尝试从元数据获取更多信息
            if 'duration' in content.metadata:
                audio_description += f", duration: {content.metadata['duration']:.1f}s"
            if 'sample_rate' in content.metadata:
                audio_description += f", sample rate: {content.metadata['sample_rate']}Hz"
            if 'channels' in content.metadata:
                audio_description += f", channels: {content.metadata['channels']}"
            
            # 如果有转录文本，添加到描述中
            if audio_transcript and audio_transcript.strip():
                audio_description += f"\n\n转录内容：\n{audio_transcript.strip()}"
                summary = f"音频转录: {audio_transcript[:50]}..." if len(audio_transcript) > 50 else f"音频转录: {audio_transcript}"
            else:
                summary = "音频内容"
            
            # 从元数据中获取语音识别相关信息
            audio_language = content.metadata.get('detected_language', 'unknown')
            audio_confidence = content.metadata.get('transcription_confidence', 0.0)
            
            return "audio", ChunkContent(
                content=audio_description,
                content_type="audio_description",
                summary=summary,
                audio_transcript=audio_transcript,
                audio_language=audio_language,
                audio_confidence=audio_confidence
            )
        
        # 处理纯文本内容（非音频文件的文本）- 优先处理index 0的文本
        elif content.text_content and index == 0:
            # 为长文本创建摘要
            text = content.text_content
            summary = self._create_text_summary(text)
            
            return "text", ChunkContent(
                content=text,
                content_type="text",
                summary=summary
            )
        
        # 处理图像内容
        elif content.image_content and len(content.image_content) > 0:
            # 调整索引计算逻辑
            image_index = index
            
            # 如果有文本内容且不是音频文件，需要调整图像索引
            if content.text_content and not content.audio_content:
                # 文本内容占用索引0，图像从索引1开始
                image_index = index - 1
            
            # 确保图像索引在有效范围内
            if image_index >= 0 and image_index < len(content.image_content):
                image_data = content.image_content[image_index]
                
                # 将图像数据转换为Base64编码
                import base64
                from PIL import Image
                import io
                
                try:
                    # 获取图像信息
                    image = Image.open(io.BytesIO(image_data))
                    image_format = image.format or "PNG"
                    image_size = f"{image.width}x{image.height}"
                    
                    # 调整压缩策略以保持更高清晰度（在65KB限制内）
                    if len(image_data) > 100000:  # 100KB以上才压缩
                        # 对于大图像，适度压缩但保持高质量
                        if image.width > 1600 or image.height > 1600:
                            # 保持高分辨率，最大1200px
                            image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                        
                        buffer = io.BytesIO()
                        # 使用高质量压缩
                        if image_format.upper() == 'PNG':
                            image.save(buffer, format='PNG', optimize=True)
                        else:
                            image.save(buffer, format='JPEG', quality=85, optimize=True)  # 质量85
                        image_data = buffer.getvalue()
                        image_format = image.format or "JPEG"
                        image_size = f"{image.width}x{image.height}"
                    
                    # Base64编码
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    
                    # 如果编码后的数据仍然太大，进行二次压缩（保持在65KB限制内）
                    if len(image_base64) > 42000:  # 42KB限制（留出更多余量）
                        # 进一步压缩，但保持较高质量
                        if image.width > 1000 or image.height > 1000:
                            image.thumbnail((1000, 1000), Image.Resampling.LANCZOS)
                        
                        buffer = io.BytesIO()
                        image.save(buffer, format="JPEG", quality=80, optimize=True)  # 质量80
                        compressed_data = buffer.getvalue()
                        image_base64 = base64.b64encode(compressed_data).decode('utf-8')
                        image_format = "JPEG"
                        image_size = f"{image.width}x{image.height}"
                        
                        # 如果还是太大，最后一次压缩
                        if len(image_base64) > 42000:
                            if image.width > 800 or image.height > 800:
                                image.thumbnail((800, 800), Image.Resampling.LANCZOS)
                            
                            buffer = io.BytesIO()
                            image.save(buffer, format="JPEG", quality=75, optimize=True)  # 质量75
                            final_data = buffer.getvalue()
                            image_base64 = base64.b64encode(final_data).decode('utf-8')
                            image_format = "JPEG"
                            image_size = f"{image.width}x{image.height}"
                            
                            # 最终检查，如果还是太大，进行最后压缩
                            if len(image_base64) > 42000:
                                if image.width > 600 or image.height > 600:
                                    image.thumbnail((600, 600), Image.Resampling.LANCZOS)
                                
                                buffer = io.BytesIO()
                                image.save(buffer, format="JPEG", quality=70, optimize=True)  # 质量70
                                ultra_final_data = buffer.getvalue()
                                image_base64 = base64.b64encode(ultra_final_data).decode('utf-8')
                                image_format = "JPEG"
                                image_size = f"{image.width}x{image.height}"
                                
                                # 最终安全检查：如果还是超过限制，强制截断（这不应该发生，但作为安全措施）
                                if len(image_base64) > 65500:  # 留出35字符余量
                                    self.logger.warning(f"Image still too large after all compression: {len(image_base64)} chars, truncating")
                                    image_base64 = image_base64[:65500]
                    
                except Exception as e:
                    self.logger.warning(f"Failed to process image {image_index}: {e}")
                    # 如果处理失败，使用原始数据
                    image_base64 = base64.b64encode(image_data).decode('utf-8')
                    image_format = "Unknown"
                    image_size = f"{len(image_data)} bytes"
                
                # 尝试从图像中提取OCR文本
                ocr_text = self._extract_image_text(image_data)
                
                # 创建图像描述
                if ocr_text and ocr_text.strip():
                    # 如果有OCR文本，使用OCR文本作为主要内容
                    image_description = f"图像 {image_index + 1} 包含文本内容：\n{ocr_text.strip()}"
                    summary = f"图像 {image_index + 1}: {ocr_text[:50]}..." if len(ocr_text) > 50 else f"图像 {image_index + 1}: {ocr_text}"
                else:
                    # 根据文件类型提供不同的描述
                    if content.file_type == '.pdf':
                        image_description = f"PDF页面 {image_index + 1} 的截图"
                        summary = f"PDF页面 {image_index + 1}"
                    elif content.file_type in ['.txt', '.md']:
                        image_description = f"文本文件的可视化截图"
                        summary = f"文本截图"
                    elif content.file_type == '.docx':
                        image_description = f"Word文档的页面截图"
                        summary = f"Word页面截图"
                    elif content.file_type in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                        image_description = f"音频文件的波形图和频谱图"
                        summary = f"音频波形图"
                    elif content.file_type in ['.mp4', '.avi', '.mov', '.mkv']:
                        image_description = f"视频关键帧 {image_index + 1}"
                        summary = f"视频帧 {image_index + 1}"
                    elif content.file_type in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                        image_description = f"图像文件内容"
                        summary = f"图像内容"
                    else:
                        image_description = f"文件内容的可视化截图 {image_index + 1}"
                        summary = f"内容截图 {image_index + 1}"
                
                # 尝试从元数据获取更多信息
                if 'images_info' in content.metadata and image_index < len(content.metadata['images_info']):
                    img_info = content.metadata['images_info'][image_index]
                    if 'size' in img_info:
                        image_description += f"\n图像尺寸: {img_info['size']}"
                    if 'format' in img_info:
                        image_description += f"\n图像格式: {img_info['format']}"
                
                # 添加图像位置信息
                if 'page_count' in content.metadata:
                    # 估算图像所在页面（简单估算）
                    estimated_page = min(image_index + 1, content.metadata['page_count'])
                    image_description += f"\n估计位置: 第 {estimated_page} 页"
                
                return "image", ChunkContent(
                    content=image_description,
                    content_type="image_description",
                    summary=summary,
                    image_data=image_base64,
                    image_format=image_format,
                    image_size=image_size,
                    ocr_text=ocr_text
                )
        
        # 处理纯文本内容（非音频文件的文本）
        elif content.text_content and index == 0:
            # 为长文本创建摘要
            text = content.text_content
            summary = self._create_text_summary(text)
            
            return "text", ChunkContent(
                content=text,
                content_type="text",
                summary=summary
            )
        
        # 处理超出范围的索引 - 生成通用描述
        else:
            # 根据文件类型生成合适的描述
            if content.file_type in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                description = f"音频文件的附加内容块 {index}"
                content_type = "audio_metadata"
            elif content.file_type in ['.mp4', '.avi', '.mov', '.mkv']:
                description = f"视频文件的附加内容块 {index}"
                content_type = "video_metadata"
            elif content.file_type == '.pdf':
                description = f"PDF文档的附加内容块 {index}"
                content_type = "document_metadata"
            elif content.file_type in ['.png', '.jpg', '.jpeg', '.gif', '.bmp']:
                description = f"图像文件的附加内容块 {index}"
                content_type = "image_metadata"
            elif content.file_type in ['.txt', '.md', '.docx', '.doc']:
                description = f"文档的附加内容块 {index}"
                content_type = "document_metadata"
            else:
                description = f"文件的附加内容块 {index}"
                content_type = "file_metadata"
            
            return content_type.split('_')[0], ChunkContent(
                content=description,
                content_type=content_type,
                summary=f"附加内容 {index}"
            )
        
        # 这个分支理论上不应该被执行到
        return "unknown", ChunkContent(
            content=f"Content chunk {index} from {content.file_type} file",
            content_type="unknown",
            summary=f"Chunk {index}"
        )
    
    def _extract_image_text(self, image_data: bytes) -> str:
        """从图像中提取OCR文本（使用改进的OCR引擎）
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            str: 提取的文本内容
        """
        try:
            # 使用改进的OCR引擎
            if not hasattr(self, '_ocr_engine'):
                from .improved_ocr_engine import ImprovedOCREngine
                self._ocr_engine = ImprovedOCREngine()
                self.logger.info(f"OCR引擎初始化完成，使用: {self._ocr_engine.primary_engine}")
            
            # 执行OCR
            extracted_text = self._ocr_engine.extract_text(image_data)
            
            if extracted_text:
                self.logger.debug(f"OCR成功提取文本: {extracted_text[:100]}...")
            else:
                self.logger.debug("OCR未提取到文本")
            
            return extracted_text
            
        except Exception as e:
            # OCR失败，回退到Tesseract
            self.logger.warning(f"改进OCR引擎失败，回退到Tesseract: {e}")
            return self._extract_image_text_fallback(image_data)
    
    def _extract_image_text_fallback(self, image_data: bytes) -> str:
        """OCR回退方案（使用Tesseract）
        
        Args:
            image_data: 图像二进制数据
            
        Returns:
            str: 提取的文本内容
        """
        try:
            # 尝试使用PIL和pytesseract进行OCR
            from PIL import Image
            import pytesseract
            import io
            
            # 将字节数据转换为PIL图像
            image = Image.open(io.BytesIO(image_data))
            
            # 执行OCR，支持中英文，使用优化参数
            # PSM 6: 假设单个统一的文本块
            # OEM 1: 使用LSTM OCR引擎
            config = '--psm 6 --oem 1'
            text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=config)
            
            # 后处理：移除多余空格，但保留必要的分隔
            processed_text = self._post_process_ocr_text(text)
            
            return processed_text
            
        except ImportError:
            # 如果没有安装OCR库，返回空字符串
            self.logger.debug("OCR libraries not available (PIL, pytesseract)")
            return ""
        except Exception as e:
            # OCR失败，返回空字符串
            self.logger.debug(f"Fallback OCR failed: {e}")
            return ""
    
    def _post_process_ocr_text(self, text: str) -> str:
        """后处理OCR文本，改善可读性"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = text.strip()
        
        # 处理中文字符间的空格
        import re
        
        # 移除中文字符之间的单个空格，但保留换行
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
    
    def _create_text_summary(self, text: str, max_length: int = 200) -> str:
        """创建文本摘要
        
        Args:
            text: 原始文本
            max_length: 摘要最大长度
            
        Returns:
            str: 文本摘要
        """
        if not text:
            return ""
        
        # 简单的摘要策略：取前几句话
        sentences = text.split('。')
        summary = ""
        
        for sentence in sentences:
            if len(summary + sentence + "。") <= max_length:
                summary += sentence + "。"
            else:
                break
        
        if not summary and len(text) > max_length:
            summary = text[:max_length] + "..."
        elif not summary:
            summary = text
        
        return summary.strip()
    
    def _generate_record_id(self, file_path: str, chunk_index: int) -> str:
        """生成记录 ID
        
        Args:
            file_path: 文件路径
            chunk_index: 块索引
            
        Returns:
            str: 唯一记录 ID
        """
        # 使用文件路径和块索引生成唯一 ID
        content = f"{file_path}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _determine_content_type(self, content: ParsedContent, index: int) -> str:
        """确定内容类型
        
        Args:
            content: 解析的内容
            index: 嵌入索引
            
        Returns:
            str: 内容类型
        """
        # 简单的内容类型确定逻辑
        if content.text_content and index == 0:
            return "text"
        elif content.image_content and index < len(content.image_content or []):
            return "image"
        elif content.audio_content:
            return "audio"
        else:
            return "unknown"
    
    def _get_content_length(self, content: ParsedContent, content_type: str) -> int:
        """获取内容长度
        
        Args:
            content: 解析的内容
            content_type: 内容类型
            
        Returns:
            int: 内容长度
        """
        if content_type == "text" and content.text_content:
            return len(content.text_content)
        elif content_type == "image" and content.image_content:
            return sum(len(img) for img in content.image_content)
        elif content_type == "audio" and content.audio_content:
            return len(content.audio_content)
        else:
            return 0
    
    async def _file_already_processed(self, file_path: str, 
                                    collection_name: Optional[str] = None) -> bool:
        """检查文件是否已处理
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            
        Returns:
            bool: 是否已处理
        """
        try:
            # 生成文件的第一个记录 ID
            record_id = self._generate_record_id(file_path, 0)
            
            # 在 Milvus 中搜索
            results = self.index_manager.search_vectors(
                query_vectors=[[0.0] * self.embedder.get_vector_dimension()],
                collection_name=collection_name,
                top_k=1,
                expr=f'id == "{record_id}"'
            )
            
            return len(results) > 0
            
        except Exception as e:
            self.logger.debug(f"Error checking if file processed: {e}")
            return False
    
    def _create_error_result(self, file_path: str, error_message: str) -> Dict[str, Any]:
        """创建错误结果
        
        Args:
            file_path: 文件路径
            error_message: 错误信息
            
        Returns:
            Dict: 错误结果
        """
        return {
            'status': 'error',
            'file_path': file_path,
            'error': error_message
        }
    
    def _create_skip_result(self, file_path: str, reason: str) -> Dict[str, Any]:
        """创建跳过结果
        
        Args:
            file_path: 文件路径
            reason: 跳过原因
            
        Returns:
            Dict: 跳过结果
        """
        return {
            'status': 'skipped',
            'file_path': file_path,
            'reason': reason
        }


class BatchProcessor:
    """批量处理器"""
    
    def __init__(self, file_processor: FileProcessor, max_concurrent: int = 10):
        self.file_processor = file_processor
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    async def process_directory(self, directory_path: str, 
                               collection_name: Optional[str] = None,
                               progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """批量处理目录中的文件
        
        Args:
            directory_path: 目录路径
            collection_name: 集合名称
            progress_callback: 进度回调函数
            
        Returns:
            Dict: 处理结果汇总
        """
        start_time = datetime.now()
        
        try:
            directory = Path(directory_path)
            if not directory.exists():
                raise ValueError(f"Directory not found: {directory_path}")
            
            # 收集所有支持的文件
            files = self._collect_files(directory)
            total_files = len(files)
            
            if total_files == 0:
                return {
                    'status': 'completed',
                    'total_files': 0,
                    'successful': 0,
                    'failed': 0,
                    'skipped': 0,
                    'processing_time': 0,
                    'results': []
                }
            
            self.logger.info(f"Found {total_files} files to process in {directory_path}")
            
            # 并发处理文件
            tasks = [
                self._process_file_with_semaphore(file_path, collection_name, i, total_files, progress_callback)
                for i, file_path in enumerate(files)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 处理异常结果
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'status': 'error',
                        'file_path': files[i],
                        'error': str(result)
                    })
                else:
                    processed_results.append(result)
            
            # 统计结果
            summary = self._summarize_results(processed_results, start_time)
            
            self.logger.info(f"Batch processing completed: {summary['successful']}/{total_files} successful")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
    
    async def _process_file_with_semaphore(self, file_path: str, collection_name: Optional[str],
                                         file_index: int, total_files: int,
                                         progress_callback: Optional[Callable]) -> Dict[str, Any]:
        """使用信号量控制并发的文件处理
        
        Args:
            file_path: 文件路径
            collection_name: 集合名称
            file_index: 文件索引
            total_files: 总文件数
            progress_callback: 进度回调函数
            
        Returns:
            Dict: 处理结果
        """
        async with self.semaphore:
            result = await self.file_processor.process_file(file_path, collection_name)
            
            # 调用进度回调
            if progress_callback:
                try:
                    progress_callback(file_index + 1, total_files, result)
                except Exception as e:
                    self.logger.warning(f"Error in progress callback: {e}")
            
            return result
    
    def _collect_files(self, directory: Path) -> List[str]:
        """收集目录中的所有支持文件
        
        Args:
            directory: 目录路径
            
        Returns:
            List[str]: 文件路径列表
        """
        supported_extensions = set(self.file_processor.config.processing.supported_extensions)
        
        files = []
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in supported_extensions:
                    files.append(str(file_path))
        
        return sorted(files)
    
    def _summarize_results(self, results: List[Dict[str, Any]], 
                          start_time: datetime) -> Dict[str, Any]:
        """汇总处理结果
        
        Args:
            results: 处理结果列表
            start_time: 开始时间
            
        Returns:
            Dict: 汇总结果
        """
        successful = sum(1 for r in results if r.get('status') == 'success')
        failed = sum(1 for r in results if r.get('status') == 'error')
        skipped = sum(1 for r in results if r.get('status') == 'skipped')
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 计算统计信息
        total_size = sum(r.get('file_size', 0) for r in results if r.get('file_size'))
        total_embeddings = sum(r.get('embeddings_count', 0) for r in results if r.get('embeddings_count'))
        
        return {
            'status': 'completed',
            'total_files': len(results),
            'successful': successful,
            'failed': failed,
            'skipped': skipped,
            'processing_time': processing_time,
            'total_size': total_size,
            'total_embeddings': total_embeddings,
            'average_time_per_file': processing_time / len(results) if results else 0,
            'results': results
        }