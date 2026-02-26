# 切片策略详解

## 切片策略概述

多模态文件索引器采用基于内容类型的智能切片策略，将不同类型的文件内容分解为独立的chunk，每个chunk对应一个向量嵌入，实现精确的多模态检索。

## 核心切片原理

### 1. 切片类型分类

系统根据内容特性将chunk分为三种主要类型：

```python
CHUNK_TYPES = {
    "text": "文本内容chunk",
    "image": "图像内容chunk", 
    "audio": "音频内容chunk"
}
```

### 2. 索引分配策略

```python
def determine_chunk_type(content: ParsedContent, index: int) -> str:
    """
    根据索引位置和内容类型确定chunk类型
    
    规则：
    - index = 0: 文本类型 (如果存在文本内容)
    - index > 0: 图像/音频类型 (按顺序分配)
    """
    if content.text_content and index == 0:
        return "text"
    elif content.image_content and index > 0:
        return "image"
    elif content.audio_content:
        return "audio"
    else:
        return "unknown"
```

## 详细切片策略

### 1. Text类型切片 (index=0)

#### 切片条件
```python
if content.text_content and index == 0:
    return "text", ChunkContent(...)
```

#### 内容处理
- **完整文本**: 存储文档的完整文本内容
- **智能摘要**: 生成200字符以内的文本摘要
- **语言检测**: 自动检测文本语言
- **编码处理**: 统一UTF-8编码

#### 文本摘要算法
```python
def create_text_summary(self, text: str, max_length: int = 200) -> str:
    """创建文本摘要"""
    if not text:
        return ""
    
    # 1. 句子分割
    sentences = text.split('。')
    summary = ""
    
    # 2. 逐句添加直到达到长度限制
    for sentence in sentences:
        if len(summary + sentence + "。") <= max_length:
            summary += sentence + "。"
        else:
            break
    
    # 3. 处理边界情况
    if not summary and len(text) > max_length:
        summary = text[:max_length] + "..."
    elif not summary:
        summary = text
    
    return summary.strip()
```

#### 存储结构
```json
{
    "content_type": "text",
    "chunk_content": "完整的文档文本内容...",
    "chunk_summary": "文档摘要，包含主要信息...",
    "content_length": 10522,
    "image_data": null,
    "ocr_text": null
}
```

### 2. Image类型切片 (index>0)

#### 切片条件
```python
elif content.image_content and index > 0 and index <= len(content.image_content):
    image_index = index - (1 if content.text_content else 0)
    return "image", ChunkContent(...)
```

#### 索引计算逻辑
```python
# 计算图像在数组中的实际索引
image_index = index - (1 if content.text_content else 0)

# 示例：
# - 有文本内容的PDF：index=1对应image_index=0 (第一张图)
# - 纯图像文件：index=0对应image_index=0 (第一张图)
```

#### 图像处理流程

##### A. 图像数据处理
```python
def process_image_data(self, image_data: bytes) -> dict:
    """处理图像数据"""
    # 1. 图像信息提取
    image = Image.open(io.BytesIO(image_data))
    image_format = image.format or "PNG"
    image_size = f"{image.width}x{image.height}"
    
    # 2. 智能压缩
    compressed_data = self.compress_image_intelligently(image, image_data)
    
    # 3. Base64编码
    image_base64 = base64.b64encode(compressed_data).decode('utf-8')
    
    return {
        "image_data": image_base64,
        "image_format": image_format,
        "image_size": image_size,
        "original_size": len(image_data),
        "compressed_size": len(compressed_data)
    }
```

##### B. 智能压缩算法
```python
def compress_image_intelligently(self, image: Image.Image, original_data: bytes) -> bytes:
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
```

##### C. OCR文本提取
```python
def extract_image_text(self, image_data: bytes) -> str:
    """从图像中提取OCR文本"""
    try:
        # 1. 图像预处理
        image = Image.open(io.BytesIO(image_data))
        
        # 2. OCR识别 (支持中英文)
        text = pytesseract.image_to_string(image, lang='chi_sim+eng')
        
        # 3. 文本清理
        cleaned_text = self.clean_ocr_text(text)
        
        return cleaned_text.strip()
        
    except ImportError:
        self.logger.debug("OCR libraries not available")
        return ""
    except Exception as e:
        self.logger.debug(f"OCR failed: {e}")
        return ""

def clean_ocr_text(self, text: str) -> str:
    """清理OCR文本"""
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    
    # 移除过短的行
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 2]
    
    return '\n'.join(lines)
```

#### 图像描述生成

##### 基于OCR的描述 (优先级高)
```python
if ocr_text and ocr_text.strip():
    image_description = f"图像 {image_index + 1} 包含文本内容：\n{ocr_text.strip()}"
    summary = f"图像 {image_index + 1}: {ocr_text[:50]}..." if len(ocr_text) > 50 else f"图像 {image_index + 1}: {ocr_text}"
```

##### 基于文件类型的描述 (OCR失败时)
```python
def generate_type_based_description(self, content_type: str, image_index: int) -> tuple:
    """根据文件类型生成图像描述"""
    descriptions = {
        '.pdf': (f"PDF页面 {image_index + 1} 的截图", f"PDF页面 {image_index + 1}"),
        '.docx': (f"Word文档的页面截图", f"Word页面截图"),
        '.txt': (f"文本文件的可视化截图", f"文本截图"),
        '.md': (f"Markdown文件的可视化截图", f"Markdown截图"),
        '.mp3': (f"音频文件的波形图和频谱图", f"音频波形图"),
        '.wav': (f"音频文件的波形图和频谱图", f"音频波形图"),
        '.mp4': (f"视频关键帧 {image_index + 1}", f"视频帧 {image_index + 1}"),
        '.avi': (f"视频关键帧 {image_index + 1}", f"视频帧 {image_index + 1}"),
        '.png': (f"图像文件内容", f"图像内容"),
        '.jpg': (f"图像文件内容", f"图像内容"),
        '.jpeg': (f"图像文件内容", f"图像内容")
    }
    
    return descriptions.get(content_type, 
                          (f"文件内容的可视化截图 {image_index + 1}", 
                           f"内容截图 {image_index + 1}"))
```

#### 元数据增强
```python
def enhance_image_metadata(self, image_description: str, content: ParsedContent, image_index: int) -> str:
    """增强图像元数据"""
    
    # 添加图像尺寸信息
    if 'images_info' in content.metadata and image_index < len(content.metadata['images_info']):
        img_info = content.metadata['images_info'][image_index]
        if 'size' in img_info:
            image_description += f"\n图像尺寸: {img_info['size']}"
        if 'format' in img_info:
            image_description += f"\n图像格式: {img_info['format']}"
    
    # 添加页面位置信息
    if 'page_count' in content.metadata:
        estimated_page = min(image_index + 1, content.metadata['page_count'])
        image_description += f"\n估计位置: 第 {estimated_page} 页"
    
    return image_description
```

#### 存储结构
```json
{
    "content_type": "image",
    "chunk_content": "PDF页面 1 的截图\n图像尺寸: 1920x1080\n估计位置: 第 1 页",
    "chunk_summary": "PDF页面 1",
    "content_length": 85,
    "image_data": "/9j/4AAQSkZJRgABAQEA...",  // Base64编码
    "image_format": "JPEG",
    "image_size": "1920x1080",
    "ocr_text": "search query\nversion update\n202509"
}
```

### 3. Audio类型切片

#### 切片条件
```python
elif content.audio_content:
    return "audio", ChunkContent(...)
```

#### 音频处理
```python
def process_audio_content(self, content: ParsedContent) -> ChunkContent:
    """处理音频内容"""
    audio_description = f"Audio content from {content.file_type} file"
    
    # 添加元数据信息
    if 'duration' in content.metadata:
        audio_description += f", duration: {content.metadata['duration']}s"
    if 'sample_rate' in content.metadata:
        audio_description += f", sample rate: {content.metadata['sample_rate']}Hz"
    if 'channels' in content.metadata:
        audio_description += f", channels: {content.metadata['channels']}"
    
    return ChunkContent(
        content=audio_description,
        content_type="audio_description",
        summary="Audio content"
    )
```

## 切片策略的优势

### 1. 精确检索
- **内容分离**: 文本和图像内容独立索引，避免相互干扰
- **类型匹配**: 查询类型与内容类型精确匹配
- **上下文保持**: 保持文档内容的逻辑关系

### 2. 多模态融合
- **OCR集成**: 图像中的文字参与文本检索
- **视觉语义**: 图像描述提供视觉语义信息
- **跨模态检索**: 文本查询可以找到相关图像

### 3. 存储优化
- **智能压缩**: 根据内容特点选择最优压缩策略
- **大小控制**: 严格控制在Milvus字段限制内
- **质量平衡**: 在存储大小和图像质量间找到平衡

### 4. 扩展性
- **类型扩展**: 易于添加新的内容类型
- **策略调整**: 可根据需求调整切片策略
- **配置驱动**: 通过配置文件控制切片行为

## 实际应用示例

### PDF文件切片示例
```
文件: report.pdf (19页)

切片结果:
├── index=0 (text): 完整PDF文本内容 (10,522字符)
├── index=1 (image): PDF页面1截图 + OCR文本
├── index=2 (image): PDF页面2截图 + OCR文本
├── ...
└── index=19 (image): PDF页面19截图 + OCR文本

总计: 20个chunk (1个文本 + 19个图像)
```

### 图像文件切片示例
```
文件: presentation.png

切片结果:
└── index=0 (image): 图像内容 + OCR文本

总计: 1个chunk (1个图像)
```

### 音频文件切片示例
```
文件: meeting_record.mp3

切片结果:
├── index=0 (audio): 音频描述信息
└── index=1 (image): 波形图可视化

总计: 2个chunk (1个音频 + 1个图像)
```

## 性能优化

### 1. 处理速度优化
- **并行处理**: 多个chunk并行生成嵌入
- **缓存机制**: 重复内容避免重复处理
- **流式处理**: 大文件分块处理

### 2. 内存优化
- **及时释放**: 处理完成后立即释放图像对象
- **分批处理**: 避免同时加载过多图像
- **压缩存储**: 减少内存占用

### 3. 质量控制
- **阈值设置**: 设置合理的压缩阈值
- **质量监控**: 监控压缩后的图像质量
- **自适应调整**: 根据内容特点调整策略