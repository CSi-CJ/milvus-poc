# API参考文档

## 概述

多模态文件索引器提供了多种API接口，包括Python SDK、CLI命令行工具和Web API。本文档详细介绍了各种接口的使用方法和参数说明。

## Python SDK

### 1. 系统管理器

#### MultimodalIndexerSystem

```python
from multimodal_indexer.system import MultimodalIndexerSystem

# 初始化系统
system = MultimodalIndexerSystem()

# 处理单个文件
result = system.process_file("./files/document.pdf")

# 批量处理目录
results = system.process_directory("./files/")

# 搜索内容
search_results = system.search("查询内容", top_k=10)

# 健康检查
health_status = system.health_check()
```

#### 方法详解

##### process_file(file_path: str) -> ProcessingResult
处理单个文件并建立索引。

**参数:**
- `file_path` (str): 文件路径

**返回:**
- `ProcessingResult`: 处理结果对象

**示例:**
```python
result = system.process_file("./document.pdf")
print(f"处理状态: {result.success}")
print(f"生成嵌入数: {result.embeddings_count}")
print(f"处理时间: {result.processing_time}秒")
```

##### process_directory(directory_path: str, max_concurrent: int = 10) -> List[ProcessingResult]
批量处理目录中的所有支持文件。

**参数:**
- `directory_path` (str): 目录路径
- `max_concurrent` (int): 最大并发数，默认10

**返回:**
- `List[ProcessingResult]`: 处理结果列表

**示例:**
```python
results = system.process_directory("./files/", max_concurrent=5)
success_count = sum(1 for r in results if r.success)
print(f"成功处理: {success_count}/{len(results)} 个文件")
```

##### search(query: str, top_k: int = 10, content_type: str = None) -> List[SearchResult]
搜索相似内容。

**参数:**
- `query` (str): 搜索查询
- `top_k` (int): 返回结果数量，默认10
- `content_type` (str): 内容类型过滤，可选值: "text", "image", "audio"

**返回:**
- `List[SearchResult]`: 搜索结果列表

**示例:**
```python
# 通用搜索
results = system.search("search query", top_k=5)

# 只搜索文本内容
text_results = system.search("project report", content_type="text")

# 只搜索图像内容
image_results = system.search("workflow", content_type="image")
```

### 2. 文件处理器

#### FileProcessor

```python
from multimodal_indexer.file_processor import FileProcessor
from multimodal_indexer.config import Config

config = Config()
processor = FileProcessor(config)

# 处理文件
result = processor.process_file("./document.pdf")
```

### 3. 索引管理器

#### IndexManager

```python
from multimodal_indexer.index_manager import IndexManager
from multimodal_indexer.config import Config

config = Config()
index_manager = IndexManager(config)

# 搜索
results = index_manager.search_similar("查询文本", top_k=10)

# 插入数据
index_manager.insert_embeddings(embeddings_data)

# 删除数据
index_manager.delete_by_file_path("./old_document.pdf")
```

### 4. 集合管理器

#### CollectionManager

```python
from multimodal_indexer.collection_manager import CollectionManager
from multimodal_indexer.config import Config

config = Config()
collection_manager = CollectionManager(config)

# 创建集合
collection_manager.create_collection()

# 检查集合是否存在
exists = collection_manager.collection_exists()

# 获取集合统计信息
stats = collection_manager.get_collection_stats()

# 删除集合
collection_manager.drop_collection()
```

## CLI命令行工具

### 基本用法

```bash
python -m multimodal_indexer.cli <command> [options]
```

### 可用命令

#### 1. process-file - 处理单个文件

```bash
python -m multimodal_indexer.cli process-file <file_path>
```

**参数:**
- `file_path`: 要处理的文件路径

**示例:**
```bash
# 处理PDF文件
python -m multimodal_indexer.cli process-file ./files/document.pdf

# 处理图像文件
python -m multimodal_indexer.cli process-file ./files/image.png

# 处理音频文件
python -m multimodal_indexer.cli process-file ./files/audio.mp3
```

#### 2. process-directory - 批量处理目录

```bash
python -m multimodal_indexer.cli process-directory <directory_path> [--max-concurrent N]
```

**参数:**
- `directory_path`: 要处理的目录路径
- `--max-concurrent`: 最大并发数，默认10

**示例:**
```bash
# 处理整个目录
python -m multimodal_indexer.cli process-directory ./files/

# 限制并发数
python -m multimodal_indexer.cli process-directory ./files/ --max-concurrent 5
```

#### 3. search - 搜索内容

```bash
python -m multimodal_indexer.cli search <query> [--top-k N] [--content-type TYPE]
```

**参数:**
- `query`: 搜索查询字符串
- `--top-k`: 返回结果数量，默认10
- `--content-type`: 内容类型过滤，可选值: text, image, audio

**示例:**
```bash
# 基本搜索
python -m multimodal_indexer.cli search "search query"

# 限制结果数量
python -m multimodal_indexer.cli search "project report" --top-k 5

# 只搜索文本内容
python -m multimodal_indexer.cli search "workflow" --content-type text

# 只搜索图像内容
python -m multimodal_indexer.cli search "chart" --content-type image
```

#### 4. health-check - 健康检查

```bash
python -m multimodal_indexer.cli health-check
```

检查系统各组件的健康状态，包括：
- Milvus数据库连接
- BGE-M3模型加载状态
- 集合状态
- 系统资源使用情况

#### 5. list-files - 列出已索引文件

```bash
python -m multimodal_indexer.cli list-files [--limit N]
```

**参数:**
- `--limit`: 限制显示数量，默认100

#### 6. delete-file - 删除文件索引

```bash
python -m multimodal_indexer.cli delete-file <file_path>
```

**参数:**
- `file_path`: 要删除索引的文件路径

#### 7. collection-info - 集合信息

```bash
python -m multimodal_indexer.cli collection-info
```

显示Milvus集合的详细信息，包括：
- 集合名称和schema版本
- 数据量统计
- 索引信息
- 存储使用情况

## Web API

### 启动Web服务

```bash
python web_ui.py
```

默认在 `http://localhost:5000` 启动Web界面。

### API端点

#### 1. 文件上传和处理

**POST /api/upload**

上传并处理文件。

**请求:**
```bash
curl -X POST -F "file=@document.pdf" http://localhost:5000/api/upload
```

**响应:**
```json
{
    "success": true,
    "message": "File processed successfully",
    "file_path": "./uploads/document.pdf",
    "embeddings_count": 20,
    "processing_time": 45.2
}
```

#### 2. 搜索API

**GET /api/search**

搜索内容。

**参数:**
- `q`: 搜索查询
- `top_k`: 结果数量，默认10
- `content_type`: 内容类型过滤

**请求:**
```bash
curl "http://localhost:5000/api/search?q=search+query&top_k=5"
```

**响应:**
```json
{
    "success": true,
    "results": [
        {
            "id": "doc1_chunk0",
            "file_name": "report.pdf",
            "content_type": "text",
            "chunk_summary": "Report overview...",
            "similarity_score": 0.95,
            "metadata": {...}
        }
    ],
    "total_results": 5,
    "query_time": 0.08
}
```

#### 3. 文件列表API

**GET /api/files**

获取已索引文件列表。

**参数:**
- `limit`: 限制数量，默认100
- `offset`: 偏移量，默认0

**请求:**
```bash
curl "http://localhost:5000/api/files?limit=20"
```

**响应:**
```json
{
    "success": true,
    "files": [
        {
            "file_path": "./files/document.pdf",
            "file_name": "document.pdf",
            "file_type": ".pdf",
            "chunks_count": 20,
            "created_at": "2024-01-15T10:30:00Z"
        }
    ],
    "total_files": 15
}
```

#### 4. 健康检查API

**GET /api/health**

检查系统健康状态。

**请求:**
```bash
curl http://localhost:5000/api/health
```

**响应:**
```json
{
    "success": true,
    "status": "healthy",
    "components": {
        "milvus": "connected",
        "bge_model": "loaded",
        "collection": "ready"
    },
    "stats": {
        "total_documents": 150,
        "total_chunks": 3420,
        "collection_size": "2.1GB"
    }
}
```

## 数据模型

### ProcessingResult

```python
@dataclass
class ProcessingResult:
    success: bool
    file_path: str
    embeddings_count: int
    processing_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None
```

### SearchResult

```python
@dataclass
class SearchResult:
    id: str
    file_path: str
    file_name: str
    content_type: str
    chunk_content: str
    chunk_summary: str
    similarity_score: float
    image_data: Optional[str] = None
    ocr_text: Optional[str] = None
    metadata: Optional[Dict] = None
```

### ParsedContent

```python
@dataclass
class ParsedContent:
    file_path: str
    file_type: str
    text_content: Optional[str]
    image_content: List[bytes]
    audio_content: Optional[bytes]
    metadata: Dict[str, Any]
```

### ChunkContent

```python
@dataclass
class ChunkContent:
    content: str
    content_type: str
    summary: str
    image_data: Optional[bytes] = None
    ocr_text: Optional[str] = None
```

## 错误处理

### 常见错误码

- `FILE_NOT_FOUND`: 文件不存在
- `UNSUPPORTED_FORMAT`: 不支持的文件格式
- `PARSING_FAILED`: 文件解析失败
- `EMBEDDING_FAILED`: 向量嵌入生成失败
- `MILVUS_CONNECTION_ERROR`: Milvus连接错误
- `MODEL_LOADING_ERROR`: 模型加载错误

### 错误响应格式

```json
{
    "success": false,
    "error_code": "PARSING_FAILED",
    "error_message": "Failed to parse PDF file: corrupted file",
    "details": {
        "file_path": "./document.pdf",
        "error_type": "PyMuPDFError"
    }
}
```

## 配置参数

### 系统配置

```json
{
    "milvus": {
        "host": "localhost",
        "port": 19530,
        "collection_name": "multimodal_files",
        "vector_dim": 1024
    },
    "embedding": {
        "multimodal_model": "BAAI/bge-m3",
        "batch_size": 12,
        "use_fp16": true,
        "max_length": 8192
    },
    "processing": {
        "max_concurrent": 10,
        "enable_ocr": true,
        "enable_speech_recognition": true,
        "image_quality": 85,
        "max_image_size": 42000
    },
    "logging": {
        "level": "INFO",
        "file": "./logs/multimodal_indexer.log",
        "max_size": "10MB",
        "backup_count": 5
    }
}
```

## 性能调优

### 1. 批量处理优化

```python
# 调整并发数
system.process_directory("./files/", max_concurrent=20)

# 批量嵌入
embedder.batch_embed(texts, batch_size=16)
```

### 2. 内存优化

```python
# 启用FP16精度
config.embedding.use_fp16 = True

# 调整批量大小
config.embedding.batch_size = 8
```

### 3. 搜索优化

```python
# 调整搜索参数
results = index_manager.search_similar(
    query="搜索内容",
    top_k=10,
    search_params={"ef": 200}  # HNSW搜索参数
)
```

## 示例代码

### 完整处理流程

```python
from multimodal_indexer.system import MultimodalIndexerSystem

# 初始化系统
system = MultimodalIndexerSystem()

# 处理文件
result = system.process_file("./document.pdf")
if result.success:
    print(f"成功处理文件，生成 {result.embeddings_count} 个嵌入")
else:
    print(f"处理失败: {result.error_message}")

# 搜索内容
search_results = system.search("search query", top_k=5)
for result in search_results:
    print(f"文件: {result.file_name}")
    print(f"相似度: {result.similarity_score:.3f}")
    print(f"摘要: {result.chunk_summary}")
    print("---")

# 健康检查
health = system.health_check()
print(f"系统状态: {health['status']}")
```

### 自定义解析器

```python
from multimodal_indexer.parsers.base import BaseFileParser
from multimodal_indexer.models import ParsedContent

class CustomParser(BaseFileParser):
    supported_extensions = ['.custom']
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.custom')
    
    def parse(self, file_path: str) -> ParsedContent:
        # 自定义解析逻辑
        with open(file_path, 'r') as f:
            content = f.read()
        
        return ParsedContent(
            file_path=file_path,
            file_type='.custom',
            text_content=content,
            image_content=[],
            audio_content=None,
            metadata={'custom_field': 'value'}
        )

# 注册自定义解析器
from multimodal_indexer.parsers.registry import FileParserRegistry
registry = FileParserRegistry()
registry.register_parser(CustomParser())
```