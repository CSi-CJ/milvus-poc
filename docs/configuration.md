# 配置指南

## 概述

多模态文件索引器通过 `config.json` 文件进行配置管理，支持环境变量覆盖和运行时配置调整。本文档详细介绍了所有配置选项和最佳实践。

## 配置文件结构

### 默认配置文件 (config.json)

```json
{
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "collection_name": "multimodal_files",
    "vector_dim": 1024,
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "index_params": {
      "M": 16,
      "efConstruction": 200
    },
    "search_params": {
      "ef": 200
    }
  },
  "embedding": {
    "multimodal_model": "BAAI/bge-m3",
    "batch_size": 12,
    "use_fp16": true,
    "max_length": 8192,
    "device": "auto",
    "cache_dir": "./models"
  },
  "processing": {
    "max_concurrent": 10,
    "enable_ocr": true,
    "enable_speech_recognition": true,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "image_quality": 85,
    "max_image_size": 42000,
    "pdf_scale_factor": 2.0
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/multimodal_indexer.log",
    "max_size": "10MB",
    "backup_count": 5,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  },
  "web": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false,
    "upload_folder": "./uploads",
    "max_file_size": "100MB"
  }
}
```

## 配置选项详解

### 1. Milvus数据库配置

#### 基本连接配置

```json
{
  "milvus": {
    "host": "localhost",           // Milvus服务器地址
    "port": 19530,                 // Milvus服务器端口
    "collection_name": "multimodal_files",  // 集合名称
    "vector_dim": 1024             // 向量维度 (必须与BGE-M3匹配)
  }
}
```

**参数说明:**
- `host`: Milvus服务器IP地址或域名
- `port`: Milvus服务器端口，默认19530
- `collection_name`: 存储数据的集合名称
- `vector_dim`: 向量维度，BGE-M3模型为1024维

#### 索引配置

```json
{
  "milvus": {
    "index_type": "HNSW",          // 索引类型
    "metric_type": "COSINE",       // 距离度量类型
    "index_params": {
      "M": 16,                     // HNSW参数：每个节点的最大连接数
      "efConstruction": 200        // HNSW参数：构建时的搜索宽度
    },
    "search_params": {
      "ef": 200                    // HNSW参数：搜索时的候选集大小
    }
  }
}
```

**索引类型选项:**
- `HNSW`: 层次化可导航小世界图 (推荐)
- `IVF_FLAT`: 倒排文件索引
- `IVF_SQ8`: 倒排文件 + 标量量化
- `IVF_PQ`: 倒排文件 + 乘积量化

**距离度量选项:**
- `COSINE`: 余弦相似度 (推荐用于文本嵌入)
- `L2`: 欧几里得距离
- `IP`: 内积距离

#### 高级配置

```json
{
  "milvus": {
    "timeout": 30,                 // 连接超时时间(秒)
    "retry_times": 3,              // 重试次数
    "secure": false,               // 是否启用TLS
    "user": "",                    // 用户名 (企业版)
    "password": "",                // 密码 (企业版)
    "db_name": "default"           // 数据库名称 (2.1+版本)
  }
}
```

### 2. 嵌入模型配置

#### BGE-M3模型配置

```json
{
  "embedding": {
    "multimodal_model": "BAAI/bge-m3",  // 模型名称
    "batch_size": 12,                   // 批处理大小
    "use_fp16": true,                   // 使用FP16精度
    "max_length": 8192,                 // 最大序列长度
    "device": "auto",                   // 设备选择
    "cache_dir": "./models"             // 模型缓存目录
  }
}
```

**参数说明:**
- `multimodal_model`: 嵌入模型名称，支持HuggingFace模型
- `batch_size`: 批处理大小，影响内存使用和处理速度
- `use_fp16`: 是否使用半精度浮点数，可减少内存使用
- `max_length`: 文本最大长度，超出部分会被截断
- `device`: 设备选择，"auto"自动选择，"cpu"强制CPU，"cuda"强制GPU
- `cache_dir`: 模型文件缓存目录

#### 性能调优参数

```json
{
  "embedding": {
    "normalize_embeddings": true,       // 是否归一化嵌入向量
    "pooling_method": "cls",           // 池化方法: cls, mean, max
    "trust_remote_code": false,        // 是否信任远程代码
    "revision": "main",                // 模型版本分支
    "torch_dtype": "float16"           // PyTorch数据类型
  }
}
```

### 3. 文件处理配置

#### 基本处理参数

```json
{
  "processing": {
    "max_concurrent": 10,              // 最大并发处理数
    "enable_ocr": true,                // 启用OCR文字识别
    "enable_speech_recognition": true, // 启用语音识别
    "chunk_size": 1000,               // 文本分块大小
    "chunk_overlap": 200              // 分块重叠大小
  }
}
```

**参数说明:**
- `max_concurrent`: 同时处理的文件数量，影响CPU和内存使用
- `enable_ocr`: 是否对图像进行OCR文字识别
- `enable_speech_recognition`: 是否对音频进行语音识别
- `chunk_size`: 文本分块的字符数
- `chunk_overlap`: 相邻分块的重叠字符数

#### 图像处理参数

```json
{
  "processing": {
    "image_quality": 85,               // JPEG压缩质量 (1-100)
    "max_image_size": 42000,          // 最大图像Base64长度
    "pdf_scale_factor": 2.0,          // PDF截图缩放因子
    "image_resize_threshold": 1600,    // 图像缩放阈值
    "compression_stages": [
      {"quality": 85, "max_size": 1200},
      {"quality": 80, "max_size": 1000},
      {"quality": 75, "max_size": 800},
      {"quality": 70, "max_size": 600}
    ]
  }
}
```

#### OCR配置

```json
{
  "processing": {
    "ocr": {
      "engine": "tesseract",           // OCR引擎
      "languages": "chi_sim+eng",      // 识别语言
      "config": "--oem 3 --psm 6",    // Tesseract配置
      "preprocessing": {
        "resize": true,                // 预处理：调整大小
        "denoise": true,              // 预处理：去噪
        "contrast": true              // 预处理：对比度增强
      }
    }
  }
}
```

#### 音频处理配置

```json
{
  "processing": {
    "audio": {
      "sample_rate": 16000,            // 采样率
      "channels": 1,                   // 声道数
      "format": "wav",                 // 音频格式
      "whisper_model": "base",         // Whisper模型大小
      "language": "auto"               // 语言检测
    }
  }
}
```

### 4. 日志配置

#### 基本日志配置

```json
{
  "logging": {
    "level": "INFO",                   // 日志级别
    "file": "./logs/multimodal_indexer.log",  // 日志文件路径
    "max_size": "10MB",               // 单个日志文件最大大小
    "backup_count": 5,                // 保留的日志文件数量
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  }
}
```

**日志级别选项:**
- `DEBUG`: 详细调试信息
- `INFO`: 一般信息 (推荐)
- `WARNING`: 警告信息
- `ERROR`: 错误信息
- `CRITICAL`: 严重错误

#### 高级日志配置

```json
{
  "logging": {
    "console_output": true,            // 是否输出到控制台
    "json_format": false,             // 是否使用JSON格式
    "include_caller": true,           // 是否包含调用者信息
    "timezone": "Asia/Shanghai",      // 时区设置
    "handlers": {
      "file": {
        "enabled": true,
        "level": "INFO",
        "formatter": "detailed"
      },
      "console": {
        "enabled": true,
        "level": "INFO",
        "formatter": "simple"
      }
    }
  }
}
```

### 5. Web服务配置

#### 基本Web配置

```json
{
  "web": {
    "host": "0.0.0.0",               // 绑定地址
    "port": 5000,                    // 端口号
    "debug": false,                  // 调试模式
    "upload_folder": "./uploads",    // 上传文件目录
    "max_file_size": "100MB"         // 最大文件大小
  }
}
```

#### 安全配置

```json
{
  "web": {
    "secret_key": "your-secret-key", // Flask密钥
    "cors_enabled": true,            // 启用CORS
    "cors_origins": ["*"],           // 允许的源
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60
    },
    "authentication": {
      "enabled": false,
      "type": "basic",               // basic, jwt, oauth
      "users": {
        "admin": "password"
      }
    }
  }
}
```

## 环境变量覆盖

系统支持通过环境变量覆盖配置文件中的设置：

### 基本格式

```bash
# 格式: MULTIMODAL_<SECTION>_<KEY>=<VALUE>
export MULTIMODAL_MILVUS_HOST=192.168.1.100
export MULTIMODAL_MILVUS_PORT=19530
export MULTIMODAL_EMBEDDING_BATCH_SIZE=16
export MULTIMODAL_PROCESSING_MAX_CONCURRENT=20
```

### 嵌套配置

```bash
# 嵌套配置使用双下划线分隔
export MULTIMODAL_MILVUS_INDEX_PARAMS__M=32
export MULTIMODAL_MILVUS_SEARCH_PARAMS__EF=400
export MULTIMODAL_PROCESSING_OCR__LANGUAGES=chi_sim+eng+fra
```

### 常用环境变量

```bash
# Milvus连接
export MULTIMODAL_MILVUS_HOST=localhost
export MULTIMODAL_MILVUS_PORT=19530
export MULTIMODAL_MILVUS_COLLECTION_NAME=my_collection

# 模型配置
export MULTIMODAL_EMBEDDING_MULTIMODAL_MODEL=BAAI/bge-m3
export MULTIMODAL_EMBEDDING_BATCH_SIZE=8
export MULTIMODAL_EMBEDDING_USE_FP16=true

# 处理配置
export MULTIMODAL_PROCESSING_MAX_CONCURRENT=5
export MULTIMODAL_PROCESSING_ENABLE_OCR=true

# 日志配置
export MULTIMODAL_LOGGING_LEVEL=DEBUG
export MULTIMODAL_LOGGING_FILE=./logs/debug.log

# Web配置
export MULTIMODAL_WEB_HOST=0.0.0.0
export MULTIMODAL_WEB_PORT=8080
export MULTIMODAL_WEB_DEBUG=false
```

## 配置验证

### 配置文件验证

```python
from multimodal_indexer.config import Config

# 加载并验证配置
config = Config()
validation_result = config.validate()

if validation_result.is_valid:
    print("配置验证通过")
else:
    print("配置验证失败:")
    for error in validation_result.errors:
        print(f"  - {error}")
```

### 配置检查工具

```bash
# 检查配置文件
python -m multimodal_indexer.cli check-config

# 检查特定配置项
python -m multimodal_indexer.cli check-config --section milvus
python -m multimodal_indexer.cli check-config --section embedding
```

## 性能调优配置

### 高性能配置 (GPU环境)

```json
{
  "embedding": {
    "batch_size": 32,
    "use_fp16": true,
    "device": "cuda"
  },
  "processing": {
    "max_concurrent": 20,
    "image_quality": 90
  },
  "milvus": {
    "index_params": {
      "M": 32,
      "efConstruction": 400
    },
    "search_params": {
      "ef": 400
    }
  }
}
```

### 内存优化配置 (低内存环境)

```json
{
  "embedding": {
    "batch_size": 4,
    "use_fp16": true,
    "device": "cpu"
  },
  "processing": {
    "max_concurrent": 2,
    "image_quality": 70,
    "max_image_size": 20000
  },
  "milvus": {
    "index_params": {
      "M": 8,
      "efConstruction": 100
    }
  }
}
```

### 生产环境配置

```json
{
  "milvus": {
    "host": "milvus-cluster.example.com",
    "port": 19530,
    "secure": true,
    "user": "production_user",
    "password": "${MILVUS_PASSWORD}",
    "timeout": 60,
    "retry_times": 5
  },
  "embedding": {
    "batch_size": 16,
    "use_fp16": true,
    "cache_dir": "/opt/models"
  },
  "processing": {
    "max_concurrent": 15,
    "enable_ocr": true,
    "enable_speech_recognition": false
  },
  "logging": {
    "level": "INFO",
    "file": "/var/log/multimodal_indexer/app.log",
    "max_size": "50MB",
    "backup_count": 10,
    "json_format": true
  },
  "web": {
    "host": "0.0.0.0",
    "port": 8080,
    "debug": false,
    "secret_key": "${FLASK_SECRET_KEY}",
    "cors_enabled": true,
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 100
    }
  }
}
```

## 配置模板

### 开发环境模板

```json
{
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "collection_name": "dev_multimodal_files"
  },
  "embedding": {
    "batch_size": 8,
    "use_fp16": false,
    "device": "cpu"
  },
  "processing": {
    "max_concurrent": 5,
    "enable_ocr": true
  },
  "logging": {
    "level": "DEBUG",
    "console_output": true
  },
  "web": {
    "debug": true,
    "port": 5000
  }
}
```

### 测试环境模板

```json
{
  "milvus": {
    "host": "test-milvus.example.com",
    "collection_name": "test_multimodal_files"
  },
  "embedding": {
    "batch_size": 12,
    "use_fp16": true
  },
  "processing": {
    "max_concurrent": 10
  },
  "logging": {
    "level": "INFO",
    "file": "./logs/test.log"
  }
}
```

## 故障排除

### 常见配置问题

1. **向量维度不匹配**
   ```
   错误: Vector dimension mismatch
   解决: 确保 vector_dim 设置为 1024 (BGE-M3)
   ```

2. **Milvus连接失败**
   ```
   错误: Failed to connect to Milvus
   解决: 检查 host 和 port 配置，确保Milvus服务运行
   ```

3. **内存不足**
   ```
   错误: CUDA out of memory
   解决: 降低 batch_size 或启用 use_fp16
   ```

4. **模型加载失败**
   ```
   错误: Model loading failed
   解决: 检查 cache_dir 权限和网络连接
   ```

### 配置诊断命令

```bash
# 诊断配置问题
python -m multimodal_indexer.cli diagnose

# 测试Milvus连接
python -m multimodal_indexer.cli test-milvus

# 测试模型加载
python -m multimodal_indexer.cli test-model

# 生成配置报告
python -m multimodal_indexer.cli config-report
```