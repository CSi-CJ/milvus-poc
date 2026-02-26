# 模型和算法详解

## 核心模型

### 1. BGE-M3 多模态嵌入模型

#### 模型概述
- **全称**: BAAI General Embedding Model - Multimodal (BGE-M3)
- **开发者**: 北京智源人工智能研究院 (BAAI)
- **模型类型**: 多模态嵌入模型
- **向量维度**: 1024维
- **支持模态**: 文本、图像、音频

#### 技术特性
```python
# 模型配置
{
    "model_name": "BAAI/bge-m3",
    "vector_dimension": 1024,
    "max_sequence_length": 8192,
    "supported_languages": ["中文", "英文", "多语言"],
    "precision": "FP16",
    "batch_size": 12
}
```

#### 模型架构
- **基础架构**: Transformer-based
- **编码器**: 多模态共享编码器
- **池化策略**: CLS token pooling
- **归一化**: L2归一化输出向量

#### 性能指标
- **文本检索**: NDCG@10 > 0.85
- **图像检索**: Recall@10 > 0.80
- **跨模态检索**: MRR > 0.75
- **推理速度**: ~50ms/batch (GPU)

### 2. OCR文字识别

#### Tesseract OCR引擎
```python
# OCR配置
{
    "engine": "pytesseract",
    "languages": "chi_sim+eng",  # 中文简体 + 英文
    "config": "--oem 3 --psm 6",
    "preprocessing": {
        "resize": True,
        "denoise": True,
        "contrast_enhancement": True
    }
}
```

#### OCR处理流程
1. **图像预处理**
   - 尺寸调整和分辨率优化
   - 噪声去除和对比度增强
   - 二值化和边缘检测

2. **文字检测**
   - 文本区域定位
   - 文字方向识别
   - 字符分割

3. **文字识别**
   - 字符识别和分类
   - 语言模型校正
   - 置信度评估

4. **后处理**
   - 文本格式化
   - 错误纠正
   - 结构化输出

### 3. 音频处理模型

#### Whisper语音识别 (可选)
```python
# Whisper配置
{
    "model": "whisper-base",
    "language": "auto",
    "task": "transcribe",
    "temperature": 0.0,
    "beam_size": 5
}
```

#### 音频特征提取
- **波形分析**: 时域特征提取
- **频谱分析**: 频域特征提取
- **MFCC特征**: 梅尔频率倒谱系数
- **可视化**: 波形图和频谱图生成

## 算法详解

### 1. 向量嵌入算法

#### 文本嵌入
```python
def embed_text(self, text: str) -> np.ndarray:
    """文本向量嵌入"""
    # 1. 文本预处理
    tokens = self.tokenizer(
        text,
        max_length=self.max_length,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )
    
    # 2. 模型推理
    with torch.no_grad():
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
    
    # 3. 池化和归一化
    pooled = self._mean_pooling(embeddings, tokens['attention_mask'])
    normalized = F.normalize(pooled, p=2, dim=1)
    
    return normalized.cpu().numpy()[0]
```

#### 图像嵌入
```python
def embed_image(self, image_data: bytes) -> np.ndarray:
    """图像向量嵌入"""
    # 1. 图像预处理
    image = Image.open(io.BytesIO(image_data))
    image = self.image_processor(image)
    
    # 2. 特征提取
    with torch.no_grad():
        features = self.vision_encoder(image.unsqueeze(0))
    
    # 3. 投影和归一化
    embeddings = self.projection_layer(features)
    normalized = F.normalize(embeddings, p=2, dim=1)
    
    return normalized.cpu().numpy()[0]
```

#### 多模态融合
```python
def embed_multimodal(self, content: ParsedContent) -> List[np.ndarray]:
    """多模态内容嵌入"""
    embeddings = []
    
    # 文本模态
    if content.text_content:
        text_emb = self.embed_text(content.text_content)
        embeddings.append(text_emb)
    
    # 图像模态
    if content.image_content:
        for image_data in content.image_content:
            # 结合图像和OCR文本
            image_emb = self.embed_image(image_data)
            ocr_text = self.extract_ocr_text(image_data)
            
            if ocr_text:
                # 多模态融合：图像 + 文本
                text_emb = self.embed_text(ocr_text)
                fused_emb = self._fuse_embeddings(image_emb, text_emb)
                embeddings.append(fused_emb)
            else:
                embeddings.append(image_emb)
    
    return embeddings
```

### 2. 相似度计算算法

#### 余弦相似度
```python
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)
```

#### HNSW索引算法
- **层次结构**: 多层图结构，上层稀疏，下层密集
- **搜索策略**: 从顶层开始，逐层向下搜索
- **插入策略**: 随机选择层数，维护连接关系
- **参数优化**:
  - M=16: 每个节点的最大连接数
  - efConstruction=200: 构建时的搜索宽度
  - ef=200: 搜索时的候选集大小

### 3. 图像处理算法

#### 高质量截图生成
```python
def generate_page_screenshot(self, page, scale_factor=2):
    """生成高质量页面截图"""
    # 1. 设置变换矩阵
    mat = fitz.Matrix(scale_factor, scale_factor)
    
    # 2. 渲染页面
    pix = page.get_pixmap(matrix=mat)
    
    # 3. 转换为PIL图像
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    
    # 4. 质量优化
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image
```

#### 智能压缩算法
```python
def compress_image(self, image: Image.Image, max_size: int = 42000) -> bytes:
    """智能图像压缩"""
    # 1. 初始压缩
    if image.width > 1600 or image.height > 1600:
        image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    
    # 2. 质量调整
    quality = 85
    while quality >= 70:
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality, optimize=True)
        
        # 3. 大小检查
        compressed_data = buffer.getvalue()
        if len(compressed_data) <= max_size:
            return compressed_data
        
        # 4. 降低质量或尺寸
        quality -= 5
        if quality < 70:
            image.thumbnail((800, 800), Image.Resampling.LANCZOS)
            quality = 75
    
    return compressed_data
```

### 4. 文本处理算法

#### 智能摘要生成
```python
def create_text_summary(self, text: str, max_length: int = 200) -> str:
    """创建文本摘要"""
    if not text or len(text) <= max_length:
        return text
    
    # 1. 句子分割
    sentences = self._split_sentences(text)
    
    # 2. 重要性评分
    scores = self._calculate_sentence_scores(sentences)
    
    # 3. 选择重要句子
    summary_sentences = self._select_top_sentences(sentences, scores, max_length)
    
    # 4. 重新排序和组合
    summary = self._reconstruct_summary(summary_sentences)
    
    return summary
```

#### 分词和标记化
```python
def tokenize_text(self, text: str) -> List[str]:
    """文本分词"""
    # 1. 预处理
    text = self._preprocess_text(text)
    
    # 2. 中英文混合分词
    tokens = []
    for segment in self._segment_by_language(text):
        if self._is_chinese(segment):
            tokens.extend(self._chinese_tokenize(segment))
        else:
            tokens.extend(self._english_tokenize(segment))
    
    # 3. 后处理
    tokens = self._postprocess_tokens(tokens)
    
    return tokens
```

## 模型优化策略

### 1. 性能优化

#### 批量处理
```python
def batch_embed(self, texts: List[str], batch_size: int = 12) -> List[np.ndarray]:
    """批量嵌入处理"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = self._embed_batch(batch)
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

#### 内存优化
- **梯度清零**: 推理时禁用梯度计算
- **精度降低**: 使用FP16精度减少内存占用
- **模型量化**: 支持INT8量化推理
- **缓存管理**: 及时释放中间结果

#### GPU加速
```python
def setup_gpu_acceleration(self):
    """设置GPU加速"""
    if torch.cuda.is_available():
        self.device = torch.device("cuda")
        self.model = self.model.to(self.device)
        self.model = self.model.half()  # FP16
        
        # 优化设置
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
```

### 2. 质量优化

#### 数据增强
- **文本增强**: 同义词替换、句式变换
- **图像增强**: 旋转、缩放、亮度调整
- **噪声注入**: 添加适量噪声提高鲁棒性

#### 模型微调
```python
def fine_tune_model(self, training_data, epochs=3, lr=1e-5):
    """模型微调"""
    optimizer = AdamW(self.model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100,
        num_training_steps=len(training_data) * epochs
    )
    
    for epoch in range(epochs):
        for batch in training_data:
            loss = self._compute_contrastive_loss(batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

### 3. 评估指标

#### 检索质量评估
```python
def evaluate_retrieval_quality(self, queries, ground_truth):
    """评估检索质量"""
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'ndcg_at_k': [],
        'mrr': []
    }
    
    for query, gt in zip(queries, ground_truth):
        results = self.search(query, top_k=10)
        
        # 计算各项指标
        metrics['precision_at_k'].append(self._precision_at_k(results, gt, k=10))
        metrics['recall_at_k'].append(self._recall_at_k(results, gt, k=10))
        metrics['ndcg_at_k'].append(self._ndcg_at_k(results, gt, k=10))
        metrics['mrr'].append(self._mean_reciprocal_rank(results, gt))
    
    return {k: np.mean(v) for k, v in metrics.items()}
```

#### 性能基准测试
- **吞吐量测试**: 每秒处理文档数
- **延迟测试**: 端到端响应时间
- **并发测试**: 多用户同时访问性能
- **资源使用**: CPU、内存、GPU利用率

## 模型部署

### 1. 模型加载策略
```python
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.loaded = False
    
    def load_model(self, model_path: str):
        """延迟加载模型"""
        if not self.loaded:
            self.model = AutoModel.from_pretrained(model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.loaded = True
    
    def unload_model(self):
        """卸载模型释放内存"""
        if self.loaded:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()
            self.loaded = False
```

### 2. 服务化部署
- **模型服务**: 独立的嵌入服务
- **负载均衡**: 多实例部署和请求分发
- **缓存策略**: 结果缓存和模型缓存
- **监控告警**: 性能监控和异常告警