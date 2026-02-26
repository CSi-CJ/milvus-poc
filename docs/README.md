# 文档索引

## 核心文档

本目录包含多模态文件索引器（Multimodal File Indexer）的核心技术文档。

### 系统架构

- **[architecture.md](architecture.md)** - 系统整体架构设计
  - 系统组件和模块划分
  - 数据流和处理流程
  - 技术栈和依赖关系

- **[high_level_design.md](high_level_design.md)** - 高层设计文档
  - 设计原则和决策
  - 系统边界和接口
  - 扩展性考虑

### 功能模块

- **[parsing.md](parsing.md)** - 文件解析器详解
  - PDF 解析器
  - 文本解析器
  - 图像解析器（OCR）
  - 音频解析器（语音识别）
  - 视频解析器（场景检测 + OCR）

- **[chunking.md](chunking.md)** - 内容切片策略
  - 文本切片算法
  - 图像切片方法
  - 多模态融合策略

- **[models.md](models.md)** - 向量嵌入模型
  - BGE-M3 模型介绍
  - 向量生成流程
  - 性能优化

### 配置和部署

- **[configuration.md](configuration.md)** - 配置指南
  - 配置文件说明
  - 参数详解
  - 环境变量

- **[deployment.md](deployment.md)** - 部署指南
  - 本地部署
  - Docker 部署
  - 生产环境配置

### API 文档

- **[api.md](api.md)** - API 参考文档
  - REST API 接口
  - Python SDK 使用
  - CLI 命令行工具

## 快速导航

| 角色 | 推荐阅读顺序 |
|------|-------------|
| 新用户 | [architecture.md](architecture.md) → [configuration.md](configuration.md) → [deployment.md](deployment.md) |
| 开发者 | [parsing.md](parsing.md) → [models.md](models.md) → [api.md](api.md) |
| 运维人员 | [deployment.md](deployment.md) → [configuration.md](configuration.md) → [architecture.md](architecture.md) |

## 文档维护

- 文档应保持与代码同步更新
- 重大功能变更需更新相关文档
- 定期审查文档的准确性和完整性
