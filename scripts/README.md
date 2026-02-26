# 工具脚本

本目录包含项目的各类工具脚本。

## 脚本列表

### 运维工具

**clean_and_reindex.py** - 重新索引工具
- 清理Milvus集合
- 重新索引所有文件
- 验证索引结果
- 使用方法：`python scripts/clean_and_reindex.py`

**verify_index.py** - 索引验证工具
- 检查Milvus索引状态
- 显示索引统计信息
- 列出已索引文件
- 使用方法：`python scripts/verify_index.py`

## 使用说明

### 重新索引
当需要重新索引所有文件时：
```bash
python scripts/clean_and_reindex.py
```

这将：
1. 清理现有Milvus集合
2. 重新处理files/目录下的所有文件
3. 验证索引结果

### 验证索引
检查当前索引状态：
```bash
python scripts/verify_index.py
```

这将显示：
- 总记录数
- 已索引的文件列表
- 每个文件的向量块数
- OCR文本统计

## 注意事项

- 运行脚本前确保Milvus服务已启动
- 重新索引会删除现有数据，请谨慎操作
- 大量文件索引可能需要较长时间
- 建议在非生产环境测试

## 添加新脚本

新增工具脚本时：
1. 添加脚本文件到此目录
2. 更新本README文档
3. 添加使用说明和注意事项
4. 确保脚本有适当的错误处理

---

*最后更新: 2026-01-14*
