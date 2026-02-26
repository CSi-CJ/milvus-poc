#!/usr/bin/env python3
"""验证Milvus索引结果"""

from pymilvus import connections, Collection
from multimodal_indexer.config import load_config

def main():
    config = load_config()
    
    # 连接Milvus
    connections.connect('default', host=config.milvus.host, port=config.milvus.port)
    
    # 获取集合
    collection = Collection(config.milvus.collection_name)
    collection.load()
    
    print('📊 Milvus索引统计')
    print('=' * 60)
    print(f'总记录数: {collection.num_entities}')
    print()
    
    # 查询所有文件
    results = collection.query(
        expr='file_name != ""',
        output_fields=['file_name', 'content_type', 'ocr_text'],
        limit=50
    )
    
    # 按文件分组
    files = {}
    for r in results:
        name = r['file_name']
        if name not in files:
            files[name] = []
        files[name].append(r)
    
    print('已索引的文件:')
    print()
    
    for i, (name, chunks) in enumerate(sorted(files.items()), 1):
        print(f'{i}. {name} ({len(chunks)}个向量块)')
        print(f'   类型: {chunks[0].get("content_type", "unknown")}')
        
        # 检查OCR文本
        ocr_texts = [c.get('ocr_text', '') for c in chunks if c.get('ocr_text')]
        if ocr_texts:
            avg_len = sum(len(t) for t in ocr_texts) // len(ocr_texts)
            print(f'   OCR文本: 是 (平均{avg_len}字符/块)')
        else:
            print('   OCR文本: 否')
        print()
    
    print('=' * 60)
    print(f'✅ 验证完成！共索引 {len(files)} 个文件，{collection.num_entities} 个向量块')
    
    connections.disconnect('default')

if __name__ == '__main__':
    main()
