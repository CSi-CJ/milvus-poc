#!/usr/bin/env python3
"""
清理Milvus集合并重新索引
正确的顺序：unload -> drop -> create -> reindex
"""

import os
import sys
import logging
from typing import List

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)  # 上一级目录
sys.path.insert(0, project_dir)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def clean_milvus_collection():
    """正确清理Milvus集合：unload -> drop"""
    print("🧹 清理Milvus集合...")
    
    try:
        from pymilvus import connections, utility, Collection
        from multimodal_indexer.config import load_config
        
        # 加载配置
        config = load_config()
        milvus_config = config.milvus
        
        # 连接到Milvus
        connections.connect(
            alias="default",
            host=milvus_config.host,
            port=milvus_config.port
        )
        print(f"✅ 已连接到Milvus: {milvus_config.host}:{milvus_config.port}")
        
        collection_name = milvus_config.collection_name
        
        # 检查集合是否存在
        if utility.has_collection(collection_name):
            print(f"📊 发现集合: {collection_name}")
            
            # 获取集合统计
            collection = Collection(collection_name)
            collection.load()  # 确保集合已加载以获取统计信息
            
            print(f"   当前记录数: {collection.num_entities}")
            
            # 步骤1: Unload集合
            print("🔄 正在unload集合...")
            collection.release()
            print("✅ 集合已unload")
            
            # 步骤2: Drop集合
            print("🗑️  正在删除集合...")
            utility.drop_collection(collection_name)
            print("✅ 集合已删除")
            
        else:
            print(f"ℹ️  集合 {collection_name} 不存在")
        
        # 断开连接
        connections.disconnect("default")
        print("✅ 已断开Milvus连接")
        
        return True
        
    except Exception as e:
        print(f"❌ 清理集合失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
        return False

def reindex_files():
    """重新索引所有文件"""
    print("\n🚀 开始重新索引文件...")
    
    # 获取files目录下的所有文件
    files_dir = "./files"
    if not os.path.exists(files_dir):
        print(f"❌ 文件目录不存在: {files_dir}")
        return False
    
    # 使用CLI命令处理整个目录
    print(f"📁 处理目录: {files_dir}")
    
    import subprocess
    
    try:
        # 运行CLI命令处理目录
        cmd = ["python", "-m", "multimodal_indexer.cli", "process-dir", files_dir]
        
        print(f"🔄 执行命令: {' '.join(cmd)}")
        print("📋 实时处理日志:")
        print("-" * 60)
        
        # 使用实时输出模式
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',  # 处理编码错误
            bufsize=1,
            universal_newlines=True
        )
        
        # 实时显示输出
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 等待进程完成
        return_code = process.poll()
        
        print("-" * 60)
        
        if return_code == 0:
            print("✅ 重新索引完成")
            return True
        else:
            print(f"❌ 重新索引失败，退出码: {return_code}")
            return False
            
    except Exception as e:
        print(f"❌ 执行重新索引失败: {e}")
        return False

def verify_results():
    """验证重新索引的结果"""
    print("\n🔍 验证重新索引结果...")
    
    try:
        from pymilvus import connections, Collection, utility
        from multimodal_indexer.config import load_config
        
        # 连接到Milvus
        config = load_config()
        milvus_config = config.milvus
        
        connections.connect(
            alias="default",
            host=milvus_config.host,
            port=milvus_config.port
        )
        
        collection_name = milvus_config.collection_name
        
        if not utility.has_collection(collection_name):
            print("❌ 集合不存在")
            return False
        
        # 获取集合信息
        collection = Collection(collection_name)
        collection.load()
        
        total_count = collection.num_entities
        print(f"📊 集合统计:")
        print(f"   总记录数: {total_count}")
        
        if total_count > 0:
            # 查询一些示例数据
            print("\n📄 示例数据:")
            
            # 查询前5条记录
            results = collection.query(
                expr='file_name != ""',
                output_fields=["file_name", "content_type", "ocr_text"],
                limit=5
            )
            
            for i, result in enumerate(results, 1):
                file_name = result.get('file_name', 'unknown')
                content_type = result.get('content_type', 'unknown')
                ocr_text = result.get('ocr_text', '')
                
                print(f"   {i}. {file_name} ({content_type})")
                if ocr_text:
                    print(f"      OCR文本长度: {len(ocr_text)} 字符")
                    print(f"      OCR预览: {ocr_text[:100]}...")
                else:
                    print("      无OCR文本")
        
        connections.disconnect("default")
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def main():
    """主函数"""
    print("🎯 清理并重新索引Milvus数据")
    print("=" * 60)
    
    # 步骤1: 清理现有集合
    print("步骤1: 清理现有集合")
    if not clean_milvus_collection():
        print("❌ 集合清理失败，终止操作")
        return
    
    # 步骤2: 重新索引文件
    print("\n步骤2: 重新索引文件")
    if not reindex_files():
        print("❌ 重新索引失败")
        return
    
    # 步骤3: 验证结果
    print("\n步骤3: 验证结果")
    if verify_results():
        print("\n🎉 重新索引成功完成！")
        print("现在Milvus中应该包含使用增强OCR处理的数据。")
    else:
        print("\n⚠️  验证过程中出现问题，请检查数据。")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断操作")
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")