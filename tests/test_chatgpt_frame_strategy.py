#!/usr/bin/env python3
"""
测试ChatGPT帧策略集成到完整系统
"""

import asyncio
import logging
from multimodal_indexer.config import load_config
from multimodal_indexer.index_manager import IndexManager
from multimodal_indexer.parsers.factory import create_default_registry
from multimodal_indexer.embedder import VectorEmbedder
from multimodal_indexer.file_processor import FileProcessor

async def test_chatgpt_frame_strategy_integration():
    """测试ChatGPT帧策略集成"""
    
    # 设置详细日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🧪 测试ChatGPT帧策略集成到完整系统")
    print("="*60)
    
    # 加载配置
    config = load_config()
    
    # 创建组件
    parser_registry = create_default_registry(config.processing.__dict__)
    embedder = VectorEmbedder(config.embedding)
    index_manager = IndexManager(config.milvus)
    
    try:
        # 清理现有数据
        print("🗑️  清理现有数据...")
        try:
            # 删除所有可能的集合
            for collection_name in ["multimodal_content", "multimodal_files"]:
                try:
                    index_manager.drop_collection(collection_name)
                    print(f"✅ 删除集合: {collection_name}")
                except Exception as e:
                    print(f"⚠️  集合 {collection_name} 不存在或删除失败: {e}")
        except Exception as e:
            print(f"⚠️  清理数据时出现错误: {e}")
        
        # 创建处理器
        processor = FileProcessor(parser_registry, embedder, index_manager, config)
        
        # 只处理视频文件
        video_file = "./files/个性化推荐.mp4"
        
        print(f"🔄 开始处理视频文件: {video_file}")
        print("   使用ChatGPT帧策略: 1 FPS + 相似度过滤 + 重点帧识别")
        
        result = await processor.process_file(video_file)
        
        if result['status'] == 'success':
            print(f"✅ 视频处理成功!")
            print(f"   - 嵌入向量数量: {result['embeddings_count']}")
            print(f"   - 处理时间: {result['processing_time']:.2f}s")
            
            # 搜索测试 - 多个查询
            test_queries = [
                "财务报告",
                "XR",
                "MindSearch",
                "早上好",
                "计划"
            ]
            
            print(f"\n🔍 搜索测试 ({len(test_queries)} 个查询)...")
            print("="*60)
            
            for query in test_queries:
                print(f"\n🔎 查询: '{query}'")
                query_vector = embedder.embed_text(query)
                search_results = index_manager.search_vectors(
                    query_vectors=[query_vector.tolist()],
                    top_k=3
                )
                
                print(f"   找到 {len(search_results)} 个结果:")
                for i, result in enumerate(search_results, 1):
                    content_type = result.get('content_type', 'unknown')
                    file_name = result.get('file_name', 'unknown')
                    score = result.get('score', 0)
                    
                    print(f"   {i}. 类型: {content_type}, 文件: {file_name}, 评分: {score:.4f}")
                    
                    # 显示OCR文本
                    ocr_text = result.get('ocr_text', '')
                    if ocr_text:
                        # 截取前100个字符显示
                        display_text = ocr_text[:100] + "..." if len(ocr_text) > 100 else ocr_text
                        print(f"      📝 OCR文本: {display_text}")
                    
                    # 显示帧策略信息
                    if 'frame_strategy' in result:
                        strategy_info = result['frame_strategy']
                        print(f"      🎯 帧策略: 相似度过滤={strategy_info.get('similarity_filtering', False)}, "
                              f"优先级排序={strategy_info.get('priority_ranking', False)}")
            
            # 显示详细的OCR提取统计
            print(f"\n📊 OCR提取统计:")
            print("="*60)
            
            # 获取所有结果查看OCR提取情况
            all_results = index_manager.search_vectors(
                query_vectors=[embedder.embed_text("").tolist()],
                top_k=50
            )
            
            total_items = len(all_results)
            items_with_ocr = sum(1 for r in all_results if r.get('ocr_text', '').strip())
            total_ocr_length = sum(len(r.get('ocr_text', '')) for r in all_results)
            
            print(f"   总项目数: {total_items}")
            print(f"   包含OCR文本的项目: {items_with_ocr}")
            print(f"   OCR覆盖率: {items_with_ocr/total_items*100 if total_items > 0 else 0:.1f}%")
            print(f"   总OCR文本长度: {total_ocr_length} 字符")
            print(f"   平均每项OCR长度: {total_ocr_length/items_with_ocr if items_with_ocr > 0 else 0:.1f} 字符")
            
            # 显示帧策略效果
            frame_strategy_items = [r for r in all_results if 'frame_strategy' in r]
            if frame_strategy_items:
                print(f"   使用帧策略的项目: {len(frame_strategy_items)}")
                print("   帧策略特性:")
                for item in frame_strategy_items[:3]:  # 显示前3个
                    strategy = item.get('frame_strategy', {})
                    ocr_info = item.get('ocr_extraction', {})
                    print(f"     - 成功帧数: {ocr_info.get('successful_frames', 0)}/{ocr_info.get('total_frames', 0)}")
                    print(f"       平均置信度: {ocr_info.get('average_confidence', 0):.3f}")
                    print(f"       相似度过滤: {strategy.get('similarity_filtering', False)}")
                    print(f"       优先级排序: {strategy.get('priority_ranking', False)}")
        else:
            print(f"❌ 视频处理失败: {result.get('error', 'unknown error')}")
            # 显示详细错误信息
            if 'traceback' in result:
                print(f"详细错误: {result['traceback']}")
    
    finally:
        index_manager.close()

if __name__ == "__main__":
    asyncio.run(test_chatgpt_frame_strategy_integration())