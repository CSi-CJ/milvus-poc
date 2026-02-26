#!/usr/bin/env python3
"""
简单的Web UI用于查看多模态文件索引系统的召回结果
"""

import os
import json
from flask import Flask, render_template, request, jsonify
from multimodal_indexer.config import load_config
from multimodal_indexer.parsers.factory import create_default_registry
from multimodal_indexer.embedder import VectorEmbedder
from multimodal_indexer.index_manager import IndexManager

app = Flask(__name__)

# 全局变量存储系统组件
config = None
embedder = None
index_manager = None

def initialize_system():
    """初始化系统组件"""
    global config, embedder, index_manager
    
    try:
        # 加载配置
        config = load_config()
        
        # 初始化嵌入器
        embedder = VectorEmbedder(config.embedding)
        
        # 初始化索引管理器
        index_manager = IndexManager(config.milvus)
        
        print("✓ 系统初始化成功")
        return True
    except Exception as e:
        print(f"✗ 系统初始化失败: {e}")
        return False

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """搜索接口"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = int(data.get('top_k', 10))
        
        if not query:
            return jsonify({'error': '查询不能为空'}), 400
        
        # 生成查询向量
        query_vector = embedder.search_embed(query)
        
        # 搜索
        results = index_manager.search_vectors(
            query_vectors=[query_vector.tolist()],
            top_k=top_k
        )
        
        # 格式化结果
        formatted_results = []
        for result in results:
            formatted_result = {
                'id': result.get('id', ''),
                'score': round(result.get('score', 0), 4),
                'distance': round(result.get('distance', 0), 4),
                'file_path': result.get('file_path', ''),
                'file_name': result.get('file_name', ''),
                'file_type': result.get('file_type', ''),
                'content_type': result.get('content_type', ''),
                'chunk_index': result.get('chunk_index', 0),
                
                # chunk内容字段
                'chunk_content': result.get('chunk_content', ''),
                'chunk_summary': result.get('chunk_summary', ''),
                'content_length': result.get('content_length', 0),
                'chunk_content_preview': result.get('chunk_content_preview', ''),
                
                # 图像数据字段
                'image_data': result.get('image_data', ''),
                'image_format': result.get('image_format', ''),
                'image_size': result.get('image_size', ''),
                'ocr_text': result.get('ocr_text', ''),
                'has_image': bool(result.get('image_data', '')),
                
                # 音频数据字段
                'audio_transcript': result.get('audio_transcript', ''),
                'audio_language': result.get('audio_language', ''),
                'audio_confidence': result.get('audio_confidence', 0.0),
                'has_audio_transcript': bool(result.get('audio_transcript', '')),
                
                # 分离的元数据
                'metadata': result.get('metadata', {}),
                'file_metadata': result.get('file_metadata', {}),
                'content_metadata': result.get('content_metadata', {})
            }
            formatted_results.append(formatted_result)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': formatted_results,
            'total': len(formatted_results)
        })
        
    except Exception as e:
        return jsonify({'error': f'搜索失败: {str(e)}'}), 500

@app.route('/stats')
def stats():
    """获取系统统计信息"""
    try:
        stats = index_manager.get_collection_stats()
        health = index_manager.health_check()
        
        return jsonify({
            'success': True,
            'collection_stats': stats,
            'health': health,
            'embedder_info': embedder.get_model_info()
        })
        
    except Exception as e:
        return jsonify({'error': f'获取统计信息失败: {str(e)}'}), 500

if __name__ == '__main__':
    # 初始化系统
    if initialize_system():
        print("🚀 启动Web UI服务器...")
        print("📱 访问地址: http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ 系统初始化失败，无法启动Web服务器")