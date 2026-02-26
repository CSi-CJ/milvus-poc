"""
命令行界面
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
import logging

try:
    import click
except ImportError:
    click = None

from .config import load_config
from .parsers.factory import create_default_registry, get_parser_info
from .embedder import VectorEmbedder
from .index_manager import IndexManager
from .file_processor import FileProcessor, BatchProcessor


def setup_logging(level: str = "INFO"):
    """设置日志"""
    # 创建更详细的日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 设置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有的处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 确保输出立即刷新
    sys.stdout.flush()


def progress_callback(current: int, total: int, result: dict):
    """进度回调函数"""
    status = result.get('status', 'unknown')
    file_path = result.get('file_path', 'unknown')
    
    if status == 'success':
        embeddings_count = result.get('embeddings_count', 0)
        processing_time = result.get('processing_time', 0)
        print(f"[{current}/{total}] ✓ {os.path.basename(file_path)} - {embeddings_count} embeddings ({processing_time:.1f}s)")
    elif status == 'error':
        error = result.get('error', 'unknown error')
        print(f"[{current}/{total}] ✗ {os.path.basename(file_path)} - {error}")
    elif status == 'skipped':
        reason = result.get('reason', 'unknown reason')
        print(f"[{current}/{total}] - {os.path.basename(file_path)} - {reason}")
    
    # 强制刷新输出
    sys.stdout.flush()


async def process_file_async(file_path: str, config_path: Optional[str] = None):
    """异步处理单个文件"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建组件
    parser_registry = create_default_registry(config.processing.__dict__)
    embedder = VectorEmbedder(config.embedding)
    index_manager = IndexManager(config.milvus)
    
    # 创建处理器
    processor = FileProcessor(parser_registry, embedder, index_manager, config)
    
    try:
        # 处理文件
        result = await processor.process_file(file_path)
        
        if result['status'] == 'success':
            print(f"✓ Successfully processed: {file_path}")
            print(f"  - Embeddings: {result['embeddings_count']}")
            print(f"  - Processing time: {result['processing_time']:.2f}s")
        elif result['status'] == 'skipped':
            print(f"- Skipped: {file_path}")
            print(f"  - Reason: {result.get('reason', 'Unknown reason')}")
        else:
            print(f"✗ Failed to process: {file_path}")
            print(f"  - Error: {result.get('error', 'Unknown error')}")
    
    finally:
        index_manager.close()


async def process_directory_async(directory_path: str, config_path: Optional[str] = None):
    """异步处理目录"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建组件
    parser_registry = create_default_registry(config.processing.__dict__)
    embedder = VectorEmbedder(config.embedding)
    index_manager = IndexManager(config.milvus)
    
    # 创建处理器
    processor = FileProcessor(parser_registry, embedder, index_manager, config)
    batch_processor = BatchProcessor(processor, config.processing.max_concurrent)
    
    try:
        print(f"Processing directory: {directory_path}")
        print(f"Max concurrent: {config.processing.max_concurrent}")
        print()
        
        # 批量处理
        result = await batch_processor.process_directory(
            directory_path, 
            progress_callback=progress_callback
        )
        
        print()
        print("=" * 50)
        print("PROCESSING SUMMARY")
        print("=" * 50)
        print(f"Total files: {result['total_files']}")
        print(f"Successful: {result['successful']}")
        print(f"Failed: {result['failed']}")
        print(f"Skipped: {result['skipped']}")
        print(f"Total processing time: {result['processing_time']:.2f}s")
        print(f"Average time per file: {result['average_time_per_file']:.2f}s")
        print(f"Total embeddings: {result['total_embeddings']}")
        
        if result['total_size'] > 0:
            size_mb = result['total_size'] / (1024 * 1024)
            print(f"Total size processed: {size_mb:.2f} MB")
    
    finally:
        index_manager.close()


async def search_async(query: str, top_k: int = 10, config_path: Optional[str] = None):
    """异步搜索"""
    # 加载配置
    config = load_config(config_path)
    
    # 创建组件
    embedder = VectorEmbedder(config.embedding)
    index_manager = IndexManager(config.milvus)
    
    try:
        print(f"Searching for: {query}")
        print(f"Top K: {top_k}")
        print()
        
        # 生成查询向量
        query_vector = embedder.embed_text(query)
        
        # 搜索
        results = index_manager.search_vectors(
            query_vectors=[query_vector.tolist()],
            top_k=top_k
        )
        
        if not results:
            print("No results found.")
            return
        
        print("SEARCH RESULTS")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('file_name', 'Unknown')}")
            print(f"   Path: {result.get('file_path', 'Unknown')}")
            print(f"   Type: {result.get('file_type', 'Unknown')}")
            print(f"   Content: {result.get('content_type', 'Unknown')}")
            print(f"   Score: {result.get('score', 0):.4f}")
            print()
    
    finally:
        index_manager.close()


def check_dependencies():
    """检查依赖"""
    print("DEPENDENCY CHECK")
    print("=" * 50)
    
    info = get_parser_info()
    
    for parser in info['parsers']:
        status_icon = "✓" if parser['status'] == 'available' else "✗"
        print(f"{status_icon} {parser['name']}")
        print(f"   Formats: {', '.join(parser['formats'])}")
        print(f"   Dependencies: {', '.join(parser['dependencies'])}")
        
        if parser['status'] != 'available' and 'install_command' in parser:
            print(f"   Install: {parser['install_command']}")
        print()


def health_check(config_path: Optional[str] = None):
    """健康检查"""
    try:
        config = load_config(config_path)
        index_manager = IndexManager(config.milvus)
        
        health = index_manager.health_check()
        
        print("HEALTH CHECK")
        print("=" * 50)
        
        if health['connected']:
            print("✓ Milvus connection: OK")
            print(f"  Host: {health['host']}:{health['port']}")
            print(f"  Database: {health['database']}")
            print(f"  Collections: {', '.join(health['collections'])}")
        else:
            print("✗ Milvus connection: FAILED")
            print(f"  Error: {health.get('error', 'Unknown error')}")
        
        index_manager.close()
        
    except Exception as e:
        print(f"✗ Health check failed: {e}")


# 如果没有安装 click，提供简单的命令行界面
if click is None:
    def main():
        """简单的命令行界面（不使用 click）"""
        if len(sys.argv) < 2:
            print("Usage:")
            print("  python -m multimodal_indexer.cli <command> [args]")
            print()
            print("Commands:")
            print("  process-file <file_path>     - Process a single file")
            print("  process-dir <directory>      - Process all files in directory")
            print("  search <query> [top_k]       - Search for similar content")
            print("  check-deps                   - Check dependencies")
            print("  health                       - Health check")
            return
        
        command = sys.argv[1]
        
        if command == "process-file":
            if len(sys.argv) < 3:
                print("Usage: process-file <file_path>")
                return
            file_path = sys.argv[2]
            setup_logging()
            asyncio.run(process_file_async(file_path))
        
        elif command == "process-dir":
            if len(sys.argv) < 3:
                print("Usage: process-dir <directory>")
                return
            directory = sys.argv[2]
            setup_logging()
            asyncio.run(process_directory_async(directory))
        
        elif command == "search":
            if len(sys.argv) < 3:
                print("Usage: search <query> [top_k]")
                return
            query = sys.argv[2]
            top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10
            setup_logging()
            asyncio.run(search_async(query, top_k))
        
        elif command == "check-deps":
            check_dependencies()
        
        elif command == "health":
            setup_logging()
            health_check()
        
        else:
            print(f"Unknown command: {command}")

else:
    # 使用 click 的完整命令行界面
    @click.group()
    @click.option('--config', '-c', help='Configuration file path')
    @click.option('--log-level', default='INFO', help='Log level')
    @click.pass_context
    def cli(ctx, config, log_level):
        """多模态文件索引器命令行工具"""
        ctx.ensure_object(dict)
        ctx.obj['config'] = config
        setup_logging(log_level)

    @cli.command()
    @click.argument('file_path', type=click.Path(exists=True))
    @click.pass_context
    def process_file(ctx, file_path):
        """处理单个文件"""
        asyncio.run(process_file_async(file_path, ctx.obj['config']))

    @cli.command()
    @click.argument('directory', type=click.Path(exists=True, file_okay=False))
    @click.pass_context
    def process_dir(ctx, directory):
        """处理目录中的所有文件"""
        asyncio.run(process_directory_async(directory, ctx.obj['config']))

    @cli.command()
    @click.argument('query')
    @click.option('--top-k', default=10, help='Number of results to return')
    @click.pass_context
    def search(ctx, query, top_k):
        """搜索相似内容"""
        asyncio.run(search_async(query, top_k, ctx.obj['config']))

    @cli.command()
    def check_deps():
        """检查依赖状态"""
        check_dependencies()

    @cli.command()
    @click.pass_context
    def health(ctx):
        """系统健康检查"""
        health_check(ctx.obj['config'])

    def main():
        cli()


if __name__ == '__main__':
    main()