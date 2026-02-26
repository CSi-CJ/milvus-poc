# Multimodal File Indexer

A document processing and semantic search system powered by [BGE-M3](https://huggingface.co/BAAI/bge-m3) embeddings and [Milvus](https://milvus.io/) vector database. It parses PDF, text, image, audio, and video files, generates vector embeddings, and enables fast similarity search across all content.

## Features

- **Multimodal Parsing** — PDF, text/markdown, images, audio, and video
- **Semantic Search** — 1024-dimensional vector embeddings via BGE-M3 with cosine similarity
- **Multi-Engine OCR** — EasyOCR, PaddleOCR, and Tesseract with automatic fallback
- **Speech Recognition** — Whisper-based audio transcription with language detection
- **Smart Video Processing** — Scene-change detection, keyframe extraction, and frame quality enhancement
- **Image Storage** — Base64-encoded images stored in Milvus, displayed directly in the Web UI
- **Concurrent Processing** — Batch file processing with configurable concurrency
- **Multiple Interfaces** — CLI, Python SDK, and Web UI/API

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for Milvus)
- (Optional) GPU with CUDA for faster embedding and OCR

### Installation

```bash
git clone <repository-url>
cd multimodal-file-indexer

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt
```

### Start Milvus

```bash
docker run -d --name milvus-standalone \
  -p 19530:19530 -p 9091:9091 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest standalone
```

### Configuration

Copy the example config and fill in your settings:

```bash
cp config/config.example.json config/config.json
```

See the [Configuration](#configuration) section for details.

### Usage

```bash
# Process a single file
python -m multimodal_indexer.cli process-file ./path/to/document.pdf

# Process all files in a directory
python -m multimodal_indexer.cli process-dir ./path/to/files/

# Search indexed content
python -m multimodal_indexer.cli search "search query"

# Launch the Web UI
python web_ui.py
# Then open http://localhost:5000
```

## Supported File Types

| Category | Formats | Capabilities |
|----------|---------|-------------|
| PDF | `.pdf` | Text extraction, page screenshots (2× resolution), OCR, metadata |
| Text | `.txt`, `.md`, `.doc`, `.docx` | Multi-encoding support (UTF-8, GBK, GB2312), format auto-detection |
| Image | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp` | Multi-engine OCR, EXIF metadata, Base64 storage |
| Audio | `.mp3`, `.wav`, `.m4a` | Whisper transcription, waveform/spectrogram visualization, language detection |
| Video | `.mp4`, `.avi`, `.mov` | Scene-change keyframe extraction, frame quality enhancement, OCR on frames |

## Architecture Overview

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   File Input     │    │   Parser Layer   │    │  Embedding Layer │
│                  │    │                  │    │                  │
│  PDF, Text,      │───▶│  PDF Parser      │───▶│  BGE-M3 Model   │
│  Image, Audio,   │    │  Text Parser     │    │  1024-dim vectors│
│  Video           │    │  Image Parser    │    │                  │
│                  │    │  Audio Parser    │    │                  │
│                  │    │  Video Parser    │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
                                                         │
                                                         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Retrieval Layer │    │   Index Layer    │    │  Storage Layer   │
│                  │    │                  │    │                  │
│  Semantic search │◀───│  HNSW indexing   │◀───│  Milvus DB       │
│  Similarity rank │    │  Metadata store  │    │  Collections     │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

Key components (under `multimodal_indexer/`):

- **parsers/** — File-type-specific parsers with a plugin registry
- **embedder.py** — Vector embedding via BGE-M3
- **index_manager.py** — Milvus vector index operations
- **file_processor.py** — Orchestrates parsing → embedding → indexing
- **collection_manager.py** — Milvus collection lifecycle management
- **config.py** — Configuration loading and validation
- **cli.py** — Command-line interface
- **system.py** — System factory and context manager

## CLI Usage

```bash
# Process a single file
python -m multimodal_indexer.cli process-file document.pdf

# Process a directory of files
python -m multimodal_indexer.cli process-dir ./documents/

# Semantic search
python -m multimodal_indexer.cli search "project report" --top-k 5

# Check parser dependencies
python -m multimodal_indexer.cli check-deps

# Milvus health check
python -m multimodal_indexer.cli health
```

Options:

```bash
python -m multimodal_indexer.cli --config path/to/config.json --log-level DEBUG <command>
```

## Web UI & API

Start the web server:

```bash
python web_ui.py
```

The Web UI is available at `http://localhost:5000` and provides a search interface with image preview support.

### API Endpoints

**Search** — `POST /search`

```bash
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "search query", "top_k": 10}'
```

**System Stats** — `GET /stats`

```bash
curl http://localhost:5000/stats
```

## Configuration

The configuration file lives at `config/config.json`. Copy from the provided template:

```bash
cp config/config.example.json config/config.json
```

Key sections:

```jsonc
{
  "milvus": {
    "host": "localhost",
    "port": 19530,
    "user": "",              // Milvus credentials (if auth enabled)
    "password": "",
    "collection_name": "multimodal_files",
    "vector_dim": 1024,
    "index_type": "HNSW",
    "metric_type": "COSINE"
  },
  "embedding": {
    "multimodal_model": "BAAI/bge-m3",
    "batch_size": 12,
    "use_fp16": true
  },
  "processing": {
    "max_concurrent": 10,
    "enable_ocr": true,
    "enable_speech_recognition": true,
    "skip_existing": true
  },
  "logging": {
    "level": "INFO"
  }
}
```

See [docs/configuration.md](./docs/configuration.md) for the full reference.

## Development

### Python SDK

```python
from multimodal_indexer.system import MultimodalIndexerSystem

with MultimodalIndexerSystem() as system:
    # Process a file
    result = system.file_processor.process_file("document.pdf")

    # Search
    query_vector = system.embedder.embed_text("search query")
    results = system.index_manager.search_vectors(
        query_vectors=[query_vector.tolist()],
        top_k=5
    )
```

### Custom Parsers

Extend `BaseFileParser` and register it:

```python
from multimodal_indexer.parsers.base import BaseFileParser
from multimodal_indexer.parsers.registry import FileParserRegistry
from multimodal_indexer.models import ParsedContent

class CustomParser(BaseFileParser):
    def can_parse(self, file_path: str) -> bool:
        return file_path.lower().endswith('.custom')

    def parse(self, file_path: str) -> ParsedContent:
        self._validate_file(file_path)
        # Your parsing logic here
        return ParsedContent(...)

registry = FileParserRegistry()
registry.register(CustomParser())
```

### Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Run tests before submitting: `pytest tests/`
4. Open a Pull Request

### Documentation

Detailed docs are in the [docs/](./docs/) directory:

- [Architecture](./docs/architecture.md)
- [API Reference](./docs/api.md)
- [Configuration](./docs/configuration.md)
- [Deployment](./docs/deployment.md)
- [Models](./docs/models.md)
- [Parsing](./docs/parsing.md)
- [Chunking Strategy](./docs/chunking.md)
- [High-Level Design](./docs/high_level_design.md)

## License

MIT License — see [LICENSE](LICENSE) for details.
