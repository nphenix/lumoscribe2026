#!/usr/bin/env python
"""检查知识库建设状态"""
import sys
# Add project root to path
PROJECT_ROOT = 'F:/lumoscribe2026'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT + '/src')
from src.application.services.vector_storage_service import VectorStorageService
from pathlib import Path
import asyncio
import json

async def check_sources():
    service = VectorStorageService()
    
    # 1. 获取向量库中的所有 source_file_id
    client = service._get_client()
    collection = client.get_collection(name='default')
    results = collection.get(limit=10000, include=['metadatas'])
    
    source_counts = {}
    for meta in results['metadatas']:
        sid = meta.get('source_file_id', 'unknown')
        source_counts[sid] = source_counts.get(sid, 0) + 1
    
    print('=== 向量库中各源文档的 chunks 数量 ===')
    for sid, count in sorted(source_counts.items()):
        print(f'{sid}: {count} chunks')
    
    total = sum(source_counts.values())
    print(f'\n总计: {total} chunks')
    
    # 2. 检查 pic_to_json 目录下的文档
    print('\n=== pic_to_json 目录下的文档 ===')
    pic_to_json_dirs = list(Path('data/intermediates').glob('*/pic_to_json'))
    for p in sorted(pic_to_json_dirs):
        md_files = list(p.glob('*.md'))
        print(f'{p.parent.name}: {len(md_files)} 个 md 文件')
        for md in md_files:
            print(f'  - {md.name}')
    
    # 3. BM25 索引文档数对比
    print('\n=== BM25 索引文档数 ===')
    bm25_files = sorted(Path('data/intermediates/kb_chunks/default').glob('*.bm25.json'))
    for f in bm25_files:
        data = json.loads(f.read_text(encoding='utf-8'))
        print(f'{f.name}: doc_count={data["doc_count"]}, docs_in_file={len(data["docs"])}')

if __name__ == '__main__':
    asyncio.run(check_sources())
