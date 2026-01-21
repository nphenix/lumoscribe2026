#!/usr/bin/env python3
"""测试各供应商的连通性。"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
from src.shared.config import get_settings
from src.shared.db import make_engine, make_session_factory
from src.domain.entities.llm_provider import LLMProvider


def test_openai_compatible(provider: LLMProvider) -> dict:
    """测试 OpenAI 兼容接口的连通性。"""
    config = json.loads(provider.config_json) or {}
    model = config.get('model', '')
    base_url = provider.base_url.rstrip('/') if provider.base_url else ''
    
    result = {
        'provider': provider.name,
        'key': provider.key,
        'type': provider.provider_type,
        'base_url': base_url,
        'model': model,
        'success': False,
        'error': None,
        'response': None,
    }
    
    if not base_url:
        result['error'] = 'Base URL 未配置'
        return result
    
    try:
        headers = {
            'Content-Type': 'application/json',
        }
        
        if provider.api_key:
            headers['Authorization'] = f'Bearer {provider.api_key}'
        
        payload = {
            'model': model or 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': 'Hi'}],
            'max_tokens': 5,
        }
        
        response = requests.post(
            f'{base_url}/chat/completions',
            headers=headers,
            json=payload,
            timeout=config.get('timeout_seconds', 60) or 60,
        )
        
        if response.status_code in (200, 429):
            result['success'] = True
            result['response'] = f'Status: {response.status_code}'
        else:
            result['error'] = f'HTTP {response.status_code}: {response.text[:200]}'
            
    except requests.exceptions.Timeout:
        result['error'] = 'Connection timed out'
    except requests.exceptions.ConnectionError as e:
        result['error'] = f'Connection failed: {str(e)[:200]}'
    except Exception as e:
        result['error'] = f'Error: {str(e)[:200]}'
    
    return result


def test_ollama(provider: LLMProvider) -> dict:
    """测试 Ollama 接口的连通性。"""
    config = json.loads(provider.config_json) or {}
    ollama_model = config.get('ollama_model', 'llama3.1')
    base_url = provider.base_url.rstrip('/') if provider.base_url else ''
    
    result = {
        'provider': provider.name,
        'key': provider.key,
        'type': provider.provider_type,
        'base_url': base_url,
        'model': ollama_model,
        'success': False,
        'error': None,
        'response': None,
    }
    
    if not base_url:
        result['error'] = 'Base URL 未配置'
        return result
    
    try:
        response = requests.get(f'{base_url}/api/tags', timeout=10)
        
        if response.status_code == 200:
            result['success'] = True
            result['response'] = 'API reachable'
            
            models_data = response.json()
            model_names = [m['name'] for m in models_data.get('models', [])]
            available_models = [m.split(':')[0] for m in model_names]
            expected_model = ollama_model.split(':')[0]
            
            if expected_model in available_models or ollama_model in model_names:
                result['response'] += f' | Model "{ollama_model}" is available'
            else:
                result['response'] += f' | Model "{ollama_model}" NOT in available models'
                result['error'] = f'Model "{ollama_model}" not found'
                result['success'] = False
        else:
            result['error'] = f'HTTP {response.status_code}: {response.text[:200]}'
            
    except requests.exceptions.Timeout:
        result['error'] = 'Connection timed out'
    except requests.exceptions.ConnectionError as e:
        result['error'] = f'Connection failed: {str(e)[:200]}'
    except Exception as e:
        result['error'] = f'Error: {str(e)[:200]}'
    
    return result


def test_flagembedding(provider: LLMProvider) -> dict:
    """测试 FlagEmbedding 接口的连通性。"""
    config = json.loads(provider.config_json) or {}
    device = config.get('device', 'cpu')
    
    result = {
        'provider': provider.name,
        'key': provider.key,
        'type': provider.provider_type,
        'device': device,
        'success': False,
        'error': None,
        'response': None,
    }
    
    if config.get('remote'):
        base_url = config.get('host', '')
        if not base_url:
            result['error'] = 'Remote mode enabled but no host configured'
            return result
        
        try:
            response = requests.get(f'{base_url}/health', timeout=10)
            if response.status_code == 200:
                result['success'] = True
                result['response'] = 'Remote service healthy'
            else:
                result['error'] = f'HTTP {response.status_code}'
        except Exception as e:
            result['error'] = str(e)[:200]
    else:
        try:
            from FlagEmbedding import FlagModel
            result['success'] = True
            result['response'] = f'Local mode | Device: {device} | Library available'
        except ImportError as e:
            result['error'] = f'Library not installed: {str(e)[:100]}'
        except Exception as e:
            result['error'] = str(e)[:200]
    
    return result


def show_provider_config(provider: LLMProvider) -> None:
    """显示 Provider 的配置信息。"""
    config = json.loads(provider.config_json) if provider.config_json else {}
    
    print(f"\n[{provider.key}] {provider.name} ({provider.provider_type})")
    print(f"  Base URL: {provider.base_url}")
    
    if config.get('model'):
        print(f"  Model: {config['model']}")
    if config.get('ollama_model'):
        print(f"  Ollama Model: {config['ollama_model']}")
    if config.get('device'):
        print(f"  Device: {config['device']}")
    
    if provider.api_key:
        print(f"  API Key: [已配置 - 明文]")
    elif provider.api_key_env:
        print(f"  API Key: [环境变量: {provider.api_key_env}]")
    else:
        print(f"  API Key: [未配置]")


def main() -> None:
    """测试各供应商的连通性。"""
    settings = get_settings()
    db_path = Path(settings.sqlite_path)
    engine = make_engine(db_path)
    session_factory = make_session_factory(engine)

    # 要测试的供应商 key
    test_keys = ['k', 'm', 'ollama', 'e', 'r']
    
    print("=" * 70)
    print("供应商连通性测试")
    print("=" * 70)
    
    with session_factory() as session:
        providers = session.query(LLMProvider).all()
        test_results = []
        
        for provider in providers:
            if provider.key not in test_keys:
                continue
            
            show_provider_config(provider)
            print(f"  测试连通性...")
            
            if provider.provider_type == 'openai_compatible':
                result = test_openai_compatible(provider)
            elif provider.provider_type == 'ollama':
                result = test_ollama(provider)
            elif provider.provider_type == 'flagembedding':
                result = test_flagembedding(provider)
            else:
                result = {
                    'provider': provider.name,
                    'key': provider.key,
                    'type': provider.provider_type,
                    'success': False,
                    'error': f'Unsupported provider type: {provider.provider_type}',
                }
            
            test_results.append(result)
            
            status = '✅ 成功' if result['success'] else '❌ 失败'
            print(f"  状态: {status}")
            if result.get('error'):
                print(f"  错误: {result['error']}")
            if result.get('response'):
                print(f"  信息: {result['response']}")
        
        # 汇总
        print("\n" + "=" * 70)
        print("测试汇总")
        print("=" * 70)
        
        success_count = sum(1 for r in test_results if r['success'])
        total_count = len(test_results)
        
        for r in test_results:
            status = '✅' if r['success'] else '❌'
            print(f"{status} [{r['key']}] {r['provider']} ({r['type']}): {r.get('error') or r.get('response', 'OK')}")
        
        print(f"\n总计: {success_count}/{total_count} 通过")
        
        if success_count < total_count:
            print("\n⚠️  存在失败的供应商，请检查配置或网络连接")
        
        return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
