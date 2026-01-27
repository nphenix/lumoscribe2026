"""T064: 大纲润色和保存功能测试。

约束（对齐项目要求）：
- 使用真实 LLM Runtime 调用（基于 T060 LangChain Agent 实现）
- 不使用任何 mock 数据
- 测试 Service 层和 API 端点的真实行为

运行：
    pytest tests/test_outline_polish.py -v --tb=short
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

import pytest

# 允许直接用 `python tests/test_outline_polish.py` 运行
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# 测试用的大纲样本
_SAMPLE_OUTLINE = """# 储能行业市场研究报告

## 1. 行业概述
- 储能技术发展现状
- 全球市场规模分析
- 主要参与者格局

## 2. 市场需求
- 电力系统需求
- 可再生能源配套需求
- 分布式能源需求

## 3. 技术路线
- 锂电池储能
- 钠离子电池
- 液流电池技术

## 4. 政策环境
- 国家政策导向
- 地方补贴政策
- 行业标准规范

## 5. 发展趋势
- 技术降本趋势
- 市场规模化趋势
- 商业模式创新趋势
"""


class TestOutlinePolishService:
    """大纲润色服务单元测试（真实 LLM 调用）。"""

    @pytest.fixture
    def setup_services(self, api_client):
        """设置服务实例（使用 conftest 中的 api_client 初始化数据库）。"""
        from src.shared.config import get_settings
        from src.shared.db import make_session_factory, make_engine
        from pathlib import Path

        settings = get_settings()
        db_path = Path(settings.sqlite_path)
        engine = make_engine(db_path)
        session_factory = make_session_factory(engine)
        db = session_factory()

        from src.application.repositories.prompt_repository import PromptRepository
        from src.application.repositories.llm_call_site_repository import LLMCallSiteRepository
        from src.application.repositories.llm_provider_repository import LLMProviderRepository
        from src.application.services.llm_runtime_service import LLMRuntimeService
        from src.application.services.outline_polish.outline_polish_service import (
            OutlinePolishService,
        )

        provider_repo = LLMProviderRepository(db)
        prompt_repo = PromptRepository(db)
        callsite_repo = LLMCallSiteRepository(db)
        llm_runtime = LLMRuntimeService(
            provider_repository=provider_repo,
            capability_repository=None,
            callsite_repository=callsite_repo,
            prompt_repository=prompt_repo,
        )
        service = OutlinePolishService(
            prompt_service=prompt_repo,
            llm_call_site_repository=callsite_repo,
            llm_runtime_service=llm_runtime,
        )

        yield service, db

        db.close()

    def test_polish_outline_success(self, setup_services):
        """测试大纲润色成功场景（使用真实 LLM）。"""
        service, db = setup_services

        from src.application.services.outline_polish.schema import OutlinePolishInput
        import asyncio

        # 构建输入
        input_data = OutlinePolishInput(
            outline=_SAMPLE_OUTLINE,
            industry="储能行业",
            report_type="市场研究报告",
            language="中文",
            style="专业、客观、数据驱动",
        )

        # 执行润色（真实 LLM 调用）
        result = asyncio.run(service.polish_outline(input_data))

        # 验证结果
        assert result.success is True, f"润色失败: {result.error}"
        assert result.output is not None, "润色结果为空"
        assert result.output.polished_outline, "润色后大纲为空"
        assert isinstance(result.output.changes_summary, list), "修改摘要应为列表"
        assert result.output.structure_integrity is True, "结构完整性检查未通过"
        assert result.output.core_keywords_preserved is True, "核心关键词未保留"

    def test_polish_outline_with_minimal_input(self, setup_services):
        """测试大纲润色最小输入场景。"""
        service, db = setup_services

        from src.application.services.outline_polish.schema import OutlinePolishInput
        import asyncio

        # 最小输入（只提供大纲内容）
        minimal_outline = """# 行业报告
## 概述
- 市场现状
## 趋势
- 发展方向
"""
        input_data = OutlinePolishInput(outline=minimal_outline)

        result = asyncio.run(service.polish_outline(input_data))

        # 最小输入也应能处理
        assert result.success is True, f"最小输入润色失败: {result.error}"
        assert result.output is not None


class TestOutlineAPIRoutes:
    """大纲 API 端点集成测试。"""

    @pytest.fixture
    def client(self, api_client):
        """使用 conftest 中的 api_client fixture（同步 TestClient）。"""
        return api_client

    def test_polish_outline_api_success(self, client):
        """测试 POST /v1/outline/polish 端点成功场景。"""
        response = client.post(
            "/v1/outline/polish",
            json={
                "outline": _SAMPLE_OUTLINE,
            },
        )

        assert response.status_code == 200, f"请求失败: {response.text}"
        data = response.json()
        assert "polished_outline" in data, "响应中缺少 polished_outline 字段"
        assert data["polished_outline"], "润色后大纲为空"
        # 验证返回的大纲仍然包含章节结构
        assert "#" in data["polished_outline"], "润色后大纲缺少 Markdown 格式"

    def test_polish_outline_api_with_options(self, client):
        """测试带选项的大纲润色 API。"""
        response = client.post(
            "/v1/outline/polish",
            json={
                "outline": _SAMPLE_OUTLINE,
                "industry": "新能源汽车行业",
                "report_type": "技术分析报告",
                "language": "中文",
                "style": "技术性、专业",
            },
        )

        assert response.status_code == 200, f"请求失败: {response.text}"
        data = response.json()
        assert "polished_outline" in data

    def test_polish_outline_api_invalid_input(self, client):
        """测试大纲润色 API 无效输入。"""
        # 空大纲
        response = client.post(
            "/v1/outline/polish",
            json={"outline": ""},
        )
        assert response.status_code == 422, "空大纲应返回 422 错误"

        # 缺少 outline 字段
        response = client.post(
            "/v1/outline/polish",
            json={},
        )
        assert response.status_code == 422, "缺少 outline 字段应返回 422 错误"

    def test_save_outline_api_success(self, client):
        """测试 POST /v1/outline/save 端点成功场景。"""
        test_filename = f"test_outline_{uuid4().hex[:8]}"
        response = client.post(
            "/v1/outline/save",
            json={
                "outline": _SAMPLE_OUTLINE,
                "filename": test_filename,
            },
        )

        assert response.status_code == 200, f"保存失败: {response.text}"
        data = response.json()
        assert "file_path" in data, "响应中缺少 file_path 字段"

        # 验证文件路径格式
        file_path = data["file_path"]
        assert "data/Templates/drafts" in file_path, f"文件路径不符合预期: {file_path}"
        assert file_path.endswith(".md"), f"文件应为 .md 格式: {file_path}"

        # 验证文件内容
        saved_path = Path(file_path)
        assert saved_path.exists(), f"文件未落盘: {file_path}"
        saved_content = saved_path.read_text(encoding="utf-8")
        assert saved_content == _SAMPLE_OUTLINE, "保存的文件内容与原始内容不一致"

        # 清理测试文件
        saved_path.unlink(missing_ok=True)

    def test_save_outline_api_filename_sanitization(self, client):
        """测试大纲保存文件名安全处理。"""
        # 测试包含特殊字符的文件名
        test_filename = "test outline (1).md"
        response = client.post(
            "/v1/outline/save",
            json={
                "outline": _SAMPLE_OUTLINE,
                "filename": test_filename,
            },
        )

        assert response.status_code == 200, f"保存失败: {response.text}"
        data = response.json()
        file_path = data["file_path"]

        # 验证特殊字符已被替换
        assert "(" not in file_path and ")" not in file_path, f"特殊字符未被处理: {file_path}"
        assert " " not in file_path, f"空格未被处理: {file_path}"

        # 清理测试文件
        saved_path = Path(file_path)
        saved_path.unlink(missing_ok=True)

    def test_save_outline_api_file_verification(self, client):
        """测试大纲保存文件钜真验证（使用临时目录）。"""
        test_filename = f"verify_test_{uuid4().hex[:8]}"
        response = client.post(
            "/v1/outline/save",
            json={
                "outline": _SAMPLE_OUTLINE,
                "filename": test_filename,
            },
        )

        assert response.status_code == 200
        data = response.json()
        file_path = data["file_path"]

        # 验证路径格式符合预期
        assert file_path.endswith(f"{test_filename}.md"), "文件名格式不正确"

        # 清理测试文件
        saved_path = Path(file_path)
        saved_path.unlink(missing_ok=True)

    def test_save_outline_api_invalid_filename(self, client):
        """测试大纲保存 API 无效文件名。"""
        # 空文件名
        response = client.post(
            "/v1/outline/save",
            json={
                "outline": _SAMPLE_OUTLINE,
                "filename": "",
            },
        )
        assert response.status_code == 422, "空文件名应返回 422 错误"

    def test_polish_and_save_workflow(self, client):
        """测试大纲润色+保存完整工作流。"""
        # Step 1: 润色大纲
        polish_response = client.post(
            "/v1/outline/polish",
            json={"outline": _SAMPLE_OUTLINE},
        )
        assert polish_response.status_code == 200
        polished = polish_response.json()["polished_outline"]

        # Step 2: 保存润色后的大纲
        save_filename = f"polished_{uuid4().hex[:8]}"
        save_response = client.post(
            "/v1/outline/save",
            json={
                "outline": polished,
                "filename": save_filename,
            },
        )
        assert save_response.status_code == 200
        file_path = save_response.json()["file_path"]

        # Step 3: 验证保存的文件
        saved_path = Path(file_path)
        assert saved_path.exists(), "保存的文件不存在"
        saved_content = saved_path.read_text(encoding="utf-8")
        assert saved_content == polished, "保存的内容与润色后内容不一致"

        # Step 4: 清理
        saved_path.unlink(missing_ok=True)


def run_t064_smoke_test():
    """T064 一键冒烟测试（润色 + 保存）。"""
    import json

    os.environ["LUMO_API_MODE"] = "full"

    print("=" * 60)
    print("T064 大纲润色和保存冒烟测试")
    print("=" * 60)

    from src.interfaces.api.app import create_app
    from fastapi.testclient import TestClient

    app = create_app()
    client = TestClient(app)

    # 1. 大纲润色测试
    print("\n[1/2] 测试大纲润色...")
    polish_resp = client.post(
        "/v1/outline/polish",
        json={"outline": _SAMPLE_OUTLINE},
    )
    assert polish_resp.status_code == 200, f"润色失败: {polish_resp.text}"
    polished = polish_resp.json()["polished_outline"]
    print(f"    润色成功，大纲长度: {len(polished)} 字符")

    # 2. 保存测试
    print("\n[2/2] 测试保存大纲...")
    test_filename = f"t064_smoke_{uuid4().hex[:8]}"
    save_resp = client.post(
        "/v1/outline/save",
        json={"outline": polished, "filename": test_filename},
    )
    assert save_resp.status_code == 200, f"保存失败: {save_resp.text}"
    file_path = save_resp.json()["file_path"]
    print(f"    保存成功: {file_path}")

    # 3. 验证
    saved_path = Path(file_path)
    if saved_path.exists():
        content = saved_path.read_text(encoding="utf-8")
        assert content == polished
        print("    验证通过")
        print(f"\n    测试文件保留路径: {file_path}")
        print("    请手动删除该文件以清理测试工件")

    print("\n" + "=" * 60)
    print("T064 冒烟测试通过！")
    print("=" * 60)

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="T064 大纲润色和保存测试")
    parser.add_argument("--smoke", action="store_true", help="运行冒烟测试")
    args = parser.parse_args()

    if args.smoke:
        run_t064_smoke_test()
    else:
        raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
