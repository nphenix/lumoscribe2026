"""T093: MinerU Cleaning Test Script.

Test flow:
1. Load MinerU config from database
2. Upload test files to MinerU
3. Wait for processing to complete
4. Download result ZIP
5. Extract to target directory

Test data: data/sources/default/*.pdf
"""

from __future__ import annotations

import zipfile
from pathlib import Path
import sys

import pytest

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestMinerUIntegration:
    """MinerU integration tests."""

    @pytest.mark.asyncio
    async def test_mineru_service(self):
        """Test complete MinerU service flow."""
        import httpx
        import asyncio
        from src.application.services.mineru_service import MinerUClient
        from src.shared.logging import configure_logging
        from src.shared.db import make_session_factory, make_engine
        from src.shared.config import get_settings
        from src.application.repositories.llm_provider_repository import LLMProviderRepository

        # Enable structured logs for observability in this integration test
        configure_logging("INFO")

        # Step 1: Load config from database
        settings = get_settings()
        engine = make_engine(settings.sqlite_path)
        session_factory = make_session_factory(engine)
        with session_factory() as session:
            repo = LLMProviderRepository(session)
            provider = repo.get_by_key("mineru")

            assert provider is not None, "mineru provider not found in database"
            assert provider.enabled, "mineru provider is not enabled"
            assert provider.base_url, "mineru base_url not configured"

            print(f"Config loaded: {provider.base_url}")

        # Step 2: Create client (loads config from database)
        client = MinerUClient()
        assert client.config.base_url, "Failed to load base_url from database"

        # Step 3: Get test files
        sources_dir = PROJECT_ROOT / "data" / "sources" / "default"
        pdf_files = list(sources_dir.glob("*.pdf"))

        if not pdf_files:
            pytest.skip("No test files found")

        # Step 4: Upload all PDFs in one batch
        test_files = sorted(pdf_files, key=lambda p: p.name)
        print(f"Test files: {len(test_files)}")

        # Step 5: Apply for upload URL
        upload_result = await client.apply_upload_urls(
            # apply_upload_urls only needs filenames; bytes are ignored by client code
            files=[(p.name, b"") for p in test_files],
            data_ids=[f"test-{i:03d}" for i in range(len(test_files))],
        )

        batch_id = upload_result["batch_id"]
        file_urls = upload_result["file_urls"]

        assert batch_id, "No batch_id returned"
        assert file_urls, "No upload URLs returned"
        assert len(file_urls) == len(test_files), f"Upload URL count mismatch: {len(file_urls)} != {len(test_files)}"

        print(f"Task submitted: {batch_id}")
        print(f"Upload URLs: {len(file_urls)}")

        # Step 6: Upload files concurrently (bounded)
        # Bounded concurrency: reduce flakiness for large uploads
        semaphore = asyncio.Semaphore(2)

        async def upload_one(path: Path, url: str) -> None:
            async with semaphore:
                ok = await client.upload_file(path, url)
                assert ok, f"Upload failed: {path.name}"

        await asyncio.gather(*(upload_one(p, u) for p, u in zip(test_files, file_urls)))
        print("All files uploaded successfully")

        # Step 7: Wait for processing to complete
        print("Waiting for processing...")
        task_info = await client.wait_for_completion(
            batch_id,
            poll_interval=5,
            max_wait=1800,
            expected_item_count=len(test_files),
        )

        assert task_info.status.value == "completed", f"Processing failed: {task_info.error_msg}"
        print(f"Processing complete, progress: {task_info.progress}%")

        # Step 8: Download result ZIPs (one per file)
        result = task_info.result or {}
        file_results = result.get("file_results")
        if not isinstance(file_results, list):
            # 兼容：从 batch 原始结构解析
            batch = result.get("batch") if isinstance(result.get("batch"), dict) else {}
            items = batch.get("extract_result") or batch.get("extract_results") or []
            file_results = items if isinstance(items, list) else []

        assert len(file_results) == len(test_files), f"Batch item count mismatch: {len(file_results)} != {len(test_files)}"

        # Step 9: Download & extract each zip to its own folder
        output_dir = PROJECT_ROOT / "data" / "intermediates" / "test_mineru" / batch_id
        output_dir.mkdir(parents=True, exist_ok=True)

        download_jobs: list[tuple[str, str]] = []
        for fr in file_results:
            if not isinstance(fr, dict):
                continue
            file_name = fr.get("file_name") or fr.get("filename") or fr.get("name") or "unknown"
            url = fr.get("download_url") or fr.get("full_zip_url") or fr.get("zip_url")
            assert url, f"No download url for {file_name}"
            download_jobs.append((str(file_name), str(url)))

        semaphore_dl = asyncio.Semaphore(2)

        async def download_and_extract(file_name: str, url: str) -> None:
            async with semaphore_dl:
                print(f"Download: {file_name}")
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(client.config.upload_timeout),
                    follow_redirects=True,
                ) as http_client:
                    resp = await http_client.get(url)
                assert resp.status_code == 200, f"Download failed for {file_name}: {resp.status_code}"

                subdir = output_dir / Path(file_name).stem
                subdir.mkdir(parents=True, exist_ok=True)
                zip_path = subdir / f"{Path(file_name).stem}.zip"
                zip_path.write_bytes(resp.content)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(subdir)

        await asyncio.gather(*(download_and_extract(fn, url) for fn, url in download_jobs))
        extracted_files = list(output_dir.rglob("*"))
        print(f"Extracted files: {len(extracted_files)}")

        await client.close()
        print("Test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "--asyncio-mode=auto"])
