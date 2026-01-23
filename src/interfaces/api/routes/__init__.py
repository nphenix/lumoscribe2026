"""API routes package.

重要：本模块必须**无副作用**（不要在 import 时自动导入各路由）。

原因：
- 部分路由（如文件上传）依赖可选运行时依赖（python-multipart）。
- KB 独立部署模式（LUMO_API_MODE=kb_admin/kb_query）不应因无关路由导入而失败。

路由挂载请在 `src/interfaces/api/app.py` 中显式导入与 include。
"""

__all__: list[str] = []
