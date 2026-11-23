"""Build a linked inventory of versioned source, configurations and documentation."""
from __future__ import annotations

import argparse
import ast
from itertools import groupby
from pathlib import Path
import subprocess
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "CODE_INDEX.md"
ROOTS = {
    "applications": "应用交付入口和合成测试",
    "configs": "运行、训练、标签和注册配置",
    "docs": "模块设计、数据准入和历史审计",
    "experiments": "实验定义",
    "extensions": "扩展模块",
    "external": "第三方适配说明",
    "reports": "历史验收报告",
    "research": "研究与案例脚本",
    "scripts": "平台调度和操作命令",
    "src": "核心包和可复用算法",
    "tests": "单元和集成测试",
    "tools": "诊断、数据准备、执行和回读工具",
    "web": "交互审阅与静态预览",
    "work": "R3/R4 应用依赖入口",
}
EXTENSIONS = {
    ".py", ".ps1", ".sh", ".cmd", ".bat", ".md", ".js", ".mjs", ".ts",
    ".tsx", ".jsx", ".html", ".css", ".json", ".yaml", ".yml", ".toml",
}
DOCUMENTS = {"pyproject.toml", "README.md", "CHANGELOG.md", "run.py"}


def source_paths():
    result = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
    )
    for name in sorted(set(result.decode("utf-8").split("\0"))):
        if not name:
            continue
        path = Path(name)
        if (
            name != "docs/CODE_INDEX.md"
            and (ROOT / path).is_file()
            and path.suffix.lower() in EXTENSIONS
            and (path.parts[0] in ROOTS or path.name in DOCUMENTS)
        ):
            yield path


def describe(path):
    stem = path.stem.replace("_", " ").replace("-", " ").strip()
    if path.name == "__init__.py":
        stem = f"{path.parent.name} 包入口"
    if path.suffix != ".py":
        return stem
    try:
        tree = ast.parse((ROOT / path).read_bytes(), filename=path.as_posix())
    except SyntaxError:
        return stem
    doc = ast.get_docstring(tree)
    if doc:
        return doc.splitlines()[0][:150].rstrip()
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not node.name.startswith("_")
    ]
    return stem + (" · " + ", ".join(definitions[:4]) if definitions else "")


def render():
    lines = [
        "# 文件索引",
        "",
        "按仓库路径列出源码、配置和文档。职责来自源码说明或函数与类名。"
        "应用入口、数据与版本状态见[模块状态](MODULE_STATUS.md)。",
        "",
        "索引由 tools/maintenance/build_code_index.py 生成。",
        "",
    ]
    for root, paths in groupby(source_paths(), key=lambda p: p.parts[0] if len(p.parts) > 1 else "根目录"):
        lines += [f"## {root}", ""]
        if root in ROOTS:
            lines += [ROOTS[root], ""]
        for parent, files in groupby(paths, key=lambda p: p.parent):
            if parent != Path(".") and parent.as_posix() != root:
                lines += [f"### {parent.as_posix()}", ""]
            for path in files:
                link = quote(path.as_posix())
                label = path.name.replace("[", "\\[").replace("]", "\\]")
                lines.append(f"- [{label}](../{link}) — {describe(path)}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the committed index.")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != content:
            parser.exit(1, "Code index is stale. Run tools/maintenance/build_code_index.py.\n")
    else:
        OUTPUT.write_text(content, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
