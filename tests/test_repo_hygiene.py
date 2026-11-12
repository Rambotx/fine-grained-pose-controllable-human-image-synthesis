"""仓库卫生测试：防止把本机路径、隐私文件或死链提交进仓库。"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 允许出现在 legacy/ 里的历史本机路径
LOCAL_PATH_PATTERN = re.compile(r"[A-Za-z]:[\\/](?:Programs|Users|ComfyUI)")
# 只扫描代码/配置；README 里的示例路径属于文档，不算硬编码
SOURCE_SUFFIXES = {".py", ".html", ".yml", ".yaml"}


def _iter_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if rel.parts and rel.parts[0] in {"legacy", ".git", "assets"}:
            continue
        if path.suffix.lower() in SOURCE_SUFFIXES:
            yield path, rel


def test_no_hardcoded_local_paths():
    offenders = []
    for path, rel in _iter_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LOCAL_PATH_PATTERN.finditer(text):
            offenders.append(f"{rel}: {match.group(0)}")
    assert not offenders, "发现硬编码的本机路径：" + "; ".join(offenders)


def test_readme_images_exist():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    links = re.findall(r'(?:!\[[^\]]*\]\(([^)]+)\)|<img[^>]+src="([^"]+)")', readme)
    missing = []
    for markdown_link, html_link in links:
        target = markdown_link or html_link
        if target.startswith(("http://", "https://", "#")):
            continue
        if not (ROOT / target).exists():
            missing.append(target)
    assert not missing, f"README 引用了不存在的文件: {missing}"


def test_sensitive_files_are_ignored():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    required = [
        "采购发票.pdf",
        "申报书.doc",
        "结题报告",
        "test_images/",
        "results/",
        ".env",
    ]
    missing = [item for item in required if item not in gitignore]
    assert not missing, f".gitignore 缺少隐私/大文件规则: {missing}"


def test_entry_points_are_importable_modules():
    for name in ("app.py", "server_api.py", "skeleton_extraction.py", "config.py", "models.py", "pose_utils.py"):
        assert (ROOT / name).is_file(), f"缺少入口文件: {name}"