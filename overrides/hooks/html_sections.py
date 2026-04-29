"""
Builds section nav dynamically from files dropped into section dirs.

- .md files  → added to config['nav'] as normal MkDocs pages (theme-wrapped)
- .html files → added as Link objects in on_nav, served as raw static files
                (their own styling intact, no MkDocs wrapper)

Drop any file into docs/dsa/, docs/ml/, etc. — it appears automatically.
"""
import re
from pathlib import Path
from mkdocs.structure.nav import Link, Section

SECTIONS = [
    ("DSA", "dsa"),
    ("ML", "ml"),
    ("Maths", "maths"),
    ("System Design", "system-design"),
]


def _html_title(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"<title>(.*?)</title>", content, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return path.stem.replace("-", " ").replace("_", " ").title()


def _md_title(path: Path) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
        m = re.match(r"^---\s*\ntitle:\s*(.+)", content, re.IGNORECASE)
        if m:
            return m.group(1).strip().strip("'\"")
        m = re.search(r"^#\s+(.+)", content, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return path.stem.replace("-", " ").replace("_", " ").title()


def on_config(config, **kwargs):
    """Build nav with only .md files — HTML files are added as Links later."""
    docs_dir = Path(config["docs_dir"])

    nav = [{"Blog": "index.md"}]

    for title, folder in SECTIONS:
        section_path = docs_dir / folder
        if not section_path.exists():
            continue

        entries = [{"Overview": f"{folder}/index.md"}]

        for md_file in sorted(section_path.glob("*.md")):
            if md_file.name == "index.md":
                continue
            entries.append({_md_title(md_file): f"{folder}/{md_file.name}"})

        nav.append({title: entries})

    config["nav"] = nav
    return config


def on_nav(nav, config, files, **kwargs):
    """Append Link entries for .html files so they open as raw static pages."""
    docs_dir = Path(config["docs_dir"])

    section_map = {title: folder for title, folder in SECTIONS}

    for item in nav.items:
        if not isinstance(item, Section):
            continue
        folder = section_map.get(item.title)
        if not folder:
            continue

        section_path = docs_dir / folder
        for html_file in sorted(section_path.glob("*.html")):
            title = _html_title(html_file)
            # Absolute path — MkDocs copies this file as-is to site/folder/file.html
            url = f"/{folder}/{html_file.name}"
            item.children.append(Link(title=title, url=url))

    return nav
