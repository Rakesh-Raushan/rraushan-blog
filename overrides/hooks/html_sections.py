"""
Builds section nav dynamically from files dropped into section dirs.

- .md files  → added to config['nav'] as normal MkDocs pages (theme-wrapped)
- .html files → added as Link objects in on_nav, served as raw static files
                (their own styling intact, no MkDocs wrapper)
- on_post_build injects <link rel="stylesheet" href="/assets/theme.css"> into
  every HTML note so colours/dark-mode are controlled from one central file.

Drop any file into docs/dsa/, docs/ml/, etc. — it appears automatically.
"""
import re
from pathlib import Path
from mkdocs.structure.nav import Link, Section


def _natural_key(path: Path):
    """Sort key that orders 2- before 10- before 100- (not lexicographic)."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", path.name)]

SECTIONS = [
    ("DSA", "dsa"),
    ("ML", "ml"),
    ("Maths", "maths"),
    ("System Design", "system-design"),
    ("Guides", "guides"),
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

        for md_file in sorted(section_path.glob("*.md"), key=_natural_key):
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
        for html_file in sorted(section_path.glob("*.html"), key=_natural_key):
            title = _html_title(html_file)
            # Absolute path — MkDocs copies this file as-is to site/folder/file.html
            url = f"/{folder}/{html_file.name}"
            item.children.append(Link(title=title, url=url))

    # Portfolio is a standalone page — insert after Blog (index 0)
    nav.items.insert(1, Link(title="Portfolio", url="/portfolio/"))

    return nav


def on_post_build(config, **kwargs):
    """
    Inject <link rel="stylesheet" href="/assets/theme.css"> into every HTML
    note in the built site output, just before </head>.

    Runs after MkDocs finishes copying files so source files are never touched.
    The injected link comes after each file's own <style> block, so theme.css
    variables override any locally defined ones via normal CSS cascade.
    """
    site_dir = Path(config["site_dir"])
    link_tag = '  <link rel="stylesheet" href="/assets/theme.css">\n'

    for _, folder in SECTIONS:
        section_path = site_dir / folder
        if not section_path.exists():
            continue

        for html_file in sorted(section_path.glob("*.html"), key=_natural_key):
            if html_file.name == "index.html":
                continue

            content = html_file.read_text(encoding="utf-8", errors="ignore")

            # Idempotent — skip if already injected
            if "/assets/theme.css" in content:
                continue

            content = content.replace("</head>", f"{link_tag}</head>", 1)
            html_file.write_text(content, encoding="utf-8")
