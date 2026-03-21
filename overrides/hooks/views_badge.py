import re

def on_post_page(output, page, config, **kwargs):
    if not page.file.src_path.startswith("posts/"):
        return output

    page_id = f"rraushan.blog/{page.url}"
    badge = (
        '\n<ul class="md-post__meta md-nav__list">'
        '<li class="md-nav__item">'
        '<div class="md-nav__link" style="padding-left:0.6rem;">'
        f'<img src="https://visitor-badge.laobi.icu/badge?page_id={page_id}&label=views" alt="Views" style="height:1.3em;vertical-align:middle;">'
        '</div>'
        '</li>'
        '</ul>'
    )

    output = re.sub(
        r'(<ul class="md-post__meta md-nav__list">.*?</ul>\s*\n\s*</nav>)',
        r'\1' + badge,
        output,
        count=1,
        flags=re.DOTALL
    )
    return output