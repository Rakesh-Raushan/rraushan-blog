import re

def on_post_page(output, page, config, **kwargs):
    if not page.file.src_path.startswith("posts/"):
        return output

    page_id = f"rraushan.blog/{page.url}"
    badge = (
        '\n<ul class="md-post__meta md-nav__list">'
        '<li class="md-nav__item">'
        '<div class="md-nav__link">'
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">'
        '<path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 '
        '11-7.5c-1.73-4.39-6-7.5-11-7.5m0 12.5c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 '
        '2.24 5 5-2.24 5-5 5m0-8a3 3 0 0 0-3 3 3 3 0 0 0 3 3 3 3 0 0 0 3-3 3 3 '
        '0 0 0-3-3z"/>'
        '</svg>'
        f'<img src="https://visitor-badge.laobi.icu/?page_id={page_id}" '
        f'alt="Views" style="height:1em;vertical-align:middle;">'
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