// Force full-page navigation for raw .html note files so Material's
// instant loading doesn't intercept and blank out the content.
document$.subscribe(function () {
  document.querySelectorAll(".md-nav__link").forEach(function (link) {
    var href = link.getAttribute("href");
    if (href && href.endsWith(".html")) {
      link.addEventListener(
        "click",
        function (e) {
          e.stopImmediatePropagation();
          e.preventDefault();
          window.location.href = this.href;
        },
        true
      );
    }
  });
});
