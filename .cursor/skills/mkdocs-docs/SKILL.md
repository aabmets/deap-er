---
name: mkdocs-docs
description: >-
  MkDocs Material and mkdocstrings rules for deap-er. Use when the user asks
  for documentation updates, when shipping a documented feature, or when
  README / docs / GitHub Pages work is in scope.
---

# deap-er docs

Do **not** edit `docs/` or `README.md` unless the user requested docs work or approved it as part of the task. Mention doc impact in the reply instead.

Exception: a DEAP bug, still-open DEAP issue, or new capability vs DEAP
must update the scoreboard via
[`deap-differences`](../deap-differences/SKILL.md). That edit is in
scope without asking.

This is **MkDocs Material + mkdocstrings**. Do not add Sphinx, RST, or Read the Docs.

## Stack

| Piece | Location |
|---|---|
| Config | `mkdocs.yml` |
| Source | `docs/` |
| Built site | `site/` (gitignored) |
| Publish | `.github/workflows/docs.yml` → GitHub Pages |
| Live URL | https://aabmets.github.io/deap-er/ |

Pages source must be **GitHub Actions**, not “Deploy from a branch → `/docs`”. `/docs` is Markdown source, not published HTML.

Preview:

```bash
uv run mkdocs build
uv run mkdocs serve
```

Do not commit `site/`.

## Page map

| Area | Path | How to edit |
|---|---|---|
| Overview | `docs/overview/` | Hand-written Markdown |
| Tutorials | `docs/tutorials/` | Hand-written Markdown; link to `../reference/…` |
| Examples | `docs/examples/` | Thin pages that include `examples/*.py` |
| API reference | `docs/reference/` | mkdocstrings stubs only |

Example include (pymdown snippets, `base_path: .`) — a fenced `python` block whose body is:

`--8<-- "examples/genetic_algorithms/onemax_detailed.py"`

Reference stub:

```markdown
# Operators

::: deap_er.operators
    options:
      show_submodules: true
```

Do not hand-copy API text into `docs/reference/`. Docstrings live on the Python objects.

Python docstrings stay **Google-style** (`Args:`, `Returns:`). `mkdocs.yml` sets `docstring_style: google`.

Admonitions use Material syntax (`!!! attention`, `!!! note`), not RST `.. attention::`. Collapsible equation blocks use `??? note "Equations"`.

Math uses **Arithmatex + KaTeX** (`pymdownx.arithmatex` with `generic: true` in `mkdocs.yml`, plus `docs/javascripts/katex.js`). Write inline math as `$...$` and display math as `$$` blocks. Do not use Sphinx `:math:` roles or `.. math::` directives. After `inspect.cleandoc`, `???` and `$$` must sit at column 0 so Material and Arithmatex can parse them.

## When code changes (docs in scope)

- Public API or docstring change → update the docstring on the object. Add a new `docs/reference/<pkg>.md` stub and a `mkdocs.yml` `nav` entry only for a **new** top-level package.
- New or renamed example script → update the matching `docs/examples/**/*.md` `--8<--` path and `nav` if you add a page.
- Tutorial / overview behavior change → edit the Markdown; keep links to reference pages.
- DEAP bug, DEAP issue, or new capability vs DEAP → follow
  [`deap-differences`](../deap-differences/SKILL.md) (inventory + both
  scoreboards). Required even when the user did not ask for docs.
- README only carries the docs URL plus the three-line Relative-to-DEAP
  scoreboard. Do not grow a second docs tree there.

New `.md` files under `docs/` must be listed in `mkdocs.yml` `nav` or they will be orphaned.
