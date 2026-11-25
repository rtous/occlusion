#!/usr/bin/env bash
# Build paper_v2/main.pdf (Springer Nature sn-jnl class). Installs a basic TeX Live on a fresh Colab VM if needed (~3 min).
set -e
cd "$(dirname "$0")"
if ! command -v pdflatex >/dev/null || ! kpsewhich lmodern.sty >/dev/null; then
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
    texlive-science lmodern >/dev/null
fi
(cd figures && pdflatex -interaction=nonstopmode -halt-on-error diagram.tex >/dev/null && rm -f diagram.aux diagram.log)
# latexmk stopped early on this class, so run the passes explicitly
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
bibtex main >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null
if grep -q "undefined" main.log; then grep "undefined" main.log; fi
echo "paper_v2/main.pdf built"
