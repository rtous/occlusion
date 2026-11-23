#!/usr/bin/env bash
# Build paper/main.pdf (installs a basic TeX Live on a fresh Colab VM if needed, ~3 min).
set -e
cd "$(dirname "$0")"
if ! command -v latexmk >/dev/null || ! kpsewhich lmodern.sty >/dev/null; then
  apt-get -qq update
  DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended lmodern latexmk >/dev/null
fi
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null && echo "paper/main.pdf built"
