#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="jet_detect"
ENV_FILE="${REPO_DIR}/conda.yaml"

export MAMBA_ROOT_PREFIX="$HOME/micromamba"
export PATH="$HOME/.local/bin:$MAMBA_ROOT_PREFIX/bin:$PATH"
export ULTRALYTICS_AGREE=True

"${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh)
micromamba create -y -n "$ENV_NAME" -f "$ENV_FILE"
micromamba run -n "$ENV_NAME" python -m pip install --upgrade pip
micromamba run -n "$ENV_NAME" python -m pip install pyremotedata

read -rp "PYREMOTEDATA_REMOTE_USERNAME: " PRD_USER
export PYREMOTEDATA_REMOTE_USERNAME="$PRD_USER"
export PYREMOTEDATA_REMOTE_URI="io.erda.au.dk"
export PYREMOTEDATA_REMOTE_DIRECTORY="/"
export PYREMOTEDATA_AUTO="yes"

read -rp "Input (-i): " INPUT
read -rp "Output directory (-o): " OUTPUT
read -rp "Weights (-w): " WEIGHTS
read -rp "Max images (-n, optional; empty for all): " N_MAX
read -rp "Extra args (optional, e.g. --conf 0.2): " EXTRA

mkdir -p "$OUTPUT"
ARGS=(-i "$INPUT" -o "$OUTPUT" -w "$WEIGHTS")
[[ -n "${N_MAX}" ]] && ARGS+=(-n "$N_MAX")
# shellcheck disable=SC2206
EXTRA_ARR=($EXTRA)
ARGS+=("${EXTRA_ARR[@]}")

cd "$REPO_DIR"
micromamba run -n "$ENV_NAME" env \
  PYREMOTEDATA_REMOTE_USERNAME="$PYREMOTEDATA_REMOTE_USERNAME" \
  PYREMOTEDATA_REMOTE_URI="$PYREMOTEDATA_REMOTE_URI" \
  PYREMOTEDATA_REMOTE_DIRECTORY="$PYREMOTEDATA_REMOTE_DIRECTORY" \
  PYREMOTEDATA_AUTO="$PYREMOTEDATA_AUTO" \
  python predict.py "${ARGS[@]}"
