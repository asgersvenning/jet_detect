#!/usr/bin/env bash
set -euo pipefail

## Get and set the correct working directory
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="jet_detect"
ENV_FILE="${REPO_DIR}/conda.yaml"

## Install micromamba
export MAMBA_ROOT_PREFIX="${HOME}/micromamba"
export PATH="${HOME}/.local/bin:${MAMBA_ROOT_PREFIX}/bin:${PATH}"

"${SHELL}" <(curl -Ls https://micro.mamba.pm/install.sh)

## Create micromamba environment
micromamba create -y -n "${ENV_NAME}" -f "${ENV_FILE}"
micromamba run -n "${ENV_NAME}" python -m pip install --upgrade pip
micromamba run -n "${ENV_NAME}" python -m pip install pyremotedata

## Get prediction configuration from user
read -rp "Input (-i): " INPUT
read -rp "Output directory (-o): " OUTPUT
read -rp "Weights (-w): " WEIGHTS
read -rp "Max images (-n, optional; empty for all): " N_MAX
read -rp "Extra args to pass (optional, e.g. --conf 0.2): " EXTRA

## Create output directory
mkdir -p "${OUTPUT}"

## Assemble prediction arguments
ARGS=(-i "${INPUT}" -o "${OUTPUT}" -w "${WEIGHTS}")
if [[ -n "${N_MAX}" ]]; then ARGS+=(-n "${N_MAX}"); fi
# shellcheck disable=SC2206
EXTRA_ARR=(${EXTRA})
ARGS+=("${EXTRA_ARR[@]}")

## Run prediction pipeline
cd "${REPO_DIR}"
micromamba run -n "${ENV_NAME}" python predict.py "${ARGS[@]}"
