#!/usr/bin/env bash

set -euo pipefail

REPO_SLUG="${SRCLIGHT_REPO_SLUG:-Matrix-aas/srclight}"
REF="${SRCLIGHT_REF:-main}"
PACKAGE_SPEC="git+https://github.com/${REPO_SLUG}.git@${REF}"

say() {
  printf '%s\n' "$*"
}

fail() {
  printf '\n%b\n' "$*"
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required command: $1"
}

install_pipx() {
  require_cmd python3
  say "pipx not found. Installing it now so we can do this like civilized people."
  python3 -m pip install --user pipx
  python3 -m pipx ensurepath >/dev/null 2>&1 || true

  if command -v pipx >/dev/null 2>&1; then
    return
  fi

  local user_base
  user_base="$(python3 -m site --user-base)"
  export PATH="${user_base}/bin:${PATH}"

  command -v pipx >/dev/null 2>&1 || fail \
    "pipx installed, but it is not on PATH yet. Restart your shell and rerun this installer."
}

existing_srclight_version() {
  if ! command -v srclight >/dev/null 2>&1; then
    return 0
  fi

  srclight --version 2>/dev/null | awk '{print $NF}' | tail -n1
}

pipx_package_spec() {
  if ! command -v pipx >/dev/null 2>&1; then
    return 0
  fi

  local pipx_json
  pipx_json="$(pipx list --json 2>/dev/null || true)"

  python3 -c '
import json
import sys

try:
    data = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)

venv = (data.get("venvs") or {}).get("srclight") or {}
meta = (venv.get("metadata") or {}).get("main_package") or {}
spec = meta.get("package_or_url") or ""
if spec:
    print(spec)
' <<<"${pipx_json}"
}

require_cmd python3

old_version="$(existing_srclight_version || true)"
if [[ "${old_version}" =~ ^0\.15\. ]]; then
  fail "You still have the old busted srclight (${old_version}) on this machine.\n\nRun this first:\n  pipx uninstall srclight\n\nThen rerun the installer and let the better timeline begin."
fi

if ! command -v pipx >/dev/null 2>&1; then
  install_pipx
fi

installed_spec="$(pipx_package_spec || true)"
if [[ -n "${installed_spec}" && "${installed_spec}" != *"${REPO_SLUG}"* ]]; then
  fail "Found an existing pipx install for srclight:\n  ${installed_spec}\n\nThat is not this fork. Uninstall that old shit first:\n  pipx uninstall srclight\n\nThen rerun:\n  curl -fsSL https://raw.githubusercontent.com/${REPO_SLUG}/main/scripts/install.sh | bash"
fi

say "Installing srclight from ${REPO_SLUG}@${REF} ..."
pipx install --force "${PACKAGE_SPEC}"

say ""
say "srclight is installed."
say "Try:"
say "  srclight --version"
say "  srclight index"
