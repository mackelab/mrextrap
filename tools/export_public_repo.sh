#!/usr/bin/env bash
# Export the release tree into a NEW directory with a fresh, single-commit git history.
#
#   tools/export_public_repo.sh DEST [--author "Name <email>"]
#
# * copies only files tracked by git in this working tree (commit everything you want to publish
#   first; untracked and ignored files are never exported),
# * runs the sensitive-file guard (tests/test_no_sensitive_files.py) on the copy,
# * creates a new git repository in DEST with ONE initial commit on branch "main",
# * never adds a remote and never pushes; the commands to publish are printed at the end.
#
# DEST must not exist or must be an empty directory. PYTHON overrides the interpreter that has
# pytest (default: python3).
set -euo pipefail

usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-1}"; }

DEST=""
AUTHOR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --author) AUTHOR="${2:?--author needs \"Name <email>\"}"; shift 2 ;;
    -*) echo "unknown option: $1" >&2; usage 1 ;;
    *) [[ -z "$DEST" ]] || usage 1; DEST="$1"; shift ;;
  esac
done
[[ -n "$DEST" ]] || usage 1

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"

git -C "$SRC" rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "$SRC is not a git work tree" >&2; exit 1; }
if [[ -e "$DEST" ]] && [[ -n "$(ls -A "$DEST" 2>/dev/null)" ]]; then
  echo "DEST exists and is not empty: $DEST" >&2; exit 1
fi
mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd)"
case "$DEST" in "$SRC"|"$SRC"/*) echo "DEST must be outside the source tree" >&2; exit 1 ;; esac

if [[ -n "$(git -C "$SRC" status --porcelain --untracked-files=no)" ]]; then
  echo "warning: tracked files have uncommitted changes; the export contains the working-tree" \
       "versions of tracked files" >&2
fi
if [[ -n "$(git -C "$SRC" ls-files --others --exclude-standard)" ]]; then
  echo "warning: untracked files are NOT exported (git add them first if they belong in the release):" >&2
  git -C "$SRC" ls-files --others --exclude-standard | sed 's/^/  /' >&2
fi

echo "[export] copying tracked files from $SRC to $DEST"
git -C "$SRC" ls-files -z | (cd "$SRC" && tar --null -T - -cf -) | tar -xf - -C "$DEST"

cd "$DEST"
git init -q -b main
git add -A

echo "[export] running the sensitive-file guard on the copy"
if ! "$PYTHON" -m pytest -q -p no:cacheprovider --noconftest tests/test_no_sensitive_files.py; then
  echo "[export] guard FAILED; nothing was committed. Fix the offending files in the source tree" \
       "and export again into a fresh directory (this one is left for inspection)." >&2
  exit 1
fi

if [[ -z "$(git config user.name || true)" ]]; then
  export GIT_COMMITTER_NAME="${AUTHOR%% <*}" GIT_COMMITTER_EMAIL="${AUTHOR##*<}"
  GIT_COMMITTER_EMAIL="${GIT_COMMITTER_EMAIL%>}"
  [[ -n "$AUTHOR" ]] || { echo "no git identity configured: pass --author \"Name <email>\"" >&2; exit 1; }
fi
git commit -q ${AUTHOR:+--author "$AUTHOR"} -m "Initial public release of MRExtrap" \
  -m "Code release accompanying the MRExtrap preprint. Development history is not included."

echo
echo "[export] done: $(git rev-list --count HEAD) commit, $(git ls-files | wc -l | tr -d ' ') files in $DEST"
cat <<MSG

Nothing was pushed. To publish (after checking the export by hand):

  cd $DEST
  git remote add origin git@github.com:mackelab/mrextrap.git
  git push -u origin main

MSG
