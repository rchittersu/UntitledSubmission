#!/usr/bin/env bash
# Build the single handoff patch (secure env -> outside), see CLAUDE.md.
#
# Usage: handoff/make_patch.sh [output-file]
#   BASE=<ref> handoff/make_patch.sh   # override base (default: origin/main)
#
# The patch contains every commit on HEAD that is not in BASE. Checks that:
#   - there are commits to send and the working tree is clean
#   - only allowed repo paths are touched (no secure code/data)
#   - no binary files are included
#   - a report exists in handoff/from_secure/
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASE="${BASE:-origin/main}"
OUT="${1:-../handoff_$(date +%Y-%m-%d).patch}"

# Paths the patch may touch (regex on repo-relative paths).
ALLOWED='^(Template/|handoff/from_secure/|CLAUDE\.md$|\.gitignore$)'

fail() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

git rev-parse --verify -q "$BASE" >/dev/null || fail "base '$BASE' not found (set BASE=<ref>)"

n_ahead=$(git rev-list --count "$BASE..HEAD")
[ "$n_ahead" -gt 0 ] || fail "no commits on HEAD beyond $BASE; commit your report first"

n_behind=$(git rev-list --count "HEAD..$BASE")
[ "$n_behind" -eq 0 ] || warn "$BASE has $n_behind commit(s) not in HEAD; consider 'git rebase $BASE' first"

if [ -n "$(git status --porcelain)" ]; then
  warn "uncommitted changes are NOT included in the patch:"
  git status --short >&2
fi

changed=$(git diff --name-only "$BASE..HEAD")

bad=$(printf '%s\n' "$changed" | grep -Ev "$ALLOWED" || true)
[ -z "$bad" ] || fail "patch touches paths outside the allowed set:
$bad
(allowed: Template/, handoff/from_secure/, CLAUDE.md, .gitignore)"

binary=$(git diff --numstat "$BASE..HEAD" | awk -F'\t' '$1=="-" && $2=="-" {print $3}')
[ -z "$binary" ] || fail "binary files are not allowed (send plot data as text instead):
$binary"

reports=$(printf '%s\n' "$changed" | grep -E '^handoff/from_secure/.+\.md$' || true)
[ -n "$reports" ] || fail "no report in handoff/from_secure/*.md; every patch needs one"

git format-patch --stdout --base="$BASE" "$BASE..HEAD" > "$OUT"

author=$(git log -1 --format='%an <%ae>')
echo "Wrote $OUT"
echo "  commits: $n_ahead, files: $(printf '%s\n' "$changed" | wc -l | tr -d ' '), lines: $(wc -l < "$OUT" | tr -d ' ')"
echo "  reports:"; printf '    %s\n' $reports
echo "  git author in patch headers: $author  (edit if internal)"
