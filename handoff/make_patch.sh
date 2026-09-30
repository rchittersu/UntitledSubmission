#!/usr/bin/env bash
# Build the single handoff patch (secure env -> outside), see CLAUDE.md.
#
# Usage: handoff/make_patch.sh [output-file]
#   BASE=<branch> handoff/make_patch.sh   # explicit base, e.g. BASE=paper-v2 (= origin/paper-v2)
#
# Base branch, if BASE is not given: the current branch's origin/* upstream, else the
# origin/* branch HEAD is closest to. The patch contains every commit on HEAD since the
# fork point from that branch. Checks that:
#   - there are commits to send and the working tree is clean
#   - only allowed repo paths are touched (no secure code/data)
#   - no binary files are included
#   - a report exists in handoff/from_secure/
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASE="${BASE:-}"
OUT="${1:-../handoff_$(date +%Y-%m-%d).patch}"

# Paths the patch may touch (regex on repo-relative paths).
ALLOWED='^(Template/|plan/|handoff/from_secure/|CLAUDE\.md$|\.gitignore$)'

fail() { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

if [ -n "$BASE" ]; then
  # "branch" means origin/branch when that exists; any other ref is used as is.
  git rev-parse --verify -q "origin/$BASE" >/dev/null && BASE="origin/$BASE"
  git rev-parse --verify -q "$BASE" >/dev/null || fail "base '$BASE' not found"
elif up=$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null) \
     && [ "${up#origin/}" != "$up" ]; then
  BASE="$up"
else
  # No upstream: pick the origin branch HEAD is closest to (fewest own commits).
  best=""
  for ref in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin/); do
    [ "$ref" = "origin/HEAD" ] || [ "$ref" = "origin" ] && continue
    mb=$(git merge-base "$ref" HEAD 2>/dev/null) || continue
    n=$(git rev-list --count "$mb..HEAD")
    if [ -z "$best" ] || [ "$n" -lt "$best" ]; then best=$n; BASE=$ref; fi
  done
  [ -n "$BASE" ] || fail "no origin/* branch shares history with HEAD (set BASE=<ref>)"
fi

# Diff against the fork point, so a base branch that moved on is harmless.
BASE_COMMIT=$(git merge-base "$BASE" HEAD) || fail "HEAD shares no history with $BASE"
echo "Base: $BASE @ $(git rev-parse --short "$BASE_COMMIT")"

n_ahead=$(git rev-list --count "$BASE_COMMIT..HEAD")
[ "$n_ahead" -gt 0 ] || fail "no commits on HEAD beyond $BASE; commit your report first"

n_behind=$(git rev-list --count "HEAD..$BASE")
[ "$n_behind" -eq 0 ] || warn "$BASE has $n_behind newer commit(s); patch is still based on the fork point (rebase onto $BASE if you want them included)"

if [ -n "$(git status --porcelain)" ]; then
  warn "uncommitted changes are NOT included in the patch:"
  git status --short >&2
fi

changed=$(git diff --name-only "$BASE_COMMIT..HEAD")

bad=$(printf '%s\n' "$changed" | grep -Ev "$ALLOWED" || true)
[ -z "$bad" ] || fail "patch touches paths outside the allowed set:
$bad
(allowed: Template/, plan/, handoff/from_secure/, CLAUDE.md, .gitignore)"

binary=$(git diff --numstat "$BASE_COMMIT..HEAD" | awk -F'\t' '$1=="-" && $2=="-" {print $3}')
[ -z "$binary" ] || fail "binary files are not allowed (send plot data as text instead):
$binary"

reports=$(printf '%s\n' "$changed" | grep -E '^handoff/from_secure/.+\.md$' || true)
[ -n "$reports" ] || fail "no report in handoff/from_secure/*.md; every patch needs one"

git format-patch --stdout --base="$BASE_COMMIT" "$BASE_COMMIT..HEAD" > "$OUT"

author=$(git log -1 --format='%an <%ae>')
echo "Wrote $OUT  (base branch: $BASE)"
echo "  commits: $n_ahead, files: $(printf '%s\n' "$changed" | wc -l | tr -d ' '), lines: $(wc -l < "$OUT" | tr -d ' ')"
echo "  reports:"; printf '    %s\n' $reports
echo "  git author in patch headers: $author  (edit if internal)"
