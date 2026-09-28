#!/usr/bin/env bash

# =====================
# Expose .cursor/skills to Claude Code via .claude/skills
# =====================

CLAUDE_SKILLS_LINK="${PROJECT_DIR}/.claude/skills"

mkdir -p "${PROJECT_DIR}/.claude"
if [[ -L "$CLAUDE_SKILLS_LINK" || ! -e "$CLAUDE_SKILLS_LINK" ]]; then
    ln -sfn ../.cursor/skills "$CLAUDE_SKILLS_LINK"
else
    >&2 echo "WARNING: '${CLAUDE_SKILLS_LINK}' is not a symlink; skipped linking .cursor/skills."
fi
