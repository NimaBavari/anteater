#!/bin/bash

if [ ! -d ".git" ]; then
    echo "Error: Not in a git repository."
    exit 1
fi

HOOKS_DIR=".git/hooks"

mkdir -p "$HOOKS_DIR"

rm -rf "$HOOKS_DIR"/* || { echo "Error: Failed to clear $HOOKS_DIR."; exit 1; }

PRE_COMMIT_FILE="$HOOKS_DIR/pre-commit"

cat > "$PRE_COMMIT_FILE" << 'EOF' || { echo "Error: Failed to write to $PRE_COMMIT_FILE."; exit 1; }
#!/bin/sh
isort -rc .
autoflake -r --in-place --remove-unused-variables .
black -l 120 .
flake8 --max-line-length 120 . --exclude .venv
mypy --disable-error-code import-not-found --explicit-package-bases .
rm -rf .mypy_cache
EOF

chmod +x "$PRE_COMMIT_FILE" || { echo "Error: Failed to make $PRE_COMMIT_FILE executable."; exit 1; }

echo "Success: pre-commit hook created."
exit 0
