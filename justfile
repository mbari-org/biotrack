#!/usr/bin/env just --justfile

# List recipes
list:
    @just --list --unsorted

# Install dependencies
setup:
    poetry install

# Check source formatting
format-check:
    poetry run ruff format --check

format:
    poetry run ruff format

# Lint source code
lint:
    poetry run ruff check --fix

# Check linting of source code
lint-check:
    poetry run ruff check

# Export poetry dependencies to requirements.txt
export-requirements:
    poetry export -f requirements.txt --output requirements.txt --without-hashes

# Run an example video
track-video:
    export PYTHONPATH=. && poetry run python examples/video.py