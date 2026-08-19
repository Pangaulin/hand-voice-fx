#!/usr/bin/env bash
# Lance Hand Vocal FX dans son environnement virtuel.
cd "$(dirname "$0")"
exec .venv/bin/python main.py "$@"