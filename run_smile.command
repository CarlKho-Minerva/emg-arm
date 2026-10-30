#!/bin/bash
# Served With a Smile: start the smile detector from Terminal, which holds the camera grant.
cd "$(dirname "$0")" && exec .venv/bin/python smile.py
