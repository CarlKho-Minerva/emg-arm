#!/bin/bash
cd "$(dirname "$0")" && .venv/bin/python smile.py --calibrate; echo; echo "done; close this window"; read -n 1
