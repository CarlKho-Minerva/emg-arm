#!/bin/bash
# Served With a Smile: start the smile detector from Terminal, which holds the camera grant.
# Output also goes to data/smile.out so the error is readable when the window closes.
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2; cd "$(dirname "$0")" && .venv/bin/python smile.py 2>&1 | tee data/smile.out
echo "smile.py ended; this window stays open. Read data/smile.out."; read -r
