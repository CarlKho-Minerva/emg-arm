#!/bin/bash
# Served With a Smile: start the smile detector from Terminal, which holds the camera grant.
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2; cd "$(dirname "$0")" && exec .venv/bin/python smile.py
