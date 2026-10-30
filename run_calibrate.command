#!/bin/bash
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2; cd "$(dirname "$0")" && .venv/bin/python smile.py --calibrate 2>/dev/null; exit
