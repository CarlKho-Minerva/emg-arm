#!/bin/bash
# Served With a Smile: the screen beside the arm, http://localhost:8790. Start it from Terminal (double-click or `open`):
# the Start buttons on the page launch the face reader from this process, and Terminal holds the camera grant.
export GLOG_minloglevel=2 TF_CPP_MIN_LOG_LEVEL=2; cd "$(dirname "$0")" && exec .venv/bin/python dash.py
