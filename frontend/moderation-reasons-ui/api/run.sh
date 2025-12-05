#!/usr/bin/env bash
export PYTHONUNBUFFERED=1
export MOCK=1
uvicorn src.ui_api:app --host 0.0.0.0 --port 8000 --reload
