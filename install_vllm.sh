#!/bin/bash
python3 -m venv vllm_env
./vllm_env/bin/pip install --no-cache-dir vllm > vllm_install.log 2>&1
