#!/bin/bash

# 确保目录存在
mkdir -p uploads
mkdir -p audio

# 启动应用
python app.py --port=$PORT