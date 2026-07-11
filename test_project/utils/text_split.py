import sys
import os

print(sys.path)

# 获取项目根目录（假设当前脚本在 src/ 子目录下）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

print(sys.path)

import config
config.load_config()

def textsplit():
    print("我是文本分割工具")