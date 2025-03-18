import numpy as np
import matplotlib.pyplot as plt
import itertools
import torch
# 指定要加载的模型文件的路径
model_path = 'MAML_TCN_231027-02.16_MODEL.pt'

# 使用torch.load加载模型
model_data = torch.load(model_path, map_location=torch.device('cpu'))

# 递归遍历模型数据并以[]和换行的格式打印
def print_data(data, indent=''):
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{indent}[{key}]")
            print_data(value, indent + '  ')
    elif isinstance(data, list):
        for item in data:
            print_data(item, indent)
    else:
        print(f"{indent}{data}")

# 打印模型数据
print_data(model_data)

