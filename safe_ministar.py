#AI编写，非真人
import torch
from safetensors.torch import save_file

# 加载 .pth 权重（state_dict）
state_dict = torch.load("ministar.pth", map_location="cpu")

# 保存为 .safetensors 格式
save_file(state_dict, "model.safetensors")

print("转换成功！已生成 model.safetensors")