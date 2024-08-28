import torch

# 创建一个示例张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])

# 对整个张量进行求和
sum_all = torch.sum(x)
print("整个张量的和:", sum_all.item())

# 沿着指定维度进行求和
sum_dim0 = torch.sum(x, dim=0)  # 沿着第0维度求和
sum_dim1 = torch.sum(x, dim=1)  # 沿着第1维度求和

print("沿着第0维度的和:", sum_dim0)
print("沿着第1维度的和:", sum_dim1)