import torch

# 创建一个示例张量
x = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9], 
                  [1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

# 沿着第一个维度（行）翻转张量
flipped_x = torch.flip(x, dims=[2])

print("原始张量:")
print(x)

print("\n翻转后的张量:")
print(flipped_x)