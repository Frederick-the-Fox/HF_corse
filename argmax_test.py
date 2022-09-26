import torch
import numpy as np 

a = np.array([[[6, 7, 8], [9, 10, 11]], [[0, 1, 2], [3, 4, 5]]])
tensor_a = torch.from_numpy(a)

print(type(tensor_a))
print(tensor_a.shape)

print(tensor_a.__dir__())

result = torch.argmax(tensor_a, dim = 0)

print(result)