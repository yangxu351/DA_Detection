# from grf import revgrad
# import torch

# alpha = torch.tensor([1.])

# x = torch.tensor([4.], requires_grad=True)
# y = torch.pow(x,2) 
# y = y+6
# y.backward()

# x_rev = torch.tensor([4.], requires_grad=True)
# y_rev = torch.pow(x_rev,2) 
# y_rev = revgrad(y_rev, alpha)
# y_rev = y_rev + 6

# y_rev.backward()

# print(f'x gradient {x.grad}')
# print(f'revsered x gradient {x_rev.grad}')

__sets={}
import math
__sets['nwpu'] = (lambda x=10, y=8: math.pow(x,2) + y)
print(__sets['nwpu']())