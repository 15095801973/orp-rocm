

import torch


feature = torch.arange(20.).view(1, 1, 4, 5)
print(feature[0][0])
## output:



h = torch.linspace(-1, 1, 4)+1e-6
w = torch.linspace(-1, 1, 5)+1e-6
h_c, w_c = torch.meshgrid(h, w)
coord = torch.stack([h_c, w_c]) # 2,5,6 坐标通道，行数，列数
coord_grid = coord.permute(1, 2, 0).unsqueeze(0).expand(1, h.shape[0], w.shape[0], 2).flip(-1) # 变成batch,行数,列数,坐标通道
# coord_grid[:,:,:,0]=coord_grid[:,:,:,0]+0.1 # 让x分量都+0.1，即让所有的采样点都往右偏移一下下
print(coord_grid)

p = torch.tensor([1.,1.]).type_as(feature).view(1,1,1,2)

## output:
# res = torch.nn.functional.grid_sample(feature, coord_grid, mode='nearest', align_corners=False, padding_mode='zeros') # 得到N,C,H,W
res2 = torch.nn.functional.grid_sample(feature, p, mode="bilinear", align_corners=True, padding_mode='zeros') # 得到N,C,H,W
res3 = torch.nn.functional.grid_sample(feature, coord_grid, mode="bilinear", align_corners=True, padding_mode='zeros') # 得到N,C,H,W
print(res3)
print(res2[0][0])