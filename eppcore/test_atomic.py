import torch
import eppcoreops

bz = 2
h = 375
w = 1242
instance = torch.ones([bz, 1, h, w], dtype=torch.int32, device=torch.device(type='cuda', index=0))
compdst = torch.zeros([bz, 200, 1, 1], dtype=torch.float32, device=torch.device(type='cuda', index=0))
compsrc = torch.zeros([bz, h, w, 1, 1], dtype=torch.float32, device=torch.device(type='cuda', index=0))
eppcoreops.epp_compression(instance, compdst, compsrc * 1.1, h, w, bz, 1, 1)

print(compdst[0,0,0,0], compdst[1,0,0,0])