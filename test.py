import torch
from components.network import StyleEncoder, StyleDecoder

gan = StyleDecoder().cuda()
w = torch.zeros(1,512,4,4).cuda()
a = torch.zeros(14,1,512).cuda()
print(gan(w,a).cpu().data.shape)

encoder = StyleEncoder().cuda()
img = torch.zeros(1,3,256,256).cuda()
a = torch.zeros(14,1,512).cuda()
print(encoder(img,a).cpu().data.shape)
