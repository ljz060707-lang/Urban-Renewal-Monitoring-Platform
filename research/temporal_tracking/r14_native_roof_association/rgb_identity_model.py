import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18,ResNet18_Weights
class RoofAppearanceEncoder(nn.Module):
 def __init__(self):
  super().__init__();self.rgb=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);self.rgb.fc=nn.Identity();self.project=nn.Sequential(nn.Linear(512,128),nn.ReLU(),nn.Linear(128,64))
  for p in self.rgb.parameters():p.requires_grad_(False)
  for p in self.rgb.layer4.parameters():p.requires_grad_(True)
  self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None]);self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
 def train(self,mode=True):
  super().train(mode)
  for m in self.rgb.modules():
   if isinstance(m,nn.BatchNorm2d):m.eval()
  return self
 def forward(self,rgba,pretrained=False,augment=False):
  x=rgba[:,:3].float()/255;mask=rgba[:,3:4].float()/255
  if augment:
   brightness=torch.rand((len(x),1,1,1),device=x.device)*.4+.8;channel=torch.rand((len(x),3,1,1),device=x.device)*.2+.9;x=(x*brightness*channel).clamp(0,1)
  x=x*(.25+.75*mask);z=self.rgb((x-self.mean)/self.std)
  return F.normalize(z.float() if pretrained else self.project(z).float(),dim=1)
