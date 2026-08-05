from __future__ import annotations
from torch import nn

class TemporalChangeHead(nn.Module):
    def __init__(self,channels): super().__init__(); self.change=nn.Conv2d(channels,1,1); self.new=nn.Conv2d(channels,1,1); self.demolition=nn.Conv2d(channels,1,1)
    def forward(self,features):
        # features [B,T,C,H,W]; output covers adjacent pairs only.
        d=(features[:,1:]-features[:,:-1]).abs(); b,t,c,h,w=d.shape; z=d.reshape(b*t,c,h,w)
        return {"dense_change_probability":self.change(z).sigmoid().reshape(b,t,h,w),"new_building_probability":self.new(z).sigmoid().reshape(b,t,h,w),"demolition_probability":self.demolition(z).sigmoid().reshape(b,t,h,w)}
