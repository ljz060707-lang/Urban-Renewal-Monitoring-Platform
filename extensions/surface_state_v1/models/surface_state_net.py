from __future__ import annotations
from torch import nn

class _Conv(nn.Sequential):
    def __init__(self,a,b): super().__init__(nn.Conv2d(a,b,3,padding=1),nn.ReLU(True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(True))

class SurfaceStateNet(nn.Module):
    """Small shared encoder with separate native-label decoders."""
    def __init__(self,fusu_classes=17,cwld_classes=4,width=24):
        super().__init__(); self.encoder=nn.Sequential(_Conv(3,width),nn.MaxPool2d(2),_Conv(width,width*2),nn.MaxPool2d(2),_Conv(width*2,width*4)); self.decoder=nn.Sequential(nn.Conv2d(width*4,width,1),nn.ReLU(True)); self.fusu_decoder_head=nn.Conv2d(width,fusu_classes,1); self.cwld_decoder_head=nn.Conv2d(width,cwld_classes,1)
    def features(self,x):
        return nn.functional.interpolate(self.decoder(self.encoder(x)),size=x.shape[-2:],mode="bilinear",align_corners=False)
    def forward_cwld(self,x):
        logits=self.cwld_decoder_head(self.features(x)); return {"logits":logits,"probabilities":logits.softmax(1)}
    def forward_fusu(self,t1,t2=None):
        a=self.fusu_decoder_head(self.features(t1)); b=self.fusu_decoder_head(self.features(t2)) if t2 is not None else None; return {"logits_t1":a,"logits_t2":b,"probabilities_t1":a.softmax(1),"probabilities_t2":b.softmax(1) if b is not None else None}
    def forward_sequence(self,images,timestamps=None,valid_mask=None):
        b,t,c,h,w=images.shape; logits=self.fusu_decoder_head(self.features(images.reshape(b*t,c,h,w))).reshape(b,t,-1,h,w); return {"logits":logits,"probabilities":logits.softmax(2)}
