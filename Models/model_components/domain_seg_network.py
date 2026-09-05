from .domain_seg_upstream import DomainSegUpstream
from .domain_seg_head import DomainSegHead

import torch.nn as nn

class DomainSegNetwork(nn.Module):
    def __init__(self, pretrained):
        super(DomainSegNetwork, self).__init__()

        # Upstream blocks
        self.DomainSegUpstream = DomainSegUpstream(pretrained)

        # DomainSeg Head
        self.DomainSegHead = DomainSegHead()
    

    def forward(self, image):
        neck, features = self.DomainSegUpstream(image)
        prediction = self.DomainSegHead(neck, features)
        return prediction