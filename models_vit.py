# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer
from timm.models.vision_transformer import PatchEmbed




class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.head2 = torch.nn.Linear(self.head.in_features + self.head.out_features, 44)
        self.head3 = torch.nn.Linear(self.head2.in_features + self.head2.out_features, 156)
        
        
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)
            self.fc_norm2 = norm_layer(self.head2.in_features)
            self.fc_norm3 = norm_layer(self.head3.in_features)
            del self.norm  # remove the original norm
        self.patch_embed = PatchEmbed(img_size=kwargs['img_size'], patch_size=kwargs['patch_size'], in_chans=kwargs['in_chans'], embed_dim=kwargs['embed_dim'])

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  #([512, 64, 1024])

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)  # ([512, 65, 1024])
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])  # x must be a tuple
            if self.training and not torch.jit.is_scripting():
                # during inference, return the average of both classifier predictions
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            if self.global_pool:
                x1 = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                out1 = self.fc_norm(x1)  # ([Batch, 1024])
                outcome1 = self.head(out1) 
                x2 = torch.cat([x1, outcome1], dim=-1)  # ([Batch, 1029])
                out2 = self.fc_norm2(x2)
                outcome2 = self.head2(out2) 
                x3 = torch.cat([x2, outcome2], dim=-1)  # ([Batch, 1085])
                out3 = self.fc_norm3(x3)
                outcome3 = self.head3(out3)
            else:
                x1 = self.norm(x)
                out1 = x1[:, 0]
                outcome1 = self.head(out1)
                x2 = torch.cat([out1, outcome1], dim=-1)
                
                outcome2 = self.head2(x2)
                out3 = torch.cat([x2, outcome2], dim=-1)
                outcome3 = self.head3(out3)

            return outcome1, outcome2, outcome3,out3


def vit_large_patch4_5mer(args,**kwargs):
    model = VisionTransformer(img_size=32, in_chans=1, patch_size=4, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch4_5mer(args,**kwargs):
    model = VisionTransformer(img_size=32, in_chans=1, patch_size=4,  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


