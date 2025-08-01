"""
# -- coding: utf-8 --
Author: Weiyu Zhang
Date: 2024
Description: Modified Masked Autoencoder with Vision Transformer for geographical data analysis.
Based on Meta's MAE implementation with custom modifications for flexible embedding dimensions.

Original source: https://github.com/facebookresearch/mae
License: Creative Commons Attribution-NonCommercial 4.0 International

Key modifications:
1. Implement an additional final layer to reduce the dimensionality of embeddings. Whether use this structure is controled by final_emb_dim.
2. Removed predefined representation dimensions from vit_large_patch16 to allow external parameter control. 

In our study, we choose this solution 2 to allow the configurable dimensions of embeddings.

Original Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the Creative Commons Attribution-NonCommercial 4.0 
International License.
"""

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False,final_emb_dim=None, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        self.final_emb_dim = final_emb_dim
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        if final_emb_dim is not None:
            embed_dim = kwargs['embed_dim']
            self.output_layer = nn.Sequential(nn.Linear(embed_dim, embed_dim, bias=False),
                                nn.LayerNorm(embed_dim),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(embed_dim, embed_dim, bias=False),
                                nn.LayerNorm(embed_dim),
                                nn.ReLU(inplace=True), # second layer
                                nn.Linear(embed_dim, final_emb_dim, bias=False),
                                nn.LayerNorm(final_emb_dim)) # output layer
            del self.norm # remove the original norm
        else:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.output_layer = norm_layer(embed_dim)
            del self.norm
    
    def get_cls_attention_map(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i in range(len(self.blocks)-1):
            x = self.blocks[i](x)
            
        x = self.blocks[-1].norm1(x)
        B, N, C = x.shape
        qkv = self.blocks[-1].attn.qkv(x).reshape(B, N, 3, self.blocks[-1].attn.num_heads, C // self.blocks[-1].attn.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)

        cls_attn_map = attn[:, :, 0, 1:].mean(dim=1)
        cls_attn_map = cls_attn_map.reshape(B, 1, self.patch_embed.img_size[0]//self.patch_embed.patch_size[0], self.patch_embed.img_size[0]//self.patch_embed.patch_size[0])

        return cls_attn_map

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        elif self.final_emb_dim is not None:
            x = self.output_layer(x)
            outcome = x[:, 0]
        else:
            x = self.output_layer(x)
            outcome = x[:, 0]

        return outcome


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
