import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

import torch.nn.functional as F

def random_indexes(size : int):
    forward_indexes = np.arange(size)

    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)

    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    #sequences[4,2,192]
    #indexes[3,2]
    c=repeat(indexes, 't b -> t b c', c=sequences.shape[-1])#[3,2,192]
    result=torch.gather(sequences, 0, c)#编码中[3,2,192]
    return result

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        #T(4),B(2),C(192)
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)#[256,2]
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)#[256,2]

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]#[4,2,192]
        print(patches)

        return patches, forward_indexes, backward_indexes

class Repeat(torch.nn.Module):
    def __init__(self, patch_size=16) -> None:
        super().__init__()
        self.patch_num=patch_size*patch_size

    def forward(self, patches : torch.Tensor):

        T, B, C = patches.shape
        #T(4),B(2),C(192)

        seq=np.arange(self.patch_num)#[0-3]
        indexes = [seq for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack(indexes, axis=-1), dtype=torch.long).to(patches.device)#[4,2]

        total=[]

        for i in range(self.patch_num):
            if i==0:
                choose_index=forward_indexes[i+1:]
            elif i==self.patch_num-1:
                choose_index=forward_indexes[:i]
            else:
                choose_index=torch.cat([forward_indexes[:i],forward_indexes[i+1:]],dim=0)
            patches_=take_indexes(patches, choose_index)#[3,2,192]
            total.append(patches_)

        return total

class SMC_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))#[1,1,192]

        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))#[4,1,192]

        self.repeat = Repeat(image_size // patch_size)

        self.patchify = torch.nn.Conv2d(in_channels=3, out_channels=emb_dim, kernel_size=patch_size, stride=patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)#[2,192,2,2]

        patches = rearrange(patches, 'b c h w -> (h w) b c')#[4,2,192]

        patches = patches + self.pos_embedding#[4,2,192]

        total_patches= self.repeat(patches)

        total_features=[]

        for item in total_patches:
            #item[3,2,192]
            cls = self.cls_token.expand(-1, item.shape[1], -1)  # [1,2,192]
            item = torch.cat([cls, item], dim=0)  # [4,2,192]
            item = rearrange(item, 't b c -> b t c')  # [2,4,192]
            features = self.layer_norm(self.transformer(item))  # [2,4,192]

            a,b,c=features.shape
            #a(2),b(4),c(192)
            noise=torch.rand(a,1,1,requires_grad=False).expand_as(features).to(features.device)
            features=features*noise

            features = rearrange(features, 'b t c -> t b c')  # [4,2,192]
            total_features.append(features)

        return total_features

class SMC_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))#[4,1,192]

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)#[192,12]

        self.predict=torch.nn.Sequential(
            torch.nn.Linear((image_size//patch_size)**2, (image_size//patch_size)**2),
            torch.nn.ReLU(),
            torch.nn.Linear((image_size // patch_size) ** 2, 1)
                            )

        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, total_features):
        final_features=[]
        for item in total_features:
            #item[4,2,192]
            item = rearrange(item, 'b t c -> t c b')  # [2,192,4]
            item = self.predict(item)  # [2,192,1]
            item = rearrange(item, 'a b c -> c a b')#[1,2,192]
            final_features.append(item)

        features=torch.cat(final_features,dim=0)#[4,2,192]
        features = features + self.pos_embedding#[4,2,192]
        features = rearrange(features, 't b c -> b t c')  # [2,4,192]
        features = self.transformer(features)  # [2,4,192]
        features = rearrange(features, 'a b c -> b a c')  # [4,2,192]
        patches = self.head(features)#[4,2,3*32*32]

        img = self.patch2img(patches)#[2,3,64,64]

        return img

class SMC_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 ) -> None:
        super().__init__()

        self.encoder = SMC_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head)
        self.decoder = SMC_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features = self.encoder(img)
        predicted_img = self.decoder(features)
        return predicted_img

class Dim_reduction_net(torch.nn.Module):
    def __init__(self,image_size=32,vectordim=128) -> None:
        super().__init__()

        self.downLinear = torch.nn.Linear(3 * image_size * image_size, vectordim)

    def forward(self, x):
        x=x.view(x.shape[0], -1)
        out=self.downLinear(x)
        return F.normalize(out, p=2, dim=1)

class RMC_Classifier(torch.nn.Module):
    def __init__(self, encoder : SMC_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, img):
        patches = self.patchify(img)#[2,192,2,2]
        patches = rearrange(patches, 'b c h w -> (h w) b c')#[4,2,192]
        patches = patches + self.pos_embedding#[4,2,192]
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)#[5,2,192]
        patches = rearrange(patches, 't b c -> b t c')#[2,5,192]
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')#[5,2,192]
        logits = self.head(features[0])
        return logits

if __name__ == '__main__':
    # shuffle = PatchShuffle(0.75)
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a)
    # print(b.shape)

    img = torch.rand(2, 3, 64, 64)
    encoder = SMC_Encoder(image_size=64, patch_size=32)
    decoder = SMC_Decoder(image_size=64, patch_size=32)
    features=encoder(img)
    out=decoder(features)

    # # print(forward_indexes.shape)
    # predicted_img, mask = decoder(features)
    # # print(predicted_img.shape)
    # # loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)
    # # print(loss)
    # classifier=RMC_Classifier(encoder)