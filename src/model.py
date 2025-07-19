import torch
import torch.nn as nn



class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, in_channels=3, emb_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_dim = emb_dim
        self.proj = nn.Conv2d(
            in_channels = in_channels,
            out_channels = self.emb_dim,
            kernel_size = self.patch_size,
            stride = patch_size,
            bias = True
        )
    
    def forward(self, x):
        out = self.proj(x)
        B,C,H,W = x.shape
        out = out.view(B, C, H*W).transpose(1,2)
        return out



class LayerNormalization(nn.Module):

    def __init__(self, embdim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(embdim))
        self.beta = nn.Parameter(torch.zeros(embdim))

    def forward(self, x):
        xmean = x.mean(dim=-1, keepdim=True)
        xstd = x.std(dim=-1, keepdim=True)
        return self.alpha * ((x-xmean)/(xstd+1e-8)) + self.beta
    


class SelfAttention(nn.Module):

    def __init__(self, embdim, num_heads):
        super().__init__()
        self.emb_dim = embdim
        self.num_heads = num_heads
        self.query = nn.Linear(embdim, embdim)
        self.key  = nn.Linear(embdim, embdim)
        self.value = nn.Linear(embdim, embdim)
        self.proj  = nn.Linear(embdim, embdim)
        assert self.emb_dim%self.num_heads==0, f"Embdim: {embdim} must be divisible by num_heads: {num_heads} "
        self.head_dim = self.emb_dim//num_heads


    @staticmethod
    def attention(q,k,v):
        head_dim = q.shape[-1]
        attention = q @ k.transpose(-2,-1) / (head_dim)**(1/2)
        attention = attention.softmax(dim=-1)
        out = attention @ v
        return out, attention

    
    def forward(self, x):
        # shape of x => (B, SEQ, EM )
        q = self.query(x)
        k = self.query(x)
        v = self.query(x)
        # shape of q,k,v => (B, SEQ, EM )
        # converting q,k,v => (B, NUM_HEAD, SEQ, HEAD_DIM)
        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        k = q.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1,2)




