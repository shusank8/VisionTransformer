import torch
import torch.nn as nn



class PatchEmbedding(nn.Module):

    def __init__(self, img_size, patch_size, in_channels=3, emb_dim=768):
        # (B, NC, H, W)
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
        # (B, H*W, C)
        # C = EMBDIM
        # (B, SEQ, EMBDIM)
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
        # SHAPE OF Q,K,V = > (B, NUM_HEADS, SEQ, HEAD_DIM)
        head_dim = q.shape[-1]
        attention = q @ k.transpose(-2,-1) / (head_dim)**(1/2)
        # SHAPE OF ATTENTION=> (B, NUM_HEADS, SEQ, SEQ)
        attention = attention.softmax(dim=-1)
        # SHAPE OF OUT => (B, NUM_HEADS, SEQ, HEAD_DIM)
        out = attention @ v
        return out, attention

    
    def forward(self, x):
        # shape of x => (B, SEQ, EM )
        B,T,C = q.shape
        q = self.query(x)
        k = self.query(x)
        v = self.query(x)
        # shape of q,k,v => (B, SEQ, EM )
        # converting q,k,v => (B, NUM_HEAD, SEQ, HEAD_DIM)

        q = q.view(q.shape[0], q.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        k = q.view(k.shape[0], k.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(v.shape[0], v.shape[1], self.num_heads, self.head_dim).transpose(1,2)

        out, attention = SelfAttention.attention(q,k,v)

        # shape of out=> (B, NUM_HEADS, SEQ, HEAD_DIM)

        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)


class FeedForward(nn.Module):
    def __init__(self, embdim):
        super().__init__()
        self.ffd = nn.Sequential(
            nn.Linear(embdim, 4*embdim),
            # use SWiGlU
            nn.ReLU(),
            nn.Linear(4*embdim, embdim)
        )

    
    def forward(self, x):
        return self.ffd(x)
    


class EncoderBlock(nn.Module):

    def __init__(self, embdim, num_heads):
        super().__init__()
        self.attn = SelfAttention(embdim, num_heads)
        self.ffd = FeedForward(embdim)
        self.layernorm1 = nn.LayerNormalization(embdim)
        self.layernorm2 = nn.LayerNormalization(embdim)
    
    def forward(self, x):
        # skip connections
        x = x + self.attn(self.layernorm1(x))
        x = x + self.ffd(self.layernorm2(x))
        return x

class Encoder(nn.Module):

    def __init__(self, layers):
        super().__init__()
        self.layers = layers

    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



class VisionTransformer(nn.Module):
    def __init__(self, patch, cls_tok, encoder):
        super().__init__()
        # self.patch = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
        # # adding cls token: not doing average
        # self.cls_token = torch.zeros(1,1, emb_dim)
        # # about to add cls token to seqlen
        # self.seq_len = patch_size+1

        # self.encoder = encoder
        self.patch = patch
        self.cls_tok = cls_tok
        self.encoder = encoder

    
    def add_cls(self,x):
        # shape of x=> (B, SEQ, EMBDIM)
        # (B, SEQ, EMBDIM)
        # (B, SEQ+1, EMBDIM)
        # (B, 1, EMBDIM)
        # (B, 1, EMDIM) => (B, EMBDIM) => (B, 2)
        B, SEQ, C = x.shape
        self.cls_tok = self.cls_tok.expand(B, -1, C)
        return torch.cat([
            self.cls_tok, x
        ], dim=1)


    def forward(self, x):
        # x shape => (B, C, H, W)
        x = self.patch(x)
        # x shape => (B, H*W, EMBDIM)
        x = self.add_cls(x)
        # shape of x => (B, SEQ+1, Embdim) added 1 cause of cls token
        x = self.encoder(x)

        # only return cls token from every batch, if classification we can add final layer over here
        # that will project x[:, 0] to probabilities of classes
        return x[:, 0]

def build_transformer(img_size, patch_size, in_channels, emb_dim, encoder_depth, num_heads):
    patch = PatchEmbedding(img_size, patch_size, in_channels, emb_dim)
    cls_token = nn.Parameter(torch.zeros(1,1,emb_dim))
    encoder_blocks = []

    for _ in range(encoder_depth):
        encoder_block = EncoderBlock(emb_dim, num_heads)
        encoder_blocks.append(encoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    vision_transformer = VisionTransformer(patch, cls_token, encoder)
    return vision_transformer





