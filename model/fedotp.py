import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast



from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {"trainer": 'GLP_OT',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0}

    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        
        # changed
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        

    def forward(self, prompts, tokenized_prompts):

        x = prompts + self.positional_embedding.type(self.dtype)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] 
        x= x@ self.text_projection

        return x

class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()

        # changed
        self.conv1 = clip_model.conv1
        self.class_embedding = clip_model.class_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_pre = clip_model.ln_pre
        self.transformer = clip_model.transformer
        self.ln_post = clip_model.ln_post
        self.proj = clip_model.proj

    def forward(self, x, vis_ctx=[]):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # changed
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]forwad
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, vis_ctx, False)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            # changed
            x = x @ self.proj


        return x.permute(1,0,2)

class PromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.n_cls = cfg.TRAIN.NUM_CLASS_PER_CLIENT
        self.n_ctx = 16 # TODO
        ctx_init = False
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # clip_imsize = clip_model.visual.input_resolution
        # cfg_imsize = cfg.INPUT.SIZE[0]
        self.N = 2
        # assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # if ctx_init:
        #     # use given words to initialize context vectors
        #     ctx_init = ctx_init.replace("_", " ")
        #     n_ctx = len(ctx_init.split(" "))
        #     prompt = clip.tokenize(ctx_init)
        #     with torch.no_grad():
        #         embedding = clip_model.token_embedding(prompt).type(self.dtype)
        #     ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
        #     self.prompt_prefix = ctx_init

        # else:
        #     # random initialization
        #     # if cfg.TRAINER.GLP_OT.CSC:
        #     if False:
        #         print("Initializing class-specific contexts")
        #         ctx_vectors = torch.empty(self.n_cls, self.n_ctx, ctx_dim, dtype=self.dtype)
        #     else:
        #         print("Initializing a generic context")
        ctx_vectors = torch.empty(self.N, self.n_ctx, ctx_dim, dtype=self.dtype) 
        nn.init.normal_(ctx_vectors, std=0.02)   # define the prompt to be trained
        self.prompt_prefix = " ".join(["X"] * self.n_ctx)    

        print(f'Initial context: "{self.prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

    def forward(self,classnames, name_lens, tokenized_prompts, embedding):
        # self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        # self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx :, :])  # CLS, EOS
        
        self.n_cls = embedding.shape[0] // self.N
        # self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        # self.class_token_position = cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION
        self.class_token_position = "end"
       
        ctx = self.ctx

        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1) 
        
        ctx = ctx.permute(1, 0, 2, 3) 
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        # prefix = self.token_prefix
        # suffix = self.token_suffix
        
        prefix = embedding[:, :1, :]
        suffix = embedding[:, 1 + self.n_ctx :, :]

        if self.class_token_position == "end":
            
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class Fedotp(nn.Module):
    def __init__(self, cfg, clip_model, device):
        super().__init__()
        self.n_cls = cfg.TRAIN.NUM_CLASS_PER_CLIENT
        self.prompt_learner = PromptLearner(cfg, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = ImageEncoder(clip_model.visual)
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.device1 = device
        self.N = 2
        self.use_uniform = True
        self.eps = 0.1
        self.max_iter = 100
        self.thresh = 0.001
        self.OT = 'COT'
        self.top_percent = 0.8
        self.max_iter = 100
        self.clip_model = clip_model

    def Sinkhorn(self, K, u, v):
        r = torch.ones_like(u)
        c = torch.ones_like(v)
        thresh = self.thresh
        for i in range(self.max_iter):
            r0 = r
            r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
            c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
            err = (r - r0).abs().mean()
            if err.item() < thresh:
                break

        T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

        return T
    
    def entropic_COT_fast(self, a, b, M, reg, numItermax=1000, stopThr=1e-9, verbose=False, log=False):
        """
        modify from ot.partial.entropic_partial_wasserstein in torch version

        """
        dx = torch.ones_like(a)
        dy = torch.ones_like(b)

        log_e = {'err': []}
        stopThr=self.thresh 

        # K = torch.exp(M / (-reg))
        K = M

        Kp = torch.matmul(torch.diag_embed(1 / a, dim1=1), K)
        Kq = torch.matmul(torch.diag_embed(1 / b, dim1=1), K.permute(0, 2, 1))

        err, cpt = 1, 0
        u = dx
        v = dy
        while (cpt < numItermax):

            v0 = v
            temp = torch.div(dx, torch.matmul(Kp, v.unsqueeze(-1)).squeeze(-1))
            u = torch.minimum(temp, dx)
            v = torch.div(dy, torch.matmul(Kq, u.unsqueeze(-1)).squeeze(-1))

            cpt = cpt + 1
            err = (v - v0).abs().mean()
            if err.item() <  stopThr:
                break
        Kprev = torch.matmul(torch.diag_embed(u,dim1=1), K)
        Kprev = torch.matmul(Kprev, torch.diag_embed(v,dim1=1))
        if log:
            return Kprev, log_e
        else:
            return Kprev

    def get_tokenized_prompts(self,classnames, prompt_prefix):
        classnames = [name.replace("_", " ") for name in classnames]   
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        # changed
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]) 
        tokenized_prompts = tokenized_prompts.repeat(self.N,1) 
        # tokenized_prompts3.view(3,100,77)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype).to(self.device) # changed

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        
        return name_lens, tokenized_prompts, embedding
    
    def forward(self, image, classnames, dataname):
        
        b = image.shape[0]
        image_features = self.image_encoder(image.type(self.dtype))
        image_feature_pool = image_features[0]
        image_features = image_features[1:]
        M = image_features.shape[0]
        self.d = image_features.shape[-1]

        name_lens, tokenized_prompts, embedding = self.get_tokenized_prompts(classnames, self.prompt_learner.prompt_prefix)
        self.n_cls = embedding.shape[0]//self.N
        prompts = self.prompt_learner(classnames,name_lens, tokenized_prompts, embedding)
        if dataname == "ImageNet":
            text_features = self.text_encoder(prompts.to(self.device1), tokenized_prompts.to(self.device1)) 
            text_features = text_features.to(self.device)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)
        else:
            text_features = self.text_encoder(prompts, tokenized_prompts)
            text_features =  text_features.contiguous().view(self.N, self.n_cls, self.d)  
            text_feature_pool = text_features.mean(dim=0)

        image_features =  F.normalize(image_features, dim=-1) 
        image_feature_pool = F.normalize(image_feature_pool, dim=-1)
        text_features = F.normalize(text_features, dim=2)
        text_feature_pool = F.normalize(text_feature_pool, dim=1)
        
        

        sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()  
        sim = sim.view(M,self.N,b*self.n_cls)
        sim = sim.permute(2,0,1)
        wdist = 1.0 - sim

        xx=torch.zeros(b*self.n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
        if self.OT == 'Sinkhorn':
            yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N)
        elif self.OT == 'COT':
            top_percent = min(torch.sum(xx).item(), self.top_percent)
            yy=torch.zeros(b*self.n_cls, self.N, dtype=sim.dtype, device=sim.device).fill_(1. / self.N) * top_percent

        with torch.no_grad():
            KK = torch.exp(-wdist / self.eps)
            if self.OT == 'Sinkhorn':
                T = self.Sinkhorn(KK,xx,yy)
            elif self.OT == 'COT':
                T = self.entropic_COT_fast(xx,yy,KK,0.01,numItermax=self.max_iter)
        if torch.isnan(T).any():
            return None


        sim_op = torch.sum(T * sim, dim=(1, 2))
        sim_op = sim_op.contiguous().view(b,self.n_cls)
        

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * sim_op  
        
        return logits,0
    
    
    
    
    