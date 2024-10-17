import torch
import torch.nn as nn
from model.prompt_net import PromptTranslator
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

    model = clip.build_model(state_dict or model.state_dict())

    return model


# class ImageEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()

#         self.conv1 = clip_model.conv1
#         self.class_embedding = clip_model.class_embedding
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_pre = clip_model.ln_pre
#         self.transformer = clip_model.transformer
#         self.ln_post = clip_model.ln_post
#         self.proj = clip_model.proj

#     def forward(self, x, vis_ctx=[]):
#         x = self.conv1(x)  # shape = [*, width, grid, grid]
#         x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
#         x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
#         x = torch.cat(
#             [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
#              x], dim=1)  # shape = [*, grid ** 2 + 1, width]forwad
#         x = x + self.positional_embedding.to(x.dtype)
#         x = self.ln_pre(x)

#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, vis_ctx, False)
#         x = x.permute(1, 0, 2)  # LND -> NLD

#         x = self.ln_post(x[:, 0, :])

#         if self.proj is not None:
#             x = x @ self.proj

#         return x


# class TextEncoder(nn.Module):
#     def __init__(self, clip_model):
#         super().__init__()
#         self.transformer = clip_model.transformer
#         self.positional_embedding = clip_model.positional_embedding
#         self.ln_final = clip_model.ln_final
#         self.text_projection = clip_model.text_projection
#         self.dtype = clip_model.dtype

#     def forward(self, prompts, tokenized_prompts, text_ctx):
#         x = prompts + self.positional_embedding.type(self.dtype)
#         x = x.permute(1, 0, 2)  # NLD -> LND
#         x = self.transformer(x, text_ctx, True)
#         x = x.permute(1, 0, 2)  # LND -> NLD
#         x = self.ln_final(x).type(self.dtype)

#         # x.shape = [batch_size, n_ctx, transformer.width]
#         # take features from the eot embedding (eot_token is the highest number in each sequence)
#         x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

#         return x




class CLIP(nn.Module):
    def __init__(self, cfg, clip_model,  device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.set_prompt_prefix()
        # ctx_dim = clip_model.ln_final.weight.shape[0]

        # self.image_encoder = ImageEncoder(clip_model.visual)
        # self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model = clip_model
        print('Token embedding: ', clip_model.token_embedding)
        
        embed_dim = self.clip_model.visual.proj.data.shape[1]
        self.prompt_learner = nn.Sequential(nn.Linear(embed_dim,embed_dim).half(), nn.Tanh(
            ), nn.Linear(embed_dim,embed_dim).half(), nn.Softmax(dim=1)).to(self.device)


    def set_prompt_prefix(self):
    
        self.prompt_prefix = "A photo of a"


    def get_tokenized_classnames(self, classnames):

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def forward(self, image, classnames,dataname):

        classnames = [name.replace("_", " ") for name in classnames]
        
        prompts_ = [self.prompt_prefix + " " + name + "." for name in classnames]
        # print(f"Prompts: {prompts_}")
        prompts_ = torch.cat([clip.tokenize(p) for p in prompts_])
        prompts_ = prompts_.to('cuda:4')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(prompts_)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits, 0


from transformers import pipeline, SiglipModel, AutoProcessor, SiglipTokenizer

class Siglip(nn.Module):
    def __init__(self, cfg, clip_model,  device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.set_prompt_prefix()
        # ctx_dim = clip_model.ln_final.weight.shape[0]

        # self.image_encoder = ImageEncoder(clip_model.visual)
        # self.text_encoder = TextEncoder(clip_model)

        self.token_embedding = clip_model.token_embedding
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.device = device
        self.clip_model = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
        self.tokenizer = SiglipTokenizer.from_pretrained('google/siglip-base-patch16-224')
        
        # embed_dim = self.clip_model.visual.proj.data.shape[1]
        embed_dim = 768
        self.prompt_learner = nn.Sequential(nn.Linear(embed_dim,embed_dim).half(), nn.Tanh(
            ), nn.Linear(embed_dim,embed_dim).half(), nn.Softmax(dim=1)).to(self.device)


    def set_prompt_prefix(self):

        self.prompt_prefix = "A photo of a"


    def get_tokenized_classnames(self, classnames):

        prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = self.tokenizer.batch_encode_plus(prompts, return_tensors='pt', padding='max_length')['input_ids']
        with torch.no_grad():
            embedding = self.token_embedding(tokenized_prompts.to(self.device)).type(self.dtype)
        # token_prefix = embedding[:, :1, :]  # SOS
        # token_suffix = embedding[:, 1 + self.n_ctx:, :]  # CLS, EOS
        return embedding, tokenized_prompts

    def forward(self, image, classnames,dataname):


        classnames = [name.replace("_", " ") for name in classnames]
        
        prompts_ = [self.prompt_prefix + " " + name + "." for name in classnames]
        # print(f"Prompts: {prompts_}")
        prompts_ = self.tokenizer.batch_encode_plus(prompts_, return_tensors='pt', padding='max_length')['input_ids']
        prompts_ = prompts_.to('cuda:4')

        with torch.no_grad():
            image_features = self.clip_model.vision_model(image).pooler_output
            text_features = self.clip_model.text_model(prompts_).pooler_output
        
        # print(image_features.shape)
        # print(text_features.shape)

        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()


        return logits, 0