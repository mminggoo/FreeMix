
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from utils.utils import visualize_correspondence, visualize_attention_map
from PIL import Image

def normalize(x, dim):
    x_mean = x.mean(dim=dim, keepdim=True)
    x_std = x.std(dim=dim, keepdim=True)
    x_normalized = (x - x_mean) / x_std
    return x_normalized

class DMSASelfAttention():
    def __init__(self,  start_step=0, end_step=50, step_idx=None, layer_idx=None, ref_masks=None, mask_weights=[1.0,1.0,1.0], style_fidelity=1, word_idx=None, structure_step=12, structure_layer=12, norm_step=50, noly_stru=False, noly_appear=False):
        """
        Args:
            start_step   : the step to start transforming self-attention to multi-reference self-attention
            end_step     : the step to end transforming self-attention to multi-reference self-attention
            step_idx     : list of the steps to transform self-attention to multi-reference self-attention
            layer_idx    : list of the layers to transform self-attention to multi-reference self-attention
            ref_masks    : masks of the input reference images
            mask_weights : mask weights for each reference masks
        """
        self.cur_step       =  0
        self.num_att_layers = -1
        self.cur_att_layer  =  0

        self.start_step   = start_step
        self.end_step     = end_step
        self.step_idx     = step_idx if step_idx is not None else list(range(start_step, end_step))
        self.layer_idx    = layer_idx

        self.structure_step = structure_step
        self.structure_layer = structure_layer
        
        self.ref_masks    = ref_masks
        self.mask_weights = mask_weights
        
        self.style_fidelity = style_fidelity


        self.cross_attns = []
        self.attention_store = None
        self.word_idx = word_idx
        self.norm_step = norm_step
        self.noly_stru = noly_stru
        self.noly_appear = noly_appear
       
    def after_step(self):
        if self.attention_store is None:
            self.attention_store = self.cross_attns
        else:
            for i in range(len(self.attention_store)):
                self.attention_store[i] += self.cross_attns[i]
        self.cross_attns = []

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        # Add
        if is_cross:
            # save cross attention map with res 16 * 16
            if attn.shape[1] == 16 * 16:
                B = len(attn) // num_heads // 2
                self.cross_attns.append(attn.reshape(-1, num_heads, *attn.shape[-2:]).mean(1)[B:B+1]) # (6, 256, 77)

        out = self.mrsa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            # Add
            self.after_step()
        return out

    # Add
    def aggregate_cross_attn_map(self, idx):
        # attn_map = torch.stack(self.cross_attns, dim=1).mean(1)  # (B, N, dim)
        attn_map = torch.stack(self.attention_store, dim=1).mean(1) / self.cur_step  # (B, N, dim)
        res = int(np.sqrt(attn_map.shape[-2]))
        attn_map = attn_map.reshape(-1, res, res, attn_map.shape[-1])
        image = attn_map[..., idx]
        if isinstance(idx, list):
            image = image.sum(-1)
        image_min = image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
        image_max = image.max(dim=1, keepdim=True)[0].max(dim=2, keepdim=True)[0]
        image = (image - image_min) / (image_max - image_min)
        return image.reshape(-1, 1, res, res)
    
    def get_ref_mask(self, ref_mask, mask_weight, H, W):
        ref_mask = ref_mask.float() * mask_weight
        ref_mask = F.interpolate(ref_mask, (H, W))
        ref_mask = ref_mask.flatten()
        return ref_mask
    
    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, idx=None, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))

        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        
        if kwargs.get("attn_batch_type") == 'mrsa':
            sim_own, sim_refs = sim[..., :H*W], sim[..., H*W:]
            sim_or = [sim_own]

            for i in range(len(self.ref_masks) - 1):
                ref_mask, mask_weight = self.ref_masks[i], self.mask_weights[i]
                ref_mask = self.get_ref_mask(ref_mask, mask_weight, H, W)
                
                sim_ref = sim_refs[..., :H*W]
                if self.cur_step >= self.norm_step:
                    sim_ref = torch.einsum("h i d, h j d -> h i j", normalize(q, dim=-2), normalize(k[:, H*W:], dim=-2)) * kwargs.get("scale")
                    ref_mask = self.get_ref_mask(self.ref_masks[i], mask_weight, H, W)

                sim_ref = sim_ref + ref_mask.masked_fill(ref_mask == 0, torch.finfo(sim.dtype).min)
                sim_or.append(sim_ref)

            sim = torch.cat(sim_or, dim=-1)
        attn = sim.softmax(-1)

        
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)


        return out

    def attn_batch2(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, idx=None, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads) 
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        
        structure_mask = self.aggregate_cross_attn_map(self.word_idx)
        structure_mask = self.get_ref_mask(torch.where(structure_mask > 0.2, 1, 0), 1, H, W)
        
        ref_mask = self.ref_masks[1]
        ref_mask = self.get_ref_mask(ref_mask, 1, H, W)
        k_selected = k[:, H*W:][:, ref_mask != 0]
        k_restored = torch.zeros_like(k[:, H*W:])
        num_repeats = H*W // sum(ref_mask != 0) + 1
        k_repeated = k_selected.repeat(1, num_repeats, 1)[:, :H*W, :]
        k[:, :H*W] = torch.where(structure_mask.unsqueeze(0).unsqueeze(-1) == 0, k[:, :H*W], k_repeated)

        v_selected = v[:, H*W:][:, ref_mask != 0]
        v_restored = torch.zeros_like(v[:, H*W:])
        num_repeats = H*W // sum(ref_mask != 0) + 1
        v_repeated = v_selected.repeat(1, num_repeats, 1)[:, :H*W, :]
        v[:, :H*W] = torch.where(structure_mask.unsqueeze(0).unsqueeze(-1) == 0, v[:, :H*W], v_repeated)


        sim = torch.einsum("h i d, h j d -> h i j", q, k[:, :H*W]) * kwargs.get("scale")


        attn = sim.softmax(-1)
        
        
        if len(attn) == 2 * len(v):
            v = torch.cat([v] * 2)
        out = torch.einsum("h i j, h j d -> h i d", attn, v[:, :H*W])
        out = rearrange(out, "(h1 h) (b n) d -> (h1 b) n (h d)", b=B, h=num_heads)

        return out

    def mrsa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Mutli-reference self-attention(MRSA) forward function
        """
        if is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
            return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        
        B = q.shape[0] // num_heads // 2 # 3
        qu, qc = q.chunk(2)
        ku, kc = k.chunk(2)
        vu, vc = v.chunk(2)

        # The first batch is the q,k,v feature of $z_t$ (own feature), and the subsequent batches are the q,k,v features of $z_t^'$ (reference featrue)
        qu_o, qu_r = qu[:num_heads], qu[num_heads:]
        qc_o, qc_r = qc[:num_heads], qc[num_heads:]
        
        ku_o, ku_r = ku[:num_heads], ku[num_heads:]
        kc_o, kc_r = kc[:num_heads], kc[num_heads:]
        
        vu_o, vu_r = vu[:num_heads], vu[num_heads:]
        vc_o, vc_r = vc[:num_heads], vc[num_heads:]

        out_u_target = self.attn_batch(qu[:num_heads], ku[:num_heads], vu[:num_heads], None, None, is_cross, place_in_unet, num_heads, **kwargs)
        
        ku_cat, vu_cat = torch.cat([ku_o, ku_r.chunk(B-1)[0], ku_r.chunk(B-1)[1]], 1), torch.cat([vu_o, vu_r.chunk(B-1)[0], vu_r.chunk(B-1)[1]], 1)
        
        if self.cur_att_layer // 2 < self.structure_layer or self.cur_step < self.structure_step:
            if self.noly_appear:
                return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
            kc_cat, vc_cat = torch.cat([kc_o, *kc_r.chunk(B-1)[:-1]], 1), torch.cat([vc_o, *vc_r.chunk(B-1)[:-1]], 1)
            out_c_target = self.attn_batch(qc_o, kc_cat, vc_cat, None, None, is_cross, place_in_unet, num_heads, attn_batch_type='mrsa', idx=0, **kwargs)
        else:
            if self.noly_stru:
                return self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
            kc_cat, vc_cat = torch.cat([kc_o, kc_r.chunk(B-1)[-1]], 1), torch.cat([vc_o, vc_r.chunk(B-1)[-1]], 1)
            out_c_target = self.attn_batch2(qc_o, kc_cat, vc_cat, None, None, is_cross, place_in_unet, num_heads, idx=1, **kwargs)
        
        out = self.sa_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        out_u, out_c = out.chunk(2)
        out_u_ref, out_c_ref = out_u[1:], out_c[1:]
        out = torch.cat([out_u_target, out_u_ref, out_c_target, out_c_ref], dim=0)
        
        return out
    
    def sa_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Original self-attention forward function
        """
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
        return out
