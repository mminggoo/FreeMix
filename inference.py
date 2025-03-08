import os,sys,argparse
from omegaconf import OmegaConf
import numpy as np

import torch
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from pytorch_lightning import seed_everything

from diffusers import DDIMScheduler

from utils.utils import load_image, load_mask
from pipelines.pipeline_stable_diffusion_freemix import StableDiffusionFreeMixPipeline
from freemix.dmsa import DMSASelfAttention
from freemix.hack_attention import hack_self_attention_to_mrsa

def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--stru_path', type=str, default="data")
    parser.add_argument('--attr_path', type=str, default="data")
    parser.add_argument('--stru_prompt', type=str, default="data")
    parser.add_argument('--attr_prompt', type=str, default="data")
    parser.add_argument('--target_prompt', type=str, default="data")
    parser.add_argument('--blend_word', type=str, default="data")
    parser.add_argument('--results_dir', type=str, default="results")
    parser.add_argument('--base_prompt', type=str, default="results")
    
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--end_step', type=int, default=50)
    parser.add_argument('--structure_layer', type=int, default=13)
    parser.add_argument('--structure_step', type=int, default=15)
    parser.add_argument('--norm_step', type=int, default=50)
    
    parser.add_argument('--model_path', type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--negative_prompt', type=str, default="lowres, bad anatomy, text, error, cropped, worst quality, low quality, normal quality, jpeg artifacts, blurry")
    parser.add_argument('--use_null_ref_prompts', default=False, action="store_true")
    parser.add_argument('--seeds', nargs='+', type=int, default=[16, 5000, 202045, 884312, 1552580])
    args = parser.parse_args()
     
    sys.path.append(os.getcwd())

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    ref_image_infos = {args.stru_path: args.stru_prompt, args.attr_path: args.attr_prompt}

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set model
    scheduler = DDIMScheduler(  beta_start=0.00085, 
                                beta_end=0.012, 
                                beta_schedule="scaled_linear", 
                                clip_sample=False, 
                                set_alpha_to_one=False
                            )
    model = StableDiffusionFreeMixPipeline.from_pretrained(args.model_path, scheduler=scheduler).to(device)
    model.safety_checker = None

    # prapare data (including prompt, mask, image, latent)
    ref_masks       = []
    ref_images      = []
    ref_prompts     = []
    ref_latents_z_0 = []
    for idx, ref_image_info in enumerate(ref_image_infos.items()):
        ref_image_path  = ref_image_info[0]
        ref_text_prompt = ref_image_info[1]
        ref_mask_path   = ref_image_path.replace('/image/', '/mask/')
        ref_mask  = load_mask(ref_mask_path, device)
        ref_image = load_image(ref_image_path, device)
        ref_masks.append(ref_mask)
        ref_images.append(ref_image)
        ref_prompts.append(ref_text_prompt)
        ref_latents_z_0.append(model.image2latent(ref_image))

        # if idx == 0:
        #     start_code, latents_list = model.invert(ref_image,
        #                                     ref_text_prompt,
        #                                     guidance_scale=1.0,
        #                                     num_inference_steps=50,)
        #     randn_latent_z_T = start_code

    # set prompt
    target_prompt = args.target_prompt
    if args.use_null_ref_prompts:  
        prompts = [target_prompt] + ([""] * (len(ref_prompts))) 
    else:
        prompts = [target_prompt] + ref_prompts
    negative_prompts = [args.negative_prompt] * len(prompts)

    word_idx = get_word_inds(target_prompt, args.blend_word, model.tokenizer)

    # set dirs
    concepts_name  = ref_image_path.split('/')[3]
    image_save_dir = os.path.join(results_dir, 'ref_images')
    mask_save_dir  = os.path.join(results_dir, 'ref_masks')
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    # set config for visualization
    viz_cfg = OmegaConf.load("configs/config_for_visualization.yaml")
    viz_cfg.results_dir = results_dir
    viz_cfg.ref_image_infos = ref_image_infos

    # save image, mask, and config
    for i, (ref_image, ref_mask) in enumerate(zip(ref_images, ref_masks)):
        save_image(ref_image*0.5+0.5, os.path.join(image_save_dir, f'image_{i}.png'))
        save_image(ref_mask.float(),  os.path.join(mask_save_dir, f'mask_{i}.png'))

    # run each seed
    for seed in args.seeds:
        seed_everything(seed)

        # hack the attention module
        mrsa = DMSASelfAttention(
                                start_step     = args.start_step,
                                end_step       = args.end_step,
                                layer_idx      = [8,9,10,11,12,13,14,15],
                                ref_masks      = ref_masks,
                                mask_weights   = [3.0, 3.0],
                                style_fidelity = 1,
                                viz_cfg        = viz_cfg,
                                word_idx       = word_idx,
                                structure_layer = args.structure_layer,
                                structure_step = args.structure_step,
                                norm_step = args.norm_step,
                                )
        hack_self_attention_to_mrsa(model, mrsa)

        # set latent
        randn_latent_z_T = torch.randn_like(ref_latents_z_0[0])   # Initialize Gaussian noise for generated image $z_T$
        
        # idx = torch.randperm(randn_latent_z_T.nelement()//4)
        # randn_latent_z_T = randn_latent_z_T.view(1, 4, -1)[:, :, idx].view(randn_latent_z_T.size())

        latents = torch.cat([randn_latent_z_T] + ref_latents_z_0) # Concatenate $z_T$ and the latent code of the reference images $z_0^'$
        
        # run freecustom
        images = model(
                    prompt=prompts,
                    latents=latents,
                    guidance_scale=7.5,
                    negative_prompt=negative_prompts,
                    ).images[0]
        images.save(os.path.join(results_dir, f"{args.base_prompt}.jpg"))
        
        # concat input images and generated image
        out_image = torch.cat([ref_image * 0.5 + 0.5 for ref_image in ref_images] + [ToTensor()(images).to(device).unsqueeze(0)], dim=0)
        save_image(out_image, os.path.join(results_dir, f"all_{seed}.png"))

