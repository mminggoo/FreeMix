<div align="center">

<h1>FreeMix: Personalized Structure and Appearance Control Without Finetuning</h1>

</div>

## 📖 Abstract
<p>
  Personalized image generation has gained significant attention with the advancement of text-to-image diffusion models. However, existing methods struggle with effectively mixing multiple visual attributes, such as structure and appearance, from different reference images. Finetuning-based methods are time-consuming and prone to overfitting, while finetuning-free approaches often suffer from feature entanglement, leading to undesired distortions. 
</p>
<p>
  To address these challenges, we propose FreeMix, a finetuning-free approach for multi-concept mixing in personalized image generation. Given a structure reference and an appearance reference, FreeMix generates a new image that integrates both attributes through Disentangle-Mixing Self-Attention (DMSA). DMSA selectively transfers structural and appearance features across different layers and time steps within a diffusion model. Extensive qualitative and quantitative experiments demonstrate that our method achieves superior structural consistency and appearance transfer compared to existing methods. In addition to personalization, FreeMix can be easily adapted to exemplar-based image editing, enabling appearance modifications.
</p>


## 🚀 Run
1. install
```
conda create -n freemix python=3.11.9 -y
conda activate freemix
pip install -r requirements.txt
```

2. run the following command to view the results
```
python freecustom_stable_diffusion.py
```

**At this point, you can already see the customized results, but you can also try the following two methods:**
1. try another config
- replace `./configs/config_stable_diffusion.yaml` with one of configuration files in the `./datasets/freecustom/multi_concept` folder. 
- run as step 2.

2. prepare your own data
- Select 2 to 3 images that represent the concepts you wish to customize, ensuring that each concept has contextual interaction.
- Use [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) or other segmentation tools to obtain concept masks for filtering out irrelevant pixels.
- Store your images and masks according to the structure in the dataset folder, making sure that the filenames and extensions of the images and masks are identical.
- Modify the `./configs/config_stable_diffusion.yaml` file by updating the "ref_image_infos" and "target_prompt" fields to align with your prepared data.
- Execute `python freecustom_stable_diffusion.py` to view the results.
- Feel free to experiment with adjusting the "seeds" and "mask_weights" fields in the `./configs/config_stable_diffusion.yaml` to achieve satisfactory results.

## 🌄 Demo of customized image generation
### multi-concept composition 
![results_of_multi_concept](docs/static/images/results_of_multi_concept.png)

### single-concept customization
![results_of_single_concept](docs/static/images/results_of_single_concept.png)

Our method excels at *rapidly* generating high-quality images with multiple concept combinations and single concept customization, without any model parameter tuning. The identity of each concept is remarkably preserved. Furthermore, our method exhibits great versatility and robustness when dealing with different categories of concepts. This versatility allows users to generate customized images that involve diverse combinations of concepts, catering to their specific needs and preferences. Best viewed on screen.

