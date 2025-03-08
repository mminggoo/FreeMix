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
# multi-concept mixing
python inference.py
```



