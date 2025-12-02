# Segmenter: Transformer for Semantic Segmentation

![Figure 1 from paper](./overview.png)

[Segmenter: Transformer for Semantic Segmentation](https://arxiv.org/abs/2105.05633)  
by Robin Strudel*, Ricardo Garcia*, Ivan Laptev and Cordelia Schmid, ICCV 2021.  
*Equal Contribution  

🔥 **Segmenter is now available on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation/tree/master/configs/segmenter).**

---

## 🔥 Meta Dataset (Used for Fully Supervised Teacher Model)

This repository has been adapted to train a fully supervised Vision Transformer–based teacher model on a unified **Meta Dataset** specifically designed for wildfire-related semantic segmentation.  

### **🧩 Meta Dataset Composition**
The Meta Dataset merges **four publicly available UAV datasets** into one unified dataset containing **four classes**:

| Class ID | Class Name      |
|----------|-----------------|
| 0        | Background      |
| 1        | Fire            |
| 2        | Burned Area     |
| 3        | Water           |

### **📚 Source Datasets Included in the Meta Dataset**

**[6] FLAME Dataset — Aerial Pile-Burn Detection (Drone/UAV)**  
D. Sharma et al.  
https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs  
Accessed: 2025-01-15  

**[7] Corsican Fire Database (CFD)**  
University of Corsica / SPE Laboratory  
https://cfdb.univ-corse.fr/index.php?newlang=english&menu=1  
Accessed: 2025-01-15  

**[8] AIWR — Aerial Image Water Resources Dataset**  
E. Menezes et al.  
https://data.mendeley.com/datasets/8dxv4yvvjk/1  
Mendeley Data, Version 1  

**[9] BurnedAreaUAV v1.1 — Burned Area Segmentation Dataset**  
C. Pinto et al.  
https://zenodo.org/records/7866087  
Zenodo, Version 1.1  

These four datasets are harmonized into a single structure, remapped into the 4-class taxonomy shown above, and prepared for transformer-based semantic segmentation.

---

## Installation

Set the dataset directory in `.bashrc`:
```sh
export DATASET=/path/to/dataset/dir
```

Install PyTorch 1.9, then:
```sh
pip install .
```

To download ADE20K:
```python
python -m segm.scripts.prepare_ade20k $DATASET
```

---

## Model Zoo

Segmenter models trained on various datasets using ViT and DeiT backbones.

### ADE20K (ViT Backbone)

<table><tr><th>Name</th><th>mIoU (SS/MS)</th><th># params</th><th>Resolution</th><th>FPS</th><th colspan="3">Download</th></tr><tr><td>Seg-T-Mask/16</td><td>38.1 / 38.8</td><td>7M</td><td>512x512</td><td>52.4</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_tiny_mask/log.txt">log</a></td></tr><tr><td>Seg-S-Mask/16</td><td>45.3 / 46.9</td><td>27M</td><td>512x512</td><td>34.8</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_small_mask/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_small_mask/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_small_mask/log.txt">log</a></td></tr><tr><td>Seg-B-Mask/16</td><td>48.5 / 50.0</td><td>106M</td><td>512x512</td><td>24.1</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_mask/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_mask/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_mask/log.txt">log</a></td></tr><tr><td>Seg-B/8</td><td>49.5 / 50.5</td><td>89M</td><td>512x512</td><td>4.2</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_patch8/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_patch8/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_patch8/log.txt">log</a></td></tr><tr><td>Seg-L-Mask/16</td><td>51.8 / 53.6</td><td>334M</td><td>640x640</td><td>-</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_large_mask_640/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_large_mask_640/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_large_mask_640/log.txt">log</a></td></tr></table>

### DeiT Backbone

<table><tr><th>Name</th><th>mIoU (SS/MS)</th><th># params</th><th>Resolution</th><th>FPS</th><th colspan="3">Download</th></tr><tr><td>Seg-B†/16</td><td>47.1 / 48.1</td><td>87M</td><td>512x512</td><td>27.3</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_linear/log.txt">log</a></td></tr><tr><td>Seg-B†-Mask/16</td><td>48.7 / 50.1</td><td>106M</td><td>512x512</td><td>24.1</td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_mask/checkpoint.pth">model</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_mask/variant.yml">config</a></td><td><a href="https://www.rocq.inria.fr/cluster-willow/rstrudel/segmenter/checkpoints/ade20k/seg_base_deit_mask/log.txt">log</a></td></tr></table>

---

## Inference

```python
python -m segm.inference --model-path seg_tiny_mask/checkpoint.pth -i images/ -o segmaps/
```

Evaluate on ADE20K:
```python
python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --singlescale
python -m segm.eval.miou seg_tiny_mask/checkpoint.pth ade20k --multiscale
```

---

## Train

Example (ADE20K):
```python
python -m segm.train --log-dir seg_tiny_mask --dataset ade20k \
  --backbone vit_tiny_patch16_384 --decoder mask_transformer
```

---

## Logs

```python
python -m segm.utils.logs logs.yml
```

---

## Attention Maps

```python
python -m segm.scripts.show_attn_map seg_tiny_mask/checkpoint.pth images/im0.jpg output_dir/ --layer-id 0 --x-patch 0 --y-patch 21 --enc
```

---

## Video Segmentation

Zero-shot DAVIS segmentation examples:

<p align="middle">
<img src="https://github.com/rstrudel/segmenter/blob/master/gifs/choreography.gif" width="350">
<img src="https://github.com/rstrudel/segmenter/blob/master/gifs/city-ride.gif" width="350">
</p>
<p align="middle">
<img src="https://github.com/rstrudel/segmenter/blob/master/gifs/car-competition.gif" width="350">
<img src="https://github.com/rstrudel/segmenter/blob/master/gifs/breakdance-flare.gif" width="350">
</p>

---

## BibTex

```
@article{strudel2021,
  title={Segmenter: Transformer for Semantic Segmentation},
  author={Strudel, Robin and Garcia, Ricardo and Laptev, Ivan and Schmid, Cordelia},
  journal={arXiv preprint arXiv:2105.05633},
  year={2021}
}
```

---

## Acknowledgements

Vision Transformer code is based on [timm](https://github.com/rwightman/pytorch-image-models).  
Training and evaluation pipeline is based on [mmsegmentation](https://github.com/open-mmlab/mmsegmentation).

