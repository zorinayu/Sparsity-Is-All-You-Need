# Sparge Attention
This repository provides the official implementation of SpargeAttn.

<div align="center"> 
<h2>SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference </h2>
<a href="https://huggingface.co/papers/2502.18137"><img src="https://img.shields.io/static/v1?label=Daily papers&message=HuggingFace&color=yellow"></a>
<a href='https://arxiv.org/abs/2502.18137'><img src='https://img.shields.io/badge/arXiv-2502.15894-b31b1b.svg'></a> &nbsp;
</div>

<div align="center">
    <a href="https://jt-zhang.github.io/" target="_blank">Jintao Zhang</a><sup></sup> | 
    <a href="https://xiang-cd.github.io/cv" target="_blank">Chendong Xiang</a><sup></sup> | 
    <a href="" target="_blank">Haofeng Huang</a><sup></sup> | 
    <a href="" target="_blank">Haocheng Xi</a><sup></sup>|
    <a href="" target="_blank">Jia Wei</a><sup></sup> | 
    <a href="https://ml.cs.tsinghua.edu.cn/~jun/index.shtml" target="_blank">Jun Zhu</a><sup></sup> |
    <a href="https://ml.cs.tsinghua.edu.cn/~jianfei" target="_blank">Jianfei Chen</a><sup></sup>
</div>

<!-- Jintao Zhang, Chendong Xiang, Haofeng Huang, Haocheng Xi, Jia Wei, Jun Zhu, Jianfei Chen -->

<p align="center">
<img src="./assets/speed_comparison.png" width="85%" alt="speed comparison.">
</p>

<p align="center">
<img src="./assets/overview.png" width="93%" alt="overview.">
</p>

## Installation
### Base environment
+ `python>=3.9`   , `torch>=2.3.0`
- `CUDA`:
  + `>=12.8` for Blackwell
  + `>=12.4` for fp8 support on Ada
  + `>=12.3` for fp8 support on Hopper
  + `>=12.0` for Ampere


### Install Package

```bash
python setup.py install   # or pip install -e .
```


## Avalible API
- `spas_sage2_attn_meansim_cuda`: SpargeAttn based on [SageAttention2](https://github.com/thu-ml/SageAttention).

- `spas_sage_attn_meansim_cuda`: SpargeAttn based on [SageAttention](https://github.com/thu-ml/SageAttention).



## Usage Examples
### CogVideoX

Tuning:  
```bash
# sequential tuning
python evaluate/cogvideo_example.py  --use_spas_sage_attn --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt --tune

# parallel tuning, this will use all gpu available on the machine 
# do hyperparameter tuning with head level parallel
# NOTE: this will use more GPU vram in the main process
python evaluate/cogvideo_example.py  --use_spas_sage_attn --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt --tune --parallel_tune
```

Inference:  
```bash
python evaluate/cogvideo_example.py  --use_spas_sage_attn --model_out_path evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt
```

> **Note:**
We provide pre-tuned hyper-parameters `CogVideoX-2b_0.06_0.07.pt` that allow running the inference script directly. However, for better performance in both speed and quality, we recommend re-tuning because the provided hyper-parameters are tuned with SpargeAttn based on SageAttention, whereas the default API is based on SageAttention2 now.

### LLama
The tuning and inference usage is similar to CogVideoX.

### Supported models
Here’s a list of the model modifications we’ve implemented so far. Our approach is universal, and we warmly welcome contributions! Feel free to submit a pull request to support more models. 🚀
| model name | example script | tuned ckpt |
| ---- | ---- | ---- |
| CogVideoX-2b | evaluate/cogvideo_example.py | evaluate/models_dict/CogVideoX-2b_0.06_0.07.pt 
| Flux  | evaluate/flux_example.py  | TBD 



## Performance
![Local Image](./assets/exp_table.png)
> **Note:** All experiments in the above Table and our paper used SpargeAttn based on SageAttention. An updated implementation based on SageAttention2, is available now. **It further offers a 30% speedup.**
<br>



<table>
  <tr>
    <td align="center">
      <img src="./assets/more_mochi_example.png" width="55%" alt="End-to-end video generation on Mochi.">
      <br>
      The quality of video generation on Mochi.
    </td>
    <td align="center">
      <img src="./assets/niah128k.png" width="100%" alt="End-to-end performance of NIAH.">
      <br>
      End-to-end performance of NIAH.
    </td>
  </tr>
</table>


<!-- <img src="./assets/visible_image.png" width="80%" alt="image generation."> -->



## Citation
**If you use this code or find our work valuable, please cite:**
```
@misc{zhang2025spargeattn,
      title={SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference}, 
      author={Jintao Zhang and Chendong Xiang and Haofeng Huang and Jia Wei and Haocheng Xi and Jun Zhu and Jianfei Chen},
      year={2025},
      eprint={2502.18137},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2502.18137}, 
}

@inproceedings{zhang2025sageattention,
      title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
      author={Zhang, Jintao and Wei, Jia and Zhang, Pengle and Zhu, Jun and Chen, Jianfei},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2025}
}

@misc{zhang2024sageattention2,
      title={SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization}, 
      author={Jintao Zhang and Haofeng Huang and Pengle Zhang and Jia Wei and Jun Zhu and Jianfei Chen},
      year={2024},
      eprint={2411.10958},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.10958}, 
}
```
