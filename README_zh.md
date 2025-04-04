
# dive-into-stable-diffusion-v3-5

[English](./README.md)

[Stable Diffusion v3.5原理介绍](https://blog.csdn.net/sinat_16020825/article/details/146406963)

## 简介

本项目是用于微调 Stable Diffusion v3.5 的训练代码，基于 [diffusers库](https://github.com/huggingface/diffusers/) 进行适配开发。


<p align="center">
    <img src="imgs/A-capybara-holding-a-sign-that-reads-Hello-World.png" width="200px" alt="A capybara holding a sign that reads Hello World"/>
</p>

项目包含以下功能模块：

- SDv3.5 模型的全量微调
- 使用 LoRA 微调 SDv3.5 模型
- 结合 DreamBooth 与 LoRA 微调 SDv3.5 模型
- 基于 DDPO 和美学评分器的 RLHF（人类反馈强化学习）微调 SDv3.5 模型
- 基于 GRPO 和美学评分器的 RLHF 微调 SDv3.5 模型
- 基于 DPO 微调 SDv1.5 模型
- 基于 ReFL 和图文匹配评分器的 RLHF 微调 SDv1.5 模型

让我们一起 Dive Into Stable Diffusion v3.5 吧！

## 环境配置

`pip install -r requirements.txt`

## 项目结构

- `datas/` 数据集目录（从 HuggingFace Hub 下载），存放训练用的图片或提示文本
- `models/` 预训练模型目录（从 huggingface.co/models 下载）
- `outputs/` 输出目录，用于保存模型预测结果和训练检查点
- `scripts/` SDv3.5 训练主脚本目录
- `src/` 核心训练流程和训练器代码
- `demo.py / demo.sh` SDv3.5 推理示例
- `requirements.txt / setup.py` 基础依赖配置
- `train*.py` 核心训练脚本

## 下载模型与数据集
从 HuggingFace 或 GitHub 下载资源：

```bash
models
|-- aesthetics-predictor-v1-vit-large-patch14
|-- clip-vit-large-patch14
|-- improved-aesthetic-predictor
`-- stable-diffusion-3.5-medium
```

从以下地址下载 improved-aesthetic-predictor：
[improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor)

```bash
datas
|-- dogs
`-- pokemon
```

## 运行

- SDv3.5 全量微调
```bash
bash scripts/train_full_finetuning_sd3.sh
```

- LoRA 微调 SDv3.5
```bash
bash scripts/train_text_to_image_lora_sd3.sh
```

- DreamBooth + LoRA 微调 SDv3.5
```bash
bash scripts/train_dreambooth_lora_sd3.sh
```

- DDPO + 美学评分器 RLHF 微调
```bash
bash scripts/train_aesthetic_ddpo_sd3.sh
```

- GRPO + 美学评分器 RLHF 微调
```bash
# 注意：这部分代码可能存在问题，还需要完善。
bash scripts/train_aesthetic_rlhf_grpo_lora_sd3.sh
```

- DPO 微调 SDv1.5
```bash
bash scripts/train_dpo_sd_v1_5.sh
```

- ReFL + 图文匹配评分器的 RLHF 微调
```bash
bash scripts/train_refl_v1_5.sh
```

## 核心参数说明

### 通用参数
- `--pretrained_model_name_or_path` 预训练模型路径
- `--output_dir` 模型输出和日志目录
- `--seed` 训练随机种子（默认不设置）

### 优化器/学习率
- `--max_train_steps` 总训练步数
- `--gradient_accumulation_steps` 梯度累积步数
- `--train_batch_size` 实际批大小（具体参考脚本说明）
- `--checkpointing_steps` 模型保存间隔步数
- `--gradient_checkpointing` 自动为 SDXL 启用梯度检查点
- `--learning_rate` 基础学习率
- `--scale_lr` 学习率缩放（推荐启用但非默认）
- `--lr_scheduler` 学习率调度器类型（默认线性预热）
- `--lr_warmup_steps` 学习率预热步数

### 数据相关
- `--dataset_name` 数据集名称（来自 HuggingFace Hub）
- `--cache_dir` 本地数据集缓存路径（需根据文件系统调整）
- `--resolution` 输入分辨率（非 SDXL 默认 512，SDXL 默认 1024）
- `--random_crop` 和 `--no_hflip` 数据增强设置
- `--dataloader_num_workers` 数据加载器工作线程数
