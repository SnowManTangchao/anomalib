<div align="center">

<img src="https://raw.githubusercontent.com/open-edge-platform/anomalib/main/docs/source/_static/images/logos/anomalib-wide-blue.png" width="600px" alt="Anomalib 标志 - 用于异常检测的深度学习库">

**一个用于基准测试、开发与部署深度学习异常检测算法的库**

英文说明见 [README.md](README.md)。

---

[核心特性](#核心特性) •
[文档](https://anomalib.readthedocs.io/en/latest/) •
[示例 Notebook](examples/notebooks) •
[许可证](LICENSE)

![python](https://img.shields.io/badge/python-3.10%2B-green)
![pytorch](https://img.shields.io/badge/pytorch-2.10%2B-orange)
![lightning](https://img.shields.io/badge/lightning-2.6%2B-blue)
![openvino](https://img.shields.io/badge/openvino-2024.0%2B-purple)

[![Pre-Merge Checks](https://github.com/open-edge-platform/anomalib/actions/workflows/pre_merge.yml/badge.svg)](https://github.com/open-edge-platform/anomalib/actions/workflows/pre_merge.yml)
[![codecov](https://codecov.io/gh/open-edge-platform/anomalib/branch/main/graph/badge.svg?token=Z6A07N1BZK)](https://codecov.io/gh/open-edge-platform/anomalib)
[![Downloads](https://static.pepy.tech/personalized-badge/anomalib?period=total&units=international_system&left_color=grey&right_color=green&left_text=PyPI%20Downloads)](https://pepy.tech/project/anomalib)
[![snyk](https://snyk.io/advisor/python/anomalib/badge.svg)](https://snyk.io/advisor/python/anomalib)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/8330/badge)](https://www.bestpractices.dev/projects/8330)

[![ReadTheDocs](https://readthedocs.org/projects/anomalib/badge/?version=latest)](https://anomalib.readthedocs.io/en/latest/?badge=latest)
[![Anomalib - Gurubase docs](https://img.shields.io/badge/Gurubase-Ask%20Anomalib%20Guru-006BFF)](https://gurubase.io/g/anomalib)

<a href="https://trendshift.io/repositories/14514" target="_blank" rel="noopener noreferrer"><img src="https://trendshift.io/api/badge/repositories/14514" alt="open-edge-platform%2Fanomalib | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>

</div>

---

> 🌟 **宣布 v2.7.0 发布！** 🌟
>
> 此小版本增加了新模型，并加固了 checkpoint 与路径处理，同时移除了计划在 2.7.0 中删除的 API。
>
> 主要变更
>
> - **新模型**：MH-PatchCore、RAD（基于检索，retrieval-based）和 FoundAD（小样本，few-shot）。
> - **安全 / API**：加载 checkpoint 时默认 `weights_only=True`。当设置了数据集的 `root` 时，元数据路径必须解析到该 root 之下。
> - **移除项**：旧版 ONNX 导出器（`dynamo=False`）已移除——请使用 `dynamo=True`（默认值），并配合 `anomalib[openvino]`。不再支持 pre-timm ViT checkpoint 迁移。PyTorch 最低版本为 2.10+（ROCm 为 2.13+）。
>
> 我们重视您的意见！请通过 [GitHub Issues](https://github.com/open-edge-platform/anomalib/issues) 或我们的 [Discussions](https://github.com/open-edge-platform/anomalib/discussions) 分享反馈。

# 👋 简介

Anomalib 是一个深度学习库，旨在汇集最先进的异常检测算法，以便在公开数据集和私有数据集上进行基准测试。Anomalib 提供了近期文献中若干异常检测算法的开箱即用实现，以及一组有助于开发并实现自定义模型的工具。该库重点关注视觉异常检测，即算法的目标是检测和/或定位数据集中图像或视频里的异常。Anomalib 会持续加入新算法以及训练/推理扩展，欢迎持续关注！

<p align="center">
  <img src="https://raw.githubusercontent.com/open-edge-platform/anomalib/main/docs/source/_static/images/readme.png" width="1000" alt="由 anomalib 做出的预测">
</p>

## 核心特性

- 简洁且模块化的 API 与 CLI，用于训练、推理和基准测试。
- 规模最大的公开集合，包含开箱即用的深度学习异常检测算法与基准数据集。
- 基于 [**Lightning**](https://www.lightning.ai/) 的模型实现，用以减少样板代码，并将实现工作限制在最必要的部分。
- 大多数模型可导出为 [**OpenVINO**](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html) 中间表示（Intermediate Representation，IR），以便在 Intel 硬件上加速推理。
- 一套[推理工具](tools)，用于快速、便捷地部署标准或自定义异常检测模型。

# 📦 安装

可以从 PyPI 安装 Anomalib。建议使用虚拟环境，以及 `uv` 或 `pip` 这类现代包安装器。

## 🚀 快速安装

进行标准安装时，可以使用 `uv` 或 `pip`。这将安装最新版本的 Anomalib 及其核心依赖。PyTorch 会按其默认行为安装，通常适用于 CPU 和标准 CUDA 环境。

```bash
# 使用 uv
uv pip install anomalib

# 或使用 pip
pip install anomalib
```

若要更精细地控制安装，例如指定 PyTorch 后端（如 XPU、CUDA 和 ROCm），或为特定模型安装额外依赖，请参阅下面的高级选项。

<details>
<summary><strong>💡 高级安装：指定硬件后端</strong></summary>

为确保与所用硬件兼容，可以在安装时指定后端。对于生产环境，以及 CPU 或标准 CUDA 以外的硬件，建议采用这种方式。

**使用 `uv`：**

```bash
# CPU 支持（默认，适用于所有平台）
uv pip install "anomalib[cpu]"

# CUDA 12.6 支持（带 NVIDIA GPU 的 Linux/Windows）
uv pip install "anomalib[cu126]"

# CUDA 13.0 支持（带 NVIDIA GPU 的 Linux/Windows）
uv pip install "anomalib[cu130]"

# ROCm 支持（带 AMD GPU 的 Linux）
uv pip install "anomalib[rocm]"

# Intel XPU 支持（带 Intel GPU 的 Linux/Windows）
uv pip install "anomalib[xpu]"
```

**使用 `pip`：**
`pip` 可以使用相同的 extra：

```bash
pip install "anomalib[cu130]"
```

</details>

<details>
<summary><strong>🧩 高级安装：额外依赖</strong></summary>

Anomalib 默认包含大多数依赖。若使用专门功能，可能还需要其他可选依赖。请记得一并包含与硬件对应的 extra。

```bash
# 示例：安装 OpenVINO 支持以及 CUDA 13.0
uv pip install "anomalib[openvino,cu130]"

# 示例：为仅 CPU 的环境安装全部可选依赖
uv pip install "anomalib[full,cpu]"
```

以下是可用的可选依赖组：

| Extra           | 说明                            | 用途                                                                                  |
| :-------------- | :------------------------------ | :------------------------------------------------------------------------------------ |
| `[openvino]`    | Intel OpenVINO 优化             | 用于在 Intel 硬件上加速推理                                                           |
| `[clip]`        | 视觉-语言模型                   | `winclip`                                                                             |
| `[vlm]`         | 视觉-语言模型后端               | 高级 VLM 功能                                                                         |
| `[loggers]`     | 实验跟踪（wandb、comet 等）     | 用于实验管理                                                                          |
| `[notebooks]`   | Jupyter notebook 支持           | 用于运行示例 notebook                                                                 |
| `[video]`       | 视频解码                        | 视频数据集，例如 Avenue                                                               |
| `[huggingface]` | Hugging Face Hub 与 safetensors | 由 Hub 提供的模型与 checkpoint                                                        |
| `[datasets]`    | 数据集存储 extra                | 诸如需要 `pyarrow` 的格式                                                             |
| `[full]`        | 可选功能依赖                    | `clip`、`datasets`、`openvino`、`loggers`、`notebooks`、`vlm`、`video`、`huggingface` |

</details>

<details>
<summary><strong>🔧 高级安装：从源码安装</strong></summary>

若要为 `anomalib` 做贡献，或使用开发版本，可以从源码安装。

**使用 `uv`：**
这是推荐给开发者的方式，因为它使用项目的 lock 文件，以获得可复现的环境。

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# 创建虚拟环境
uv venv

# 针对特定后端与 lock 文件同步（例如 CPU）
uv sync --extra cpu

# 或用于其他后端，例如 CUDA 13.0
uv sync --extra cu130

# 搭建完整的开发环境
uv sync --extra dev --extra cpu
```

**使用 `pip`：**

```bash
git clone https://github.com/open-edge-platform/anomalib.git
cd anomalib

# 以可编辑模式安装，并指定特定后端
pip install -e ".[cpu]"

# 连同开发依赖一起安装
pip install -e ".[dev,cpu]"
```

</details>

# 🧠 训练

Anomalib 同时支持基于 API 和基于 CLI 的训练方式：

## 🔌 Python API

```python
from anomalib.data import MVTecAD
from anomalib.models import Patchcore
from anomalib.engine import Engine

# 初始化各组件
datamodule = MVTecAD()
model = Patchcore()
engine = Engine()

# 训练模型
engine.fit(datamodule=datamodule, model=model)
```

## ⌨️ 命令行

```bash
# 使用默认设置进行训练
anomalib train --model Patchcore --data anomalib.data.MVTecAD

# 使用自定义类别进行训练
anomalib train --model Patchcore --data anomalib.data.MVTecAD --data.category transistor

# 使用配置文件进行训练
anomalib train --config path/to/config.yaml
```

# 🤖 推理

Anomalib 提供多种推理方式，包括 Torch、Lightning、Gradio 和 OpenVINO。入门步骤如下：

## 🔌 Python API

```python
# 加载模型并做出预测
predictions = engine.predict(
    datamodule=datamodule,
    model=model,
    ckpt_path="path/to/checkpoint.ckpt",
)
```

## ⌨️ 命令行

```bash
# 基本预测
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt

# 返回预测结果
anomalib predict --model anomalib.models.Patchcore \
                 --data anomalib.data.MVTecAD \
                 --ckpt_path path/to/model.ckpt \
                 --return_predictions true
```

> 📘 **说明：** 有关包括 Gradio 和 OpenVINO 在内的高级推理选项，请查阅我们的[推理文档](https://anomalib.readthedocs.io)。

# 在 Intel GPU 上训练

> [!Note]
> 目前，Intel GPU 上仅支持单 GPU 训练。
> 这些命令已在 Arc 750 和 Arc 770 上测试通过。

请确保已安装带 XPU 支持的 PyTorch。更多信息请参阅 [PyTorch XPU 文档](https://pytorch.org/docs/stable/notes/get_start_xpu.html)。

## 🔌 API

```python
from anomalib.data import MVTecAD
from anomalib.engine import Engine, SingleXPUStrategy, XPUAccelerator
from anomalib.models import Stfpm

engine = Engine(
    strategy=SingleXPUStrategy(),
    accelerator=XPUAccelerator(),
)
engine.train(Stfpm(), datamodule=MVTecAD())
```

## ⌨️ CLI

```bash
anomalib train --model Padim --data MVTecAD --trainer.accelerator xpu --trainer.strategy xpu_single
```

# 🧪 实验管理

通过 [PyTorch Lightning loggers](https://lightning.ai/docs/pytorch/stable/extensions/logging.html)，使用常见的日志平台跟踪实验：

- 📊 Weights & Biases
- 📈 Comet.ml
- 📉 TensorBoard

在配置文件中启用日志，以跟踪：

- 超参数
- 指标
- 模型图
- 测试预测

> 📘 **说明：** 有关日志配置，请参阅我们的[日志文档](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/loggers/index.html)。

# 📊 基准测试

在不同数据集上评估并比较模型性能：

```bash
# 使用默认配置运行基准测试
anomalib benchmark --config tools/experimental/benchmarking/sample.yaml
```

> 💡 **提示：** 请在各模型对应的 README 文件中查看其性能：
>
> - [Patchcore 结果](src/anomalib/models/image/patchcore/README.md#mvtec-ad-dataset)
> - [其他模型](src/anomalib/models/)

# Anomalib Studio

> [!IMPORTANT]
> Anomalib Studio 目前正在积极开发中，应视为预发布版本。功能可能发生变化，部分功能可能不完整或不稳定。在迈向稳定版本的过程中，我们欢迎反馈与贡献。

Anomalib Studio 是一个低代码/无代码 Web 应用，用户可以用它训练并部署异常检测模型。它让用户能够在自己的运行环境中使用 Anomalib 的功能。用户可以连接 USB 摄像头和 IP 摄像头，或使用一个图像文件夹，作为训练流程的输入。该工具可以通过 ROS 消息、MQTT 等方式，把结果直接输出到工业流水线。

<p align="center">
  <img src="docs/source/_static/images/anomalib_studio.png" alt="Anomalib Studio" />
</p>

Anomalib Studio 的源代码位于 [application](application) 目录中。

Anomalib Studio 有两种发行形式：

1. [Docker 容器](application/docker)
2. [独立应用](application/ui)

关于每一种的更多信息，请参阅各自的 README 文件。

## 开始使用开发构建

### 配置后端依赖

> [!NOTE]
> 这里假定您已经安装了 `uv`。如果尚未安装，请参阅[安装指南](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
cd application/backend
uv sync --extra xpu # 或者，CUDA 13.0 使用 uv sync --extra cu130，CPU 使用 uv sync --extra cpu
```

### 配置前端依赖

```bash
cd application/ui
npm install
```

### 运行应用

> [!IMPORTANT]
> 运行应用之前，应先安装后端和前端依赖。

运行后端服务器：

```bash
cd application/backend
./run.sh
```

运行前端服务器：

```bash
cd application/ui
npm run start
```

打开 [http://localhost:3000](http://localhost:3000) 即可看到应用。

# ✍️ 参考文献

如果 Anomalib 对您的研究或工作有帮助，请引用：

```tex
@inproceedings{akcay2022anomalib,
  title={Anomalib: A deep learning library for anomaly detection},
  author={Akcay, Samet and Ameln, Dick and Vaidya, Ashwin and Lakshmanan, Barath and Ahuja, Nilesh and Genc, Utku},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={1706--1710},
  year={2022},
  organization={IEEE}
}
```

# 👥 参与贡献

我们欢迎贡献！请查看[贡献指南](CONTRIBUTING.md)以开始参与。

<p align="center">
  <a href="https://github.com/open-edge-platform/anomalib/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=open-edge-platform/anomalib" alt="open-edge-platform/anomalib 的贡献者" />
  </a>
</p>

<p align="center">
  <b>感谢所有贡献者！</b>
</p>
