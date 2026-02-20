# DynaCLR

Self-supervised contrastive learning for cellular dynamics from time-lapse microscopy.

Part of the [VisCy](https://github.com/mehta-lab/VisCy) monorepo.

> **DynaCLR: Dynamic Contrastive Learning of Representations for label-free assessment of live cells**
> Eduardo Hirata-Miyasaki, Shao-Chun Hsu, Tazim Buksh, Talley Lambert, Madhura Bhave, Syuan-Ming Guo, Manu Prakash, Shalin B. Mehta
>
> [arXiv:2506.18420](https://arxiv.org/abs/2506.18420)

## Installation

```bash
# From the VisCy monorepo root
uv pip install -e "applications/dynacrl"

# With evaluation extras (PHATE, UMAP, etc.)
uv pip install -e "applications/dynacrl[eval]"
```

## Examples

| Example | Description |
|---------|-------------|
| [Quick start](examples/quickstart/) | Get started with model inference |
| [Infection analysis](examples/demos/infection_analysis/) | Compare ImageNet vs DynaCLR embeddings for cell infection |
| [Embedding explorer](examples/demos/embedding_explorer/) | Interactive web-based embedding visualization |
| [Classical sampling](examples/data_preparation/classical_sampling/) | Generate pseudo-tracks for classical triplet sampling |
| [Configs](examples/configs/) | Training, prediction, and ONNX export configs |

## Datasets and Models

- [Test datasets](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_data/)
- [Pre-trained models](https://public.czbiohub.org/comp.micro/viscy/DynaCLR_models/)
