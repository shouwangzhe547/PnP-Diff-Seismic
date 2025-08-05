# PnP-Diff-Seismic


PnP-Diff-Seismic is detailed in the paper titled "Plug-and-Play Post-Stack Seismic Inversion with Denoising Diffusion Model", which is currently under review.  Updates regarding the paper's publication status will be provided in due course.

This code is based on the [Deepinv](https://github.com/deepinv/deepinv),
[PyLops](https://github.com/PyLops/pylops) and [DPIR](https://github.com/yuanzhi-zhu/DiffPIR).


## Getting Started

### Requirements

The code requires **Python >= 3.9** and the following dependencies:

```
numpy==1.23.5
deepinv==0.2.2
pylops==2.3.1
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118
tqdm==4.67.1
scipy==1.13.1
scikit-image==0.24.0
```

### Installation

1. Clone this repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Code

To reproduce the main results of the paper, simply run:

```bash
jupyter notebook
```

and open the file:

```
PnP-Diff-Seismic.ipynb
```

All results will be generated automatically.

## Contact

For questions or collaboration, please contact:

**Lu Li**  
📧 2023710149@yangtzeu.edu.cn
