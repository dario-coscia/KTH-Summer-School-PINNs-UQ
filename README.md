# KTH-Summer-School-PINNs-PINA

This repository contains the material for the lecture *Uncertainty Quantification in Scientific Machine Learning* at the Physics-Informed Neural Networks and Applications summer school at KTH 2025. Slides available [here](https://www.canva.com/design/DAGql727GXw/JZ3gbH91MNLNTjCDirxb-g/edit?utm_content=DAGql727GXw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton).

## Getting Started:
 
Follow the steps below to set up and run the scripts.

## Installation:
```bash
git clone https://github.com/dario-coscia/KTH-Summer-School-PINNs-UQ.git
cd KTH-Summer-School-PINNs-UQ
bash create_env.sh
source venv/bin/activate
```

## Running Scripts:
```bash
python scripts/[file].py
```
where `[file]` might be:
* [bayes_by_backprop](scripts/bayes_by_backprop.py)
* [deep_ensemble](scripts/deep_ensemble.py)
* [hmc](scripts/hmc.py)
* [mc_dropout](scripts/mc_dropout.py)
* [variational_dropout](scripts/variational_dropout.py)

