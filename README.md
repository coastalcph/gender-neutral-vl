# Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models

This is the code to replicate the experiments described in the paper (to appear in EMNLP23):
>Laura Cabello, Emanuele Bugliarello, Stephanie Brandl and Desmond Elliott. [Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models](https://arxiv.org/abs/2310.17530). In _Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing_.

## Repository Setup

You can clone this repository issuing: <br>
`git clone git@github.com:coastalcph/gender-neutral-vl.git`

1\. Create a fresh conda environment and install all dependencies.
```text
conda create -n genvlm python=3.9
conda activate genvlm
pip install -r requirements.txt
```

2\. Install PyTorch
```text
conda install pytorch=1.12.0=py3.9_cuda11.3_cudnn8.3.2_0 torchvision=0.13.0=py39_cu113 cudatoolkit=11.3 -c pytorch
```

Following steps are required in order to run code from [VOLTA](https://github.com/e-bug/volta/tree/main):

3\. Install [apex](https://github.com/NVIDIA/apex).
If you use a cluster, you may want to first run commands like the following:
```text
module load cuda/10.1.105
module load gcc/8.3.0-cuda
```

4\. Setup the `refer` submodule for Referring Expression Comprehension:
```
cd src/LXMERT/volta/tools/refer; make
```

5\. Install this codebase as a package in this environment.
```text
python setup.py develop
```

## Repository Config

The main configuration needed to run the scripts in the [experiments/](experiments/) folder is stored in [main.config](main.config). Please, edit this file at your own convenience.


## Data

You can download the preprocessed **gender-neutral** data from [here](https://sid.erda.dk/sharelink/evl10QgflV). These data files are used for continued pretraining on gender-neutral data.

Details on the method used to generate this data can be found in the paper. The mappings between gendered words and neutral words is in [Mappings.csv](src/bias/Mappings.csv) (and in Appendix A). The code to reproduce our preprocessing pipeline or apply it to your own data is stored in [src/preprocessing/](src/preprocessing). Scripts to run the code are in [experiments/preprocessing/](experiments/preprocessing).

Lists of common nouns that co-occur with gender entities in the corresponding training data are stored in [src/preprocessing/top_objects/](src/preprocessing/top_objects). The top-N *objects* are used to evaluate bias amplification (N=100 to measure bias in pretraining, N=50 to measure bias in downstream tasks). See Section 4.2 and Section 5.3 for details.

\* Note that we use the same COCO train split used for pretraining [LXMERT](https://github.com/airsplay/lxmert#pre-training), which is different from the original COCO train split or the Karpathy split.

\* Note that our CC3M files map captions to image ids obtained from filenames as done in [VOLTA](https://github.com/e-bug/volta/tree/main/data/conceptual_captions#conceptual-captions).

## Models

Our pretrained models can be downloaded from [here](https://sid.erda.dk/sharelink/aQxK7MVAjw), where `third_party/` contains the original weights, while `Pretrain_{CC3M,COCO}_neutral/` contain the weights after continued pretraining on gender-neutral data.

Model configuration depend on the model family. Files are stored in:

* LXMERT: [src/LXMERT/volta/config/](src/LXMERT/volta/config/)
* ALBEF: [src/ALBEF/configs/](src/ALBEF/configs)
* BLIP: [src/BLIP/configs/](src/BLIP/configs)

## Training and Evaluation

We provide bash scripts to train (i.e. continued pretraining or fine-tuning) and evaluate models in [experiments/](experiments).
These include the following models, as specified in our experimental setup (Section 5):

![Alt text](image.png)


Task configuration files are stored in:

* LXMERT: [src/LXMERT/volta/config_tasks/](src/LXMERT/volta/config_tasks/)
* ALBEF: [src/ALBEF/configs/](src/ALBEF/configs)
* BLIP: [src/BLIP/configs/](src/BLIP/configs)


Code to plot results is shared in Jupyter Notebooks in [notebooks/](notebooks).


## License 

This work is licensed under the MIT license. See [`LICENSE`](LICENSE) for details. 
Third-party software and data sets are subject to their respective licenses. <br>
If you find our code/data/models or ideas useful in your research, please consider citing the paper:
```
@inproceedings{cabello-etal-2023-evaluating,
    title = "Evaluating Bias and Fairness in Gender-Neutral Pretrained Vision-and-Language Models",
    author = "Cabello, Laura  and
      Bugliarello, Emanuele   and
      Brandl, Stephanie  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore, Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/2310.17530",
}
```


## Acknowledgement

Our codebase heavily relies on these excellent repositories:
- [VOLTA](https://github.com/e-bug/volta/tree/main)
- [LXMERT](https://github.com/airsplay/lxmert)
- [ALBEF](https://github.com/salesforce/ALBEF)
- [BLIP](https://github.com/salesforce/BLIP)
- [directional-bias-amp](https://github.com/princetonvisualai/directional-bias-amp/tree/main)
