# McQueen: A Transformer-based multimodal query rewrite benchmark

## Setup
```bash
# Create python environment (optional)
conda create -n vlt5 python=3.7
source activate vlt5

# Install python dependencies
pip install -r requirements.txt

# Download T5/BART backbone checkpoint
python download_backbones.py

# Run feature extraction to extract image features
./feature_extraction

# Train VL-T5
./VL-T5/
    src/
        modeling_t5.py modeling_bart.py                       <= VL-T5/VL-BART model classes
        pretrain.py, pretrain_data.py, pretrain_model.py      <= pretraining
        vqa.py, vqa_data.py vqa_model.py ...                  <= fine-tuning on downstream tasks (ex. VQA, GQA, NLVR2)
        multitask.py, multitask_data.py multiask_model.py     <= multitask learning on 7 downstream tasks
        param.py                                              <= (argparse) configuration
        tokenization.py                                       <= custom tokenizer
        utils.py, dist_utils.py                               <= utility functions
    snap/                                                     <= store weight checkpoints
    scripts/                                                  <= bash scripts for pretraining and finetuning
```

## Dataset
The image files can be found in [link]()

The textual files can be found in [link]()



## Download Pre-trained models / Pre-extracted features
We host model checkpoints and features via google drive.
We recommend using [gdrive](https://github.com/prasmussen/gdrive) to download them.

## Pretrained Models
- Download `snap/` from [Google Drive](https://drive.google.com/drive/folders/1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph?usp=sharing)
```bash
gdrive download 1_SBj4sZ0gUqfBon1gFBiNRAmfHv5w_ph --recursive
```


## Downstream tasks

### [Query Rewrite]
First replace the generation_utils.py to the Huggingface transformers package installed in your device.
```bash
mv generation_utils.py [your path]/transformers/
```

Then start fine-tuning
```bash
# Finetuning with 4 gpus
cd VL-T5/
bash scripts/QueryRewrite_VLT5.sh 4
bash scripts/QueryRewrite_VLBart.sh 4
```
