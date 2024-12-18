# Prefix-LoRA: Combining Prefix Tuning with LoRA
- 代码核心部分与注释详见./seq2seq/train_bart.py 与 ./seq2seq/finetune.py
- (部分参考自 https://github.com/XiangLi1999/PrefixTuning.git)
-----------------------------------------------------
## Prerequisites:
- Python 3.8
- Pytorch-lightning 0.9.0
- Pytorch
  
## Setup:

``cd transformer; pip install -e .; Download xsum dataset``

-----------------------------------------------------
# Train with BART model
## Hybrid Method:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning yes --use_lora yes --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```
## Prefix-Tuning:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning yes --use_lora no --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```
## LoRA:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning no --use_lora yes --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```
# Solving NLU Problems 
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning {same as training} --use_lora {same as training} --mode xsum --do_train no --prefix_model_path {checkpoint_path} --preseqlen {same as training} --mid_dim {same as training}
```
# Interfaces
## Training Interfaces are as follow:
<img src="run.png" alt="run.png">
<img src="train.png" alt="train.png">

## Solving NLU Problems Interfaces are as follow:
<img src="decode.png" alt="decode.png">
