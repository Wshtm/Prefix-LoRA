# Prefix-LoRA: Combining Prefix Tuning with LoRA

- (代码部分参考自 https://github.com/XiangLi1999/PrefixTuning.git)
-----------------------------------------------------
## Prerequisites:
- Python 3.8
- Pytorch-lightning 1.0
- Pytorch
  
## Setup:

``cd transformer; pip install -e .; Download xsum dataset``

-----------------------------------------------------
# Train & Solving NLU Problems
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

# Interfaces
### Interfaces are as follow:
<img src="run.png" alt="run.png">
<img src="train.png" alt="train.png">
