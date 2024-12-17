# Prefix-LoRA: Combining Prefix Tuning with LoRA

-----------------------------------------------------
## Setup:

``cd transformer; pip install -e .``

-----------------------------------------------------
## Train via Hybrid Method:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning yes --use_lora yes --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```
## Train via Prefix-Tuning:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning yes --use_lora no --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```
## Train via LoRA:
```python
cd seq2seq; 

python train_bart.py --use_prefix_tuning no --use_lora yes --mode xsum --preseqlen 200 --do_train yes --fp16 yes --bsz 2  --epoch 30  --gradient_accumulation_step 3 --learning_rate 0.00005  --mid_dim 800
```


