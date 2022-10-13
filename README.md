install requirements
```
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
```
train our model:
```
python train_abstractor.py --wandb_log
```

decode

```
python decode_abstractor.py  --model_dir ckpts
```


evaluate 

```
python eval_full_model.py  --decode_dir ckpts 
```