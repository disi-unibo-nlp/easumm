install requirements
```
pip install -r requirements.txt
pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric
```

Download extracted events with DeepEventMine:
```
cd deep_event_mine
gdown 1x3oHfAKdtYfTEKuLPFTV_b2foA-VEMSx
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