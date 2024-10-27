# CVMambaIR

### install Dep
```
uv sync

```

### train
```
CUDA_VISIBLE_DEVICES=1,2,3 && uv run torchrun --nproc_per_node=3 --master_port=2414 run/train.py -opt run/options/train/train_CVMambaIR_RealDN.yml --launcher pytorch

```