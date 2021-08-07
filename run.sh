#单机单卡下进行训练
python -m paddle.distributed.launch --gpus "0" main.py --mode train