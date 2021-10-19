
# python main.py --batch-size 64 --saved-dir trained_models/imagenet/

# python main.py --isrsna --batch-size 16 --saved-dir trained_models/rsna/

# python main.py  --dist-url 'tcp://211.184.186.64:16023' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --saved-dir trained_models/imagenet/ --batch-size 256

python main.py  --isrsna --dist-url 'tcp://211.184.186.64:16023' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --saved-dir trained_models/rsna/ --batch-size 32
