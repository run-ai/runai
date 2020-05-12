import runai.elastic

def init(global_batch_size, max_gpu_batch_size, gpus=None):
    if gpus is None:
        import torch.cuda
        gpus = torch.cuda.device_count()

    runai.elastic._init(global_batch_size, max_gpu_batch_size, gpus)
