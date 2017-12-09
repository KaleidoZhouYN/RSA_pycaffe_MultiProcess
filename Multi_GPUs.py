import multiprocessing
from multiprocessing import Process,freeze_support,Pool
import Work

def multiprocess(img_list,gpu_ids):
    gpu_num = len(gpu_ids)
    
    multiprocessing.freeze_support()
    pool = multiprocessing.Pool()
    
    rsa = []
    img_lists = []
    num_per_gpu = len(img_list) / gpu_num + 1
    for gpu_idx in range(gpu_num):
        start = gpu_idx*num_per_gpu
        end = (gpu_idx+1)*num_per_gpu 
        end = len(img_list) if (end > len(img_list)) else end
        img_lists.append(img_list[start:end])
        pool.apply_async(work,args=(gpu_ids[gpu_idx],img_lists[gpu_idx]))
    
    pool.close()
    pool.join()