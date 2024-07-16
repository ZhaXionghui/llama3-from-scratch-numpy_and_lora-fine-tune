# 将llama3的权重从.pth格式转换为.npy格式
import torch
import numpy as np
import os
def convert_llama3_pth_npz(pth_file, npz_home):
    # 加载llama3的.pth文件
    model = torch.load(pth_file)
    
    dict = {}
    keys = list(model.keys())
    # tok_embeddings
    key = list(model.keys())[0]
    print(key)
    # tensor_to_numpy
    dict[key] = model[key].detach().to(torch.float32).numpy()
    np.savez_compressed(npz_home + f"llama3.8b.shuke.{key}.npz", **dict)
    
    dict = {}
    # 遍历每一个层，将数据转为numpy格式保存到字典中
    cur_layer = '0'
    print(cur_layer)
    # print(keys[len(keys)-2])
    for key in keys[1:len(keys)-1]:
        if key.split('.')[1] != cur_layer:
            np.savez_compressed(npz_home + f"llama3.8b.shuke.layer.{cur_layer}.npz", **dict)
            cur_layer = key.split('.')[1]
            print(cur_layer)
            print(key)
            dict = {}
            dict[key] = model[key].detach().to(torch.float32).numpy()
        else:
            print(key)
            dict[key] = model[key].detach().to(torch.float32).numpy()
    
    np.savez_compressed(npz_home + f"llama3.8b.shuke.{key}.npz", **dict)

    key = keys[len(keys)-1]
    print(key)
    dict = {}
    dict[key] = model[key].detach().to(torch.float32).numpy()
    np.savez_compressed(npz_home + f"llama3.8b.shuke.{key}.npz", **dict)


Model_Home = "../Meta-Llama-3/Meta-Llama-3-8B/original/"
Npz_Home = Model_Home + "shuke/"

# 若路径不存在则创建
if not os.path.exists(Npz_Home):
    os.makedirs(Npz_Home)

# tokenizer_path = Model_Home + "tokenizer.model"
convert_llama3_pth_npz(Model_Home+'consolidated.00.pth', Npz_Home)


