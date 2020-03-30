#TODO: ADD TEST PIPELINES

import os
from collections import OrderedDict
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, models
from model import *
# import pretrainedmodels
import faiss

from resnet import *
import pickle
import glob

# DATA_ROOT = './datasets/xuelang_round1_test_a_20180709'
# DATA_ROOT = './datasets/xuelang_round1_test_b'
DATA_ROOT = '/data/sdv2/taobao/data/embedding/val/'
# RESULT_FILE = 'result.pkl'


def test_and_generate_result(imgs,
                             feat_dim,
                             embed_dim,
                             batch_k,
                             normalize=True,
                             ckpt_name='resnet101',
                             img_size=224,
                             is_multi_gpu=False,
                             save_file=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
            # transforms.RandomResizedCrop(224),
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

    # os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    # is_use_cuda = torch.cuda.is_available()

    model = resnet50()
    try:
        model.fc = nn.Linear(model.fc.in_features, feat_dim)
    except NameError as e:
        print("Error: current works only with model having fc layer as the last layer, try modify the code")
        exit(-1)

    model = MarginNet(base_net=model, emb_dim=embed_dim, batch_k=batch_k, feat_dim=feat_dim,
                      normalize=normalize)

    state_dict = torch.load(ckpt_name, map_location='cpu')['state_dict']

    if is_multi_gpu:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()

    test_files_list = imgs
    embed_result = []
    for _file in test_files_list:
        file_name = _file
        if not file_name.endswith('.jpg'):
            continue
        img_tensor = data_transform(Image.open(_file).convert('RGB')).unsqueeze(0)
        img_tensor = img_tensor.cuda()
        output = model.embed(img_tensor).detach().squeeze(0)
        embed_result.append(output.cpu().numpy().tolist())

    if save_file is not None:
        with open(os.path.join('./results', save_file), 'wb') as fd:
            pickle.dump(embed_result, fd)
    return embed_result

def embed_match_eval(imgs, embed, embed_dim=256, k_nearest=3):
    Index = faiss.IndexFlatL2(embed_dim)
    embed_ = np.array(embed).astype('float32')
    # Index.add(embed_)
    categories = np.array([img.split('/')[-2] for img in imgs])
    images = np.array([img.split('/')[-1] for img in imgs])
    total_result = []
    gallery_cate = []
    gallery_imgs = []
    for cate, emb, img in zip(*(categories, embed_, images)):
        emb = np.array(emb).astype('float32')
        emb = emb[np.newaxis, :]
        if not img.startswith('v'):
            Index.add(emb)
            gallery_cate.append(cate)
            gallery_imgs.append(img)

    gallery_cate = np.array(gallery_cate)
    gallery_imgs = np.array(gallery_imgs)

    for cate, emb, img in zip(*(categories, embed_, images)):
        emb = np.array(emb).astype('float32')
        emb = emb[np.newaxis, :]
        if img.startswith('v'):
            D, I = Index.search(emb, k_nearest)
            real_idx = np.squeeze(I)
            result = gallery_cate[real_idx] == cate
            if isinstance(result, bool):
                result = [result]
            else:
                result = result.tolist()
            total_result.extend(result)
            print(img, gallery_imgs[real_idx][gallery_cate[real_idx] != cate])
    acc = np.sum(total_result) / len(total_result)
    print('accuracy: {:.2f}%'.format(acc*100))


if __name__ == '__main__':
    imgs = glob.glob(os.path.join(DATA_ROOT, '*/*.jpg'))
    embed = test_and_generate_result(imgs, 512, 128, 5, ckpt_name='/data/sdv2/taobao/embedding/checkpoints/normalize_checkpoint_12.pth.tar')
    embed_match_eval(imgs, embed, 128, 1)
