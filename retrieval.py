#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cv2
import faiss
import time, glob
import pickle, json
import numpy as np
from mmdet.apis import inference_detector, init_detector
from PIL import Image
from model import *
from torchvision import transforms
from resnet import *


class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.float32):
            return float(o)


def gallery_results_from_pkl(imgs,
                             pkl_file,
                             score_thr,
                             out_json=None):
    with open(pkl_file, 'rb') as f:
        results = pickle.load(f)
    assert len(imgs) == len(results)

    category_dict = {}
    for i, (img, result) in enumerate(zip(imgs, results)):
        filename = img.split('/')[-1][:-4]
        item_id, img_name = filename.split('_')

        for idx, (bboxes, embeds) in enumerate(zip(*result)):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox, embed in zip(bboxes, embeds):
                    conf = bbox[4]
                    if conf > score_thr:
                        category_dict.setdefault(category_id, []).append(
                            [item_id] + [img_name] + list(bbox) + [embed]
                        )
    if out_json is not None:
        with open(out_json, 'w') as f:
            json.dump(category_dict, f, indent=4, cls=NumpyEncoder)
    return category_dict


def get_embed_model(feat_dim,
                    embed_dim,
                    batch_k,
                    ckpt_name,
                    model_name='resnet50',
                    normalize=True):
    if model_name == 'resnet50':
        model = resnet50()
    else:
        raise NotImplementedError

    try:
        model.fc = nn.Linear(model.fc.in_features, feat_dim)
    except NameError as e:
        print("Error: current works only with model having fc layer as the last layer, try modify the code")
        exit(-1)

    model = MarginNet(base_net=model, emb_dim=embed_dim, batch_k=batch_k, feat_dim=feat_dim,
                      normalize=normalize)

    state_dict = torch.load(ckpt_name, map_location='cpu')['state_dict']

    model.load_state_dict(state_dict)

    model = model.cuda()
    model.eval()
    return model


def get_detection_model(cfg, ckpt):
    print('\nLoading model...')
    model = init_detector(cfg, ckpt, device='cuda:0')
    return model


def gallery_results(imgs,
                    model,
                    embed_model,
                    score_thr,
                    data_transform,
                    out_json=None):
    category_dict = {}
    for i, img in enumerate(imgs):
        item_id = img.split('/')[-2]
        img_name = img.split('/')[-1][:-4]
        result = inference_detector(model, img)
        img_data = Image.open(img).convert('RGB')

        for idx, bboxes in enumerate(result):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox in bboxes:
                    conf = bbox[4]
                    if conf > score_thr:
                        cut_img = img_data.crop(bbox[:4])
                        img_tensor = data_transform(cut_img).unsqueeze(0)
                        img_tensor = img_tensor.cuda()
                        output = embed_model.embed(img_tensor).detach().squeeze(0)
                        output = output.cpu().numpy().tolist()

                        category_dict.setdefault(category_id, []).append(
                            [item_id] + [img_name] + list(bbox) + [output]
                        )
    if out_json is not None:
        with open(out_json, 'w') as f:
            json.dump(category_dict, f, indent=4, cls=NumpyEncoder)
    return category_dict


def capture_video_imgs(video_file, interval=40):
    vc = cv2.VideoCapture(video_file)
    video_id = video_file.split('/')[-1][:-4]
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
    inds = np.arange(0, 400, interval)
    c = 0
    imgs_dict = {}
    while rval:
        rval, frame = vc.read()
        if c in inds:
            imgs_dict[c] = frame
        c += 1
        cv2.waitKey(1)
    vc.release()
    return video_id, imgs_dict


def single_video_query(video_imgs,
                       gallery_dict,
                       model,
                       model_embed,
                       data_transform,
                       score_thr,
                       k_nearest,
                       iou_thr):
    frame_res = []
    for frame_index, img in video_imgs.items():
        result = inference_detector(model, img)
        img_data = Image.open(img).convert('RGB')
        for idx, bboxes in enumerate(result):
            category_id = idx + 1
            if len(bboxes) != 0:
                for bbox in bboxes:
                    conf = bbox[4]
                    if conf > score_thr:
                        cut_img = img_data.crop(bbox[:4])
                        img_tensor = data_transform(cut_img).unsqueeze(0)
                        img_tensor = img_tensor.cuda()
                        output = model_embed.embed(img_tensor).detach().squeeze(0)
                        output = output.cpu().numpy().tolist()
                        frame_res.append(
                            [frame_index] + [category_id] + list(bbox) + [output]
                        )
    gallery_res = []
    item_id_lst = []
    for res in frame_res:
        query_embed = res[-1][np.newaxis, :].astype('float32')
        gallerys = gallery_dict[res[1]]
        gallery_embeds = [lst[-1] for lst in gallerys]
        gallery_embeds = np.array(gallery_embeds).astype('float32')

        d = gallery_embeds.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(gallery_embeds)
        D, I = index.search(query_embed, k_nearest)  # Distance, Index
        topk = [gallerys[ind] for ind in I[0]]
        item_id_lst.extend([k[0] for k in topk])
        gallery_res.append(topk)

    # choose item_id according to frequency
    item_id = max(item_id_lst, key=item_id_lst.count)
    frame_bbox, item_bbox = [], []
    for frame, gallery in zip(frame_res, gallery_res):  # 1 vs k
        matched_items = [item for item in gallery if item[0] == item_id]
        if len(matched_items) > 0:
            scores = [item[6] for item in matched_items]
            inx = int(np.argmax(np.array(scores)))
            best_item = matched_items[inx]
            item_bbox.append(best_item)
            frame_bbox.append(frame)
    assert len(frame_bbox) == len(item_bbox)  # 1 vs 1

    # choose frame_index according to frequency
    frame_inds = [frame[0] for frame in frame_bbox]
    frame_index = max(frame_inds, key=frame_inds.count)
    keep_inds = np.where(np.array(frame_inds) == frame_index)
    frame_bbox = np.array(frame_bbox)[keep_inds]
    item_bbox = np.array(item_bbox)[keep_inds]

    # merge overlap frame_bbox based on IOU
    # sort (frame, item) pairs by frame bbox scores
    pairs = sorted(zip(frame_bbox, item_bbox),
                   key=lambda x: x[0][6], reverse=True)
    best_frame_bbox = []
    best_item_bbox = []
    for i, (frame, item) in enumerate(pairs):
        if i == 0:
            best_frame_bbox.append(frame)
            best_item_bbox.append(item)
        else:
            exist_bboxes = [bbox[2:6] for bbox in best_frame_bbox]
            iou = bboxes_iou(frame[2:6], exist_bboxes)
            if np.all(iou < iou_thr):
                best_frame_bbox.append(frame)
                best_item_bbox.append(item)
            else:
                ind = int(np.argmax(iou))
                # merge frame bbox coordinates by mean value
                best_frame_bbox[ind][2:6] = np.mean(
                    [best_frame_bbox[ind][2:6], frame[2:6]], axis=0)
                # keep the item bbox with highest score
                best_item_bbox[ind] = \
                    best_item_bbox[ind] if best_item_bbox[ind][6] > item[6] else item
    assert len(best_frame_bbox) == len(best_item_bbox)

    result_lst = []
    for frame, item in zip(best_frame_bbox, best_item_bbox):
        d = {
            "img_name": item[1],
            "item_bbox": list(map(int, item[2:6])),
            "frame_bbox": list(map(int, frame[2:6]))
        }
        result_lst.append(d)
    output = {
        "item_id": item_id,
        "frame_index": frame_index,
        "result": result_lst
    }
    return output


def bboxes_iou(boxes1, boxes2):
    """
    boxes: [xmin, ymin, xmax, ymax] format coordinates.
    """
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, 0.0)

    return ious


if __name__ == '__main__':
    imgs = glob.glob('/tcdata/test_dataset_3w/image/*/*.jpg')[:1000]
    videos = glob.glob('/tcdata/test_dataset_3w/video/*.mp4')[:10]

    cfg = './taobao_configs/faster_rcnn_r50_fpn_triplet.py'
    ckpt = './taobao_models/published-8429440b.pth'
    ckpt_embed = r'/data/sdv2/taobao/embedding/checkpoints/normalize_checkpoint_12.pth.tar'

    model = get_detection_model(cfg, ckpt)
    model_embed = get_embed_model(1024, 256, 5, ckpt_embed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        # transforms.RandomResizedCrop(224),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize])

    print('\nPreparing gallery datasets...')
    st = time.time()
    gallery_dict = gallery_results(imgs, model, score_thr=0.5,
                                   out_json='gallery_results.json')
    print('Total Cost Time: {}s'.format(time.time() - st))
    # with open('gallery_results.json', 'r') as f:
    #     gallery_dict = json.load(f)

    print('\nStarting video predict...')
    st = time.time()
    final_result = {}
    for i, video in enumerate(videos):
        video_id, video_imgs = capture_video_imgs(video, interval=80)
        output = single_video_query(
            video_imgs, gallery_dict, model,
            score_thr=0.5, k_nearest=3, iou_thr=0.5
        )
        final_result[video_id] = output
    print('Average Cost Time: {}s'.format((time.time() - st) / len(videos)))
    print(final_result)

    with open('result.json', 'w') as f:
        json.dump(final_result, f, indent=4, cls=NumpyEncoder)
