# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import base64
import os
import os.path as osp
import shutil
import time
from datetime import datetime
import glob
import traceback
import sys
import pickle
import cv2
import mmcv
import numpy as np
from mmaction.apis import inference_recognizer, init_recognizer
from operator import itemgetter
from pathlib import Path
import pandas as pd
import torch
from mmcv.parallel.collate import collate
import requests
import json
from mmaction.core import OutputHook
from mmaction.datasets.pipelines import Compose


# poseNet_skeleton_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9],
#                    [6, 8], [8, 10], [5, 11], [6, 12], [11, 12], [11, 13],
#                    [13, 15], [12, 14], [14, 16]]
# def openpose_to_posenet(arr):
#
#     result = np.zeros((17,3), dtype=np.float32)
#     for i,idx in enumerate(openPose_to_poseNet):
#         result[i] = (arr[idx])
#     return result


def posec3d_ntu120_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu120_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_r50_u48_240e_ntu120_xsub_keypoint-6736b03f.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map


def posec3d_ntu60_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_r50_u48_240e_ntu60_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_r50_u48_240e_ntu60_xsub_keypoint-f3adabf1.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map


def posec3d_hmdb_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_kinetics400_pretrained_r50_u48_120e_hmdb51_split1_keypoint-76ffdd8b.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_hmdb.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map


def posec3d_ucf_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/posec3d/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/slowonly_kinetics400_pretrained_r50_u48_120e_ucf101_split1_keypoint-cae8aa4a.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ucf.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    posec3d_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return posec3d_model, label_map


def stgcn_ntu60_model(device='cpu'):
    model_config_file = f'{Path(__file__).parent}/model_configs/skeleton/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
    model_ckpt_file = f'{Path(__file__).parent}/model_ckpts/stgcn_80e_ntu60_xsub_keypoint-e7bb9653.pth'
    class_names_file = f'{Path(__file__).parent}/model_configs/skeleton/label_map_ntu120.txt'

    # init action model and classmap
    config = mmcv.Config.fromfile(model_config_file)
    stgcn_model = init_recognizer(config, model_ckpt_file, device)
    label_map = [x.strip() for x in open(class_names_file).readlines()]
    return stgcn_model, label_map


def pose_inference(instance_pose_data, posec3d_model, posec3d_label_map, test_pipeline, pose_queue=None, orig_h=256,
                   orig_w=456, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    openPose_to_poseNet = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
    # pose_data_shapes = np.array([xr[1].shape[0] for xr in instance_pose_data])
    # instance_pose_data = np.array(instance_pose_data,dtype=object)[pose_data_shapes > 0]
    num_frame = len(instance_pose_data)
    if num_frame == 0:
        if pose_queue is not None:
            pose_queue.put('No Pose Info')
            return None
        return pd.DataFrame(columns=['label', 0]).set_index('label')
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    # num_person = max([len(x) for x in instance_pose_data])
    num_person = 1  # todo: fixed for time being, but need to be dynamic
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, (ts, pose_data) in enumerate(instance_pose_data):
        keypoint[0, i] = pose_data[0, openPose_to_poseNet, :2]
        keypoint_score[0, i] = pose_data[0, openPose_to_poseNet, 2]

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    data = test_pipeline(fake_anno)
    data = collate([data], samples_per_gpu=1)
    start_time = time.time()
    with torch.no_grad():
        scores = posec3d_model(return_loss=False, **data)[0]

    inf_time = time.time() - start_time
    num_classes = scores.shape[-1]
    score_tuples = tuple(zip(range(num_classes), scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)
    results = score_sorted[:5]
    # results = inference_recognizer(posec3d_model, fake_anno)
    # del fake_anno, keypoint, keypoint_score
    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], float(xr[1])) for xr in results],
                             columns=['label', 0]).set_index(
        'label')
    if pose_queue is not None:
        pose_activity_name = df_scores[0].sort_values(ascending=False).round(3).iloc[:3].to_dict()
        pose_activity_name = '|'.join([f'{kv}-{pose_activity_name[kv]}' for kv in pose_activity_name])
        # pose_queue.put(pose_activity_name)
        pose_queue.put(str(round(inf_time, 3)))
        return None
    return df_scores


def pose_inference_v2(instance_pose_data, posec3d_model, posec3d_label_map, test_pipeline, pose_queue=None, orig_h=256,
                      orig_w=456, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    openPose_to_poseNet = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
    # pose_data_shapes = np.array([xr[1].shape[0] for xr in instance_pose_data])
    # instance_pose_data = np.array(instance_pose_data,dtype=object)[pose_data_shapes > 0]
    num_frame = len(instance_pose_data)
    if num_frame == 0:
        if pose_queue is not None:
            pose_queue.put('No Pose Info')
            return None
        return pd.DataFrame(columns=['label', 0]).set_index('label')
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    # num_person = max([len(x) for x in instance_pose_data])
    num_person = 1  # todo: fixed for time being, but need to be dynamic
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, (ts, pose_data) in enumerate(instance_pose_data):
        keypoint[0, i] = pose_data[0, openPose_to_poseNet, :2]
        keypoint_score[0, i] = pose_data[0, openPose_to_poseNet, 2]

    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    start_time = time.time()
    results = inference_recognizer(posec3d_model, fake_anno)
    inf_time = time.time() - start_time
    # del fake_anno, keypoint, keypoint_score
    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], float(xr[1])) for xr in results],
                             columns=['label', 0]).set_index(
        'label')
    if pose_queue is not None:
        pose_activity_name = df_scores[0].sort_values(ascending=False).round(3).iloc[:3].to_dict()
        pose_activity_name = '|'.join([f'{kv}-{pose_activity_name[kv]}' for kv in pose_activity_name])
        # pose_queue.put(pose_activity_name)
        pose_queue.put(str(round(inf_time, 3)))
        return None
    return df_scores


def pose_inference_remote(instance_pose_data, pose_request_url, pose_model_name, pose_queue=None, orig_h=256,
                          orig_w=456, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    openPose_to_poseNet = np.array([0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10])
    num_frame = len(instance_pose_data)
    if num_frame == 0:
        if pose_queue is not None:
            pose_queue.put('No Pose Info')
            return None
        return pd.DataFrame(columns=['label', 0]).set_index('label')

    num_person = 1  # todo: fixed for time being, but need to be dynamic
    num_keypoint = 17
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, (ts, pose_data) in enumerate(instance_pose_data):
        keypoint[0, i] = pose_data[0, openPose_to_poseNet, :2]
        keypoint_score[0, i] = pose_data[0, openPose_to_poseNet, 2]

    start_time = time.time()
    request_dict = {
        'keypoints': base64.encodebytes(pickle.dumps(keypoint)).decode(),
        'keypoints_score': base64.encodebytes(pickle.dumps(keypoint_score)).decode(),
        'pose_model_name': pose_model_name
    }
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(pose_request_url, data=json.dumps(request_dict), headers=headers)
    response_dict = json.loads(response.text)
    pose_activity = f"\n{pose_model_name} {response_dict['pose_activity']}"
    inf_time = time.time() - start_time
    if pose_queue is not None:
        pose_queue.put((pose_model_name, pose_activity))
        # pose_queue.put(str(round(inf_time,3)))
    return None

def pose_inference_server(keypoints, keypoints_score, posec3d_model, posec3d_label_map, orig_h=256, orig_w=456, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    num_frame = keypoints.shape[1]
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    fake_anno['keypoint'] = keypoints
    fake_anno['keypoint_score'] = keypoints_score
    start_time = time.time()
    results = inference_recognizer(posec3d_model, fake_anno)
    inf_time = time.time()-start_time
    # del fake_anno, keypoint, keypoint_score
    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], float(xr[1])) for xr in results], columns=['label', 0]).set_index(
        'label')
    pose_activity_name = df_scores[0].sort_values(ascending=False).round(2).iloc[:5].to_dict()
    pose_activity_name = '\n'.join([f'{kv}-{pose_activity_name[kv]}' for kv in pose_activity_name])
    # pose_queue.put(pose_activity_name)
    pose_inf_time = f'{round(inf_time,1)}secs'
    return pose_inf_time+'\n\n'+pose_activity_name


def pose_inference_iphone(instance_pose_data, posec3d_model, posec3d_label_map, logger, orig_h=1, orig_w=1, short_side=480):
    w, h = mmcv.rescale_size((orig_w, orig_h), (short_side, np.Inf))
    logger.info(f"Expected width height, {w}{h}")
    # if pose_queue.qsize() <= 0.:
    #     return "No Pose Info"
    # instance_pose_data = np.array([pose_queue.get() for _ in range(pose_queue.size())][::-1])
    num_frame = len(instance_pose_data)
    logger.info(f"Got {num_frame} frames")
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        start_index=0,
        modality='Pose',
        total_frames=num_frame)
    num_keypoint = 17
    num_person = 1
    keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                              dtype=np.float16)
    for i, pose_data in enumerate(instance_pose_data):
        keypoint[0, i] = pose_data[:,:2]
        # logger.info(f"Created keypoint info")
        keypoint_score[0, i] = pose_data[:, 2]
    logger.info(f"Created all keypoint info")
    fake_anno['keypoint'] = keypoint
    fake_anno['keypoint_score'] = keypoint_score
    start_time = time.time()
    results = inference_recognizer(posec3d_model, fake_anno)
    inf_time = time.time()-start_time
    # del fake_anno, keypoint, keypoint_score
    df_scores = pd.DataFrame([(posec3d_label_map[xr[0]], float(xr[1])) for xr in results], columns=['label', 'score'])
    df_scores = df_scores.sort_values(by=['score'],ascending=False).iloc[:5]
    # logger.info(f"Got results from model {df_scores.values.tolist()}")
    return df_scores.values.tolist()

    # pose_activity_name = '\n'.join([f'{kv}-{pose_activity_name[kv]}' for kv in pose_activity_name])
    # # pose_queue.put(pose_activity_name)
    # pose_inf_time = f'{round(inf_time,1)}secs'
    # return pose_inf_time+'\n\n'+pose_activity_name

