'''
This is main server script to handle remote pose based action recognition requests
'''

import datetime

# basic libraries
import queue
import threading
import time
import traceback
import numpy as np
import cv2
import os
import json
import sys
import signal
import base64
import pickle
from flask import Flask, request
from multiprocessing import Queue

# Custom libraries
from sensing.utils import get_logger
from otc_models import get_model
from otc_models.posec3d import pose_inference_server, pose_inference_iphone

DEVICE = 'cuda:1'
QUEUE_MAXSIZE = 5*30
app = Flask(__name__)
pose_queue = Queue(maxsize=QUEUE_MAXSIZE)
audio_queue = Queue(maxsize=QUEUE_MAXSIZE)

vax_input_queue = Queue(maxsize=QUEUE_MAXSIZE)
vax_output_queue = Queue(maxsize=QUEUE_MAXSIZE)

@app.route("/pose_inference",methods=["POST"])
def pose_inference_request():
    request_json = request.get_json()
    keypoints_encoded = request_json["keypoints"]
    keypoints_score_encoded = request_json["keypoints_score"]

    keypoints = pickle.loads(base64.b64decode(keypoints_encoded.encode()))
    keypoint_scores = pickle.loads(base64.b64decode(keypoints_score_encoded.encode()))
    pose_model_name = request_json["pose_model_name"]
    pose_activity_inference = pose_inference_server(
        keypoints,
        keypoint_scores,
        pose_otc_models[pose_model_name][0],
        pose_otc_models[pose_model_name][1],
    )
    print(f"{datetime.datetime.now()} {pose_model_name}-{pose_activity_inference.split('secs')[0]}s")
    response_dict = {
        'pose_activity':pose_activity_inference
    }
    return json.dumps(response_dict)

# @app.route("/pose_inference_iphone",methods=["POST"])
# def pose_inference_iphone():
#     request_json = request.get_json()
#     pose_model_name = request_json["pose_model_name"]
#     pose_activity_inference = pose_inference_iphone(
#         pose_queue,
#         pose_otc_models[pose_model_name][0],
#         pose_otc_models[pose_model_name][1],
#     )
#     print(f"{datetime.datetime.now()} {pose_model_name}-{pose_activity_inference.split('secs')[0]}s")
#     response_dict = {
#         'pose_activity': pose_activity_inference
#     }
#     return json.dumps(response_dict)

@app.route("/pose_info_iphone",methods=["POST"])
def get_pose_iphone():
    request_json = request.get_json()
    pose_data = np.array([[xr, yr, cr] for (xr, yr, cr) in
                          zip(
                              request_json['bodyParts_x'], request_json['bodyParts_y'], request_json['bodyParts_c'])])
    if pose_queue.qsize()==QUEUE_MAXSIZE:
        # pose_model_name = 'posec3d_ntu120'
        # pose_activity_inference = pose_inference_iphone(
        #     pose_queue,
        #     pose_otc_models[pose_model_name][0],
        #     pose_otc_models[pose_model_name][1],
        # )

        _ = pose_queue.get()
    pose_queue.put(pose_data)
    print(f"Got pose data at {request_json['pose_ts']}")
    response_dict = {
        'status': 200
    }
    return json.dumps(response_dict)

@app.route("/raw_info_iphone",methods=["POST"])
def get_raw_iphone():
    request_json = request.get_json()
    pose_data = np.array([[xr, yr, cr] for (xr, yr, cr) in
                          zip(
                              request_json['bodyParts_x'], request_json['bodyParts_y'], request_json['bodyParts_c'])])
    audio_data = np.array([[label,score] for (label,score) in zip(request_json['audioLabel'], request_json['audioConfidence'])])
    if vax_input_queue.qsize() == QUEUE_MAXSIZE:
        _ = vax_input_queue.get()
    vax_input_queue.put((request_json['audio_ts'],audio_data,pose_data))
    print(f"Got raw data at {request_json['audio_ts']}:{str(request_json)}")
    if vax_output_queue.qsize()>0:
        vax_ts, vax_prediction, vax_score = vax_output_queue.get()
    else:
        vax_ts, vax_prediction, vax_score = '','',0
    response_dict = {
        'status': 200,
        'vax_ts_str':"10:00:00",
        'vax_prediction':"TBD",
        'vax_score':0.76
    }
    return json.dumps(response_dict)


if __name__=='__main__':
    # initialize logger
    logger = get_logger('pose_server_runner', 'cache/logs/pose_server/')
    logger.info("------------ New pose server ------------")

    # initialize all pose models
    pose_otc_model_names = ['posec3d_ntu120', 'posec3d_hmdb', 'posec3d_ucf']
    pose_otc_model_devices = ['cuda:0', 'cuda:1', 'cuda:1']
    pose_otc_models = {xr: get_model(xr,device=dr) for (xr,dr) in zip(pose_otc_model_names,pose_otc_model_devices)}

    app.run('0.0.0.0',port=9090,threaded=True,debug=False)
