'''
This is main server script to handle remote pose based action recognition requests
'''

import datetime

# basic libraries
import queue
import threading
import time
import traceback
import pandas as pd
import numpy as np
import cv2
import os
import json
import sys
import signal
import base64
import pickle
from flask import Flask, request
from multiprocessing import Queue, Process
from queue import Empty as EmptyQueueException, Queue as localQueue
import torch
# torch.multiprocessing.set_start_method('spawn')
# Custom libraries
from sensing.utils import get_logger
from otc_models import get_model
from otc_models.posec3d import pose_inference_server, pose_inference_iphone
from get_ensemble_prediction import get_ensemble_prediction
app = Flask(__name__)


@app.route("/raw_info_iphone", methods=["POST"])
def get_raw_iphone():
    request_json = request.get_json()
    pose_data = np.array([[xr, yr, cr] for (xr, yr, cr) in
                          zip(
                              request_json['bodyParts_x'], request_json['bodyParts_y'], request_json['bodyParts_c'])])
    audio_data = np.array(
        [[label, score] for (label, score) in zip(request_json['audioLabel'], request_json['audioConfidence'])])
    if vax_input_queue.qsize() == QUEUE_MAXSIZE:
        _ = vax_input_queue.get()
    vax_input_queue.put((request_json['audio_ts'], audio_data, pose_data))
    # logger.info(f"Got raw data at {pd.to_datetime(request_json['audio_ts'], unit='ns')}")

    try:
        vax_ts, vax_prediction, vax_score = vax_output_queue.get_nowait()
    except:
        pass
    response_dict = {
        'status': 200,
        'vax_ts_str': vax_ts,
        'vax_prediction': vax_prediction,
        'vax_score': round(vax_score,3)
    }
    return json.dumps(response_dict)


def run_vax_prediction(input_queue, output_queue):
    logger = get_logger("vax_service")
    # initialize all pose models
    pose_otc_model_names = ['posec3d_ntu120', 'posec3d_hmdb', 'posec3d_ucf']
    pose_otc_model_devices = ['cuda:1', 'cuda:1', 'cuda:1']
    pose_otc_models = {xr: get_model(xr, device=dr) for (xr, dr) in zip(pose_otc_model_names, pose_otc_model_devices)}

    logger.info("Models initiated")
    # send a message to output queue to enable server run
    # output_queue.put("Vax service started Complete..")
    local_arr = []
    try:
        while True:
            # if input_queue.qsize()>0:
            try:
                queue_data = input_queue.get_nowait()
            except EmptyQueueException:
                # logger.info("No queue packets")
                time.sleep(0.05)
                continue
            local_arr.append(queue_data)
            # logger.info("Got a queue packet...")
            if len(local_arr) > 100:
                # logger.info("Enough queue packets")
                instance_pose_data = np.array([local_arr[i][2] for i in range(100)])
                audio_data = local_arr[0][1]
                ts = local_arr[0][0]
                ts_str = pd.to_datetime(ts,unit='ns').strftime("%H:%M:%S")
                instance_ensemble_input = {
                    'yamnet': audio_data
                }
                logger.info(f"{ts_str}: Got audio instance data")
                start_time = time.time()
                instance_ensemble_input.update({
                    model_name: pose_inference_iphone(instance_pose_data,
                                                      pose_otc_models[model_name][0],
                                                      pose_otc_models[model_name][1], logger) for model_name in pose_otc_model_names
                })
                logger.info(f"{ts_str}: ensemble input in {time.time()-start_time:3f} secs.")

                start_time = time.time()
                instance_final_pred, instance_final_score = get_ensemble_prediction(instance_ensemble_input, logger)
                logger.info(f"{ts_str}: ensemble output ({instance_final_pred}-{instance_final_score}) in {time.time() - start_time:3f} secs.")
                output_queue.put((ts_str, instance_final_pred, instance_final_score))
                local_arr = list()
    except Exception as e:
        logger.info(traceback.format_exc())

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    QUEUE_MAXSIZE = 5 * 30


    vax_input_queue = Queue(maxsize=QUEUE_MAXSIZE)
    vax_output_queue = Queue(maxsize=QUEUE_MAXSIZE)
    vax_ts, vax_prediction, vax_score = '','NA',0.
    # initialize logger
    logger = get_logger('pose_server_runner', 'cache/logs/pose_server/')
    logger.info("------------ New pose server ------------")

    # initialize vax service handler
    output_handler = Process(target=run_vax_prediction,
                             args=(vax_input_queue, vax_output_queue))
    output_handler.start()

    time.sleep(5)
    # init_service_message = vax_output_queue.get()
    # logger.info(f"Service start message, {init_service_message}")
    #
    app.logger.disabled=True
    app.run('0.0.0.0', port=9090, threaded=True, debug=True)
