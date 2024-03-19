# !/usr/bin/env python3
'''
This is main file to run data collection from depthcam. It collect rgb, and depth information.
Developer: Prasoon Patidar
Created: 5th March, 2022
'''
import datetime

# basic libraries
import queue
import threading
import time
import traceback
import numpy as np
import cv2
import depthai as dai
import os
import json
import sys
import signal
from queue import Queue
import psutil
import librosa

# Custom libraries
from sensing.utils import get_logger, get_screen_size
from sensing.oakdlite.config import Config
from sensing.oakdlite.depth_recorder import DepthRecorderThread
from sensing.oakdlite.rgb_recorder import RGBRecorderThread
from sensing.oakdlite.pose_recorder import PoseRecorderThread
from sensing.oakdlite.poseestimators import get_poseestimator
from sensing.oakdlite.run import pose_model_config, create_vpu_pipeline, create_keypoint_frame
from sensing.audio.run_audio import ReaderThread as audioReaderThread, Device as audioDevice

from otc_models import get_model
from otc_models.yamnet import audio_inference
from otc_models.posec3d import pose_inference
from util import aggregate_ts_scores, add_text_to_image

CHECKPOINT_FREQ = 60


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


signal.signal(signal.SIGTERM, sigterm_handler)

if __name__ == '__main__':

    # initialize logger
    logger = get_logger('av_runner', 'cache/logs/av/')
    logger.info("------------ New Audio Run ------------")
    config_file = 'dc_config.json'
    visualize = True
    run_config = json.load(open(config_file, 'r'))
    max_duration = run_config['duration_in_mins'] * 60
    t_data_collection_start = datetime.datetime.now()
    start_time = time.time()
    experiment_dir = run_config["experiment_dir"]
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # setup audio device

    # initialize queues
    sensor_queue = Queue()
    if visualize:
        viz_queue = Queue()
    else:
        viz_queue = None

    # initialize audio device
    audioSensor = audioDevice(run_config, sensor_queue, logger, viz_queue)
    # check if available
    if audioSensor.is_available():
        logger.info(f"- Found Sensor {audioSensor.name}-")
        audioSensor.startReader()
    else:
        print("Audio sensor not available, exiting...")
    # audio_frames = np.zeros((100, audioSensor.num_channels))
    audio_frames = np.zeros((100, 1))
    audio_window_name = 'Audio FFt'
    screen_width, screen_height = get_screen_size()
    cv2.namedWindow(audio_window_name)
    cv2.moveWindow(audio_window_name, screen_width // 2, 0)

    # initialize action recognition models

    # get audio based inferences

    audio_otc_model_names = ['yamnet']
    audio_otc_models = {xr: get_model(xr) for xr in audio_otc_model_names}
    audio_otc_queue = Queue()
    audio_curr_time = time.time()
    audio_activity_interval = 2
    audio_activity = 'None'

    audio_instance_data = []
    # start running VPU pipeline
    # Connect to device and start pipeline
    try:

        while True:
            if viz_queue.qsize() > 0:
                # self.logger.info(f"Running viz data from {AUDIO_DEVICE_NAME} sensor...")
                audio_frame = np.concatenate([viz_queue.get() for _ in range(viz_queue.qsize())])
                audio_instance_data.append(audio_frame)
                if time.time() > audio_curr_time + audio_activity_interval:
                    for audio_model_name in audio_otc_models:
                        audio_th = threading.Thread(target=audio_inference,
                                                    args=((np.concatenate(audio_instance_data),
                                                           audioSensor.sampling_rate),
                                                          audio_otc_models[audio_model_name][0],
                                                          audio_otc_models[audio_model_name][1], audio_otc_queue))
                        audio_th.start()
                        audio_curr_time = time.time()
                        audio_instance_data = list()
                        pass
                if audio_otc_queue.qsize() > 0:
                    audio_activity = [audio_otc_queue.get() for _ in range(audio_otc_queue.qsize())][-1]
                audio_frames = np.concatenate([audio_frames, audio_frame])
                audio_frames = audio_frames[-40000:]
                S_fft = np.abs(librosa.stft(y=audio_frames.T, n_fft=256))
                S_dB = librosa.amplitude_to_db(S_fft, ref=np.min).mean(axis=0)
                img_col = cv2.applyColorMap(S_dB.astype(np.uint8), cv2.COLORMAP_JET)
                img_col = add_text_to_image(img_col,
                                            audio_activity,
                                            bg_color_rgb=(255, 255, 255),
                                            font_color_rgb=(255, 0, 0),
                                            top_left_xy=(0, 0),
                                            font_scale=0.5,
                                            )
                # img_col = cv2.putText(img_col, audio_activity, (0, img_col.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                #                       0.5, (0, 0, 0), 1, cv2.LINE_AA)

                cv2.imshow(audio_window_name, img_col)
                # if cv2.waitKey(1) == 27:
                #     break  # esc to quit
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
                if cv2.getWindowProperty(audio_window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
        logger.info(f"Exiting gracefully...")
    except Exception as e:
        logger.info(f"Exiting with error...")
        logger.info(traceback.print_exc())
        audioSensor.stopWriter()
        audioSensor.stopReader()
        cv2.destroyWindow(audio_window_name)
        logger.info(f"Stopped {audioSensor.name}")
    finally:
        logger.info("Pose thread joined")
        logger.info("Depth thread joined")
        logger.info("RGB thread joined")
        cv2.destroyAllWindows()
        logger.info(f"Oakdlite Sensor closed successfully")
        audioSensor.stopWriter()
        audioSensor.stopReader()
        logger.info(f"Audio Collection Complete {audioSensor.name}")
