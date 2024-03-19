'''
Inference engine for yamnet model
'''
from __future__ import division, print_function

import sys

import numpy as np
import resampy
import soundfile as sf
import tensorflow as tf
import shutil
import otc_models.model_configs.yamnet.params as yamnet_params
import otc_models.model_configs.yamnet.yamnet as yamnet_model
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from util import aggregate_ts_scores


def get_yamnet_model(device='cpu'):
    model_file = 'model_ckpts/yamnet.h5'
    class_names_file = 'model_configs/yamnet/yamnet_class_map.csv'
    params = yamnet_params.Params()
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights(f"{Path(__file__).parent}/{model_file}")
    yamnet_classes = yamnet_model.class_names(f"{Path(__file__).parent}/{class_names_file}")
    return yamnet, yamnet_classes


def audio_inference(instance_audio_data, yamnet, yamnet_classes, audio_queue=None):
    params = yamnet_params.Params()
    # wav_data, sr = sf.read(audio_file, dtype=np.int16)
    waveform, sr = instance_audio_data
    # assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
    # waveform = wav_data / 32768.0  # Convert to [-1.0, +1.0]
    # waveform = waveform.astype('float32')

    # Convert to mono and the sample rate expected by YAMNet.
    if len(waveform.shape) > 1:
        waveform = np.mean(waveform, axis=1)
    if sr != params.sample_rate:
        waveform = resampy.resample(waveform, sr, params.sample_rate)

    # Predict YAMNet classes.
    scores, embeddings, spectrogram = yamnet(waveform)
    df_scores = pd.DataFrame(scores.numpy().T, index=yamnet_classes)
    if audio_queue is not None:
        audio_activity_name = aggregate_ts_scores(df_scores)[1].sort_values(ascending=False).round(3).iloc[:5].to_dict()
        audio_activity_name = '\n'.join([f'{kv}-{audio_activity_name[kv]}' for kv in audio_activity_name])
        audio_queue.put(audio_activity_name)
        return None

    return df_scores
