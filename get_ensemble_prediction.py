from collections import Counter
import numpy as np
import pandas as pd
import glob
import pickle
from copy import deepcopy
from sklearn.cluster import OPTICS
import warnings
from scipy.spatial.distance import cosine
import hdbscan

ensemble_file = '/home/prasoon/P10.pb'
context_activities = {
    'Kitchen': ['Baking', 'Blender', 'Chopping+Grating', 'Chopping', 'CookingOnStove', 'FridgeOpen', 'Grating',
                'Microwave', 'WashingDishes'],
    'Bathroom': ['HairBrush', 'HairDryer', 'HandWash', 'Shaver In Use', 'Shower', 'ToilerFlushing', 'Toothbrush'],
    'LivingRoom': ['Coughing', 'Drinking', 'Eating', 'Drinking/Eating', 'Exercising', 'Knocking', 'Talking', 'Vacuum',
                   'Walking',
                   'WatchingTV']
}
context='LivingRoom'

prediction_dict, video_ensemble, audio_ensemble = pickle.load(open(ensemble_file,'rb'))

def get_ensemble_prediction(test_prediction_data, logger):
    #video prediction
    video_model_list = ('yamnet','posec3d_ntu120','posec3d_hmdb','posec3d_ucf')
    consideration_threshold = video_ensemble['consideration_threshold']
    consideration_label_count = video_ensemble['consideration_label_count']
    ma_match = video_ensemble['model_match']
    ma_match_all = pd.DataFrame(np.ones_like(ma_match), index=ma_match.index, columns=ma_match.columns)
    ma_clu = video_ensemble['model_activity_cluster']
    video_pred_detailed = {}
    for model in test_prediction_data:
        if model not in video_model_list:
            continue
        video_pred_detailed[model] = {}
        input_df = pd.DataFrame(test_prediction_data[model],columns=['index',1]).set_index('index')
        input_df[1] = input_df[1].astype(float)
        input_top_df = input_df[input_df[1] > consideration_threshold]
        # logger.info(input_top_df)
        passed_activities = ma_match_all.columns[np.where(ma_match_all.loc[model, :].values > 0.)[0]]
        for activity in passed_activities:
            train_clu, dummy_test_df = ma_clu[activity][model]
            for label in input_top_df.index.values:
                if label in dummy_test_df.columns:
                    dummy_test_df[label] = input_top_df.loc[label, 1]
            test_label, test_distance = -1, 1.0
            try:
                if np.sum(dummy_test_df.values) > 0:
                    test_label, test_prob = hdbscan.approximate_predict(train_clu, dummy_test_df.values.reshape(1, -1))
                    test_label, test_distance = test_label[0], 1 - test_prob[0]
            except:
                pass
            if (test_label >= 0) & (test_distance <= 1.):
                video_pred_detailed[model][activity] = 1 - test_distance
    df_video_pred = pd.DataFrame.from_dict(video_pred_detailed)
    video_pred, video_score = 'Undetected',0.
    if df_video_pred.shape[0] > 0.:
        video_prediction_score = np.nansum(df_video_pred.values, axis=1).max()
        activity_scores = np.nansum(df_video_pred.values, axis=1)
        video_prediction = df_video_pred.index[activity_scores.argmax()]
        video_pred = video_prediction
        video_score = video_prediction_score
    # logger.info(f"Video Prediction: {video_pred}:{video_score:3f}")
    #audio prediction
    audio_model_list = ('yamnet',)
    consideration_threshold = audio_ensemble['consideration_threshold']
    consideration_label_count = audio_ensemble['consideration_label_count']
    ma_match = audio_ensemble['model_match']
    ma_match_all = pd.DataFrame(np.ones_like(ma_match), index=ma_match.index, columns=ma_match.columns)
    ma_clu = audio_ensemble['model_activity_cluster']
    audio_pred_detailed = {}
    for model in test_prediction_data:
        if model not in audio_model_list:
            continue
        audio_pred_detailed[model] = {}
        input_df = pd.DataFrame(test_prediction_data[model],columns=['index',1]).set_index('index')
        input_df[1] = input_df[1].astype(float)
        input_top_df = input_df[input_df[1] > consideration_threshold]
        passed_activities = ma_match_all.columns[np.where(ma_match_all.loc[model, :].values > 0.)[0]]
        for activity in passed_activities:
            train_clu, dummy_test_df = ma_clu[activity][model]
            for label in input_top_df.index.values:
                if label in dummy_test_df.columns:
                    dummy_test_df[label] = input_top_df.loc[label, 1]
            test_label, test_distance = -1, 1.0
            try:
                if np.sum(dummy_test_df.values) > 0:
                    test_label, test_prob = hdbscan.approximate_predict(train_clu, dummy_test_df.values.reshape(1, -1))
                    test_label, test_distance = test_label[0], 1 - test_prob[0]
            except:
                pass
            if (test_label >= 0) & (test_distance <= 1.):
                audio_pred_detailed[model][activity] = 1 - test_distance
    df_audio_pred = pd.DataFrame.from_dict(audio_pred_detailed)
    # df_audio_pred = df_audio_pred[
    #     df_audio_pred.index.isin(context_activities[context])]
    audio_pred, audio_score = 'Undetected',0.
    if df_audio_pred.shape[0] > 0.:
        audio_prediction_score = np.nansum(df_audio_pred.values, axis=1).max()
        activity_scores = np.nansum(df_audio_pred.values, axis=1)
        audio_prediction = df_audio_pred.index[activity_scores.argmax()]
        audio_pred = audio_prediction
        audio_score = audio_prediction_score

    # logger.info(f"Audio Prediction: {audio_pred}:{audio_score:3f}")
    cutoffs  = {
                    'audio_high': 0.8,
                    'audio_low': 0.5,
                    'video_high':2.8,
                    'video_low': 2.5,}
    if (audio_score > cutoffs['audio_high']):
        instance_final_pred, instance_final_score = audio_pred, audio_score
    elif (audio_score > cutoffs['audio_low']) & (video_pred == 'Undetected'):
        instance_final_pred, instance_final_score = audio_pred, audio_score
    elif (video_score > cutoffs['video_high']):
        instance_final_pred, instance_final_score = video_pred, video_score
    elif (video_score > cutoffs['video_low']) & (audio_pred == 'Undetected'):
        instance_final_pred, instance_final_score = video_pred, video_score
    else:
        instance_final_pred, instance_final_score = 'Undetected', 0.
    return instance_final_pred, instance_final_score



def merge_dicts(dicts):
    """
    The merge_dicts function takes a list of dictionaries as input and returns a single dictionary.
    The function combines the values for each key in the dictionaries, resulting in one final dictionary.

    Args:
        dicts: Pass a list of dictionaries to the function

    Returns:
        A dictionary

    Doc Author:
        Trelent
    """
    c = Counter()
    for d in dicts:
        c.update(d)
    return dict(c)

def aggregate_audio_ts_scores(df_audio):
    ptile_10 = int(df_audio.shape[1] / 10)
    if ptile_10 > 0:
        df_audio = df_audio.iloc[:, ptile_10:-ptile_10]
    else:
        df_audio = df_audio.copy()
    df_audio_out = pd.DataFrame(np.percentile(df_audio.values,50,axis=1), index=df_audio.index, columns=[1]).sort_values([1],ascending=False)
    df_audio_out[1] = df_audio_out[1]/df_audio_out[1].sum()
    return df_audio_out
