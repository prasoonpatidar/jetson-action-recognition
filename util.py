"""
Utility functions for optics trainers
"""
from collections import Counter
import numpy as np
import pandas as pd
import cv2
from typing import Optional, Tuple
import glob
# from external_mapping import instance_video_map

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

def jaccard_score_custom(list1, list2):
    """
    The jaccard_score_custom function takes two lists as input and returns the jaccard score between them.
    The jaccard score is defined as the intersection of two sets divided by their union.
    If either list is empty, then the function will return 0.

    Args:
        list1: Represent the list of words in a document
        list2: Compare the list of predicted labels with

    Returns:
        The jaccard score between two lists

    Doc Author:
        Trelent
    """

    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    if union==0:
        return 0.
    return float(intersection) / union

def aggregate_ts_scores(df_otc_out):
    """
    The aggregate_ts_scores function takes in a dataframe of audio features and returns a dataframe with the
    audio features aggregated by percentile. The function also sorts the values in descending order, so that
    the most important feature is listed first.

    Args:
        df_otc_out: Store the audio features for each video

    Returns:
        A dataframe with the mean of each row, sorted by descending values

    Doc Author:
        Trelent
    """
    ptile_10 = int(df_otc_out.shape[1] / 10)
    if ptile_10 > 0:
        df_otc_out = df_otc_out.iloc[:, ptile_10:-ptile_10]
    else:
        df_otc_out = df_otc_out.copy()
    df_out = pd.DataFrame(np.percentile(df_otc_out.values,50,axis=1), index=df_otc_out.index, columns=[1]).sort_values([1],ascending=False)
    df_out[1] = df_out[1]/df_out[1].sum()
    return df_out

def add_text_to_image(
    image_rgb: np.ndarray,
    label: str,
    top_left_xy: Tuple = (0, 0),
    font_scale: float = 1,
    font_thickness: float = 1,
    font_face=cv2.FONT_HERSHEY_SIMPLEX,
    font_color_rgb: Tuple = (0, 0, 255),
    bg_color_rgb: Optional[Tuple] = None,
    outline_color_rgb: Optional[Tuple] = None,
    line_spacing: float = 1,
):
    """
    Adds text (including multi line text) to images.
    You can also control background color, outline color, and line spacing.

    outline color and line spacing adopted from: https://gist.github.com/EricCousineau-TRI/596f04c83da9b82d0389d3ea1d782592
    Examples:
        image = 200 * np.ones((550, 410, 3), dtype=np.uint8)
        image = add_text_to_image(
            image,
            "New line\nDouble new line\n\nLine too longggggggggggggggggggggg",
            top_left_xy=(0, 10),
        )
        image = add_text_to_image(
            image,
            "Different font scale",
            font_scale=0.5,
            font_color_rgb=(0, 255, 0),
            top_left_xy=(0, 150),
        )
        image = add_text_to_image(
            image,
            "New line with bg\nDouble new line\n\nLine too longggggggggggggggggggggg",
            bg_color_rgb=(255, 255, 255),
            font_color_rgb=(255, 0, 0),
            top_left_xy=(0, 200),
        )
        image = add_text_to_image(
            image,
            "Different line specing,\noutline and font face",
            font_color_rgb=(0, 255, 255),
            outline_color_rgb=(0, 0, 0),
            top_left_xy=(0, 350),
            line_spacing=1.5,
            font_face=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        )
    """
    OUTLINE_FONT_THICKNESS = 3 * font_thickness

    im_h, im_w = image_rgb.shape[:2]

    for line in label.splitlines():
        x, y = top_left_xy

        # ====== get text size
        if outline_color_rgb is None:
            get_text_size_font_thickness = font_thickness
        else:
            get_text_size_font_thickness = OUTLINE_FONT_THICKNESS

        (line_width, line_height_no_baseline), baseline = cv2.getTextSize(
            line,
            font_face,
            font_scale,
            get_text_size_font_thickness,
        )
        line_height = line_height_no_baseline + baseline

        if bg_color_rgb is not None and line:
            # === get actual mask sizes with regard to image crop
            if im_h - (y + line_height) <= 0:
                sz_h = max(im_h - y, 0)
            else:
                sz_h = line_height

            if im_w - (x + line_width) <= 0:
                sz_w = max(im_w - x, 0)
            else:
                sz_w = line_width

            # ==== add mask to image
            if sz_h > 0 and sz_w > 0:
                bg_mask = np.zeros((sz_h, sz_w, 3), np.uint8)
                bg_mask[:, :] = np.array(bg_color_rgb)
                image_rgb[
                    y : y + sz_h,
                    x : x + sz_w,
                ] = bg_mask

        # === add outline text to image
        if outline_color_rgb is not None:
            image_rgb = cv2.putText(
                image_rgb,
                line,
                (x, y + line_height_no_baseline),  # putText start bottom-left
                font_face,
                font_scale,
                outline_color_rgb,
                OUTLINE_FONT_THICKNESS,
                cv2.LINE_AA,
            )
        # === add text to image
        image_rgb = cv2.putText(
            image_rgb,
            line,
            (x, y + line_height_no_baseline),  # putText start bottom-left
            font_face,
            font_scale,
            font_color_rgb,
            font_thickness,
            cv2.LINE_AA,
        )
        top_left_xy = (x, y + int(line_height * line_spacing))

    return image_rgb
