# pylint: disable=no-member
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.cluster
from fuzzywuzzy import fuzz, process
from metaflow import FlowSpec, Parameter, step
from paddleocr import PaddleOCR
from PIL import Image

# from mmocr.apis import MMOCRInferencer


def bbox_angle(bbox):
    df = pd.DataFrame(bbox, columns=["x", "y"])
    slope, *_ = scipy.stats.linregress(df.x, df.y)
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


# result = result[0]
# image = Image.open(img_path).convert("RGB")
# boxes = [line[0] for line in result]
# txts = [line[1][0] for line in result]
# scores = [line[1][1] for line in result]
# im_show = draw_ocr(
#     image, boxes, txts, scores, font_path="/usr/share/fonts/TTF/Roboto-Medium.ttf"
# )
# im_show = Image.fromarray(im_show)
# im_show.save("result.jpg")


def paddle_results_to_df(result):
    df = pd.DataFrame(result, columns=["bbox", "inference"])
    df[["inference_text", "inference_score"]] = pd.DataFrame(
        df["inference"].tolist(), index=df.index
    )

    df.drop(columns="inference")
    return df


def make_polygons_2d(polygons):
    return np.concatenate(polygons).reshape(-1, 2).astype(np.float32)


def warp_perspective(img, angle=0, scale=1, tx=0, ty=0):
    # clear_output(wait=True)

    rows, cols, _ = img.shape
    center = (cols // 2, rows // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotation_matrix[:, 2] += [tx, ty]

    return cv2.warpAffine(img, rotation_matrix, (cols, rows))


def crop_image(image, bbox):
    """
    Crop an image (in numpy representation) to the given bounding box.

    Args:
    - image (numpy.ndarray): The input image in numpy format (height, width, channels)
    - bbox (list or tuple): The bounding box as (x_min, y_min, x_max, y_max)

    Returns:
    - cropped_image (numpy.ndarray): The cropped image
    """
    x_min, y_min, x_max, y_max = bbox
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


def rollmin(s):
    return (
        s.rolling(10, center=True).min().fillna(method="bfill").fillna(method="ffill")
    )


def rollmax(s):
    return (
        s.rolling(10, center=True).max().fillna(method="bfill").fillna(method="ffill")
    )


# def super_bounding_box(xy):
#     xx, yy = xy[:, 0], xy[:, 1]

#     return np.array(
#         [
#             [xx.min(), yy.min()],
#             [xx.max(), yy.min()],
#             [xx.max(), yy.max()],
#             [xx.min(), yy.max()],
#         ]
#     )


def best_match(text):
    markers = {"SPEED", "WATTS", "CADENCE", "CALORIES", "TARGETS", "DISTANCE"}

    # calculate the best match out of the possible markers
    best = process.extractOne(text, markers)
    return best[1] if best else 0


def super_bounding_box(xy):
    xx, yy = xy[:, 0], xy[:, 1]

    return [xx.min(), xx.max(), yy.min(), yy.max()]


def process_group(group):
    landmarks = (
        group.assign(match_score=group["inference_text"].apply(best_match))
        .query("match_score >= 90")
        .query("inference_score >= 0.98")
        .assign(angle=lambda f: f.bbox.apply(bbox_angle))
    )
    a = make_polygons_2d(landmarks.bbox.to_numpy())
    left, right, top, bottom = super_bounding_box(a)
    return {
        "angle": landmarks.angle.median(),
        "resize_width": 1000,
        "left": left,
        "right": right,
        "top": top,
        "bottom": bottom,
    }


def resize_image(image, width):
    """
    Resize an image (in numpy representation) to a given width while preserving the aspect ratio.

    Args:
    - image (numpy.ndarray): The input image in numpy format (height, width, channels)
    - width (int): The desired width of the resized image

    Returns:
    - resized_image (numpy.ndarray): The resized image
    """
    original_height, original_width = image.shape[:2]
    aspect_ratio = float(original_height) / float(original_width)
    new_height = int(width * aspect_ratio)

    resized_image = cv2.resize(
        image, (width, new_height), interpolation=cv2.INTER_LINEAR
    )

    return resized_image


def save_numpy_image_to_png(image, file_path):
    """
    Save a NumPy array representing an image to a PNG file.

    :param image: NumPy array representing the image
    :param file_path: Path of the output PNG file
    """
    # cv2.imwrite(file_path, image)
    image.save(file_path)


def crop_to_content(image, threshold=0.5):
    while True:
        rows, cols = image.shape
        top_row = image[0, :]
        bottom_row = image[-1, :]
        left_col = image[:, 0]
        right_col = image[:, -1]

        top_dark_pixels = np.sum(top_row < threshold)
        bottom_dark_pixels = np.sum(bottom_row < threshold)
        left_dark_pixels = np.sum(left_col < threshold)
        right_dark_pixels = np.sum(right_col < threshold)

        max_dark_pixels = max(
            top_dark_pixels, bottom_dark_pixels, left_dark_pixels, right_dark_pixels
        )

        if max_dark_pixels == 0:
            # No more dark pixels on the edges, stop cropping
            break

        if max_dark_pixels == top_dark_pixels:
            # Remove the top row
            image = image[1:, :]
        elif max_dark_pixels == bottom_dark_pixels:
            # Remove the bottom row
            image = image[:-1, :]
        elif max_dark_pixels == left_dark_pixels:
            # Remove the left column
            image = image[:, 1:]
        else:  # max_dark_pixels == right_dark_pixels
            # Remove the right column
            image = image[:, :-1]

    return image


class BBox(object):
    def __init__(self, polygon):
        a = np.asarray(polygon).reshape(-1, 2).astype(np.float32)
        xx, yy = a[:, 0], a[:, 1]
        self.left, self.right = xx.min(), xx.max()
        self.bottom, self.top = yy.min(), yy.max()
        self.width = self.right - self.left
        self.height = self.top - self.bottom


def implied_bbox(bbox, xl, xr):
    return list(
        map(
            round,
            (
                bbox.left + xl * bbox.width,
                bbox.top + 0.25 * bbox.height,
                bbox.right + xr * bbox.width,
                bbox.top + 2.7 * bbox.height,
            ),
        )
    )


def extract_landmark_images(image, df, config):
    out = dict()
    for word, xl, xr in config:
        rec = df.query("rec_texts == @word").squeeze()
        if rec.empty:
            continue
        bbox = BBox(rec.det_polygons)
        digit_bbox = implied_bbox(bbox, xl, xr)
        out[word] = crop_image(image, digit_bbox)

    return out


def run_ssocr(path):
    return (
        subprocess.run(["./ssocr-2.22.2/ssocr", "-d", "-1", path], capture_output=True)
        .stdout.decode("utf-8")
        .strip()
    )


def best_match_2(text):
    markers = {
        "SPEED",
        "WATTS",
        "CADENCE",
        "CALORIES",
        "TARGETS",
        "DISTANCE",
        "MPH",
        "MI",
        "HEARTRATE",
        "TIME",
    }

    # calculate the best match out of the possible markers
    best = process.extractOne(text, markers, score_cutoff=90)
    if best is None:
        return ""
    return best[0]


def compute_distance(bbox1, bbox2):
    try:
        # Compute centers of mass
        center1 = bbox1.mean(axis=0)
        center2 = bbox2.mean(axis=0)

        # Compute Euclidean distance between centers
        distance = np.linalg.norm(center1 - center2)
    except:
        return 1000

    return distance


def compute_angle(bbox1, bbox2):
    try:
        # Compute centers of mass
        center1 = bbox1.mean(axis=0)
        center2 = bbox2.mean(axis=0)

        # Compute angle relative to bbox1
        diff = center2 - center1
        angle = math.atan2(diff[1], diff[0]) * 180 / math.pi
    except:
        return 180

    return angle


def process_group_2(group):
    landmarks = group.assign(
        inference_clean=group["inference_text"].apply(best_match_2)
    )

    markers = {
        "SPEED",
        "WATTS",
        "CADENCE",
        "CALORIES",
        "DISTANCE",
        "TIME",
    }
    candidate_numbers = (
        landmarks.query("inference_clean.str.len() == 0")
        .reset_index(drop=True)
        .assign(box_id=lambda f: range(len(f)))
    )

    out = []
    for marker in markers:
        rec = landmarks.query("inference_clean == @marker").squeeze()

        _df = (
            candidate_numbers.assign(
                distance_from_ref=lambda f: [
                    compute_distance(rec.bbox, e) for e in f.bbox
                ],
                angle_from_ref=lambda f: [compute_angle(rec.bbox, e) for e in f.bbox],
            )
            .assign(
                belowness_score=lambda f: f.distance_from_ref
                + abs(f.angle_from_ref - 90)
            )
            .sort_values("belowness_score")
        )

        out.append(_df.head(1).assign(marker=marker))

    return pd.concat(out).loc[
        :, ["marker", "inference_text", "inference_score", "belowness_score", "box_id"]
    ]


class BuildMonitorDatasetFlow(FlowSpec):
    video_file = Parameter("video-file")
    start_time = Parameter("start-time", default="00:00")
    end_time = Parameter("end-time", default=None)
    duration_seconds = Parameter("duration-seconds", type=int, default=-1)
    frame_template = Parameter("frame-template", default="output_%03d.png")
    mmocr_detector = Parameter("mmocr-detector", default="DBNetpp")
    mmocr_recognizer = Parameter("mmocr-recognizer", default="ABINet")
    mmocr_batch_size = Parameter("mmocr-batch-size", default=8, type=int)
    image_width = Parameter("image-width", default=1000)
    clean_up = Parameter("clean-up", type=bool, default=True)

    @step
    def start(self):
        self.working_dir = tempfile.mkdtemp()
        print(f"{self.working_dir=}")
        self.next(self.unpack_video_to_frames)

    @step
    def unpack_video_to_frames(self):
        self.frames_dir = f"{self.working_dir}/frames"
        subprocess.run(
            [
                "./extract-frames.sh",
                self.video_file,
                self.frames_dir,
                self.start_time,
                str(self.duration_seconds),
                self.frame_template,
            ]
        )

        self.all_frames = glob.glob(f"{self.frames_dir}/*.png")
        self.next(self.text_detect_phase1)

    @step
    def text_detect_phase1(self):
        self.phase1_dir = f"{self.working_dir}/phase1-detection"
        # ocr = MMOCRInferencer(det=self.mmocr_detector, rec=self.mmocr_recognizer)

        # _unused = ocr(
        #     self.frames_dir,
        #     batch_size=self.mmocr_batch_size,
        #     out_dir=self.phase1_dir,
        #     save_pred=True,
        # )

        # Paddleocr supports Chinese, English, French, German, Korean and Japanese.
        # You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
        # to switch the language model in order.
        ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # need to run only once to download and load model into memory

        results = []
        for frame in self.all_frames:
            result = paddle_results_to_df(ocr.ocr(frame, cls=True)[0]).assign(
                frame=frame
            )
            results.append(result)

        self.phase_1_detection_results = pd.concat(results).reset_index(drop=True)

        self.next(self.postprocess_phase1_results)

    @step
    def postprocess_phase1_results(self):
        markers = {"SPEED", "WATTS", "CADENCE", "CALORIES", "TARGETS", "DISTANCE"}

        # FIXFIX: Feed in parameters, there's hardcoding occurring in the fns
        # underneath here.
        records = []
        for fg, sdf in self.phase_1_detection_results.groupby("frame"):
            # Defeated by pandas here bc I couldn't figure out how to assign a
            # list-type value to a single-row df. Fell back to for loop.
            records.append(dict(frame=fg) | process_group(sdf))

        self.phase_1_results = pd.DataFrame.from_records(records).assign(
            crop_left=lambda f: rollmin(f.left),
            crop_right=lambda f: rollmax(f.right),
            crop_top=lambda f: rollmin(f.top),
            crop_bottom=lambda f: rollmax(f.bottom),
        )

        self.next(self.reprocess_frames)

    @step
    def reprocess_frames(self):
        self.post_frames_dir = f"{self.working_dir}/postprocessed-frames"
        os.makedirs(self.post_frames_dir, exist_ok=True)
        for rec in self.phase_1_results.fillna(method="backfill").itertuples():
            img = Image.open(rec.frame)
            crop_left, crop_right, crop_top, crop_bottom = (
                rec.crop_left,
                rec.crop_right,
                rec.crop_top,
                rec.crop_bottom,
            )
            crop_width = abs(crop_left - crop_right)
            crop_height = abs(crop_top - crop_bottom)
            crop_right += int(round(crop_width * 0.1))
            crop_bottom += int(round(crop_height * 0.5))
            cropped_img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            resize_height = int(
                cropped_img.height * (rec.resize_width / cropped_img.width)
            )
            resized_img = cropped_img.resize(
                (rec.resize_width, resize_height), Image.LANCZOS
            )
            rotated_img = resized_img.rotate(rec.angle, Image.BICUBIC)
            rotated_img.save(f"{self.post_frames_dir}/{os.path.basename(rec.frame)}")

        self.next(self.text_detect_phase2)

    @step
    def text_detect_phase2(self):
        self.phase2_dir = f"{self.working_dir}/phase2-detection"

        ocr = PaddleOCR(
            use_angle_cls=True, lang="en"
        )  # need to run only once to download and load model into memory

        results = []
        for frame in self.all_frames:
            frame = f"{self.post_frames_dir}/{os.path.basename(frame)}"
            print(frame)
            result = paddle_results_to_df(ocr.ocr(frame, cls=True)[0]).assign(
                frame=frame
            )
            results.append(result)

        self.phase_2_detection_results = pd.concat(results).reset_index(drop=True)

        # ocr = MMOCRInferencer(det=self.mmocr_detector, rec=self.mmocr_recognizer)
        # _unused = ocr(
        #     self.post_frames_dir,
        #     batch_size=self.mmocr_batch_size,
        #     out_dir=self.phase2_dir,
        #     save_pred=True,
        # )
        self.next(self.postprocess_phase2_results)

    @step
    def postprocess_phase2_results(self):
        df = self.phase_2_detection_results.assign(
            frame=lambda f: [os.path.basename(e) for e in f.frame],
            bbox=lambda f: [np.array(e) for e in f.bbox],
        )
        df.info()
        print([type(e) for e in df.bbox])

        df = (
            df.groupby("frame")
            .apply(process_group_2)
            .reset_index()
            .drop(columns="level_1")
            .sort_values(["frame", "box_id", "belowness_score"])
            .assign(
                is_duplicate=lambda f: f.duplicated(["frame", "box_id"], keep="first"),
            )
        )

        df.loc[df.is_duplicate, ["inference_text", "inference_score"]] = np.nan

        self.ocr_df = df
        self.ocr_df.to_csv("ocr.csv", index=False)

        self.next(self.end)

    @step
    def end(self):
        if self.clean_up:
            shutil.rmtree(self.working_dir)
        print("Success!")


if __name__ == "__main__":
    BuildMonitorDatasetFlow()

# @step
# def postprocess_phase1_results(self):
#     records = []
#     for fn in glob.glob(f"{self.phase1_dir}/preds/*.json"):
#         df = pd.read_json(fn)

#         polygons = df.query(
#             "rec_texts in ('speed', 'cadence', 'distance', 'calories')"
#         ).det_polygons.to_numpy()
#         if len(polygons) < 4:
#             records.append(
#                 dict(
#                     prediction_file=fn,
#                     bbox=None,
#                     rotation=None,
#                 )
#             )
#             continue

#         xy_coords = make_polygons_2d(polygons)
#         bbox_df = pd.DataFrame(super_bounding_box(xy_coords), columns=["x", "y"])
#         width = bbox_df.x.max() - bbox_df.x.min()
#         height = bbox_df.y.max() - bbox_df.y.min()
#         bbox = np.round(
#             np.asarray(
#                 [
#                     bbox_df.x.min() - 0.05 * width,
#                     bbox_df.y.min() - 0.05 * height,
#                     bbox_df.x.max() + 0.10 * width,
#                     bbox_df.y.max() + 0.45 * height,
#                 ]
#             )
#         ).astype(int)
#         slope, *_ = scipy.stats.linregress(xy_coords)
#         angle_radians = np.arctan(slope)
#         angle_degrees = np.degrees(angle_radians)
#         record = dict(
#             prediction_file=fn,
#             bbox=bbox,
#             rotation=angle_degrees,
#         )
#         records.append(record)

#     self.phase_1_results = pd.DataFrame.from_records(records)

#     self.next(self.reprocess_frames)

# @step
# def postprocess_phase2_results(self):
#     landmark_config = (
#         ("speed", 0, 1.1),
#         ("watts", 0, 0.75),
#         ("cadence", 0, 0.20),
#         ("distance", 0, 0.75),
#         ("time", -1.25, 1.25),
#         ("calories", -0.15, 0.25),
#     )

#     print("Cutting out pieces of file")
#     cutout_image_rows = []
#     cutout_image_files = []
#     for fn in glob.glob(f"{self.phase2_dir}/preds/*.json"):
#         image_fn = os.path.basename(fn).replace(".json", ".png")
#         image = cv2.imread(f"{self.post_frames_dir}/{image_fn}")
#         df = pd.read_json(fn)
#         landmarks = extract_landmark_images(image, df, landmark_config)

#         # Write out landmark images to disk.
#         self.cutouts_dir = self.working_dir + "/cutouts"
#         os.makedirs(self.cutouts_dir, exist_ok=True)
#         _cutout_files = []
#         for word, image in landmarks.items():
#             _imfile = self.cutouts_dir + f"/{word}-{image_fn}"
#             _cutout_files.append(_imfile)
#             save_numpy_image_to_png(image, _imfile)
#         cutout_image_files.extend(_cutout_files)
#         cutout_image_rows.append(_cutout_files)

#     print("Building up some pixels for clustering")
#     self.min_pixels = 1e6
#     # Build k-means model to threshold images.
#     pixels = None
#     while True:
#         _imfile = random.choice(cutout_image_files)
#         _image = cv2.imread(_imfile)
#         _pixels = _image.reshape(-1, 3)
#         if pixels is None:
#             pixels = _pixels
#         else:
#             pixels = np.concatenate([pixels, _pixels])
#         if len(pixels) > self.min_pixels:
#             break

#     print("Building k-means model")
#     # Train K-means model
#     self.kmeans = sklearn.cluster.KMeans(n_clusters=2)
#     self.kmeans.fit(pixels)

#     print("Thresholding and OCRing images")
#     self.thresholded_cutouts_dir = self.working_dir + "/thresholded-cutouts"
#     os.makedirs(self.thresholded_cutouts_dir, exist_ok=True)
#     records = []
#     for _imrow in cutout_image_rows:
#         inferences = dict()
#         for _imfile in _imrow:
#             _imbase = os.path.basename(_imfile)
#             _outfile = self.thresholded_cutouts_dir + f"/{_imbase}"
#             word = _imbase.split("-")[0]
#             _image = cv2.imread(_imfile)
#             labels = self.kmeans.predict(_image.reshape(-1, 3))

#             # Make the "minority" label zero, whichever direction it's in.
#             if (labels == 0).mean() > 0.5:
#                 label = 1 - labels

#             segmented_image = labels.reshape(_image.shape[:2])
#             try:
#                 segmented_image = crop_to_content(segmented_image)
#             except IndexError:
#                 pass
#             save_numpy_image_to_png(
#                 (segmented_image * 255).astype(np.uint8), _outfile
#             )
#             digits = run_ssocr(_outfile)
#             inferences[word] = digits
#             records.append(inferences)

#     self.ocr_df = pd.DataFrame.from_records(records)

#     self.next(self.end)
