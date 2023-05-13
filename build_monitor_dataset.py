# pylint: disable=no-member
import glob
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
from metaflow import FlowSpec, Parameter, step
from mmocr.apis import MMOCRInferencer


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


def super_bounding_box(xy):
    xx, yy = xy[:, 0], xy[:, 1]

    return np.array(
        [
            [xx.min(), yy.min()],
            [xx.max(), yy.min()],
            [xx.max(), yy.max()],
            [xx.min(), yy.max()],
        ]
    )


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
    cv2.imwrite(file_path, image)


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
        self.next(self.text_detect_phase1)

    @step
    def text_detect_phase1(self):
        self.phase1_dir = f"{self.working_dir}/phase1-detection"
        ocr = MMOCRInferencer(det=self.mmocr_detector, rec=self.mmocr_recognizer)
        _unused = ocr(
            self.frames_dir,
            batch_size=self.mmocr_batch_size,
            out_dir=self.phase1_dir,
            save_pred=True,
        )
        self.next(self.postprocess_phase1_results)

    @step
    def postprocess_phase1_results(self):
        records = []
        for fn in glob.glob(f"{self.phase1_dir}/preds/*.json"):
            df = pd.read_json(fn)

            polygons = df.query(
                "rec_texts in ('speed', 'cadence', 'distance', 'calories')"
            ).det_polygons.to_numpy()
            if len(polygons) < 4:
                records.append(
                    dict(
                        prediction_file=fn,
                        bbox=None,
                        rotation=None,
                    )
                )
                continue

            xy_coords = make_polygons_2d(polygons)
            bbox_df = pd.DataFrame(super_bounding_box(xy_coords), columns=["x", "y"])
            width = bbox_df.x.max() - bbox_df.x.min()
            height = bbox_df.y.max() - bbox_df.y.min()
            bbox = np.round(
                np.asarray(
                    [
                        bbox_df.x.min() - 0.05 * width,
                        bbox_df.y.min() - 0.05 * height,
                        bbox_df.x.max() + 0.10 * width,
                        bbox_df.y.max() + 0.45 * height,
                    ]
                )
            ).astype(int)
            slope, *_ = scipy.stats.linregress(xy_coords)
            angle_radians = np.arctan(slope)
            angle_degrees = np.degrees(angle_radians)
            record = dict(
                prediction_file=fn,
                bbox=bbox,
                rotation=angle_degrees,
            )
            records.append(record)

        self.phase_1_results = pd.DataFrame.from_records(records)

        self.next(self.reprocess_frames)

    @step
    def reprocess_frames(self):
        self.post_frames_dir = f"{self.working_dir}/postprocessed-frames"
        os.makedirs(self.post_frames_dir, exist_ok=True)
        for rec in self.phase_1_results.fillna(method="backfill").itertuples():
            image_fn = os.path.basename(rec.prediction_file).replace(".json", ".png")
            image = cv2.imread(f"{self.frames_dir}/{image_fn}")
            cropped_image = crop_image(image, rec.bbox)
            rotated_image = warp_perspective(cropped_image, angle=rec.rotation)
            resized_image = resize_image(rotated_image, self.image_width)
            save_numpy_image_to_png(resized_image, f"{self.post_frames_dir}/{image_fn}")

        self.next(self.text_detect_phase2)

    @step
    def text_detect_phase2(self):
        self.phase2_dir = f"{self.working_dir}/phase2-detection"
        ocr = MMOCRInferencer(det=self.mmocr_detector, rec=self.mmocr_recognizer)
        _unused = ocr(
            self.post_frames_dir,
            batch_size=self.mmocr_batch_size,
            out_dir=self.phase2_dir,
            save_pred=True,
        )
        self.next(self.postprocess_phase2_results)

    @step
    def postprocess_phase2_results(self):
        landmark_config = (
            ("speed", 0, 1.1),
            ("watts", 0, 0.75),
            ("cadence", 0, 0.20),
            ("distance", 0, 0.75),
            ("time", -1.25, 1.25),
            ("calories", -0.15, 0.25),
        )

        print("Cutting out pieces of file")
        cutout_image_rows = []
        cutout_image_files = []
        for fn in glob.glob(f"{self.phase2_dir}/preds/*.json"):
            image_fn = os.path.basename(fn).replace(".json", ".png")
            image = cv2.imread(f"{self.post_frames_dir}/{image_fn}")
            df = pd.read_json(fn)
            landmarks = extract_landmark_images(image, df, landmark_config)

            # Write out landmark images to disk.
            self.cutouts_dir = self.working_dir + "/cutouts"
            os.makedirs(self.cutouts_dir, exist_ok=True)
            _cutout_files = []
            for word, image in landmarks.items():
                _imfile = self.cutouts_dir + f"/{word}-{image_fn}"
                _cutout_files.append(_imfile)
                save_numpy_image_to_png(image, _imfile)
            cutout_image_files.extend(_cutout_files)
            cutout_image_rows.append(_cutout_files)

        print("Building up some pixels for clustering")
        self.min_pixels = 1e6
        # Build k-means model to threshold images.
        pixels = None
        while True:
            _imfile = random.choice(cutout_image_files)
            _image = cv2.imread(_imfile)
            _pixels = _image.reshape(-1, 3)
            if pixels is None:
                pixels = _pixels
            else:
                pixels = np.concatenate([pixels, _pixels])
            if len(pixels) > self.min_pixels:
                break

        print("Building k-means model")
        # Train K-means model
        self.kmeans = sklearn.cluster.KMeans(n_clusters=2)
        self.kmeans.fit(pixels)

        print("Thresholding and OCRing images")
        self.thresholded_cutouts_dir = self.working_dir + "/thresholded-cutouts"
        os.makedirs(self.thresholded_cutouts_dir, exist_ok=True)
        records = []
        for _imrow in cutout_image_rows:
            inferences = dict()
            for _imfile in _imrow:
                _imbase = os.path.basename(_imfile)
                _outfile = self.thresholded_cutouts_dir + f"/{_imbase}"
                word = _imbase.split("-")[0]
                _image = cv2.imread(_imfile)
                labels = self.kmeans.predict(_image.reshape(-1, 3))

                # Make the "minority" label zero, whichever direction it's in.
                if (labels == 0).mean() > 0.5:
                    label = 1 - labels

                segmented_image = labels.reshape(_image.shape[:2])
                try:
                    segmented_image = crop_to_content(segmented_image)
                except IndexError:
                    pass
                save_numpy_image_to_png(
                    (segmented_image * 255).astype(np.uint8), _outfile
                )
                digits = run_ssocr(_outfile)
                inferences[word] = digits
                records.append(inferences)

        self.ocr_df = pd.DataFrame.from_records(records)

        self.next(self.end)

    @step
    def end(self):
        if self.clean_up:
            shutil.rmtree(self.working_dir)
        print("Success!")


if __name__ == "__main__":
    BuildMonitorDatasetFlow()
