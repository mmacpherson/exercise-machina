import os
import subprocess
import tempfile

import cv2
import numpy as np
import paddleocr
import pandas as pd
import PIL
import scipy.stats
from fuzzywuzzy import process as fw_process, fuzz

_OCR = None


def get_ocr():
    global _OCR
    if _OCR is None:
        _OCR = paddleocr.PaddleOCR(
            # use_gpu=False,
            # use_gpu=True,
            use_angle_cls=True,
            lang="en",
            show_log=False,
        )

    return _OCR


def unpack_video_to_frames(args):
    return subprocess.run(["./extract-frames.sh"] + list(args))


def paddle_results_to_df(result):
    df = pd.DataFrame(result, columns=["bbox", "inference"])
    df[["inference_text", "inference_score"]] = pd.DataFrame(
        df["inference"].tolist(), index=df.index
    )

    df.drop(columns="inference")
    return df


def ocr_frames(frames):
    ocr = get_ocr()
    results = []
    for frame in frames:
        ocr_results = ocr.ocr(frame, cls=True)
        if ocr_results:
            result = paddle_results_to_df(ocr_results[0]).assign(frame=frame)
        else:
            result = pd.DataFrame(dict(frame=frame), index=[0])
        results.append(result)

    return pd.concat(results).reset_index(drop=True)


def locate_display(frame):
    ocr = get_ocr()
    ocr_results = ocr.ocr(frame, cls=True)
    if ocr_results:
        return paddle_results_to_df(ocr_results[0])

    return


def score_crop(a, cf):
    l, t, r, b = cf
    perimeter = np.concatenate(
        [
            a[t:b, l].flatten(),
            a[t:b, r].flatten(),
            a[t, l:r].flatten(),
            a[t, l:r].flatten(),
        ]
    )
    score = perimeter.mean() + perimeter.std()

    return score


def tidy_crop_frame(a, cf, w=15):
    best_score = score_crop(a, cf)
    best_crop = cf

    # print(f"{cf=}")
    for ix in range(len(cf)):
        dr = 1 if (ix >= 2) else -1
        for d in range(-w * dr, w * dr):
            cfp = cf.copy()
            cfp[ix] += d
            score = score_crop(a, cfp)
            if score < best_score:
                # print(f"{best_score=} {ix=} {d=} {cfp=}")
                best_score = score
                best_crop = cfp

    return best_crop


def make_polygons_2d(polygons):
    return np.concatenate(polygons).reshape(-1, 2).astype(np.float32)


def best_match(text):
    markers = {"SPEED", "WATTS", "CADENCE", "CALORIES", "TARGETS", "DISTANCE"}

    # calculate the best match out of the possible markers
    best = fw_process.extractOne(text, markers, scorer=fuzz.ratio)
    return best if best else (None, 0)


def super_bounding_box(xy):
    xx, yy = xy[:, 0], xy[:, 1]

    return [xx.min(), xx.max(), yy.min(), yy.max()]


def bbox_angle(bbox):
    df = pd.DataFrame(bbox, columns=["x", "y"])
    slope, *_ = scipy.stats.linregress(df.x, df.y)
    angle_radians = np.arctan(slope)
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def crop_to_numbers(df):
    landmarks = (
        df.assign(  # .assign(match_score=lambda f: f.inference_text.apply(best_match))
            _scores=lambda f: f.inference_text.apply(best_match),
            match_text=lambda f: f._scores.apply(lambda x: x[0]),
            match_score=lambda f: f._scores.apply(lambda x: x[1]),
        )
        .query("match_score >= 90")
        .query("inference_score >= 0.98")
        .sort_values(["match_text", "match_score", "inference_score"])
        .drop_duplicates(subset="match_text")
        .assign(angle=lambda f: f.bbox.apply(bbox_angle))
        .drop(columns=["_scores", "inference"])
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


def run_ssocr(image, params, commands):
    # Define the command and parameters
    # cmd = ["./ssocr-2.22.2/ssocr"]
    cmd = ["./ssocr-2.23.1/ssocr"]

    # Add the parameters to the command
    for key, value in params.items():
        if value is None:
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")

    for _cmd in commands:
        cmd.extend(_cmd.split())

    # If the input is a numpy array, write it to a temporary file
    if isinstance(image, str):
        cmd.append(image)
    else:
        # Create a temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_file_path = temp_file.name
        # Write the image to the temporary file
        cv2.imwrite(temp_file_path, image)
        cmd.append(temp_file_path)

    # print(f"{cmd=}")

    # Run the command and get the output
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Delete the temporary file if it was used
    if not isinstance(image, str):
        os.remove(temp_file_path)

    # Check for errors
    if result.stderr:
        # print(f"Error: {result.stderr.decode()}")
        pass

    # Return the output
    return result.stdout.decode().strip()


def compute_distance(bbox1, bbox2):
    center1 = bbox1.mean(axis=0)
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


def plausible_number(w, thresh=0.5):
    return (sum(1 for c in w if c.isdigit() or c in ":.") / len(w)) > thresh


def locate_numbers(
    df,
    landmarks=(
        "SPEED",
        "WATTS",
        "CADENCE",
        "CALORIES",
        "DISTANCE",
        "TIME",
    ),
    min_score=0.9,
):
    candidates_df = (
        df.query("inference_score >= @min_score")
        .loc[lambda f: f.inference_text.apply(plausible_number)]
        .reset_index(drop=True)
        .assign(box_id=lambda f: range(len(f)))
    )

    out = []
    for landmark in landmarks:
        rec = df.query("inference_text == @landmark")
        if landmark == "TIME":
            # TIME matches two places typically, we want the one more to the left.
            rec = (
                rec.assign(
                    bbox_left=lambda f: [e[:, 0].min() for e in f.bbox]
                ).sort_values("bbox_left")
            ).head(1)
        assert len(rec) == 1, rec
        rec = rec.squeeze()

        _df = (
            candidates_df.assign(
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

        out.append(_df.head(1).assign(landmark=landmark, landmark_bbox=[rec.bbox]))

    return (
        pd.concat(out)
        .sort_values(["box_id", "belowness_score"])
        .assign(
            inference_text=lambda f: f.inference_text.where(
                ~f.duplicated(subset="box_id"), None
            ),
            inference_score=lambda f: f.inference_score.where(
                f.inference_text.notnull(), None
            ),
            bbox=lambda f: f.bbox.where(f.inference_text.notnull(), None),
        )
        .loc[
            :,
            [
                "landmark",
                "inference_text",
                "inference_score",
                # "distance_from_ref",
                # "angle_from_ref",
                # "belowness_score",
                # "box_id",
                "landmark_bbox",
                "bbox",
            ],
        ]
        .rename(columns={"bbox": "match_bbox"})
        .sort_values("landmark")
        .reset_index(drop=True)
    )


def ssocr_subimage(im, bbox, selection, ssocr_config):
    max_x, max_y = im.size
    xy = make_polygons_2d(bbox)

    l, t, r, b = (xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max())
    w = abs(l - r)
    h = abs(t - b)

    s = selection
    init_crop_frame = [
        max(0, int(round(e)))
        for e in (
            min(l + s[0] * w, max_x),
            min(t + s[1] * h, max_y),
            min(r + s[2] * w, max_x),
            min(b + s[3] * h, max_y),
        )
    ]

    # crop_frame = tidy_crop_frame(np.asarray(im), init_crop_frame)
    crop_frame = init_crop_frame
    imc = im.crop(crop_frame)

    return imc, run_ssocr(np.asarray(imc), *ssocr_config)


config = {
    "SPEED": dict(
        sel=(-0.2, 1.25, 1.03, 2.7),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
    "WATTS": dict(
        sel=(-0.2, 1.1, 0.6, 2.5),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
    "CADENCE": dict(
        sel=(0, 1.25, 0.20, 2.9),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
    "DISTANCE": dict(
        sel=(0, 1.3, 0.75, 3.25),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
    "TIME": dict(
        sel=(-1.2, 1.2, 1.45, 2.75),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
    "CALORIES": dict(
        sel=(0, 1.2, 0.2, 2.5),
        ssocr_conf=(
            {"number-digits": -1, "threshold": 40},
            [],
        ),
    ),
}
