import cv2
import numpy as np
import scipy.optimize
from skimage.transform import AffineTransform, warp
from sklearn.cluster import KMeans
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks


def create_templates():
    templates = []
    for i in range(10):
        img = np.zeros((5, 3), dtype=np.uint8)
        if i != 1 and i != 4:
            img[0, 1] = 1
            img[4, 1] = 1
        if i != 1 and i != 2 and i != 3 and i != 7:
            img[2, 1] = 1
        if i != 5 and i != 6:
            img[1, 2] = 1
            img[1, 0] = 1
        if i != 2:
            img[3, 0] = 1
            img[3, 2] = 1
        if i != 1 and i != 2 and i != 3 and i != 7:
            img[2, 0] = 1
            img[2, 2] = 1
        templates.append(img)
    return templates


def preprocess_image(image, grayscale_output=True):
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)

    if grayscale_output:
        probabilities = kmeans.transform(pixels)[:, 1] / np.sum(
            kmeans.transform(pixels), axis=1
        )
        probabilities_image = probabilities.reshape(image.shape[:-1])

        # Ensure that the background has lower intensity values
        if np.mean(probabilities_image) > 0.5:
            probabilities_image = 1 - probabilities_image

        return probabilities_image
    else:
        binary_image = kmeans.labels_.reshape(image.shape[:-1])

        if np.mean(binary_image) > 0.5:
            binary_image = 1 - binary_image

        return binary_image


def image_score(transformed_image, templates):
    max_scores = []
    for template in templates:
        result = cv2.matchTemplate(
            transformed_image, template.astype(np.float32), cv2.TM_CCOEFF_NORMED
        )
        max_scores.append(np.max(result))
    return np.mean(max_scores)


def image_transform(params, image):
    tform = AffineTransform(
        rotation=params[0],
        scale=(params[1], params[2]),
        translation=(params[3], params[4]),
    )
    transformed_image = warp(image, tform.inverse)
    return transformed_image


def objective(params, *args):
    transformed_image = image_transform(params, args[0])
    return -image_score(transformed_image, args[1])


def straighten_image(image, templates):
    init_params = [0, 1, 1, 0, 0]
    result = scipy.optimize.minimize(
        objective, init_params, args=(image, templates), method="Powell"
    )
    best_params = result.x
    straightened_image = image_transform(best_params, image)
    return straightened_image


def straighten_image_without_templates(
    image, canny_low_threshold=50, canny_high_threshold=150
):
    edges = canny(
        image, low_threshold=canny_low_threshold, high_threshold=canny_high_threshold
    )
    hspace, angles, distances = hough_line(edges)

    # Find the most dominant lines
    _, _, angles = hough_line_peaks(hspace, angles, distances)

    # Compute the mean angle of the dominant lines
    mean_angle = np.mean(angles)

    # Calculate the angle to rotate the image in the range of -45 to 45 degrees
    rotation_angle = np.rad2deg(mean_angle) % 180
    if rotation_angle > 90:
        rotation_angle -= 180

    # Apply the rotation to the image
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
    straightened_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))

    return straightened_image


def sig_image_score(transformed_image):
    # Compute the score for the transformed image.
    # In this case, we look for the maximum sum of squared gradients.
    sobel_x = cv2.Sobel(transformed_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(transformed_image, cv2.CV_64F, 0, 1, ksize=5)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return np.sum(gradient_magnitude)


def sig_image_transform(params, image):
    # Apply the affine transform to the image.
    tform = AffineTransform(
        rotation=params[0],
        scale=(params[1], params[2]),
        translation=(params[3], params[4]),
    )
    transformed_image = warp(image, tform.inverse)
    return transformed_image


def sig_objective(params, *args):
    transformed_image = sig_image_transform(params, args[0])
    return -sig_image_score(transformed_image)


def straighten_image_gradient(image):
    # Initialize the optimization parameters.
    init_params = [
        0,
        1,
        1,
        0,
        0,
    ]  # rotation, scale_x, scale_y, translation_x, translation_y
    result = scipy.optimize.minimize(
        sig_objective, init_params, args=(image,), method="Powell"
    )
    best_params = result.x
    straightened_image = sig_image_transform(best_params, image)
    return straightened_image


def ocr_seven_segment_image(
    image, templates, digit_width=3, digit_height=5, threshold=0.8
):
    recognized_digits = []
    for x in range(0, image.shape[1] - digit_width + 1, digit_width):
        best_score = -1
        best_digit = -1
        for idx, template in enumerate(templates):
            result = cv2.matchTemplate(
                image[:, x : x + digit_width],
                template.astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )
            score = np.max(result)
            if score > best_score:
                best_score = score
                best_digit = idx
        if best_score > threshold:
            recognized_digits.append(best_digit)
    return recognized_digits


def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocess the image
    preprocessed_image = preprocess_image(image, grayscale_output=False)

    # Convert the preprocessed binary image to the expected format
    preprocessed_image_gray = (preprocessed_image * 255).astype(np.uint8)

    # # Create templates for seven-segment LCD digits
    # templates = create_templates()

    # Straighten the image
    # straightened_image = straighten_image_without_templates(preprocessed_image_gray)
    straightened_image = straighten_image_gradient(preprocessed_image_gray)

    # # OCR the straightened image
    # recognized_digits = ocr_seven_segment_image(straightened_image, templates)

    # return recognized_digits
    return preprocessed_image_gray, straightened_image


if __name__ == "__main__":
    image_path = "path/to/image.jpg"
    recognized_digits = process_image(image_path)
    print("Recognized digits:", recognized_digits)
