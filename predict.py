import numpy as np
import tensorflow as tf
import cv2
from PIL import Image


# Load CNN model
model = tf.keras.models.load_model(
    "models/handwritten_digit_cnn.keras"
)


def preprocess_digit(digit):
    """Prepare one digit for the CNN."""

    height, width = digit.shape
    size = max(height, width)

    # Create square canvas
    canvas = np.zeros(
        (size, size),
        dtype=np.uint8
    )

    x_offset = (size - width) // 2
    y_offset = (size - height) // 2

    canvas[
        y_offset:y_offset + height,
        x_offset:x_offset + width
    ] = digit

    # Resize to 64x64
    digit = cv2.resize(
        canvas,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    # Normalize
    digit = digit.astype(np.float32) / 255.0

    # CNN input shape
    digit = digit.reshape(
        1, 64, 64, 1
    )

    return digit


# -------------------------------------------------
# SINGLE DIGIT
# -------------------------------------------------

def predict_digit(image):
    """Predict one handwritten digit."""

    image = np.array(image)

    # RGB → grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Blur noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Convert digit to white
    # and background to black
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    # Find contours
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Remove tiny noise
    contours = [
        c for c in contours
        if cv2.contourArea(c) > 50
    ]

    if not contours:
        raise ValueError(
            "Could not detect the digit. "
            "Try a clearer image."
        )

    # Get bounding box around all detected parts
    x_min = min(
        cv2.boundingRect(c)[0]
        for c in contours
    )

    y_min = min(
        cv2.boundingRect(c)[1]
        for c in contours
    )

    x_max = max(
        cv2.boundingRect(c)[0]
        + cv2.boundingRect(c)[2]
        for c in contours
    )

    y_max = max(
        cv2.boundingRect(c)[1]
        + cv2.boundingRect(c)[3]
        for c in contours
    )

    # Padding
    padding = 15

    x_min = max(
        0,
        x_min - padding
    )

    y_min = max(
        0,
        y_min - padding
    )

    x_max = min(
        binary.shape[1],
        x_max + padding
    )

    y_max = min(
        binary.shape[0],
        y_max + padding
    )

    digit = binary[
        y_min:y_max,
        x_min:x_max
    ]

    processed = preprocess_digit(digit)

    # Prediction
    probabilities = model.predict(
        processed,
        verbose=0
    )[0]

    prediction = int(
        np.argmax(probabilities)
    )

    confidence = (
        probabilities[prediction] * 100
    )

    return (
        prediction,
        round(float(confidence), 2)
    )


# -------------------------------------------------
# MULTIPLE DIGITS
# -------------------------------------------------

def segment_digits(image):
    """Detect digits from the main handwritten line."""

    image = np.array(image)

    # RGB -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce camera noise
    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive threshold works better
    # with uneven lighting
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # -----------------------------------------
    # FIND THE MAIN HORIZONTAL WRITING LINE
    # -----------------------------------------

    row_counts = np.sum(binary > 0, axis=1)

    max_count = np.max(row_counts)

    # Rows containing significant handwriting
    threshold = max(
        5,
        max_count * 0.30
    )

    active_rows = row_counts > threshold

    # Find groups of consecutive active rows
    groups = []

    start = None

    for i, active in enumerate(active_rows):

        if active and start is None:
            start = i

        elif not active and start is not None:
            groups.append(
                (start, i - 1)
            )
            start = None

    if start is not None:
        groups.append(
            (start, len(active_rows) - 1)
        )

    if not groups:
        raise ValueError(
            "Could not find the handwritten digits."
        )

    # Choose the group with the most ink
    best_group = max(
        groups,
        key=lambda g: np.sum(
            row_counts[g[0]:g[1] + 1]
        )
    )

    y1, y2 = best_group

    # Add vertical padding
    padding_y = 15

    y1 = max(
        0,
        y1 - padding_y
    )

    y2 = min(
        binary.shape[0],
        y2 + padding_y
    )

    # Crop to the writing line
    line = binary[y1:y2, :]

    # -----------------------------------------
    # REMOVE SMALL NOISE
    # -----------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    line = cv2.morphologyEx(
        line,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # -----------------------------------------
    # FIND DIGIT CONTOURS
    # -----------------------------------------

    contours, _ = cv2.findContours(
        line,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        x, y, w, h = cv2.boundingRect(
            contour
        )

        area = cv2.contourArea(contour)

        # Ignore tiny noise
        if area < 40:
            continue

        # Ignore tiny objects
        if w < 8 or h < 15:
            continue

        # Ignore extremely wide objects
        if w > line.shape[1] * 0.25:
            continue

        # Ignore extremely tall objects
        if h > line.shape[0] * 0.95:
            continue

        boxes.append(
            (x, y + y1, w, h)
        )

    # -----------------------------------------
    # SORT LEFT -> RIGHT
    # -----------------------------------------

    boxes.sort(
        key=lambda box: box[0]
    )

    return binary, boxes

def predict_digits(image):
    """Predict multiple handwritten digits."""

    binary, boxes = segment_digits(image)

    if len(boxes) == 0:
        raise ValueError(
            "No digits detected. "
            "Try a clearer image."
        )

    results = []

    for x, y, w, h in boxes:

        padding = 15

        x1 = max(
            0,
            x - padding
        )

        y1 = max(
            0,
            y - padding
        )

        x2 = min(
            binary.shape[1],
            x + w + padding
        )

        y2 = min(
            binary.shape[0],
            y + h + padding
        )

        digit = binary[
            y1:y2,
            x1:x2
        ]

        processed = preprocess_digit(
            digit
        )

        probabilities = model.predict(
            processed,
            verbose=0
        )[0]

        prediction = int(
            np.argmax(probabilities)
        )

        confidence = (
            probabilities[prediction]
            * 100
        )

        results.append({
            "digit": prediction,
            "confidence": round(
                float(confidence),
                2
            ),
            "box": (x, y, w, h)
        })

    return results
