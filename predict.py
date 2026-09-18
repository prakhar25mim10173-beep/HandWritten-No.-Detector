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


# SINGLE DIGIT

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

    # Convert digit to white and background to black
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

# MULTIPLE DIGITS

def _boxes_overlap_horizontally(box_a, box_b, min_overlap_ratio=0.5):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    left = max(ax, bx)
    right = min(ax + aw, bx + bw)
    overlap = max(0, right - left)
    smaller_width = min(aw, bw)
    if smaller_width == 0:
        return False
    return (overlap / smaller_width) >= min_overlap_ratio


def _boxes_overlap_vertically(box_a, box_b, min_overlap_ratio=0.2):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    top = max(ay, by)
    bottom = min(ay + ah, by + bh)
    overlap = max(0, bottom - top)
    smaller_height = min(ah, bh)
    if smaller_height == 0:
        return False
    return (overlap / smaller_height) >= min_overlap_ratio


def _merge_overlapping_boxes(boxes):

    boxes = list(boxes)
    merged_any = True
    while merged_any:
        merged_any = False
        result = []
        used = [False] * len(boxes)
        for i in range(len(boxes)):
            if used[i]:
                continue
            current = boxes[i]
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                if _boxes_overlap_horizontally(current, boxes[j]) and \
                   _boxes_overlap_vertically(current, boxes[j]):
                    cx, cy, cw, ch = current
                    ox, oy, ow, oh = boxes[j]
                    nx = min(cx, ox)
                    ny = min(cy, oy)
                    nx2 = max(cx + cw, ox + ow)
                    ny2 = max(cy + ch, oy + oh)
                    current = (nx, ny, nx2 - nx, ny2 - ny)
                    used[j] = True
                    merged_any = True
            result.append(current)
        boxes = result
    return boxes


def segment_digits(image, debug=False):

    image = np.array(image)

    # RGB -> grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    # Reduce camera noise
    gray_blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Adaptive threshold works better with uneven lighting
    binary = cv2.adaptiveThreshold(
        gray_blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # FIND THE MAIN HORIZONTAL WRITING LINE

    row_counts = np.sum(binary > 0, axis=1)

    max_count = np.max(row_counts)

    threshold = max(
        5,
        max_count * 0.30
    )

    active_rows = row_counts > threshold

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

    best_group = max(
        groups,
        key=lambda g: np.sum(
            row_counts[g[0]:g[1] + 1]
        )
    )

    y1, y2 = best_group

    padding_y = 15

    y1 = max(0, y1 - padding_y)
    y2 = min(binary.shape[0], y2 + padding_y)

    # Crop to the writing line
    line = binary[y1:y2, :]

    # BRIDGE SMALL GAPS (close, not open)

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    line_closed = cv2.morphologyEx(
        line,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )

    # FIND DIGIT CONTOURS

    contours, _ = cv2.findContours(
        line_closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    raw_boxes = [cv2.boundingRect(c) for c in contours]
    num_raw_components = len(raw_boxes)

    if not raw_boxes:
        raise ValueError(
            "Could not find the handwritten digits."
        )

    # ADAPTIVE AREA FILTER (relative to this image's median)

    areas = [w * h for (x, y, w, h) in raw_boxes]
    median_area = float(np.median(areas))
    area_floor = max(20.0, median_area * 0.15)

    area_filtered = [b for b, a in zip(raw_boxes, areas) if a >= area_floor]
    if not area_filtered:
        area_filtered = raw_boxes

    # ADAPTIVE HEIGHT FILTER (relative to tallest surviving box)

    max_height = max(h for (x, y, w, h) in area_filtered)
    height_filtered = [
        b for b in area_filtered
        if b[3] >= max_height * 0.25
    ]
    if not height_filtered:
        height_filtered = area_filtered

    line_width = line.shape[1]
    width_filtered = [
        b for b in height_filtered
        if b[2] <= line_width * 0.5
    ]
    if not width_filtered:
        width_filtered = height_filtered

    # MERGE FRAGMENTS OF THE SAME DIGIT

    merged = _merge_overlapping_boxes(width_filtered)

    # SORT LEFT -> RIGHT, RE-APPLY Y OFFSET

    merged.sort(key=lambda box: box[0])

    boxes = [
        (x, y + y1, w, h)
        for (x, y, w, h) in merged
    ]

    debug_info = None

    if debug:
        boxes_drawn = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
        for (x, y, w, h) in boxes:
            cv2.rectangle(boxes_drawn, (x, y), (x + w, y + h), (0, 255, 0), 2)

        debug_info = {
            "gray": gray,
            "binary_full": binary,
            "line_region": line,
            "line_closed": line_closed,
            "boxes_drawn": boxes_drawn,
            "num_raw_components": num_raw_components,
            "num_final_digits": len(boxes),
            "y_range": (y1, y2),
        }

    return binary, boxes, debug_info


def _adaptive_horizontal_padding(boxes, index, base_padding=15):

    x, y, w, h = boxes[index]

    left_pad = base_padding
    if index > 0:
        prev_x, prev_y, prev_w, prev_h = boxes[index - 1]
        gap = x - (prev_x + prev_w)
        left_pad = min(base_padding, max(0, gap // 2))

    right_pad = base_padding
    if index < len(boxes) - 1:
        next_x, next_y, next_w, next_h = boxes[index + 1]
        gap = next_x - (x + w)
        right_pad = min(base_padding, max(0, gap // 2))

    return left_pad, right_pad


def predict_digits(image, debug=False):
    """Predict multiple handwritten digits."""

    binary, boxes, debug_info = segment_digits(image, debug=debug)

    if len(boxes) == 0:
        raise ValueError(
            "No digits detected. "
            "Try a clearer image."
        )

    results = []
    debug_crops = [] if debug else None

    for i, (x, y, w, h) in enumerate(boxes):

        vertical_padding = 15
        left_pad, right_pad = _adaptive_horizontal_padding(boxes, i, base_padding=15)

        x1 = max(0, x - left_pad)
        y1 = max(0, y - vertical_padding)
        x2 = min(binary.shape[1], x + w + right_pad)
        y2 = min(binary.shape[0], y + h + vertical_padding)

        digit = binary[y1:y2, x1:x2]

        if debug:
            debug_crops.append(digit)

        processed = preprocess_digit(digit)

        probabilities = model.predict(
            processed,
            verbose=0
        )[0]

        prediction = int(np.argmax(probabilities))
        confidence = probabilities[prediction] * 100

        results.append({
            "digit": prediction,
            "confidence": round(float(confidence), 2),
            "box": (x, y, w, h)
        })

    if debug and debug_info is not None:
        debug_info["crops"] = debug_crops

    return results, debug_info
