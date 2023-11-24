from pathlib import Path
import nncf
from datetime import datetime

from zipfile import ZipFile
import matplotlib.pyplot as plt
import openvino as ov
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from urllib.request import urlretrieve
import numpy as np

from notebooks.utils.memory_logger import MemoryLogger

DATA_PATH = Path("data")
CALIBRATION_DATA_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip"

model_url = "https://www.kaggle.com/models/google/film/frameworks/tensorFlow2/variations/film/versions/1"
MODEL_PATH = Path("models/model.xml")
QUANTIZED_MODEL_DIR = Path("models/quantized")
IMAGES = {
    "https://raw.githubusercontent.com/google-research/frame-interpolation/main/photos/one.png": Path("data/one.png"),
    "https://raw.githubusercontent.com/google-research/frame-interpolation/main/photos/two.png": Path("data/two.png")
}


core = ov.Core()
device = "CPU"


def get_ov_model():
    inputs = dict(
        x0=tf.keras.layers.Input(shape=(None, None, 3)),
        x1=tf.keras.layers.Input(shape=(None, None, 3)),
        time=tf.keras.layers.Input(shape=(1)),
    )
    film_layer = hub.KerasLayer(model_url)(inputs)
    film_model = tf.keras.Model(inputs=inputs, outputs=film_layer)

    if not MODEL_PATH.exists():
        converted_model = ov.convert_model(film_model)
        ov.save_model(converted_model, MODEL_PATH)
    else:
        converted_model = core.read_model(MODEL_PATH)

    return converted_model


def prepare_input_image(filepath, crop_size=None, target_size=None, scale_factor=1, transpose=False):
    img = cv2.imread(str(filepath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img.astype(np.float32) / 255  # normalize to [0, 1]
    if crop_size is not None:
        img = img[:crop_size[0], :crop_size[1]]
    if target_size is not None:
        img = cv2.resize(img, (target_size[1], target_size[0]))
    if scale_factor != 1:
        img = cv2.resize(img, (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor)))
    if transpose:
        img = np.transpose(img, [1, 0, 2])
    img = img[np.newaxis, ...]  # add batch dim
    return img


def prepare_model_input(img1, img2, time):
    return {
        "x0": img1,
        "x1": img2,
        "time": np.array([[time]], dtype=np.float32)
    }


def get_demo_input():
    input_images = []
    for img_url in IMAGES:
        if not IMAGES[img_url].exists():
            urlretrieve(img_url, IMAGES[img_url])
        filename = str(IMAGES[img_url])
        img = prepare_input_image(filename)
        input_images.append(img)
    return prepare_model_input(*input_images, 0.5)


def infer(compiled_model, result_dir, show=False):
    model_input = get_demo_input()
    start_time = datetime.now()
    result = compiled_model(model_input)["image"]
    print(f"Time: {(datetime.now() - start_time).total_seconds()}")
    image = result[0]
    image = np.clip(image, 0, 1)

    cv2.imwrite(str(result_dir / "demo.png"), (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))
    if show:
        plt.imshow(image)
        plt.show()


def collect_calibration_data():
    calibration_data_path = DATA_PATH / "calibration_data"
    calibration_data_path.mkdir(exist_ok=True)

    import sys
    sys.path.append("../utils")
    from notebook_utils import download_file
    zip_filepath = download_file(CALIBRATION_DATA_URL, directory=calibration_data_path,
                                 show_progress=True)

    calibration_images_path = calibration_data_path / "other-data"
    if not calibration_images_path.exists():
        with ZipFile(zip_filepath, "r") as zip_ref:
            zip_ref.extractall(calibration_data_path)

    calibration_data = []
    for image_pair_dir in calibration_images_path.iterdir():
        # for target_size in [None, (768, 1024)]:
        for target_size in [(768, 1024)]:
            # for transpose in [False, True]:
            for transpose in [False]:
                # for scale_factor in [0.5, 1, 2]:
                for scale_factor in [1]:
                    image1 = prepare_input_image(image_pair_dir / "frame10.png",
                                                 scale_factor=scale_factor, transpose=transpose,
                                                 target_size=target_size)
                    image2 = prepare_input_image(image_pair_dir / "frame11.png",
                                                 scale_factor=scale_factor, transpose=transpose,
                                                 target_size=target_size)
                    # for time in [0.25, 0.5, 0.75]:
                    for time in [0.5]:
                        calibration_data.append(prepare_model_input(image1, image2, time))

    # calibration_data = []
    # for images_dir in Path("data/calibration_data/8x_interpolation").iterdir():
    #     image1 = prepare_input_image(images_dir / "0120.png", crop_size=(768, 1024))
    #     image2 = prepare_input_image(images_dir / "0128.png", crop_size=(768, 1024))
    #     # for time in [0.25, 0.5, 0.75]:
    #     for time in [0.5]:
    #         calibration_data.append(prepare_model_input(image1, image2, time))

    return calibration_data


def quantize(ov_model, quantized_model_path):
    if not quantized_model_path.exists():
        calibration_data = collect_calibration_data()
        quantized_ov_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(calibration_data),
            fast_bias_correction=True
        )
        ov.save_model(quantized_ov_model, quantized_model_path)
    else:
        quantized_ov_model = core.read_model(quantized_model_path)
    return quantized_ov_model


ov_model = get_ov_model()

# compiled_model = core.compile_model(MODEL_PATH, device)
# infer(compiled_model, Path("./"))

quantized_model_path = QUANTIZED_MODEL_DIR / "dataset1/time0.5_size1024x768/model.xml"
# quantized_model_path = QUANTIZED_MODEL_DIR / "dataset2/120-128-time0.50-crop1024x768/model.xml"
memory_logger = MemoryLogger(quantized_model_path.parent).start_logging()
quantize(ov_model, quantized_model_path)
memory_logger.stop_logging()

compiled_quantized_model = core.compile_model(quantized_model_path, device)
infer(compiled_quantized_model, quantized_model_path.parent)

# FP32: 4.85
# INT8: 2.7
