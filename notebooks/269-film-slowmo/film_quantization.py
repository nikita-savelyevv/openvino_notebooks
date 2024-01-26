from pathlib import Path

from tqdm import tqdm

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

from skimage.metrics import structural_similarity

from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
from notebooks.utils.memory_logger import MemoryLogger

DATA_PATH = Path("data")
CALIBRATION_DATA_URL = "https://vision.middlebury.edu/flow/data/comp/zip/other-color-twoframes.zip"


validation_data_dir = Path("/home/nsavel/workspace/datasets/vimeo_triplet/sequences")

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


def infer(compiled_model, result_dir=None, show=False, write=True, model_input=None, verbose=True):
    if model_input is None:
        model_input = get_demo_input()
    start_time = datetime.now()
    result = compiled_model(model_input)["image"]
    if verbose:
        print(f"Time: {(datetime.now() - start_time).total_seconds()}")
    image = result[0]
    image = np.clip(image, 0, 1)

    if write:
        cv2.imwrite(str(result_dir / "demo.png"), (cv2.cvtColor(image, cv2.COLOR_RGB2BGR) * 255).astype(np.uint8))
    if show:
        plt.imshow(image)
        plt.show()
    return image


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
        for target_size in [(768//2, 1024//2)]:
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


def collect_calibration_data_v2(calibration_dataset_size, frame_step):
    calibration_data = []

    data_dir = Path("/home/nsavel/workspace/datasets/DAVIS/JPEGImages/Full-Resolution")
    sub_dirs = list(data_dir.glob("*"))
    np.random.seed(42)
    sub_dirs = np.random.choice(sub_dirs, calibration_dataset_size, replace=True)
    for i in range(calibration_dataset_size):
        sub_dir = sub_dirs[i]
        filepaths = sorted(list(sub_dir.glob("*.jpg")))
        # frame_step = np.random.randint(low=1, high=3 + 1, size=1)[0]
        random_file_ind = np.random.randint(low=0, high=len(filepaths) - frame_step, size=1)[0]
        filepath1 = filepaths[random_file_ind]
        filepath2 = filepaths[random_file_ind + frame_step]
        # filepath2 = filepaths[random_file_ind]

        image1 = prepare_input_image(filepath1, target_size=(768 // 2, 1024 // 2))
        image2 = prepare_input_image(filepath2, target_size=(768 // 2, 1024 // 2))
        calibration_data.append(prepare_model_input(image1, image2, 0.5))

    return calibration_data


def collect_calibration_data_v3(calibration_dataset_size):
    calibration_data = []

    data_dir = Path("/home/nsavel/workspace/datasets/vimeo_triplet/sequences")
    sub_dirs = list(data_dir.rglob("**"))
    np.random.seed(42)
    sub_dirs = np.random.choice(sub_dirs, calibration_dataset_size, replace=False)
    for sub_dir in sub_dirs:
        image1 = prepare_input_image(sub_dir / "im1.png")
        image2 = prepare_input_image(sub_dir / "im3.png")
        calibration_data.append(prepare_model_input(image1, image2, 0.5))

    return calibration_data


def quantize(ov_model, quantized_model_path, calibration_dataset_size):
    if not quantized_model_path.exists():
        # calibration_data = collect_calibration_data()
        calibration_data = collect_calibration_data_v2(calibration_dataset_size, frame_step=1)
        # calibration_data = collect_calibration_data_v3(calibration_dataset_size)
        quantized_ov_model = nncf.quantize(
            ov_model,
            nncf.Dataset(calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(calibration_data),
            fast_bias_correction=True,
            ignored_scope=nncf.IgnoredScope(patterns=[
                "^.*resize",
                # "^.*warp",
                "interpolate_bilinear/*",
                # "interpolate_bilinear/d",   # +
                # "interpolate_bilinear/s",   # +
                # "interpolate_bilinear/S",   # +
                # "interpolate_bilinear/m",   # +-
                # "interpolate_bilinear/R",   # +
                # "interpolate_bilinear/r",   # +
                # "interpolate_bilinear/g",   # +-
                # "interpolate_bilinear/i",   # +
            ])
        )
        ov.save_model(quantized_ov_model, quantized_model_path)
    else:
        quantized_ov_model = core.read_model(quantized_model_path)
    return quantized_ov_model


def validate(ov_model, image_dirs, verbose=False):
    if isinstance(ov_model, ov.Model):
        ov_model = core.compile_model(ov_model)

    scores = []
    for sub_dir in tqdm(image_dirs, disable=not verbose, desc="Validating"):
        image1 = prepare_input_image(sub_dir / "im1.png", scale_factor=1.25)
        image2 = prepare_input_image(sub_dir / "im2.png", scale_factor=1.25)
        image3 = prepare_input_image(sub_dir / "im3.png", scale_factor=1.25)
        result = infer(ov_model, model_input=prepare_model_input(image1, image3, 0.5), write=False, verbose=False)
        ssim = structural_similarity(image2[0], result, data_range=1, channel_axis=2)
        scores.append(ssim)
    return np.mean(scores), scores


def qwac(ov_model, calibration_size=100, test_size=100):
    calibration_dataset = collect_calibration_data_v3(calibration_size)
    np.random.seed(41)
    validation_dataset = np.random.choice(list(validation_data_dir.rglob("**")), test_size, replace=False).tolist()

    for max_drop in [0.1, 0.05, 0.025, 0.01, 0.005]:
        quantized_model = nncf.quantize_with_accuracy_control(
            ov_model,
            nncf.Dataset(calibration_dataset),
            nncf.Dataset(validation_dataset),
            validation_fn=validate,
            max_drop=max_drop,
            preset=nncf.QuantizationPreset.MIXED,
            advanced_quantization_parameters=nncf.AdvancedQuantizationParameters(),
            # advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(tune_hyperparams=True)
        )

        save_dir = Path(f"qwac_att3/{max_drop:.3f}")
        ov.save_model(quantized_model, save_dir / "model.xml")
        infer(core.compile_model(quantized_model), save_dir)


ov_model = get_ov_model()

# qwac(ov_model)
# exit(0)

# compiled_model = core.compile_model(MODEL_PATH, device)
# infer(compiled_model, Path("./"))

quantized_model_path = \
    QUANTIZED_MODEL_DIR / "dataset3/size10_359-512/model.xml"
# quantized_model_path = Path("qwac_att3/0.050/model.xml")

validation_dataset_size = 100
np.random.seed(41)
validation_dirs = np.random.choice(list(validation_data_dir.rglob("**")), validation_dataset_size, replace=False)
# print(validate(ov_model, validation_dirs, verbose=True)[0])
print(validate(core.compile_model(quantized_model_path), validation_dirs, verbose=True)[0])
# print(validate(core.compile_model("./models/quantized/dataset3/size10_359-512/ignored-scopes/interpolate_bilinear_resize/model.xml"), validation_dirs, verbose=True)[0])
# print(validate(core.compile_model("qwac/0.050/model.xml"), validation_dirs, verbose=True)[0])
# exit(0)

# quantized_model_path = QUANTIZED_MODEL_DIR / "dataset2/120-128-time0.50-crop1024x768/model.xml"
memory_logger = MemoryLogger(quantized_model_path.parent).start_logging()
quantize(ov_model, quantized_model_path, calibration_dataset_size=10)
memory_logger.stop_logging()

compiled_quantized_model = core.compile_model(quantized_model_path, device)
infer(compiled_quantized_model, quantized_model_path.parent)

# FP32: 4.85
# INT8: 2.7
