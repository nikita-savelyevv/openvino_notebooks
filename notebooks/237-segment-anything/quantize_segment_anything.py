import pickle
import sys

import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List
from zipfile import ZipFile

import torch
import torch.utils.data as data
import openvino as ov
import cv2
import numpy as np
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry
from tqdm import tqdm

import nncf
from utils import preprocess_image, postprocess_masks, resizer, show_mask, show_points, automatic_mask_generation, \
    draw_anns, SamExportableModel

sys.path.append("../utils")
from notebook_utils import download_file


checkpoint = "sam_vit_b_01ec64.pth"
model_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
model_type = "vit_b"

DEVICE = "CPU"

download_file(model_url)


sam = sam_model_registry[model_type](checkpoint=checkpoint)

core = ov.Core()


ov_encoder_path = Path("encoder/sam_image_encoder.xml")
ov_encoder_path_compressed = Path("encoder/sam_image_encoder_compressed.xml")
# ov_encoder_path = Path("encoder/sam_image_encoder_int8.xml")
ov_decoder_path = Path("decoder/sam_mask_predictor.xml")
ov_decoder_path_compressed = Path("decoder/sam_mask_predictor_compressed.xml")

COLLECT_CALIBRATION_DATA = False


@contextmanager
def calibration_data_collection():
    global COLLECT_CALIBRATION_DATA
    try:
        COLLECT_CALIBRATION_DATA = True
        yield
    finally:
        COLLECT_CALIBRATION_DATA = False


def get_encoder():
    if not ov_encoder_path.exists():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            ov_encoder_model = ov.convert_model(sam.image_encoder, example_input=torch.zeros(1,3,1024,1024), input=([1,3,1024,1024],))
        ov.save_model(ov_encoder_model, ov_encoder_path)
    else:
        ov_encoder_model = core.read_model(ov_encoder_path)
    ov_encoder = core.compile_model(ov_encoder_model, DEVICE)
    return ov_encoder


def get_decoder():
    if not ov_decoder_path.exists():
        exportable_model = SamExportableModel(sam, return_single_mask=True)
        embed_dim = sam.prompt_encoder.embed_dim
        embed_size = sam.prompt_encoder.image_embedding_size
        dummy_inputs = {
            "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
            "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
            "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        }
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            ov_model = ov.convert_model(exportable_model, example_input=dummy_inputs)
        ov.save_model(ov_model, ov_decoder_path)
    else:
        ov_model = core.read_model(ov_decoder_path)
    ov_decoder = core.compile_model(ov_model, DEVICE)
    return ov_decoder


def point_inference(ov_decoder, image, image_embeddings, input_point, input_label, show_figure, save_figure_path):
    coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    coord = resizer.apply_coords(coord, image.shape[:2]).astype(np.float32)

    inputs = {
        "image_embeddings": image_embeddings,
        "point_coords": coord,
        "point_labels": label,
    }

    results = ov_decoder(inputs)

    masks = results[ov_decoder.output(0)]
    masks = postprocess_masks(masks, image.shape[:-1])
    masks = masks > 0.0

    if save_figure_path is None and not show_figure:
        return
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.tight_layout()
    if save_figure_path is not None:
        plt.savefig(save_figure_path)
    if show_figure:
        plt.show()
    plt.cla()


def automatic_inference(ov_encoder, ov_decoder, image, show_figure, save_figure_path=None):
    prediction = automatic_mask_generation(ov_encoder, ov_decoder, image)

    out = draw_anns(image, prediction).astype(np.uint8)
    if save_figure_path is None and not show_figure:
        return

    if save_figure_path is not None:
        cv2.imwrite(str(save_figure_path), out[:, :, ::-1])
    plt.imshow(out)
    plt.axis('off')
    plt.tight_layout()
    if show_figure:
        plt.show()
    plt.cla()


def collect_calibration_data_for_decoder(ov_encoder, ov_decoder, calibration_dataset_size: int,
                                         calibration_cache_path: Path,
                                         call_automatic_inference: bool, call_point_inference: bool):
    np.random.seed(42)

    assert call_point_inference or call_automatic_inference
    DATA_URL = "https://ultralytics.com/assets/coco128.zip"
    OUT_DIR = Path('.')

    download_file(DATA_URL, directory=OUT_DIR, show_progress=True)

    if not (OUT_DIR / "coco128/images/train2017").exists():
        with ZipFile('coco128.zip', "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)

    class COCOLoader(data.Dataset):
        def __init__(self, images_path):
            self.images = list(Path(images_path).iterdir())

        def __getitem__(self, index):
            if isinstance(index, slice):
                return [self.read_image(image_path) for image_path in self.images[index]]
            return self.read_image(self.images[index])

        def read_image(self, image_path):
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

        def __len__(self):
            return len(self.images)

    class DecoderWrapper:
        def __init__(self, ov_decoder: ov.CompiledModel, forward_data: List):
            self.ov_decoder = ov_decoder
            self.forward_data = forward_data

        def __call__(self, inputs):
            if COLLECT_CALIBRATION_DATA:
                self.forward_data.append(inputs)
            return self.ov_decoder(inputs)

        def output(self, idx):
            return self.ov_decoder.output(idx)

    if not calibration_cache_path.exists():
        calibration_data = []
        ov_decoder_wrapped = DecoderWrapper(ov_decoder, calibration_data)
        coco_dataset = COCOLoader(OUT_DIR / 'coco128/images/train2017')
        with calibration_data_collection():
            for image in tqdm(coco_dataset[:calibration_dataset_size], desc="Collecting calibration data"):
                if call_automatic_inference:
                    automatic_inference(ov_encoder, ov_decoder_wrapped, image, show_figure=False, save_figure_path=None)
                if call_point_inference:
                    n_points = np.random.randint(1, 64)
                    h, w = image.shape[:2]
                    ys, xs = np.random.randint(0, h, n_points), np.random.randint(0, w, n_points)
                    input_points = np.stack((ys, xs), axis=1)
                    input_label = np.ones(n_points)
                    image_embeddings = ov_encoder(preprocess_image(image))[ov_encoder.output(0)]
                    point_inference(ov_decoder_wrapped, image, image_embeddings, input_points, input_label,
                                    show_figure=False, save_figure_path=None)
        calibration_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_cache_path, "wb") as f:
            pickle.dump(calibration_data, f)
    else:
        with open(calibration_cache_path, "rb") as f:
            calibration_data = pickle.load(f)

    return calibration_data


def quantize_decoder(ov_encoder, ov_decoder, calibration_dataset_size, calibration_cache_path, save_model_path: Path,
                     sq_alpha, call_automatic_inference, call_point_inference):
    if save_model_path.exists():
        quantized_ov_decoder = core.read_model(save_model_path)
    else:
        calibration_data = collect_calibration_data_for_decoder(
            ov_encoder, ov_decoder, calibration_dataset_size, calibration_cache_path,
            call_automatic_inference, call_point_inference)
        quantized_ov_decoder = nncf.quantize(
            core.read_model(ov_decoder_path),
            calibration_dataset=nncf.Dataset(calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(calibration_data),
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=sq_alpha)
        )
        ov.save_model(quantized_ov_decoder, save_model_path)
    compiled_quantized_ov_decoder = core.compile_model(quantized_ov_decoder, DEVICE)
    return compiled_quantized_ov_decoder


class TimedDecoderWrapper:
    def __init__(self, ov_decoder):
        self.ov_decoder = ov_decoder
        self.total_time = 0
        self.call_idx = 0

    def __call__(self, inputs):
        # for k, v in inputs.items():
        #     save_path = Path(f"inputs/{k}/{self.call_idx}.npy")
        #     save_path.parent.mkdir(parents=True, exist_ok=True)
        #     np.save(save_path, v.astype(np.float32))
        # self.call_idx += 1

        # for k, v in inputs.items():
        #     print(k, v.shape)
        # exit(0)

        start_time = datetime.now()
        result = self.ov_decoder(inputs)
        self.total_time += (datetime.now() - start_time).total_seconds()
        return result

    def output(self, idx):
        return self.ov_decoder.output(idx)


ov_encoder = get_encoder()
ov_decoder = get_decoder()

results_path = Path("./")

calibration_dataset_size = 128
sq_alpha = 0.50
call_point_inference = bool(1)
call_automatic_inference = bool(0)
quantized_model_path = Path(f"decoder/quantized/size{calibration_dataset_size}_sq{sq_alpha:.2f}"
                            f"{'_ai' if call_automatic_inference else ''}"
                            f"{'_pi' if call_point_inference else ''}")
calibration_cache_path = Path(f"calibration_data/coco{calibration_dataset_size}"
                              f"{'_ai' if call_automatic_inference else ''}"
                              f"{'_pi' if call_point_inference else ''}.pkl")
results_path = quantized_model_path
ov_decoder = quantize_decoder(ov_encoder, ov_decoder,
                              calibration_dataset_size=calibration_dataset_size,
                              calibration_cache_path=calibration_cache_path,
                              save_model_path=quantized_model_path / "sam_mask_predictor_quantized.xml",
                              sq_alpha=sq_alpha,
                              call_point_inference=call_point_inference,
                              call_automatic_inference=call_automatic_inference)


# ov.save_model(nncf.compress_weights(core.read_model(ov_decoder_path)), ov_decoder_path_compressed)
# ov_decoder = core.compile_model(ov_decoder_path_compressed, DEVICE)

# ov.save_model(nncf.compress_weights(core.read_model(ov_encoder_path)), ov_encoder_path_compressed)
# ov_encoder = core.compile_model(ov_encoder_path_compressed, DEVICE)

ov_decoder = TimedDecoderWrapper(ov_decoder)

image = cv2.imread('truck.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

preprocessed_image = preprocess_image(image)
start_time = datetime.now()
encoding_results = ov_encoder(preprocessed_image)

image_embeddings = encoding_results[ov_encoder.output(0)]

input_point = np.array([[500, 375], [1125, 625], [575, 750], [1405, 575]])
input_label = np.array([1, 1, 1, 1])

ov_decoder.total_time = 0
point_inference(ov_decoder, image, image_embeddings, input_point, input_label,
                show_figure=False,
                save_figure_path=results_path / "points_prediction.jpg")
print("Point inference time:", ov_decoder.total_time)

ov_decoder.total_time = 0
automatic_inference(ov_encoder, ov_decoder, image,
                    show_figure=False,
                    save_figure_path=results_path / "automatic_prediction.jpg")
print("Automatic inference time:", ov_decoder.total_time)


# Decoder
# FP32: 13.16
# Compressed: 13.34 0.99x
# Quantized: 11.29  1.17x

# Encoder
# FP32: 0.94 sec.               ; 1.09 FPS
# Compressed: 0.97 sec. x0.97   ; 1.08 FPS
# Quantized: 0.69 sec. x1.36    ; 1.48 FPS
