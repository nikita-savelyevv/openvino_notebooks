import datetime
import pickle
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import cv2
import openvino as ov
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from ultralytics import FastSAM

import torch.utils.data as data

import nncf
from notebooks.utils.notebook_utils import download_file

COLLECT_CALIBRATION_DATA = False
calibration_data = []

core = ov.Core()
DEVICE = "CPU"


@contextmanager
def calibration_data_collection():
    global COLLECT_CALIBRATION_DATA
    try:
        COLLECT_CALIBRATION_DATA = True
        yield
    finally:
        COLLECT_CALIBRATION_DATA = False


class OVWrapper:
    def __init__(self, ov_model, device="CPU", stride=32) -> None:
        self.model = core.read_model(ov_model)
        self.compiled_model = core.compile_model(self.model, device_name=device)

        self.stride = stride
        self.pt = True
        self.fp16 = False
        self.names = {0: "object"}

    def __call__(self, im, **_):
        if COLLECT_CALIBRATION_DATA:
            calibration_data.append(im)

        result = self.compiled_model(im)
        return torch.from_numpy(result[0]), torch.from_numpy(result[1])


def collect_calibration_data_for_decoder(model, calibration_dataset_size: int,
                                         calibration_cache_path: Path):
    global calibration_data

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

    if not calibration_cache_path.exists():
        coco_dataset = COCOLoader(OUT_DIR / 'coco128/images/train2017')
        with calibration_data_collection():
            for image in tqdm(coco_dataset[:calibration_dataset_size], desc="Collecting calibration data"):
                model(image, device=DEVICE, retina_masks=True, imgsz=640, conf=0.6, iou=0.9)
        calibration_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(calibration_cache_path, "wb") as f:
            pickle.dump(calibration_data, f)
    else:
        with open(calibration_cache_path, "rb") as f:
            calibration_data = pickle.load(f)

    return calibration_data


def quantize(model, save_model_path: Path, calibration_cache_path: Path,
             calibration_dataset_size: int, preset: nncf.QuantizationPreset):
    if save_model_path.exists():
        quantized_ov_decoder = core.read_model(save_model_path)
    else:
        calibration_data = collect_calibration_data_for_decoder(
            model, calibration_dataset_size, calibration_cache_path)
        quantized_ov_decoder = nncf.quantize(
            model.predictor.model.model,
            calibration_dataset=nncf.Dataset(calibration_data),
            preset=preset,
            subset_size=len(calibration_data),
            fast_bias_correction=True,
            ignored_scope=nncf.IgnoredScope(
                types=["Multiply", "Subtract", "Sigmoid"],  # ignore operations
                names=[
                    "/model.22/dfl/conv/Conv",  # in the post-processing subgraph
                    "/model.22/Add",
                    "/model.22/Add_1",
                    "/model.22/Add_2",
                    "/model.22/Add_3",
                    "/model.22/Add_4",
                    "/model.22/Add_5",
                    "/model.22/Add_6",
                    "/model.22/Add_7",
                    "/model.22/Add_8",
                    "/model.22/Add_9",
                    "/model.22/Add_10",
                ],
            )
        )
        ov.save_model(quantized_ov_decoder, save_model_path)
    compiled_quantized_ov_decoder = core.compile_model(quantized_ov_decoder, DEVICE)
    return compiled_quantized_ov_decoder


model_name = "FastSAM-x"
model = FastSAM(model_name)

# Run inference on an image
image_uri = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco_bike.jpg"
results = model(image_uri, device="cpu", retina_masks=True, imgsz=1024, conf=0.6, iou=0.9)


# instance segmentation model
ov_model_path = Path(f"{model_name}_openvino_model/{model_name}.xml")
if not ov_model_path.exists():
    ov_model = model.export(format="openvino", dynamic=True, half=False)

wrapped_model = OVWrapper(ov_model_path, device=DEVICE, stride=model.predictor.model.stride)
model.predictor.model = wrapped_model

results_path = Path("./")

calibration_dataset_size = 128
preset = nncf.QuantizationPreset.MIXED
call_point_inference = bool(1)
call_automatic_inference = bool(0)
quantized_model_path = Path(f"{model_name}_quantized/size{calibration_dataset_size}_{preset.value}_ign-scope_tmp")
calibration_cache_path = Path(f"calibration_data/coco{calibration_dataset_size}.pkl")
quantize(model, quantized_model_path / "FastSAM-x.xml", calibration_cache_path,
         calibration_dataset_size=calibration_dataset_size,
         preset=preset)
wrapped_model = OVWrapper(quantized_model_path / "FastSAM-x.xml", device=DEVICE, stride=model.predictor.model.stride)
model.predictor.model = wrapped_model
results_path = quantized_model_path

start_time = datetime.datetime.now()
for _ in range(100):
    ov_results = model(image_uri, device=DEVICE, retina_masks=True, imgsz=640, conf=0.6, iou=0.9)
print("Segmented in", (datetime.datetime.now() - start_time))

cv2.imwrite(str(results_path / "result.jpg"), ov_results[0].plot())

# FP32: 9.92 FPS, latency 9.35 FPS, 14.64 sec
# size128_mixed_ign-scope: 33.34 FPS, latency 27.15 FPS, 8.75 sec