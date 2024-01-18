import datetime
import json
from pathlib import Path

from transformers import CLIPProcessor, CLIPModel, AutoTokenizer
import requests
from io import BytesIO
import numpy as np
from PIL import Image
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
import logging
import nncf
from openvino.runtime import Core
from imagenetv2_pytorch import ImageNetV2Dataset
from itertools import islice

from validation_utils import imagenet_classes, imagenet_templates

nncf.set_log_level(logging.ERROR)


fp16_model_path = 'clip-vit-base-patch16.xml'

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
max_length = model.config.text_config.max_position_embeddings
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch16")


def check_text_data(data):
    """
    Check if the given data is text-based.
    """
    if isinstance(data, str):
        return True
    if isinstance(data, list):
        return all(isinstance(x, str) for x in data)
    return False


def get_pil_from_url(url):
    """
    Downloads and converts an image from a URL to a PIL Image object.
    """
    response = requests.get(url, verify=False, timeout=20)
    image = Image.open(BytesIO(response.content))
    return image.convert("RGB")


def collate_fn(example, image_column="image_url", text_column="caption"):
    """
    Preprocesses an example by loading and transforming image and text data.
    Checks if the text data in the example is valid by calling the `check_text_data` function.
    Downloads the image specified by the URL in the image_column by calling the `get_pil_from_url` function.
    If there is any error during the download process, returns None.
    Returns the preprocessed inputs with transformed image and text data.
    """
    assert len(example) == 1
    example = example[0]

    if not check_text_data(example[text_column]):
        raise ValueError("Text data is not valid")

    url = example[image_column]
    try:
        image = get_pil_from_url(url)
        h, w = image.size
        if h == 1 or w == 1:
            return None
    except Exception:
        return None

    inputs = processor(text=example[text_column], images=[image], return_tensors="pt", padding=True)
    if inputs['input_ids'].shape[1] > max_length:
        return None
    return inputs


def prepare_calibration_data(dataloader, init_steps):
    """
    This function prepares calibration data from a dataloader for a specified number of initialization steps.
    It iterates over the dataloader, fetching batches and storing the relevant data.
    """
    data = []
    print(f"Fetching {init_steps} samples for the initialization...")
    with tqdm(total=init_steps) as pbar:
        for batch in dataloader:
            if len(data) == init_steps:
                break
            if batch:
                pbar.update(1)
                with torch.no_grad():
                    data.append(
                        {
                            "pixel_values": batch["pixel_values"].to("cpu"),
                            "input_ids": batch["input_ids"].to("cpu"),
                            "attention_mask": batch["attention_mask"].to("cpu")
                        }
                    )
    return data


def prepare_dataset(opt_init_steps):
    """
    Prepares a vision-text dataset for quantization.
    """
    dataset = load_dataset("conceptual_captions")
    train_dataset = dataset["train"].shuffle(seed=42)
    dataloader = torch.utils.data.DataLoader(train_dataset, collate_fn=collate_fn, batch_size=1)
    calibration_data = prepare_calibration_data(dataloader, opt_init_steps)
    return calibration_data


def quantize(calibration_data, ov_model):
    quantized_model = nncf.quantize(
        model=ov_model,
        calibration_dataset=nncf.Dataset(calibration_data),
        model_type=nncf.ModelType.TRANSFORMER,
        subset_size=len(calibration_data)
    )
    return quantized_model


def validate(ov_model, test_dataset_size):
    # Inspired by https://github.com/openai/CLIP/blob/main/notebooks/Prompt_Engineering_for_ImageNet.ipynb
    text_descriptions = [f"This is a photo of a {label}" for label in imagenet_classes]

    def preprocess(pil_images):
        inputs = dict(processor(text=text_descriptions, images=pil_images, return_tensors="pt", padding=True))
        assert inputs['input_ids'].shape[1] <= max_length, inputs['input_ids'].shape[1]
        return inputs

    def accuracy(output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


    images = ImageNetV2Dataset(transform=preprocess)
    torch.manual_seed(42)
    loader = torch.utils.data.DataLoader(images, batch_size=1, num_workers=0, collate_fn=lambda it: it[0], shuffle=True)

    top1, top5 = 0, 0
    test_dataset_size = len(loader) if test_dataset_size == -1 else test_dataset_size
    for i, (inputs, target) in enumerate(tqdm(islice(loader, test_dataset_size), total=test_dataset_size,
                                              desc="Validation")):
        logits = ov_model(inputs)[ov_model.output(0)]

        # measure accuracy
        acc1, acc5 = accuracy(
            output=torch.from_numpy(logits),
            target=torch.from_numpy(np.array([target], dtype=np.int64)),
            topk=(1, 5))
        top1 += acc1
        top5 += acc5

    top1 = (top1 / test_dataset_size) * 100
    top5 = (top5 / test_dataset_size) * 100

    return top1, top5


core = Core()
calibration_data = prepare_dataset(opt_init_steps=1000)
ov_model = core.read_model(fp16_model_path)

test_dataset_size = 1000

top1_fp32, top5_fp32 = validate(core.compile_model(ov_model), test_dataset_size)

save_dir = Path("metrics") / f"test_{test_dataset_size}"
metrics_per_size = []
start_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
for i, calibration_dataset_size in enumerate(
    list(range(1, 20, 1)) +
    list(range(20, 50, 2)) +
    list(range(50, 100, 5)) +
    list(range(100, 1001, 50))
    # [1, 2]
):
    calibration_data_subset = calibration_data[:calibration_dataset_size]
    quantized_model = quantize(calibration_data_subset, ov_model)
    top1, top5 = validate(core.compile_model(quantized_model), test_dataset_size)

    metrics_dict = {
        "calibration_dataset_size": calibration_dataset_size,
        "top1_int8": top1,
        "top5_int8": top5
    }

    if i == 0:
        metrics_dict["top1_fp32"] = top1_fp32
        metrics_dict["top5_fp32"] = top5_fp32

    print(f"\nSize: {calibration_dataset_size}. Metrics: {metrics_dict}\n")
    metrics_per_size.append(metrics_dict)

    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / f"metrics_{start_time}.json".replace(':', '%'), "w") as f:
        json.dump(metrics_per_size, f, indent=4)
