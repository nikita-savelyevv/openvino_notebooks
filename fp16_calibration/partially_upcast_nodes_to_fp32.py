import gc
from typing import List, Dict, Union, Tuple, Callable

import numpy as np
from openvino._pyopenvino import Node
from openvino.runtime import Model
from openvino.runtime.op import Parameter, Constant
from openvino.runtime.opset12 import matmul, convolution
from openvino.runtime.utils.types import get_element_type

import openvino as ov
from tqdm import tqdm

ops_to_track_map = {
    'Convolution': convolution,
    'MatMul': matmul
}


def rt_info_name_to_keep_orig_precision():
    return 'precise_0'
    # return 'disable_fp16_compression_0'


def get_thresholds_per_op():
    return {
        'Convolution': (0.1, 0.003, 0.00),
        # 'MatMul': (0.1, 1e-6, 1e-6),
        'MatMul': (0.1, 0.04, 0.03),
    }


def partially_upcast_nodes_to_fp32(orig_model: Model, example_input: Union[List, Dict], half_type: str,
                                   batch_size: int = -1, thresholds_per_op: Dict[str, Tuple] = None,
                                   upcast_ratio: float = 0.1,
                                   verbose: bool = False) -> Model:
    assert half_type in ("f16", "bf16")
    device = "GPU" if half_type == "f16" else "CPU"

    nodes_to_track_names = get_nodes_to_track(orig_model)
    # node_to_upcast_names = []
    nodes_sqnrs = []
    batch_size = len(nodes_to_track_names) if batch_size == -1 else batch_size
    for i in tqdm(range(0, len(nodes_to_track_names), batch_size), disable=not verbose):
        model = orig_model.clone()
        name_to_node_map = {op.get_friendly_name(): op for op in model.get_ops()}
        nodes_to_track_names_batch = nodes_to_track_names[i: i + batch_size]
        nodes_to_track_batch = [name_to_node_map[node_name] for node_name in nodes_to_track_names_batch]

        insert_results_for_tracked_ops(model, nodes_to_track_batch)
        fp16_full_net_infer_values_batch = infer_full_net(nodes_to_track_batch, model, example_input)

        fp16_infer_values_batch = infer_nodes(nodes_to_track_batch, fp16_full_net_infer_values_batch, device, half_type)
        fp32_infer_values_batch = infer_nodes(nodes_to_track_batch, fp16_full_net_infer_values_batch, device, "f32")

        # node_to_upcast_names_batch = get_nodes_with_errors(nodes_to_track_batch, fp16_infer_values_batch,
        #                                                       fp32_infer_values_batch, thresholds_per_op, verbose)
        # node_to_upcast_names.extend(node_to_upcast_names_batch)
        nodes_sqnrs.extend(get_nodes_sqnrs(nodes_to_track_batch, fp16_infer_values_batch, fp32_infer_values_batch))

        del fp16_full_net_infer_values_batch, fp16_infer_values_batch, fp32_infer_values_batch, model, name_to_node_map
        gc.collect()

    node_names = [it[0] for it in nodes_sqnrs]
    node_sqnrs = np.array([it[1] for it in nodes_sqnrs], dtype=np.float32)
    sqnr_quantile = np.quantile(node_sqnrs, upcast_ratio)
    node_to_upcast_names = [node_names[i] for i in np.where(node_sqnrs <= sqnr_quantile)[0]]
    if verbose:
        print(f"SQNR {upcast_ratio:.2f}-quantile equals {sqnr_quantile:.2f}. "
              f"Upcasted {len(node_to_upcast_names)} of {len(node_names)} considered nodes:")
        print("\n".join(node_to_upcast_names))

    new_model = orig_model.clone()
    mark_nodes_to_upcast_to_fp32(new_model, node_to_upcast_names)
    return new_model


def get_nodes_to_track(model: Model) -> List:
    # get operations of interest
    nodes_to_track = []
    for i, op in enumerate(model.get_ordered_ops()):
        if op.get_type_name() not in ops_to_track_map.keys() or \
                any(map(lambda input: input.get_node().get_type_name() == 'Result', op.output(0).get_target_inputs())):
            continue
        nodes_to_track.append(op.get_friendly_name())
    return nodes_to_track


def insert_results_for_tracked_ops(model, nodes_to_track: List) -> (List, List, List):
    # additional outputs to track inputs and output values of operations of interest
    outputs = []
    for i, op in enumerate(nodes_to_track):
        outputs.append(op.output(0))
        node_0 = op.input_value(0).get_node()
        node_1 = op.input_value(1).get_node()
        for node in [node_0, node_1]:
            if node.get_type_name() != 'Constant' and not is_decompression_convert(node):  # for Consts we can take inputs from ov::Model
                outputs.append(node.output(0))
    model.add_outputs(outputs)


def get_const_value_from_ovmodel(node: Union[Constant, Node]) -> np.ndarray:
    if node.get_type_name() == 'Constant':
        assert node.get_element_type() == ov.Type.f32
        return node.get_data()
    elif is_decompression_convert(node):
        # if model is compressed and constant values flow through decompression convert
        const_node = node.input_value(0).get_node()
        assert const_node.get_type_name() == 'Constant'
        assert const_node.get_element_type().is_real()
        return np.array(node.input_value(0).get_node().get_data(), dtype=np.float32)
    else:
        raise Exception(
            f'Cannot get const values from ov.Model for node {node.get_friendly_name()} with type {node.get_type_name()}')


def is_decompression_convert(node: Node) -> bool:
    if node.get_type_name() != 'Convert':
        return False
    if len(node.get_rt_info()['is_decompression_0'].aslist()) > 0:
        return True
    return False


def infer_full_net(nodes_to_track: List[Node], orig_model: ov.Model, example_inputs: List) -> List[Tuple]:
    core = ov.Core()
    exec_net = core.compile_model(orig_model, "CPU", config={"INFERENCE_PRECISION_HINT": "f32"})
    request = exec_net.create_infer_request()
    results = request.infer(example_inputs)

    results_map = {}
    for key, val in results.items():
        for input_val in key.node.input_values():
            node_name = input_val.get_node().get_friendly_name()
            if input_val.get_node().get_type_name() == 'Constant' or is_decompression_convert(input_val.get_node()):
                results_map[node_name] = get_const_value_from_ovmodel(input_val.get_node())
            else:
                results_map[node_name] = val

    node_data_values = []  # each item contains a tuple with node output values and all input values
    for node in nodes_to_track:
        res_item = [results_map[node.get_friendly_name()]]
        for input_val in node.input_values():
            if input_val.get_node().get_type_name() == 'Constant' or is_decompression_convert(input_val.get_node()):
                res_item.append(get_const_value_from_ovmodel(input_val.get_node()))
            else:
                res_item.append(results_map[input_val.get_node().get_friendly_name()])
        node_data_values.append(tuple(res_item))
    del request, exec_net, results, results_map
    return node_data_values


def infer_nodes(nodes_to_track: List[Node], node_data_values: List[Tuple], device: str, precision: str) -> List:
    results = []
    for node, value in zip(nodes_to_track, node_data_values):
        results.append(infer_tracked_op(node, value[1:], device, precision))
    return results


def infer_tracked_op(op: Node, input_vals: Tuple, device: str, precision: str) -> np.ndarray:
    parameters = []
    for input_val in input_vals:
        parameters.append(Parameter(get_element_type(input_val.dtype), ov.PartialShape(input_val.shape)))

    if op.get_type_name() not in ops_to_track_map.keys():
        # todo: implement for other ops
        raise NotImplementedError(f"inference track for operations {op.get_type_name()} are not implemented yet")

    new_op = ops_to_track_map[op.get_type_name()](*parameters, **op.get_attributes())
    ov_model = ov.Model([new_op], parameters)

    exec_net = ov.Core().compile_model(ov_model, device, config={"INFERENCE_PRECISION_HINT": precision})
    request = exec_net.create_infer_request()
    result = request.infer(input_vals)
    assert len(result) == 1
    del request, exec_net, ov_model
    return result[0]


def get_nodes_with_errors(nodes: List[Node], fp16_infer_vals: List, fp32_infer_vals: List, thresholds: None,
                          verbose: bool = False) -> List[str]:
    nodes_with_errors = []
    for node, fp16_val, fp32_val in zip(nodes, fp16_infer_vals, fp32_infer_vals):
        if compare_tensors(node, fp16_val, fp32_val, thresholds, verbose):
            nodes_with_errors.append(node.get_friendly_name())
    return nodes_with_errors


def get_nodes_sqnrs(nodes: List[Node], fp16_infer_vals: List, fp32_infer_vals: List) -> List[Tuple[str, float]]:
    nodes_sqnrs = []
    for node, fp16_val, fp32_val in zip(nodes, fp16_infer_vals, fp32_infer_vals):
        sqnr = compute_sqnr(fp32_val, fp16_val)
        nodes_sqnrs.append((node.get_friendly_name(), sqnr))
    return nodes_sqnrs


def is_model_partially_upcasted(model) -> bool:
    for node in model.get_ordered_ops():
        if node.get_type_name() not in ops_to_track_map.keys():
            continue
        if rt_info_name_to_keep_orig_precision() in node.get_rt_info().keys():
            return True
    return False


def mark_nodes_to_upcast_to_fp32(model: ov.Model, nodes_with_errors: List[str]) -> None:
    for node in model.get_ordered_ops():
        if node.get_friendly_name() in nodes_with_errors:
            node.get_rt_info()[rt_info_name_to_keep_orig_precision()] = ''


def compute_sqnr(x, y):
    # x -- original, y -- quantized

    x = np.nan_to_num(x, posinf=np.finfo(x.dtype).max)
    y = np.nan_to_num(y, posinf=np.finfo(y.dtype).max)

    Ps = np.linalg.norm(x)
    Pn = np.nan_to_num(np.linalg.norm(x - y), posinf=np.finfo(np.float32).max)
    return 20 * np.log10(Ps / Pn)


def compare_tensors(node: Node, a: np.ndarray, b: np.ndarray, new_thresholds_per_op, verbose: bool = False) -> bool:
    """
    If values differ more than a certain metric then function returns True
    """
    assert np.array_equal(a.shape, b.shape), f'Shapes differ {a.shape} and {b.shape}'
    out_size = int(np.prod(a.shape))
    a_, b_ = np.reshape(a, out_size), np.reshape(b, out_size)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rel_error = np.abs(2 * (a_ - b_) / (np.abs(a_) + abs(b_)))

    mean_rel_error = np.mean(rel_error)
    thresholds_map = get_thresholds_per_op()
    if new_thresholds_per_op is not None:
        thresholds_map.update(new_thresholds_per_op)
    thresholds = thresholds_map[node.get_type_name()]
    rel_threshold = thresholds[0]
    rel_threshold_ratio = thresholds[1]
    rel_tol = thresholds[2]

    rel_diff_ratio = np.size(np.where(rel_error >= rel_threshold)) / out_size
    result = False
    if not(mean_rel_error < rel_tol) and rel_diff_ratio > rel_threshold_ratio:  # "not (...)" due to nans
        if verbose:
            print(f'Upcasted node {node.get_friendly_name()} with {rel_threshold:.2f} '
                  f'rel2_diff_ratio {rel_diff_ratio:.6f} and mean_rel_error {mean_rel_error:.6f}')
        result = True

    # node_name = node.get_friendly_name().replace('/', '%')
    # from pathlib import Path
    # outcome1 = int(not(mean_rel_error < rel_tol))
    # outcome2 = int(rel_diff_ratio > rel_threshold_ratio)
    # save_dir = Path("activations/tiny-sd-unet")
    # save_dir.mkdir(exist_ok=True)
    # filepath_fp16 = save_dir / f"{node_name}_fp16_{outcome1}_{outcome2}.npy"
    # filepath_fp32 = save_dir / f"{node_name}_fp32_{outcome1}_{outcome2}.npy"
    # np.save(filepath_fp16, a)
    # np.save(filepath_fp32, b)

    return result
