import gc
from dataclasses import dataclass
from typing import List, Dict, Union

import numpy as np
from openvino._pyopenvino import Node
from openvino.runtime import Model
from openvino.runtime.op import Parameter, Constant
import openvino.runtime.opset12 as opset
from openvino.runtime.utils.types import get_element_type

import openvino as ov
from tqdm import tqdm

OPERATION_TYPE_MAP = {
    'MatMul': opset.matmul,
    'Convolution': opset.convolution,
    'Softmax': opset.softmax,
    'MVN': opset.mvn,
    'Multiply': opset.multiply,
    'Divide': opset.divide,
    'Add': opset.add,
    'Subtract': opset.subtract,
    'Concat': opset.concat,
    'Power': opset.power,
    'Transpose': opset.transpose,
    'Broadcast': opset.broadcast,
    'ShapeOf': opset.shape_of
}


ORIGINAL_PRECISION_RT_INFO_NAME = 'precise_0'


@dataclass
class TrackedNodeInfo:
    """
    Data associated with a node tracked for upcasting
    """
    node: Node          # Target node to track
    snr: float = None          # SNR of the target node
    input_nodes: List[Node] = None              # Input nodes of the target node
    node_output: ov.runtime.Output = None               # Output object of the target node
    input_node_outputs: Dict[Node, ov.runtime.Output] = None    # Outputs of non-const inputs of the target node
    node_result_full_precision: np.ndarray = None               # Result of the node in full precision
    node_result_half_precision: np.ndarray = None               # Result of the node in half precision
    input_results_full_precision: np.ndarray = None             # Results of the target node inputs in full precision


def partially_upcast_nodes_to_fp32(orig_model: Model, example_input: Union[List, Dict], half_type: str,
                                   batch_size: int = -1, operation_types: List[str] = None, upcast_ratio: float = 0.1,
                                   verbose: bool = False) -> Model:
    """
    Transform a model to upcast some nodes to be executed in full precision instead of half precision. These nodes are
    marked with runtime info flag.
    Nodes are selected based on Signal-to-Noise Ratio (SNR) metric: upcast_ratio fraction of tracked nodes with the
    lowest SNR are marked for full precision execution.
    
    :param orig_model: Model to process
    :param example_input: Example input for model inference
    :param half_type: Either 'f16' or 'bf16'
    :param batch_size: Number of nodes to process together during a single model inference. The lower the value is,
        the less memory footprint is, but the larger is the processing time. The value of -1 is used to disable
        batching.
    :param operation_types: Types of operations to consider. If None, MatMuls and Convolutions are considered.
    :param upcast_ratio: Fraction of nodes to upcast (with the lowest SNR). 0 - do not upcast anything, 1 - upcast every
        operation of the given types.
    :param verbose: If True, prints progress output.
    :return: Upcasted OV model with some nodes marked for full precision execution.
    """
    if half_type not in ("f16", "bf16"):
        raise ValueError(f"Half type must be either 'f16' or 'bf16'. Got {half_type}.")
    if operation_types is None:
        operation_types = ["MatMul", "Convolution"]
    for op_type in operation_types:
        if op_type not in OPERATION_TYPE_MAP:
            raise ValueError(f"Operation type must be one of the following {OPERATION_TYPE_MAP.keys()}. "
                             f"Got {op_type}")

    device = "GPU" if half_type == "f16" else "CPU"

    nodes_to_track_names = get_nodes_to_track(orig_model, operation_types)
    node_names_and_snrs = []
    batch_size = len(nodes_to_track_names) if batch_size == -1 or batch_size > len(nodes_to_track_names) else batch_size
    for i in tqdm(range(0, len(nodes_to_track_names), batch_size), desc="Processing",
                  disable=not verbose or batch_size == len(nodes_to_track_names)):
        model = orig_model.clone()
        name_to_node_map = {op.get_friendly_name(): op for op in model.get_ops()}
        nodes_to_track_batch = [TrackedNodeInfo(name_to_node_map[node_name]) for node_name in
                                nodes_to_track_names[i: i + batch_size]]
        # for node_info in nodes_to_track_batch:
        #     node_names_and_snrs.append((node_info.node.get_friendly_name(), 100500))
        # continue

        # Add outputs for non-constant inputs of tracked nodes
        insert_outputs_for_tracked_ops(model, nodes_to_track_batch)
        # Infer model to collect tracked operation results and results of their inputs in full precision
        infer_full_net(nodes_to_track_batch, model, example_input)
        # Infer nodes in half precision one by one using full precision inputs, collect half precision results
        infer_nodes(nodes_to_track_batch, device, half_type)

        # Compute operation SNR based on full precision and half precision results
        for node_info in nodes_to_track_batch:
            snr = compute_snr(node_info.node_result_full_precision, node_info.node_result_half_precision)
            node_names_and_snrs.append((node_info.node.get_friendly_name(), snr))

        gc.collect()

    node_names = [it[0] for it in node_names_and_snrs]
    node_snrs = np.array([it[1] for it in node_names_and_snrs], dtype=np.float32)
    snr_quantile = np.quantile(node_snrs, upcast_ratio)
    node_to_upcast_names = [node_names[i] for i in np.where(node_snrs <= snr_quantile)[0]]
    if verbose:
        print(f"Upcasting {len(node_to_upcast_names)}/{len(node_names)} nodes with SNR less than {snr_quantile:.2f}.")
        for node_name, node_snr in node_names_and_snrs:
            print(node_name, node_snr)

    new_model = orig_model.clone()
    mark_nodes_to_upcast_to_fp32(new_model, node_to_upcast_names)
    return new_model


def get_nodes_to_track(model: Model, operation_types: List[str]) -> List:
    nodes_to_track = []
    for i, op in enumerate(model.get_ordered_ops()):
        if op.get_type_name() in operation_types and \
                all(map(lambda input: input.get_node().get_type_name() != 'Result', op.output(0).get_target_inputs())):
            nodes_to_track.append(op.get_friendly_name())
    return nodes_to_track


def insert_outputs_for_tracked_ops(model: Model, nodes_to_track: List[TrackedNodeInfo]) -> None:
    outputs = []
    for node_info in nodes_to_track:
        node = node_info.node
        outputs.append(node.output(0))
        node_info.node_output = outputs[-1]
        node_info.input_nodes = []
        node_info.input_node_outputs = {}
        for inp_value in node.input_values():
            child_node = inp_value.get_node()
            node_info.input_nodes.append(child_node)
            # Do not add outputs for constant nodes
            if child_node.get_type_name() != 'Constant' and not is_constant_path(child_node):
                outputs.append(child_node.output(0))
                node_info.input_node_outputs[child_node] = outputs[-1]
    model.add_outputs(outputs)


def get_const_value_from_ovmodel(node: Union[Constant, Node]) -> np.ndarray:
    if node.get_type_name() == 'Constant':
        assert node.get_element_type() == ov.Type.f32, f"{node.get_friendly_name()}, {node.get_element_type()}"
        return node.get_data()
    elif is_constant_path(node):
        # If model is compressed and constant values flow through decompression convert
        const_node = node.input_value(0).get_node()
        assert const_node.get_type_name() == 'Constant'
        assert const_node.get_element_type().is_real()
        return np.array(node.input_value(0).get_node().get_data(), dtype=np.float32)
    else:
        raise Exception(
            f'Cannot get const values from ov.Model for {node.get_friendly_name()} with type {node.get_type_name()}')


def is_constant_path(node: Node) -> bool:
    if node.get_type_name() != 'Convert':
        return False
    if len(node.get_rt_info()['is_decompression_0'].aslist()) > 0:
        return True
    if node.input_value(0).get_node().get_type_name() == 'Constant':
        return True
    return False


def infer_full_net(nodes_to_track: List[TrackedNodeInfo], orig_model: Model, example_inputs: List) -> None:
    def get_node_identificator_name(node: Node, output: ov.runtime.Output):
        """
        This return a node id string for matching nodes with their results.
        Try to use any_name of an output if possible, otherwise use friendly name. This is done this way because
        (1) for some op types (e.g., Gather) its key in a results dict does not equal associated node's friendly name
        (2) some outputs do not have any_name property
        """
        try:
            id_name = output.any_name
        except RuntimeError as e:
            if "Attempt to get a name for a Tensor without names" in str(e):
                id_name = node.get_friendly_name()
            else:
                raise e
        return id_name

    core = ov.Core()
    exec_net = core.compile_model(orig_model, "CPU", config={"INFERENCE_PRECISION_HINT": "f32"})
    request = exec_net.create_infer_request()
    results = request.infer(example_inputs)

    results_map = {}
    for key, val in results.items():
        assert len(key.node.input_values()) == 1    # TODO: remove for loop if true
        for input_val in key.node.input_values():
            node = input_val.get_node()
            id_name = get_node_identificator_name(node, key)
            if node.get_type_name() == 'Constant' or is_constant_path(node):
                pass    # TODO: remove if condition if this is not needed
                # results_map[id_name] = get_const_value_from_ovmodel(node)
            else:
                assert id_name not in results_map, f"{id_name} {results_map[id_name]} {val}"
                results_map[id_name] = val

    for node_info in nodes_to_track:
        node_result = results_map[get_node_identificator_name(node_info.node, node_info.node_output)]
        node_info.node_result_full_precision = node_result
        node_info.input_results_full_precision = []
        for input_node in node_info.input_nodes:
            if input_node.get_type_name() == 'Constant' or is_constant_path(input_node):
                # If input is constant, retrieve its value from model
                node_info.input_results_full_precision.append(get_const_value_from_ovmodel(input_node))
            else:
                # If input is not constant, retrieve its input from inference results
                input_node_output = node_info.input_node_outputs[input_node]
                input_result = results_map[get_node_identificator_name(input_node, input_node_output)]
                node_info.input_results_full_precision.append(input_result)
    del request, exec_net, results, results_map


def infer_nodes(nodes_to_track: List[TrackedNodeInfo], device: str, precision: str) -> None:
    for node_info in nodes_to_track:
        infer_tracked_op(node_info, device, precision)


def infer_tracked_op(node_info: TrackedNodeInfo, device: str, precision: str) -> None:
    parameters = []
    input_values = node_info.input_results_full_precision
    for input_value in input_values:
        parameters.append(Parameter(get_element_type(input_value.dtype), ov.PartialShape(input_value.shape)))

    node = node_info.node
    try:
        call_attributes = node.get_attributes()
        if "m_pythondiv" in call_attributes:    # This for some reason is needed for Divide op
            del call_attributes["m_pythondiv"]
        new_op = OPERATION_TYPE_MAP[node.get_type_name()](*parameters, **call_attributes)
    except Exception as e:
        print("Operation inference error", node.get_friendly_name(), parameters, node.get_attributes())
        raise e
    ov_model = ov.Model([new_op], parameters)

    exec_net = ov.Core().compile_model(ov_model, device, config={"INFERENCE_PRECISION_HINT": precision})
    request = exec_net.create_infer_request()
    result = request.infer(input_values)
    node_info.node_result_half_precision = result[0]
    assert len(result) == 1
    del request, exec_net, ov_model


def is_model_partially_upcasted(model) -> bool:
    for node in model.get_ordered_ops():
        if node.get_type_name() not in OPERATION_TYPE_MAP.keys():
            continue
        if ORIGINAL_PRECISION_RT_INFO_NAME in node.get_rt_info().keys():
            return True
    return False


def mark_nodes_to_upcast_to_fp32(model: ov.Model, nodes_with_errors: List[str]) -> None:
    nodes_to_mark = set(nodes_with_errors)
    for node in model.get_ordered_ops():
        if node.get_friendly_name() in nodes_to_mark:
            node.get_rt_info()[ORIGINAL_PRECISION_RT_INFO_NAME] = ''
            nodes_to_mark.remove(node.get_friendly_name())
    assert len(nodes_to_mark) == 0, nodes_to_mark


def compute_snr(x, y):
    # x -- original value (full precision), y -- value with noise (half precision)

    x, y = x.astype(np.float32), y.astype(np.float32)
    max_value = np.finfo(np.float32).max

    if np.prod(x.shape) != np.prod(y.shape):
        print('Shape mismatch. Returning max value', x.shape, y.shape)
        return max_value

    x = np.nan_to_num(x, posinf=max_value)
    y = np.nan_to_num(y, posinf=max_value)

    Ps = np.linalg.norm(x)
    Pn = np.nan_to_num(np.linalg.norm(x - y), posinf=max_value)
    snr = np.nan_to_num(20 * np.log10(Ps / Pn), posinf=max_value)
    return snr


# def compare_tensors(node: Node, a: np.ndarray, b: np.ndarray, new_thresholds_per_op, verbose: bool = False) -> bool:
#     """
#     If values differ more than a certain metric then function returns True
#     """
#     assert np.array_equal(a.shape, b.shape), f'Shapes differ {a.shape} and {b.shape}'
#     out_size = int(np.prod(a.shape))
#     a_, b_ = np.reshape(a, out_size), np.reshape(b, out_size)
#
#     import warnings
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         rel_error = np.abs(2 * (a_ - b_) / (np.abs(a_) + abs(b_)))
#
#     mean_rel_error = np.mean(rel_error)
#     thresholds_map = get_thresholds_per_op()
#     if new_thresholds_per_op is not None:
#         thresholds_map.update(new_thresholds_per_op)
#     thresholds = thresholds_map[node.get_type_name()]
#     rel_threshold = thresholds[0]
#     rel_threshold_ratio = thresholds[1]
#     rel_tol = thresholds[2]
#
#     rel_diff_ratio = np.size(np.where(rel_error >= rel_threshold)) / out_size
#     result = False
#     if not(mean_rel_error < rel_tol) and rel_diff_ratio > rel_threshold_ratio:  # "not (...)" due to nans
#         if verbose:
#             print(f'Upcasted node {node.get_friendly_name()} with {rel_threshold:.2f} '
#                   f'rel2_diff_ratio {rel_diff_ratio:.6f} and mean_rel_error {mean_rel_error:.6f}')
#         result = True
#
#     node_name = node.get_friendly_name().replace('/', '%')
#     from pathlib import Path
#     outcome1 = int(not(mean_rel_error < rel_tol))
#     outcome2 = int(rel_diff_ratio > rel_threshold_ratio)
#     save_dir = Path("activations/gpt-neox-20b")
#     save_dir.mkdir(parents=True, exist_ok=True)
#     filepath_fp16 = save_dir / f"{node_name}_fp16_{outcome1}_{outcome2}.npy"
#     filepath_fp32 = save_dir / f"{node_name}_fp32_{outcome1}_{outcome2}.npy"
#     np.save(filepath_fp16, a)
#     np.save(filepath_fp32, b)
#
#     return result
