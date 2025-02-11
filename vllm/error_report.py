import enum
import json

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sequence import ExecuteModelRequest, SequenceData
from vllm.version import __version__ as VLLM_VERSION
from vllm.worker.worker_base import ModelExecutionError
from vllm.model_executor.sampling_metadata import SequenceGroupMetadata

logger = init_logger(__name__)

import torch


# Hacky way to make sure we can serialize the object in JSON format
def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


def prepare_object_to_dump(obj):
    """
    Recursively iterate over a Python object and replace PyTorch tensor fields
    with a string representation of their shape.

    Args:
        obj: The object to iterate over.

    Returns:
        A dictionary copy of the object with tensors replaced by their shapes.
    """
    if isinstance(obj, dict):
        return {k: prepare_object_to_dump(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [prepare_object_to_dump(v) for v in obj]
    elif isinstance(obj, set):
        return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return [prepare_object_to_dump(v) for v in obj]
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    # elif isinstance(obj, SequenceGroupMetadata):
    #     out_dict = dict()
    #     for k, v in obj.__dict__.items():
    #         if k == '':
    #             pass
    #         else:
    #             out_dict[k] = prepare_object_to_dump(v)
    #     return out_dict
    elif isinstance(obj, SequenceData):
        # Custom __repr__ to omit some parameters
        return (f"SequenceData("
                f"prompt_token_ids_len={len(obj._prompt_token_ids)}, "
                f"output_token_ids_len={len(obj.output_token_ids)}, "
                f"cumulative_logprob={obj.cumulative_logprob}, "
                f"get_num_computed_tokens={obj.get_num_computed_tokens()}")

    elif isinstance(obj, torch.Tensor):
        # We annonymize tensor
        return f"Tensor(shape={obj.shape}, device={obj.device}, dtype={obj.dtype})"
    elif hasattr(obj, '__dict__'):
        obj_dict = dict({'class': type(obj).__name__})
        obj_dict.update(obj.__dict__)
        return prepare_object_to_dump(obj_dict)
    else:
        if is_jsonable(obj):
            return obj
        else:
            return repr(obj)


def dump_engine_exception(err: BaseException, config: VllmConfig,
                          execute_model_req: ExecuteModelRequest,
                          use_cached_outputs: bool, engine_version: int):

    assert engine_version == 0 or engine_version == 1

    print("###############################")

    logger.error(
        "V%s LLM engine (v%s) crashed with config: %s, "
        "use_cached_outputs=%s, ",
        engine_version,
        VLLM_VERSION,
        config,
        use_cached_outputs,
    )

    if isinstance(err, ModelExecutionError):
        # print(err.model_input.__dict__)
        # err_dict = recursive_dict(err.model_input)
        try:
            err_json = prepare_object_to_dump(err.model_input)

            logger.error(json.dumps(err_json))
            print(json.dumps(err_json, indent=2))
        except BaseException as err:
            print(repr(err))
            print("Error on prepare object to dump")

    batch = execute_model_req.seq_group_metadata_list
    requests_count = len(batch)
    requests_prompt_token_ids_lenghts = ', '.join([
        str(len(r.seq_data[idx].prompt_token_ids))
        for idx, r in enumerate(batch)
    ])
    requests_ids = ', '.join([str(r.request_id) for r in batch])
    logger.error(
        "Batch info: requests_count=%s, "
        "requests_prompt_token_ids_lenghts=(%s), "
        "requests_ids=(%s)", requests_count, requests_prompt_token_ids_lenghts,
        requests_ids)

    for idx, r in enumerate(batch):
        logger.error(
            "Errored Batch request #%s: request_id=%s "
            "prompt_token_ids_lengths=%s, "
            "params=%s, "
            "lora_request=%s, prompt_adapter_request=%s ", idx, r.request_id,
            str(len(r.seq_data[idx].prompt_token_ids)), r.sampling_params,
            r.lora_request, r.prompt_adapter_request)
