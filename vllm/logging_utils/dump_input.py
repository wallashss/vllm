# SPDX-License-Identifier: Apache-2.0

import enum
import json

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return "'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ', '.join({f'{str(k)}: {prepare_object_to_dump(v)}' \
            for k, v in obj.items()})
        return f'{{{dict_str}}}'
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return (f"Tensor(shape={obj.shape}, "
                f"device={obj.device},"
                f"dtype={obj.dtype})")
    elif hasattr(obj, 'anon_repr'):
        return obj.anon_repr()
    elif hasattr(obj, '__dict__'):
        items = obj.__dict__.items()
        dict_str = ','.join([f'{str(k)}={prepare_object_to_dump(v)}' \
            for k, v in items])
        return (f"{type(obj).__name__}({dict_str})")
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(err: BaseException, config: VllmConfig):

    logger.error("Dumping input data")

    logger.error(
        "V1 LLM engine (v%s) with config: %s, ",
        VLLM_VERSION,
        config,
    )

    from vllm.v1.engine.core import ModelExecutionError
    if isinstance(err, ModelExecutionError):
        try:
            if err.scheduler_output is not None:
                dump_obj = prepare_object_to_dump(err.scheduler_output)
                logger.error("Dumping scheduler output for model execution:")
                logger.error(dump_obj)
            if err.scheduler_stats is not None:
                logger.error(err.scheduler_stats)
        except BaseException as exception:
            logger.error("Error preparing object to dump")
            logger.error(repr(exception))
