# coding=utf-8
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runtime registration of MiMo-V2-Flash into NxDI + vLLM-neuron.
#
# Contrib models are not built into the NxDI MODEL_TYPES registry or
# vLLM-neuron's architecture resolver.  This module patches both at
# import time so that ``vllm serve`` can load MiMo-V2-Flash without
# modifying any installed packages.
#
# Usage (from the bench script or any launcher):
#
#   export NXDI_CONTRIB_MIMO_V2_FLASH_SRC=/path/to/this/src
#   python -c "import register_vllm" && python -m vllm.entrypoints.openai.api_server ...
#
# Or, more conveniently, as a combined launcher:
#
#   python register_vllm.py --model /path/to/model --tensor-parallel-size 64 ...
#
# All extra CLI arguments are forwarded to vLLM's ``api_server``.

import logging
import os
import sys

logger = logging.getLogger("mimo_v2_flash.register")

# ------------------------------------------------------------------
# 1. Ensure this contrib's src/ is importable
# ------------------------------------------------------------------
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ------------------------------------------------------------------
# 2. Import the Neuron model class
# ------------------------------------------------------------------
from modeling_mimo_v2 import NeuronMiMoV2ForCausalLM  # noqa: E402

# ------------------------------------------------------------------
# 3. Register in NxDI MODEL_TYPES
# ------------------------------------------------------------------
from neuronx_distributed_inference.utils.constants import MODEL_TYPES  # noqa: E402

MODEL_TYPES["mimo_v2_flash"] = {"causal-lm": NeuronMiMoV2ForCausalLM}
logger.info(
    "Registered mimo_v2_flash in NxDI MODEL_TYPES (now %d models)",
    len(MODEL_TYPES),
)

# ------------------------------------------------------------------
# 4. Patch vLLM-neuron architecture resolver
# ------------------------------------------------------------------
try:
    import vllm_neuron.worker.neuronx_distributed_model_loader as _loader  # noqa: E402

    _original_get_cls = _loader._get_neuron_model_cls

    def _patched_get_neuron_model_cls(architecture: str):
        """Intercept MiMoV2Flash* and return our contrib class."""
        if "MiMoV2Flash" in architecture or "mimov2flash" in architecture.lower():
            logger.info("Resolved %s -> NeuronMiMoV2ForCausalLM", architecture)
            return NeuronMiMoV2ForCausalLM
        return _original_get_cls(architecture)

    _loader._get_neuron_model_cls = _patched_get_neuron_model_cls
    logger.info("Patched vLLM-neuron _get_neuron_model_cls")
except ImportError:
    logger.warning("vllm_neuron not found; skipping vLLM-neuron patch")

# ------------------------------------------------------------------
# 5. Register in vLLM ModelRegistry so architecture resolution works
# ------------------------------------------------------------------
# Without this, vLLM falls back to a default model (Qwen3-0.6B) when it
# encounters the unknown architecture MiMoV2FlashForCausalLM, which
# causes model_config.model to resolve incorrectly.
try:
    from vllm.model_executor.models import ModelRegistry  # noqa: E402

    # Register using the Qwen3MoE class as a stand-in.  vLLM-neuron never
    # instantiates the vLLM model class (it uses NxDI's model class instead),
    # but the registry entry must exist so that inspect_model_cls() succeeds
    # and vLLM preserves the user-supplied --model path.
    ModelRegistry.register_model(
        "MiMoV2FlashForCausalLM",
        "vllm.model_executor.models.qwen3_moe:Qwen3MoeForCausalLM",
    )
    logger.info("Registered MiMoV2FlashForCausalLM in vLLM ModelRegistry")
except Exception as e:
    logger.warning("Could not register in vLLM ModelRegistry: %s", e)


# ------------------------------------------------------------------
# 6. Optional: launch vLLM when run as a script
# ------------------------------------------------------------------
def main():
    """Launch vLLM api_server with MiMo-V2-Flash registered."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    # Strip this script from argv so vLLM sees only its own flags
    sys.argv = sys.argv[1:] if len(sys.argv) > 1 else ["vllm"]

    from vllm.entrypoints.openai.api_server import (
        FlexibleArgumentParser,
        make_arg_parser,
        run_server,
    )
    import uvloop

    parser = FlexibleArgumentParser(
        description="vLLM server with MiMo-V2-Flash support"
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args()
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
