import logging
import os

# A logger for this file
log = logging.getLogger(__name__)


# TODO remove need for global pth
experi_dir_flocking = "..."
experi_dir_aggregation = "..."
experi_dir_dispersion = "..."
experi_dir_random_walk = "..."



# Trying to import optional libraries
try:
    import jax

    os.environ["JAX_AVAILABLE"] = "True"

    # Check for GPU availability in JAX
    jax_devices = jax.devices()
    if any("gpu" in str(d.device_kind).lower() for d in jax_devices):
        os.environ["J_DEVICE"] = "gpu"
    else:
        os.environ["J_DEVICE"] = "cpu"

    # Set CPU as default for safety reasons
    jax.config.update(
        "jax_platform_name", "cpu"
    )  # Important to keep in mind https://github.com/jax-ml/jax/discussions/10399#discussioncomment-2608688


except ImportError:
    log.error("Could not import `jax`.")
    os.environ["JAX_AVAILABLE"] = "False"

try:
    import torch

    os.environ["TORCH_AVAILABLE"] = "True"

    # Set CPU as default for safety reasons
    torch.set_default_device("cpu")

    # Check for GPU availability in PyTorch
    if torch.cuda.is_available():
        os.environ["T_DEVICE"] = "cuda:0"
    else:
        os.environ["T_DEVICE"] = "cpu"
except ImportError:
    log.error("Could not import `torch`.")
    os.environ["TORCH_AVAILABLE"] = "False"


class LoggerWriter:
    """Redirects stdout and stderr to a logger."""

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, message):
        """Write message to logger, handling newlines."""
        if message.strip():  # Avoid empty messages
            self.logger.log(self.level, message.strip())

    def flush(self):
        """Flush method for compatibility."""
        pass


def reset_seeds(seed, print_precision_val=4):
    """
    Resets the random seed for reproducibility in Python, NumPy, JAX (if available) and Torch (if available).

    :param seed: The seed value.
    :param print_precision_val: The floating precision for print/log.
    """
    import random

    random.seed(seed)  # Reset Python seed

    import numpy as np

    # Reset NumPy seed
    np.random.seed(seed)
    # Set printing options
    np.set_printoptions(precision=print_precision_val, suppress=True)

    try:
        import jax

        # Reset JAX seed
        jax.random.PRNGKey(seed)
        # Set printing options
        jax.numpy.set_printoptions(precision=print_precision_val, suppress=True)
    except ImportError:
        log.info("Could not reset the `jax` seed.")

    try:
        import torch

        # Reset Torch seed
        torch.manual_seed(seed)
        # Set printing options
        torch.set_printoptions(precision=print_precision_val)
    except ImportError:
        log.info("Could not reset the `torch` seed.")
