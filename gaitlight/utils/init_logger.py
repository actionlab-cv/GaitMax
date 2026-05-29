import logging
import os
from typing import Literal

from rich.logging import RichHandler

logger = logging.getLogger(__name__)


def init_logger(
        log_level: str = Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        hidden_non_zero: bool = False, raise_non_zero: bool = True,
):
    # log level
    log_level = getattr(logging, log_level.upper(), logging.INFO)

    # config root logger
    root_l = logging.getLogger()
    root_l.setLevel(log_level)
    root_h = RichHandler()
    root_f = logging.Formatter('%(name)s > %(message)s')
    root_h.setFormatter(root_f)
    root_l.handlers = [root_h]

    # capture warnings
    logging.captureWarnings(True)

    # non-main
    non_main = int(os.environ.get('LOCAL_RANK', '0')) != 0 or int(os.environ.get('NODE_RANK', '0')) != 0

    if non_main and hidden_non_zero:
        "hidden"
        root_l.handlers.clear()
        root_h = logging.NullHandler()
        root_l.addHandler(root_h)

    if non_main and raise_non_zero:
        "raise level"
        log_level += 10
        root_l.setLevel(log_level)

    # host logger
    l_host = ['lightning.pytorch', 'torchvision', 'kornia', 'dinov2', 'py.warnings', 'wandb']
    for _l in l_host:
        _l = logging.getLogger(_l)
        _l.setLevel(log_level)
        _l.propagate = False
        _l.handlers.clear()
        _l.addHandler(root_h)

    # shadow logger
    l_shadow = ['wandb.sdk', 'asyncio']
    for _l in l_shadow:
        _l = logging.getLogger(_l)
        _l.setLevel(logging.CRITICAL)
        _l.propagate = False
        _l.handlers.clear()
        _l.addHandler(root_h)

    # gait-light logger
    gl_l = logging.getLogger('gaitlight')
    gl_l.setLevel(log_level)

    return gl_l
