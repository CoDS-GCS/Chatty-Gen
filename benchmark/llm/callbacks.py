"""
Code adapted from langchain library
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import (
    Generator,
    Optional,
)

from langchain_core.tracers.context import register_configure_hook

from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_community.callbacks.tracers.comet import CometTracer
from langchain_community.callbacks.tracers.wandb import WandbTracer

from langchain_google_vertexai.callbacks import VertexAICallbackHandler

logger = logging.getLogger(__name__)

openai_callback_var: ContextVar[Optional[OpenAICallbackHandler]] = ContextVar(
    "openai_callback", default=None
)
google_callback_var: ContextVar[Optional[VertexAICallbackHandler]] = ContextVar(
    "google_callback", default=None
)
wandb_tracing_callback_var: ContextVar[Optional[WandbTracer]] = ContextVar(  # noqa: E501
    "tracing_wandb_callback", default=None
)
comet_tracing_callback_var: ContextVar[Optional[CometTracer]] = ContextVar(  # noqa: E501
    "tracing_comet_callback", default=None
)

register_configure_hook(openai_callback_var, True)
register_configure_hook(google_callback_var, True)
register_configure_hook(
    wandb_tracing_callback_var, True, WandbTracer, "LANGCHAIN_WANDB_TRACING"
)
register_configure_hook(
    comet_tracing_callback_var, True, CometTracer, "LANGCHAIN_COMET_TRACING"
)


@contextmanager
def get_openai_callback() -> Generator[OpenAICallbackHandler, None, None]:
    """Get the OpenAI callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        OpenAICallbackHandler: The OpenAI callback handler.

    Example:
        >>> with get_openai_callback() as cb:
        ...     # Use the OpenAI callback handler
    """
    cb = OpenAICallbackHandler()
    openai_callback_var.set(cb)
    yield cb
    openai_callback_var.set(None)


@contextmanager
def get_google_callback() -> Generator[VertexAICallbackHandler, None, None]:
    """Get the Google callback handler in a context manager.
    which conveniently exposes token and cost information.

    Returns:
        VertexAICallbackHandler: The Google callback handler.

    Example:
        >>> with get_google_callback() as cb:
        ...     # Use the Google callback handler
    """
    cb = VertexAICallbackHandler()
    google_callback_var.set(cb)
    yield cb
    google_callback_var.set(None)


@contextmanager
def wandb_tracing_enabled(
    session_name: str = "default",
) -> Generator[None, None, None]:
    """Get the WandbTracer in a context manager.

    Args:
        session_name (str, optional): The name of the session.
            Defaults to "default".

    Returns:
        None

    Example:
        >>> with wandb_tracing_enabled() as session:
        ...     # Use the WandbTracer session
    """
    cb = WandbTracer()
    wandb_tracing_callback_var.set(cb)
    yield None
    wandb_tracing_callback_var.set(None)
