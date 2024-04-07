from __future__ import annotations

import os
import tiktoken
import copy
import json
import inspect
import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    Set,
    overload,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.pydantic_v1 import PrivateAttr
from openllm_core._schemas import CompletionChunk
from langchain_core.outputs import Generation, GenerationChunk, LLMResult, RunInfo

# token counting
tiktoken_cache_dir = "../tiktoken-cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
openai_embedding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def get_num_tokens(prompt):
    return len(openai_embedding.encode(prompt))

def update_token_usage(
    keys: Set[str], response: Dict[str, Any], token_usage: Dict[str, Any]
) -> None:
    """Update token usage."""
    _keys_to_use = keys.intersection(response["usage"])
    for _key in _keys_to_use:
        if _key not in token_usage:
            token_usage[_key] = response["usage"][_key]
        else:
            token_usage[_key] += response["usage"][_key]

if TYPE_CHECKING:
    import openllm


ServerType = Literal["http", "grpc"]


class IdentifyingParams(TypedDict):
    """Parameters for identifying a model as a typed dict."""

    model_name: str
    model_id: Optional[str]
    server_url: Optional[str]
    server_type: Optional[ServerType]
    embedded: bool
    llm_kwargs: Dict[str, Any]


logger = logging.getLogger(__name__)


class OpenLLM(LLM):
    """OpenLLM, supporting both in-process model
    instance and remote OpenLLM servers.

    To use, you should have the openllm library installed:

    .. code-block:: bash

        pip install openllm

    Learn more at: https://github.com/bentoml/openllm

    Example running an LLM model locally managed by OpenLLM:
        .. code-block:: python

            from langchain_community.llms import OpenLLM
            llm = OpenLLM(
                model_name='flan-t5',
                model_id='google/flan-t5-large',
            )
            llm("What is the difference between a duck and a goose?")

    For all available supported models, you can run 'openllm models'.

    If you have a OpenLLM server running, you can also use it remotely:
        .. code-block:: python

            from langchain_community.llms import OpenLLM
            llm = OpenLLM(server_url='http://localhost:3000')
            llm("What is the difference between a duck and a goose?")
    """

    model_name: Optional[str] = None
    """Model name to use. See 'openllm models' for all available models."""
    model_id: Optional[str] = None
    """Model Id to use. If not provided, will use the default model for the model name.
    See 'openllm models' for all available model variants."""
    server_url: Optional[str] = None
    """Optional server URL that currently runs a LLMServer with 'openllm start'."""
    server_type: ServerType = "http"
    """Optional server type. Either 'http' or 'grpc'."""
    embedded: bool = True
    """Initialize this LLM instance in current process by default. Should
    only set to False when using in conjunction with BentoML Service."""
    llm_kwargs: Dict[str, Any]
    """Keyword arguments to be passed to openllm.LLM"""

    _runner: Optional[openllm.LLMRunner] = PrivateAttr(default=None)
    _client: Union[
        openllm.client.HTTPClient, openllm.client.GrpcClient, None
    ] = PrivateAttr(default=None)

    class Config:
        extra = "forbid"

    @overload
    def __init__(
        self,
        model_name: Optional[str] = ...,
        *,
        model_id: Optional[str] = ...,
        embedded: Literal[True, False] = ...,
        **llm_kwargs: Any,
    ) -> None:
        ...

    @overload
    def __init__(
        self,
        *,
        server_url: str = ...,
        server_type: Literal["grpc", "http"] = ...,
        **llm_kwargs: Any,
    ) -> None:
        ...

    def __init__(
        self,
        model_name: Optional[str] = None,
        *,
        model_id: Optional[str] = None,
        server_url: Optional[str] = None,
        server_type: Literal["grpc", "http"] = "http",
        embedded: bool = True,
        **llm_kwargs: Any,
    ):
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm.'"
            ) from e

        llm_kwargs = llm_kwargs or {}

        if server_url is not None:
            logger.debug("'server_url' is provided, returning a openllm.Client")
            assert (
                model_id is None and model_name is None
            ), "'server_url' and {'model_id', 'model_name'} are mutually exclusive"
            client_cls = (
                openllm.client.HTTPClient
                if server_type == "http"
                else openllm.client.GrpcClient
            )
            client = client_cls(server_url)

            super().__init__(
                **{
                    "server_url": server_url,
                    "server_type": server_type,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._runner = None  # type: ignore
            self._client = client
        else:
            assert model_name is not None, "Must provide 'model_name' or 'server_url'"
            # since the LLM are relatively huge, we don't actually want to convert the
            # Runner with embedded when running the server. Instead, we will only set
            # the init_local here so that LangChain users can still use the LLM
            # in-process. Wrt to BentoML users, setting embedded=False is the expected
            # behaviour to invoke the runners remotely.
            # We need to also enable ensure_available to download and setup the model.
            runner = openllm.Runner(
                model_name=model_name,
                model_id=model_id,
                init_local=embedded,
                ensure_available=True,
                **llm_kwargs,
            )
            super().__init__(
                **{
                    "model_name": model_name,
                    "model_id": model_id,
                    "embedded": embedded,
                    "llm_kwargs": llm_kwargs,
                }
            )
            self._client = None  # type: ignore
            self._runner = runner

    @property
    def runner(self) -> openllm.LLMRunner:
        """
        Get the underlying openllm.LLMRunner instance for integration with BentoML.

        Example:
        .. code-block:: python

            llm = OpenLLM(
                model_name='flan-t5',
                model_id='google/flan-t5-large',
                embedded=False,
            )
            tools = load_tools(["serpapi", "llm-math"], llm=llm)
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
            )
            svc = bentoml.Service("langchain-openllm", runners=[llm.runner])

            @svc.api(input=Text(), output=Text())
            def chat(input_text: str):
                return agent.run(input_text)
        """
        if self._runner is None:
            raise ValueError("OpenLLM must be initialized locally with 'model_name'")
        return self._runner

    @property
    def _identifying_params(self) -> IdentifyingParams:
        """Get the identifying parameters."""
        if self._client is not None:
            self.llm_kwargs.update(self._client._config)
            model_name = self._client._metadata.model_name
            model_id = self._client._metadata.model_id
        else:
            if self._runner is None:
                raise ValueError("Runner must be initialized.")
            model_name = self.model_name
            model_id = self.model_id
            try:
                self.llm_kwargs.update(
                    json.loads(self._runner.identifying_params["configuration"])
                )
            except (TypeError, json.JSONDecodeError):
                pass
        return IdentifyingParams(
            server_url=self.server_url,
            server_type=self.server_type,
            embedded=self.embedded,
            llm_kwargs=self.llm_kwargs,
            model_name=model_name,
            model_id=model_id,
        )

    @property
    def _llm_type(self) -> str:
        return "openllm_client" if self._client else "openllm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        config = openllm.AutoConfig.for_model(
            self._identifying_params["model_name"], **copied
        )
        if self._client:
            res = self._client.generate(
                prompt, **config.model_dump(flatten=True)
            ).outputs[0]
        else:
            assert self._runner is not None
            res = self._runner(prompt, **config.model_dump(flatten=True))
        if isinstance(res, dict) and "text" in res:
            return res["text"]
        elif isinstance(res, CompletionChunk):
            return res.text
        elif isinstance(res, str):
            return res
        else:
            raise ValueError(
                "Expected result to be a dict with key 'text' or a string. "
                f"Received {res}"
            )

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            import openllm
        except ImportError as e:
            raise ImportError(
                "Could not import openllm. Make sure to install it with "
                "'pip install openllm'."
            ) from e

        copied = copy.deepcopy(self.llm_kwargs)
        copied.update(kwargs)
        config = openllm.AutoConfig.for_model(
            self._identifying_params["model_name"], **copied
        )
        if self._client:
            async_client = openllm.client.AsyncHTTPClient(self.server_url)
            res = (
                await async_client.generate(prompt, **config.model_dump(flatten=True))
            ).outputs[0]
        else:
            assert self._runner is not None
            (
                prompt,
                generate_kwargs,
                postprocess_kwargs,
            ) = self._runner.llm.sanitize_parameters(prompt, **kwargs)
            generated_result = await self._runner.generate.async_run(
                prompt, **generate_kwargs
            )
            res = self._runner.llm.postprocess_generate(
                prompt, generated_result, **postprocess_kwargs
            )

        if isinstance(res, dict) and "text" in res:
            return res["text"]
        elif isinstance(res, CompletionChunk):
            return res.text
        elif isinstance(res, str):
            return res
        else:
            raise ValueError(
                "Expected result to be a dict with key 'text' or a string. "
                f"Received {res}"
            )
    

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        # TODO: add caching here.

        _keys = {"completion_tokens", "prompt_tokens", "total_tokens"}
        token_usage = dict()
        generations = []
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")
        for prompt in prompts:
            prompt_tokens = get_num_tokens(prompt)
            text = (
                self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else self._call(prompt, stop=stop, **kwargs)
            )
            completion_tokens = get_num_tokens(text)
            total_tokens = prompt_tokens + completion_tokens

            token_used = {
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens
                }
            }
            generations.append([Generation(text=text)])
            update_token_usage(_keys, token_used, token_usage)
        llm_output = {"token_usage": token_usage, "model_name": self.model_name}
        return LLMResult(generations=generations, llm_output=llm_output)

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        generations = []
        new_arg_supported = inspect.signature(self._acall).parameters.get("run_manager")
        for prompt in prompts:
            text = (
                await self._acall(prompt, stop=stop, run_manager=run_manager, **kwargs)
                if new_arg_supported
                else await self._acall(prompt, stop=stop, **kwargs)
            )
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)
