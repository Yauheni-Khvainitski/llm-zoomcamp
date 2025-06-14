"""
OpenAI client for RAG system.

Handles communication with OpenAI's API for generating responses.
"""

import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from openai import OpenAI

from ..config import OPENAI_MODEL, get_env_file_path, get_openai_api_key

logger = logging.getLogger(__name__)


class OpenAIClient:
    """Client for interacting with OpenAI's API."""

    def __init__(self, api_key: str = None, model: str = OPENAI_MODEL, load_env: bool = True):
        """
        Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key (loads from env if not provided)
            model: Default model to use
            load_env: Whether to load environment variables
        """
        self.model = model

        if load_env:
            self._load_environment()

        # Get API key
        self.api_key = api_key or get_openai_api_key()
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and not found in environment")

        # Initialize OpenAI client
        try:
            self.client = OpenAI(api_key=self.api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        try:
            env_file_path = get_env_file_path()
            load_dotenv(dotenv_path=env_file_path, verbose=True, override=True)
            logger.debug(f"Loaded environment from {env_file_path}")
        except Exception as e:
            logger.warning(f"Could not load .env file: {e}")

    def get_response(
        self, prompt: str, model: str = None, temperature: float = 0.0, max_tokens: Optional[int] = None, **kwargs
    ) -> str:
        """
        Get a response from OpenAI for the given prompt.

        Args:
            prompt: The prompt to send to the model
            model: Model to use (defaults to instance model)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters for the API call

        Returns:
            The model's response text
        """
        if model is None:
            model = self.model

        try:
            # Prepare the messages
            messages = [{"role": "user", "content": prompt}]

            # Prepare API parameters
            api_params = {"model": model, "messages": messages, "temperature": temperature, **kwargs}

            if max_tokens is not None:
                api_params["max_tokens"] = max_tokens

            # Make the API call
            response = self.client.chat.completions.create(**api_params)

            # Extract the response text
            response_text = response.choices[0].message.content

            logger.debug(f"Generated response with {len(response_text)} characters")
            return response_text

        except Exception as e:
            logger.error(f"Error getting response from OpenAI: {e}")
            raise

    def get_response_with_usage(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """
        Get a response from OpenAI with usage information.

        Args:
            prompt: The prompt to send to the model
            model: Model to use (defaults to instance model)
            **kwargs: Additional parameters for the API call

        Returns:
            Dictionary with response text and usage information
        """
        if model is None:
            model = self.model

        try:
            messages = [{"role": "user", "content": prompt}]

            response = self.client.chat.completions.create(model=model, messages=messages, **kwargs)

            return {
                "response": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            }

        except Exception as e:
            logger.error(f"Error getting response with usage from OpenAI: {e}")
            raise

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated number of tokens
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int, model: str = None) -> Dict[str, float]:
        """
        Calculate the cost of an API call.

        Args:
            prompt_tokens: Number of prompt tokens
            completion_tokens: Number of completion tokens
            model: Model used (defaults to instance model)

        Returns:
            Dictionary with cost breakdown
        """
        if model is None:
            model = self.model

        # Pricing for GPT-4o (as of the notebook)
        # These prices may need to be updated
        pricing = {
            "gpt-4o": {"input": 2.5 / 1_000_000, "output": 10.0 / 1_000_000},  # $2.5 per 1M tokens  # $10 per 1M tokens
            "gpt-4": {"input": 30.0 / 1_000_000, "output": 60.0 / 1_000_000},
            "gpt-3.5-turbo": {"input": 1.0 / 1_000_000, "output": 2.0 / 1_000_000},
        }

        # Get pricing for the model (default to gpt-4o)
        model_pricing = pricing.get(model, pricing["gpt-4o"])

        input_cost = prompt_tokens * model_pricing["input"]
        output_cost = completion_tokens * model_pricing["output"]
        total_cost = input_cost + output_cost

        return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": total_cost, "model": model}

    def set_model(self, model: str) -> None:
        """
        Set the default model for this client.

        Args:
            model: Model name to use as default
        """
        self.model = model
        logger.info(f"Set default model to {model}")

    def list_available_models(self) -> list:
        """
        List available models (requires API call).

        Returns:
            List of available model names
        """
        try:
            models = self.client.models.list()
            return [model.id for model in models.data]
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            raise
