"""
Tests for the OpenAIClient class.
"""

import os
import unittest
from unittest.mock import Mock, patch

from ..llm.openai_client import OpenAIClient


class TestOpenAIClient(unittest.TestCase):
    """Test suite for the OpenAIClient class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_api_key = "test-api-key-12345"
        self.test_model = "gpt-4o"

    @patch("rag.llm.openai_client.OpenAI")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "env-api-key"})
    def test_init_with_env_api_key(self, mock_openai):
        """Test initialization with API key from environment."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(load_env=False)  # Skip .env loading

        self.assertEqual(client.api_key, "env-api-key")
        self.assertEqual(client.model, "gpt-4o")  # Default model
        mock_openai.assert_called_once_with(api_key="env-api-key")

    @patch("rag.llm.openai_client.OpenAI")
    def test_init_with_provided_api_key(self, mock_openai):
        """Test initialization with provided API key."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, model=self.test_model, load_env=False)

        self.assertEqual(client.api_key, self.test_api_key)
        self.assertEqual(client.model, self.test_model)
        mock_openai.assert_called_once_with(api_key=self.test_api_key)

    @patch.dict(os.environ, {}, clear=True)
    def test_init_no_api_key_raises_error(self):
        """Test initialization without API key raises ValueError."""
        with self.assertRaises(ValueError) as context:
            OpenAIClient(load_env=False)

        self.assertIn("OpenAI API key not provided", str(context.exception))

    @patch("rag.llm.openai_client.OpenAI")
    def test_init_openai_client_error(self, mock_openai):
        """Test initialization when OpenAI client creation fails."""
        mock_openai.side_effect = Exception("OpenAI initialization failed")

        with self.assertRaises(Exception) as context:
            OpenAIClient(api_key=self.test_api_key, load_env=False)

        self.assertIn("OpenAI initialization failed", str(context.exception))

    @patch("rag.llm.openai_client.load_dotenv")
    @patch("rag.llm.openai_client.OpenAI")
    def test_load_environment_success(self, mock_openai, mock_load_dotenv):
        """Test successful environment loading."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            client = OpenAIClient(load_env=True)  # noqa: F841

        mock_load_dotenv.assert_called_once()

    @patch("rag.llm.openai_client.load_dotenv")
    @patch("rag.llm.openai_client.OpenAI")
    def test_load_environment_failure(self, mock_openai, mock_load_dotenv):
        """Test environment loading failure (should not crash)."""
        mock_load_dotenv.side_effect = Exception("Failed to load .env")
        mock_client = Mock()
        mock_openai.return_value = mock_client

        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            # Should not raise exception
            client = OpenAIClient(load_env=True)  # noqa: F841

    @patch("rag.llm.openai_client.OpenAI")
    def test_get_response_success(self, mock_openai):
        """Test successful response generation."""
        # Setup mock
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "This is a test response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        # Test
        client = OpenAIClient(api_key=self.test_api_key, load_env=False)
        result = client.get_response("Test prompt")

        # Assertions
        self.assertEqual(result, "This is a test response")
        mock_client.chat.completions.create.assert_called_once()

        # Check call arguments
        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-4o")
        self.assertEqual(call_args[1]["messages"][0]["role"], "user")
        self.assertEqual(call_args[1]["messages"][0]["content"], "Test prompt")
        self.assertEqual(call_args[1]["temperature"], 0.0)

    @patch("rag.llm.openai_client.OpenAI")
    def test_get_response_with_custom_parameters(self, mock_openai):
        """Test response generation with custom parameters."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()

        mock_message.content = "Custom response"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)
        result = client.get_response("Test prompt", model="gpt-3.5-turbo", temperature=0.7, max_tokens=100, top_p=0.9)

        self.assertEqual(result, "Custom response")

        call_args = mock_client.chat.completions.create.call_args
        self.assertEqual(call_args[1]["model"], "gpt-3.5-turbo")
        self.assertEqual(call_args[1]["temperature"], 0.7)
        self.assertEqual(call_args[1]["max_tokens"], 100)
        self.assertEqual(call_args[1]["top_p"], 0.9)

    @patch("rag.llm.openai_client.OpenAI")
    def test_get_response_api_error(self, mock_openai):
        """Test response generation with API error."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        with self.assertRaises(Exception) as context:
            client.get_response("Test prompt")

        self.assertIn("API Error", str(context.exception))

    @patch("rag.llm.openai_client.OpenAI")
    def test_get_response_with_usage(self, mock_openai):
        """Test getting response with usage information."""
        mock_client = Mock()
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_usage = Mock()

        mock_message.content = "Test response"
        mock_choice.message = mock_message
        mock_choice.finish_reason = "stop"
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30

        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_response.model = "gpt-4o"

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)
        result = client.get_response_with_usage("Test prompt")

        expected = {
            "response": "Test response",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "model": "gpt-4o",
            "finish_reason": "stop",
        }

        self.assertEqual(result, expected)

    @patch("rag.llm.openai_client.OpenAI")
    def test_estimate_tokens(self, mock_openai):
        """Test token estimation."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        # Test with known text
        text = "This is a test text with exactly twenty characters."  # 50 chars
        estimated = client.estimate_tokens(text)

        # Should estimate ~4 chars per token
        expected = len(text) // 4
        self.assertEqual(estimated, expected)

    @patch("rag.llm.openai_client.OpenAI")
    def test_calculate_cost_gpt4o(self, mock_openai):
        """Test cost calculation for GPT-4o."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, model="gpt-4o", load_env=False)

        cost = client.calculate_cost(1000, 500)  # 1000 prompt, 500 completion tokens

        expected_input_cost = 1000 * (2.5 / 1_000_000)
        expected_output_cost = 500 * (10.0 / 1_000_000)
        expected_total = expected_input_cost + expected_output_cost

        self.assertEqual(cost["input_cost"], expected_input_cost)
        self.assertEqual(cost["output_cost"], expected_output_cost)
        self.assertEqual(cost["total_cost"], expected_total)
        self.assertEqual(cost["model"], "gpt-4o")

    @patch("rag.llm.openai_client.OpenAI")
    def test_calculate_cost_custom_model(self, mock_openai):
        """Test cost calculation for custom model."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        cost = client.calculate_cost(1000, 500, model="gpt-3.5-turbo")

        expected_input_cost = 1000 * (1.0 / 1_000_000)
        expected_output_cost = 500 * (2.0 / 1_000_000)
        expected_total = expected_input_cost + expected_output_cost

        self.assertEqual(cost["input_cost"], expected_input_cost)
        self.assertEqual(cost["output_cost"], expected_output_cost)
        self.assertEqual(cost["total_cost"], expected_total)
        self.assertEqual(cost["model"], "gpt-3.5-turbo")

    @patch("rag.llm.openai_client.OpenAI")
    def test_calculate_cost_unknown_model(self, mock_openai):
        """Test cost calculation for unknown model (should use gpt-4o pricing)."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        cost = client.calculate_cost(1000, 500, model="unknown-model")

        # Should use gpt-4o pricing as default
        expected_input_cost = 1000 * (2.5 / 1_000_000)
        expected_output_cost = 500 * (10.0 / 1_000_000)

        self.assertEqual(cost["input_cost"], expected_input_cost)
        self.assertEqual(cost["output_cost"], expected_output_cost)
        self.assertEqual(cost["model"], "unknown-model")

    @patch("rag.llm.openai_client.OpenAI")
    def test_set_model(self, mock_openai):
        """Test setting default model."""
        mock_client = Mock()
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        self.assertEqual(client.model, "gpt-4o")  # Default

        client.set_model("gpt-3.5-turbo")
        self.assertEqual(client.model, "gpt-3.5-turbo")

    @patch("rag.llm.openai_client.OpenAI")
    def test_list_available_models_success(self, mock_openai):
        """Test listing available models successfully."""
        mock_client = Mock()
        mock_models = Mock()
        mock_model1 = Mock()
        mock_model1.id = "gpt-4o"
        mock_model2 = Mock()
        mock_model2.id = "gpt-3.5-turbo"

        mock_models.data = [mock_model1, mock_model2]
        mock_client.models.list.return_value = mock_models
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)
        models = client.list_available_models()

        self.assertEqual(models, ["gpt-4o", "gpt-3.5-turbo"])
        mock_client.models.list.assert_called_once()

    @patch("rag.llm.openai_client.OpenAI")
    def test_list_available_models_error(self, mock_openai):
        """Test listing available models with error."""
        mock_client = Mock()
        mock_client.models.list.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        client = OpenAIClient(api_key=self.test_api_key, load_env=False)

        with self.assertRaises(Exception) as context:
            client.list_available_models()

        self.assertIn("API Error", str(context.exception))


if __name__ == "__main__":
    unittest.main()
