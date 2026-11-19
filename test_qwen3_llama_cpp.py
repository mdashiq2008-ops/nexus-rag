import json
import unittest
from io import BytesIO
from unittest.mock import Mock, patch

from qwen3_llama_cpp import (
    Qwen3LlamaCppConfig,
    Qwen3LlamaCppGenerator,
)


class Qwen3LlamaCppTests(unittest.TestCase):
    def test_server_command_uses_configured_hugging_face_gguf_and_cpu(self) -> None:
        config = Qwen3LlamaCppConfig(
            context_size=2048,
            threads=1,
            cache_dir=None,
        )
        command = Qwen3LlamaCppGenerator(config)._build_server_command(
            "llama-server.exe",
            9001,
        )

        self.assertIn("bartowski/Qwen_Qwen3-1.7B-GGUF", command)
        self.assertIn("Qwen_Qwen3-1.7B-Q4_K_M.gguf", command)
        self.assertEqual(command[command.index("--ctx-size") + 1], "2048")
        self.assertEqual(command[command.index("--threads") + 1], "1")
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "0")
        self.assertEqual(command[command.index("--port") + 1], "9001")

    def test_calls_local_openai_compatible_chat_api_and_unloads_server(self) -> None:
        generator = Qwen3LlamaCppGenerator(
            Qwen3LlamaCppConfig(keep_server_running=False)
        )
        generator._ensure_server = Mock(
            side_effect=lambda: setattr(generator, "_port", 43210)
        )
        generator.close = Mock()
        response = BytesIO(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": '{"status":"insufficient_evidence"}'
                            }
                        }
                    ]
                }
            ).encode("utf-8")
        )

        with patch(
            "qwen3_llama_cpp.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            result = generator("Only use cited passages.")

        self.assertEqual(result, '{"status":"insufficient_evidence"}')
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "http://127.0.0.1:43210/v1/chat/completions",
        )
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertTrue(payload["messages"][0]["content"].endswith("/no_think"))
        generator.close.assert_called_once_with()

    def test_counts_tokens_with_local_model_tokenizer(self) -> None:
        generator = Qwen3LlamaCppGenerator()
        generator._ensure_server = Mock(
            side_effect=lambda: setattr(generator, "_port", 43210)
        )
        response = BytesIO(json.dumps({"tokens": [10, 11, 12]}).encode("utf-8"))

        with patch(
            "qwen3_llama_cpp.urllib.request.urlopen",
            return_value=response,
        ) as urlopen:
            count = generator.count_tokens("context text")

        self.assertEqual(count, 3)
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:43210/tokenize")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {
                "content": "context text",
                "add_special": False,
                "parse_special": False,
            },
        )

    def test_empty_token_count_does_not_start_server(self) -> None:
        generator = Qwen3LlamaCppGenerator()
        generator._ensure_server = Mock()

        self.assertEqual(generator.count_tokens(""), 0)
        generator._ensure_server.assert_not_called()

    def test_validates_resource_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "context_size"):
            Qwen3LlamaCppConfig(context_size=0)
        with self.assertRaisesRegex(ValueError, "threads"):
            Qwen3LlamaCppConfig(threads=0)
        with self.assertRaisesRegex(ValueError, "port"):
            Qwen3LlamaCppConfig(port=65536)


if __name__ == "__main__":
    unittest.main()
