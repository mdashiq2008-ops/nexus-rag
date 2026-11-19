from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO


@dataclass(frozen=True)
class Qwen3LlamaCppConfig:
    model_repo: str = "bartowski/Qwen_Qwen3-1.7B-GGUF"
    model_file: str = "Qwen_Qwen3-1.7B-Q4_K_M.gguf"
    model_alias: str = "Qwen3-1.7B-Q4_K_M"
    server_executable: str | Path = (
        r"C:\Users\SONY\AppData\Local\nexus-rag\llama.cpp\b12503\llama-server.exe"
    )
    context_size: int = 4096
    threads: int = 2
    max_tokens: int = 256
    startup_timeout_seconds: float = 900
    request_timeout_seconds: float = 300
    port: int | None = None
    cache_dir: Path | None = None
    keep_server_running: bool = False

    def __post_init__(self) -> None:
        if self.context_size <= 0:
            raise ValueError("context_size must be greater than zero")
        if self.threads <= 0:
            raise ValueError("threads must be greater than zero")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")
        if self.startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be greater than zero")
        if self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be greater than zero")
        if self.port is not None and not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        if not self.model_repo or not self.model_file or not self.model_alias:
            raise ValueError("model_repo, model_file, and model_alias are required")


class Qwen3LlamaCppGenerator:
    """Local, transient llama.cpp server implementing the answer generator API."""

    def __init__(self, config: Qwen3LlamaCppConfig | None = None) -> None:
        self.config = config or Qwen3LlamaCppConfig()
        self._process: subprocess.Popen[bytes] | None = None
        self._log_directory: tempfile.TemporaryDirectory[str] | None = None
        self._log_stream: TextIO | None = None
        self._port: int | None = None

    def __call__(self, prompt: str, /) -> str:
        try:
            self._ensure_server()
            return self._complete(prompt)
        finally:
            if not self.config.keep_server_running:
                self.close()

    def close(self) -> None:
        process = self._process
        self._process = None
        self._port = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except ProcessLookupError:
                pass
            if process.poll() is None:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        if self._log_stream is not None:
            self._log_stream.close()
            self._log_stream = None
        if self._log_directory is not None:
            self._log_directory.cleanup()
            self._log_directory = None

    def _ensure_server(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        if self._process is not None:
            self.close()

        executable = self._resolve_server_executable()
        port = self.config.port or self._find_free_port()
        self._port = port
        self._log_directory = tempfile.TemporaryDirectory(
            prefix="nexus-qwen3-server-"
        )
        log_path = Path(self._log_directory.name) / "llama-server.log"
        self._log_stream = log_path.open("w", encoding="utf-8")
        environment = os.environ.copy()
        if self.config.cache_dir is not None:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(self.config.cache_dir)

        command = self._build_server_command(executable, port)
        try:
            self._process = subprocess.Popen(
                command,
                stdout=self._log_stream,
                stderr=subprocess.STDOUT,
                env=environment,
            )
        except OSError as exc:
            self.close()
            raise RuntimeError(
                f"Unable to start llama-server using {executable!r}"
            ) from exc
        self._wait_until_ready()

    def _build_server_command(self, executable: str, port: int) -> list[str]:
        return [
            executable,
            "--hf-repo",
            self.config.model_repo,
            "--hf-file",
            self.config.model_file,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(self.config.context_size),
            "--threads",
            str(self.config.threads),
            "--n-gpu-layers",
            "0",
            "--parallel",
            "1",
            "--jinja",
            "--no-mmproj",
            "--alias",
            self.config.model_alias,
        ]

    def _wait_until_ready(self) -> None:
        if self._process is None or self._port is None:
            raise RuntimeError("llama-server was not started")
        health_url = f"http://127.0.0.1:{self._port}/health"
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            exit_code = self._process.poll()
            if exit_code is not None:
                detail = self._read_server_log()
                self.close()
                raise RuntimeError(
                    "llama-server exited during startup "
                    f"(code {exit_code}). {detail}"
                )
            try:
                with urllib.request.urlopen(health_url, timeout=1):
                    return
            except (TimeoutError, urllib.error.URLError):
                time.sleep(0.25)
        detail = self._read_server_log()
        self.close()
        raise TimeoutError(
            "llama-server did not become ready before the startup timeout. "
            f"{detail}"
        )

    def _complete(self, prompt: str) -> str:
        if self._port is None:
            raise RuntimeError("llama-server is not running")
        request_body = {
            "model": self.config.model_alias,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n/no_think",
                }
            ],
            "temperature": 0.1,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/v1/chat/completions",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Local llama.cpp API returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Unable to reach the local llama.cpp API"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError("Local llama.cpp API returned invalid JSON")
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("Local llama.cpp API returned no choices")
        choice = choices[0]
        if not isinstance(choice, dict):
            raise RuntimeError("Local llama.cpp API returned an invalid choice")
        message = choice.get("message")
        if not isinstance(message, dict):
            raise RuntimeError("Local llama.cpp API returned no message")
        content = message.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Local llama.cpp API returned no text content")
        return content

    def count_tokens(self, text: str, /) -> int:
        if not text:
            return 0
        self._ensure_server()
        if self._port is None:
            raise RuntimeError("llama-server is not running")
        request_body = {
            "content": text,
            "add_special": False,
            "parse_special": False,
        }
        request = urllib.request.Request(
            f"http://127.0.0.1:{self._port}/tokenize",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_seconds,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"Local llama.cpp tokenizer returned HTTP {exc.code}: "
                f"{detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                "Unable to reach the local llama.cpp tokenizer"
            ) from exc
        if not isinstance(result, dict):
            raise RuntimeError("Local llama.cpp tokenizer returned invalid JSON")
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or any(
            type(token) is not int for token in tokens
        ):
            raise RuntimeError(
                "Local llama.cpp tokenizer returned an invalid token list"
            )
        return len(tokens)

    def _read_server_log(self) -> str:
        if self._log_stream is not None:
            self._log_stream.flush()
        if self._log_directory is None:
            return ""
        log_path = Path(self._log_directory.name) / "llama-server.log"
        if not log_path.is_file():
            return ""
        return log_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-2000:]

    def _resolve_server_executable(self) -> str:
        configured = str(self.config.server_executable)
        executable = shutil.which(configured)
        if executable is not None:
            return executable
        path = Path(configured)
        if path.is_file():
            return str(path.resolve())
        raise FileNotFoundError(
            "llama-server was not found. Install the CPU llama.cpp server "
            "and add its executable to PATH or set server_executable."
        )

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind(("127.0.0.1", 0))
            return int(server_socket.getsockname()[1])
