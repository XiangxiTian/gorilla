from __future__ import annotations

from typing import Dict, List

from bfcl_eval.model_handler.local_inference.llama import LlamaHandler
from bfcl_eval.model_handler.utils import system_prompt_pre_processing_chat_model


class LlamaTopKHandler(LlamaHandler):
    """
    Llama handler with a simple built-in top-k tool retrieval step.

    This class is intended for finetuned Llama models that you want to evaluate
    on BFCL while first restricting the available tools/functions to the
    top-k most relevant ones for the current query.

    Retrieval here is intentionally lightweight and self-contained (no extra
    dependencies). It uses a simple keyword-overlap scoring between the
    user query and each tool's name/description. You can customize or replace
    `_retrieve_top_k_functions` with a more sophisticated retriever
    (e.g., embeddings, BM25) if desired.
    """

    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype: str = "bfloat16",
        top_k: int = 10,
        **kwargs,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model id for your finetuned Llama.
            temperature: Sampling temperature.
            registry_name: Internal BFCL registry name.
            is_fc_model: Whether BFCL should treat this as FC mode.
            dtype: Model dtype for vLLM/sglang backend.
            top_k: Number of tools to keep after retrieval.
        """
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            registry_name=registry_name,
            is_fc_model=is_fc_model,
            dtype=dtype,
            **kwargs,
        )
        self.top_k = max(1, int(top_k))

    # ------------------------------------------------------------------
    # Prompting path override: restrict tools before building system prompt
    # ------------------------------------------------------------------

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        """
        Override the OSSHandler implementation to:
          1. Extract the user query from the test entry.
          2. Run a simple retrieval over all available functions.
          3. Keep only the top-k functions.
          4. Build the system prompt using the reduced function set.
        """
        functions: List[Dict] = test_entry["function"]
        test_entry_id: str = test_entry["id"]

        user_query = self._extract_user_query(test_entry)
        if user_query:
            selected_functions = self._retrieve_top_k_functions(
                functions, user_query, self.top_k
            )
        else:
            # If we cannot reliably extract a query, fall back to truncation.
            selected_functions = (
                functions[: self.top_k] if len(functions) > self.top_k else functions
            )

        # Build the system prompt using only the selected functions.
        test_entry["question"][0] = system_prompt_pre_processing_chat_model(
            test_entry["question"][0],
            selected_functions,
            test_entry_id,
        )

        # Keep the (possibly reduced) function list in the inference data.
        return {
            "message": [],
            "function": selected_functions,
        }

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def _extract_user_query(self, test_entry: dict) -> str:
        """
        Extract the user query text from the test entry.

        We look at the first turn of `question` and return the first `user`
        role content we find. This is sufficient for BFCL single-turn and
        most multi-turn setups, and keeps the implementation minimal.
        """
        question = test_entry.get("question", [])
        if not question:
            return ""

        first_turn = question[0]

        # Multi-message turn: [ {role, content}, ... ]
        if isinstance(first_turn, list):
            for msg in first_turn:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return str(msg.get("content", "")).strip()

        # Single message turn: {role, content}
        if isinstance(first_turn, dict) and first_turn.get("role") == "user":
            return str(first_turn.get("content", "")).strip()

        return ""

    @staticmethod
    def _retrieve_top_k_functions(
        functions: List[Dict], user_query: str, top_k: int
    ) -> List[Dict]:
        """
        Simple, dependency-free retrieval over tools/functions.

        Scoring heuristic:
            - Lowercase, whitespace-split tokenization.
            - Score = number of shared tokens between query and
              (tool name + description).

        Returns the top-k highest scoring tools (stable for ties via index).
        """
        if len(functions) <= top_k:
            return functions

        query_tokens = _tokenize(user_query)
        if not query_tokens:
            return functions[:top_k]

        scored = []
        for idx, fn in enumerate(functions):
            name = str(fn.get("name", "")).lower()
            desc = str(fn.get("description", "")).lower()
            doc = f"{name} {desc}"
            doc_tokens = _tokenize(doc)
            score = len(query_tokens & doc_tokens)
            scored.append((score, idx, fn))

        # Sort by score descending, then by original index to keep determinism.
        scored.sort(key=lambda x: (-x[0], x[1]))

        # If all scores are zero (no lexical overlap), just fall back to first top_k.
        if scored[0][0] == 0:
            return functions[:top_k]

        return [entry[2] for entry in scored[:top_k]]


def _tokenize(text: str) -> set[str]:
    """
    Very simple tokenizer: lowercase + split on whitespace, strip punctuation.
    """
    import re

    if not text:
        return set()

    # Replace non-alphanumeric characters with spaces, then split.
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return set(normalized.split())

