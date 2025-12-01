# Guide: Adding Top-K Tool Retrieval to BFCL

This guide explains how to add a top-k tool retrieval step before using your LLM for tool selection in the Berkeley Function Calling Leaderboard (BFCL).

## Overview

When you have a large number of available tools, it can be beneficial to first retrieve the top-k most relevant tools based on the user query, then pass only those tools to the LLM. This approach:

1. **Reduces context size** - Only the most relevant tools are sent to the LLM
2. **Improves accuracy** - LLM focuses on relevant tools, reducing confusion
3. **Lowers costs** - Fewer tokens in the prompt
4. **Enables scalability** - Can handle thousands of tools efficiently

## Architecture

The retrieval happens in the `_compile_tools` method, which is called before each query. The flow is:

```
User Query → Retrieve Top-K Tools → Compile Tools → Send to LLM → Get Function Call
```

## Implementation Steps

### Step 1: Create Your Model Handler

Create a new handler file in `bfcl_eval/model_handler/api_inference/` (for API models) or `bfcl_eval/model_handler/local_inference/` (for local models).

**Example: `bfcl_eval/model_handler/api_inference/openai_with_retrieval.py`**

```python
from bfcl_eval.model_handler.api_inference.openai_response import OpenAIResponsesHandler
from bfcl_eval.model_handler.utils import convert_to_tool
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.constants.enums import ModelStyle

class OpenAIWithRetrievalHandler(OpenAIResponsesHandler):
    """
    OpenAI handler with top-k tool retrieval.
    
    This handler retrieves the top-k most relevant tools before passing them to the LLM.
    """
    
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        top_k: int = 10,  # Number of tools to retrieve
        retrieval_method: str = "embedding",  # "embedding" or "bm25"
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.top_k = top_k
        self.retrieval_method = retrieval_method
        
        # Initialize retrieval system (embedding model, BM25 index, etc.)
        self._initialize_retrieval()
    
    def _initialize_retrieval(self):
        """Initialize your retrieval system here."""
        if self.retrieval_method == "embedding":
            # Example: Initialize embedding model
            # from sentence_transformers import SentenceTransformer
            # self.retriever = SentenceTransformer('all-MiniLM-L6-v2')
            pass
        elif self.retrieval_method == "bm25":
            # Example: Initialize BM25 index
            # from rank_bm25 import BM25Okapi
            # self.bm25_index = None  # Will be built on-the-fly
            pass
    
    def _retrieve_top_k_tools(
        self, 
        all_tools: list[dict], 
        user_query: str
    ) -> list[dict]:
        """
        Retrieve the top-k most relevant tools based on the user query.
        
        Args:
            all_tools: List of all available tools
            user_query: The user's query/question
            
        Returns:
            List of top-k most relevant tools
        """
        if len(all_tools) <= self.top_k:
            # If we have fewer tools than top_k, return all
            return all_tools
        
        # Implement your retrieval logic here
        if self.retrieval_method == "embedding":
            return self._retrieve_with_embeddings(all_tools, user_query)
        elif self.retrieval_method == "bm25":
            return self._retrieve_with_bm25(all_tools, user_query)
        else:
            # Fallback: return first top_k tools
            return all_tools[:self.top_k]
    
    def _retrieve_with_embeddings(
        self, 
        all_tools: list[dict], 
        user_query: str
    ) -> list[dict]:
        """
        Retrieve tools using embedding-based similarity.
        
        You'll need to:
        1. Create embeddings for each tool (name + description)
        2. Create embedding for user query
        3. Compute cosine similarity
        4. Return top-k tools
        """
        # TODO: Implement embedding-based retrieval
        # Example pseudocode:
        # tool_texts = [f"{tool['name']} {tool.get('description', '')}" for tool in all_tools]
        # tool_embeddings = self.retriever.encode(tool_texts)
        # query_embedding = self.retriever.encode([user_query])
        # similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
        # top_indices = np.argsort(similarities)[-self.top_k:][::-1]
        # return [all_tools[i] for i in top_indices]
        
        # Placeholder: return first top_k
        return all_tools[:self.top_k]
    
    def _retrieve_with_bm25(
        self, 
        all_tools: list[dict], 
        user_query: str
    ) -> list[dict]:
        """
        Retrieve tools using BM25 ranking.
        
        You'll need to:
        1. Build BM25 index from tool names and descriptions
        2. Query the index with user query
        3. Return top-k tools
        """
        # TODO: Implement BM25-based retrieval
        # Example pseudocode:
        # tool_texts = [f"{tool['name']} {tool.get('description', '')}" for tool in all_tools]
        # tokenized_tools = [text.lower().split() for text in tool_texts]
        # self.bm25_index = BM25Okapi(tokenized_tools)
        # query_tokens = user_query.lower().split()
        # scores = self.bm25_index.get_scores(query_tokens)
        # top_indices = np.argsort(scores)[-self.top_k:][::-1]
        # return [all_tools[i] for i in top_indices]
        
        # Placeholder: return first top_k
        return all_tools[:self.top_k]
    
    def _extract_user_query(self, test_entry: dict) -> str:
        """
        Extract the user query from the test entry.
        
        For single-turn: extract from test_entry["question"][0]
        For multi-turn: extract from the current turn's user message
        """
        # For single-turn, get the user message
        if isinstance(test_entry["question"], list) and len(test_entry["question"]) > 0:
            first_turn = test_entry["question"][0]
            if isinstance(first_turn, list):
                # Multi-turn format
                for msg in first_turn:
                    if msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(first_turn, dict):
                # Single message format
                if first_turn.get("role") == "user":
                    return first_turn.get("content", "")
        
        # Fallback: return empty string
        return ""
    
    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        """
        Override the base _compile_tools to add retrieval step.
        
        This method:
        1. Gets all available tools from test_entry
        2. Extracts the user query
        3. Retrieves top-k most relevant tools
        4. Compiles only those tools for the LLM
        """
        all_functions: list = test_entry["function"]
        
        # Extract user query for retrieval
        user_query = self._extract_user_query(test_entry)
        
        # Retrieve top-k tools
        if user_query:
            retrieved_functions = self._retrieve_top_k_tools(all_functions, user_query)
        else:
            # If we can't extract query, use all tools (or first top_k)
            retrieved_functions = all_functions[:self.top_k] if len(all_functions) > self.top_k else all_functions
        
        # Convert retrieved tools to the format expected by the model
        tools = convert_to_tool(
            retrieved_functions, 
            GORILLA_TO_OPENAPI, 
            self.model_style
        )
        
        inference_data["tools"] = tools
        
        # Store metadata about retrieval (useful for debugging)
        inference_data["retrieval_metadata"] = {
            "total_tools": len(all_functions),
            "retrieved_tools": len(retrieved_functions),
            "top_k": self.top_k,
            "user_query": user_query,
        }
        
        return inference_data
```

### Step 2: Update Model Configuration

Add your model to `bfcl_eval/constants/model_config.py`:

```python
MODEL_CONFIG_MAP = {
    # ... existing models ...
    
    "gpt-4o-with-retrieval": ModelConfig(
        model_name="gpt-4o-2024-11-20",
        display_name="GPT-4o (with Top-K Retrieval)",
        url="https://platform.openai.com/docs/models/gpt-4o",
        org="OpenAI",
        license="Proprietary",
        model_handler="OpenAIWithRetrievalHandler",
        input_price=2.50,
        output_price=10.00,
        is_fc_model=True,
        underscore_to_dot=False,
    ),
}
```

### Step 3: Update Supported Models

Add your model to `bfcl_eval/constants/supported_models.py`:

```python
SUPPORTED_MODELS = [
    # ... existing models ...
    "gpt-4o-with-retrieval",
]
```

And update `SUPPORTED_MODELS.md`:

```markdown
| GPT-4o (with Top-K Retrieval) | Function Calling | OpenAI | gpt-4o-with-retrieval |
```

### Step 4: Import Your Handler

Make sure your handler is imported in `bfcl_eval/constants/model_config.py`:

```python
from bfcl_eval.model_handler.api_inference.openai_with_retrieval import (
    OpenAIWithRetrievalHandler,
)
```

### Step 5: Handle Multi-Turn Scenarios

For multi-turn conversations, you may want to retrieve tools based on the current turn's query. The `_compile_tools` method is called for each turn, so you can access the current turn's messages from `inference_data`.

**Example for multi-turn:**

```python
def _extract_user_query_multi_turn(
    self, 
    inference_data: dict, 
    test_entry: dict,
    current_turn_idx: int = None
) -> str:
    """Extract user query for multi-turn scenarios."""
    # For multi-turn, you might want to consider the conversation history
    # or just the current turn's user message
    
    if "message" in inference_data and len(inference_data["message"]) > 0:
        # Get the last user message from the conversation
        for msg in reversed(inference_data["message"]):
            if msg.get("role") == "user":
                return msg.get("content", "")
    
    # Fallback to test_entry
    return self._extract_user_query(test_entry)
```

## Retrieval Methods

### Option 1: Embedding-Based Retrieval

Uses semantic similarity between tool descriptions and user queries.

**Pros:**
- Captures semantic meaning
- Works well with paraphrased queries
- Can use pre-trained models

**Cons:**
- Requires embedding model (adds dependency)
- Slower than keyword-based methods
- May need GPU for large-scale retrieval

**Example Implementation:**

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def _initialize_retrieval(self):
    if self.retrieval_method == "embedding":
        self.retriever = SentenceTransformer('all-MiniLM-L6-v2')

def _retrieve_with_embeddings(self, all_tools, user_query):
    # Create text representation for each tool
    tool_texts = []
    for tool in all_tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        # Combine name and description
        tool_text = f"{name}. {desc}"
        tool_texts.append(tool_text)
    
    # Create embeddings
    tool_embeddings = self.retriever.encode(tool_texts)
    query_embedding = self.retriever.encode([user_query])
    
    # Compute similarities
    similarities = cosine_similarity(query_embedding, tool_embeddings)[0]
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[-self.top_k:][::-1]
    
    return [all_tools[i] for i in top_indices]
```

### Option 2: BM25 Retrieval

Uses keyword-based ranking (TF-IDF variant).

**Pros:**
- Fast and lightweight
- No model dependencies
- Good for exact keyword matches

**Cons:**
- Doesn't capture semantic similarity
- May miss relevant tools with different wording

**Example Implementation:**

```python
from rank_bm25 import BM25Okapi
import numpy as np

def _retrieve_with_bm25(self, all_tools, user_query):
    # Create tokenized tool texts
    tool_texts = []
    for tool in all_tools:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        tool_text = f"{name} {desc}"
        tool_texts.append(tool_text.lower())
    
    # Tokenize
    tokenized_tools = [text.split() for text in tool_texts]
    
    # Build BM25 index
    bm25 = BM25Okapi(tokenized_tools)
    
    # Query
    query_tokens = user_query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    # Get top-k indices
    top_indices = np.argsort(scores)[-self.top_k:][::-1]
    
    return [all_tools[i] for i in top_indices]
```

### Option 3: Hybrid Approach

Combine multiple retrieval methods for better results.

```python
def _retrieve_top_k_tools(self, all_tools, user_query):
    # Get results from multiple methods
    embedding_results = self._retrieve_with_embeddings(all_tools, user_query)
    bm25_results = self._retrieve_with_bm25(all_tools, user_query)
    
    # Combine and deduplicate (weighted by method)
    # ... implementation ...
```

## Testing Your Implementation

1. **Test with a small subset:**
   ```bash
   bfcl generate --model gpt-4o-with-retrieval --test-category simple_python --run-ids
   ```

2. **Check retrieval metadata:**
   Look at the `inference_log` in the result files to see:
   - How many tools were retrieved
   - What the user query was
   - Which tools were selected

3. **Compare performance:**
   Run the same test with and without retrieval to compare:
   - Accuracy
   - Token usage
   - Latency

## Submitting to the Leaderboard

1. **Create a Pull Request:**
   - Include your handler implementation
   - Update model configuration
   - Add documentation

2. **Provide Results:**
   - Run evaluations on standard test categories
   - Include comparison with baseline (same model without retrieval)
   - Document the retrieval method and parameters

3. **Best Practices:**
   - Make retrieval configurable (top_k parameter)
   - Handle edge cases (empty queries, fewer tools than top_k)
   - Add logging for debugging
   - Consider caching embeddings/indexes for performance

## Example: Complete Implementation

See `bfcl_eval/model_handler/api_inference/example_with_retrieval.py` for a complete working example.

## Questions?

- Join the [Discord](https://discord.gg/grXXvj9Whz) `#leaderboard` channel
- Check existing handlers for reference
- Review the [Contributing Guide](./CONTRIBUTING.md)
