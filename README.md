# RLM_Wikidata

**RLM_Wikidata** is a Recursive Language Model (RLM) harness over a frozen Wikidata graph. The model answers a question by writing Python in a REPL: it searches entities, follows relations, reads qualifiers and references, keeps intermediate results in variables, and returns an answer checked exactly against the graph.

Wikidata describes more than 120 million entities in over 300 languages, and AI systems have no good way to explore it. SPARQL takes expertise and pasting graph data into a context window degrades as the question grows. Recursive Language Models (RLMs; Zhang, Kraska & Khattab, 2026) fit this task better: the model explores the graph in code and calls itself on the parts that matter.

This harness generated **[Wikidata-Search-Traces](https://huggingface.co/datasets/AI-BRIDGES/wikidata-search-traces)**, an open corpus of 10,235 reasoning trajectories over Wikidata, available on Hugging Face.

## The harness

- **One cell per turn.** The model replies with one Python block. The harness runs it in a persistent namespace, so variables carry over between turns, and returns the printed output, truncated to 10,000 characters. The model ends the run with `FINAL(value)`.
- **Graph functions.** The model reads the graph only through 13 functions: `search_entity`, `search_property`, `claims`, `references`, `describe`, `labels`, `label`, `descriptions`, `edges`, `count_edges`, `degree`, `name` and `has`. Every call is logged.
- **Sub-calls.** `llm_query` and `llm_map` send text to a second instance of the model, for semantic judgements over material the run has already read. Sub-calls have no graph access.
- **Grounding checks.** The harness rejects a final answer whose shape does not match the answer format, or that names an entity the run never read.
- **Full archive.** Every run keeps the messages, reasoning, executed code, graph reads, sub-calls, timing and the final answer.
- **Exact scoring.** Every answer is an entity (QID), a year, a quantity, a number, a string or a list of entities, so a deterministic scorer compares it with the reference answer. No language model judges correctness.

We worked over a frozen Wikidata graph. The harness works with any OpenAI-compatible endpoint; the reference configuration serves Qwen3.8-27B-FP8 with vLLM on one GPU.

## Evaluation

We evaluated five systems on 100 questions built with the pipeline: 50 single-entity and 50 multi-hop. The questions are in [`tasks/eval/`](tasks/eval/).

- **Single-entity types:** language 6, neighbour 5, profile 5, qualifier 6, rank 5, reference 6, value_date 5, value_item 6, value_quantity 6.
- **Multi-hop types:** identity 24, year 12, quantity 7, count 4, group 3.
- **Scoring:** exact match after canonicalisation, with no judge. Entities must match as QIDs. Quantities must match both the amount and the unit label, with `null` when the value has no unit.

### Systems

| System | Model | Setup |
| :---- | :---- | :---- |
| Qwen3.8-27B + RLM | Qwen3.8-27B (FP8, served with vLLM on one H100) | RLM harness |
| gpt-6-luna + RLM | gpt-6-luna | RLM harness |
| glm-5.3-flash agent | glm-5.3-flash | tool-calling agent |
| gpt-6-luna agent | gpt-6-luna | tool-calling agent |
| gemini-3.1-flash-lite agent | gemini-3.1-flash-lite | tool-calling agent |

- **Shared:** every system uses the same 13 graph functions over the same frozen graph, with low reasoning effort. Every system gets the same instruction: "The question has an answer. Do not stop until you find one."
- **RLM harness:** the model writes Python in a REPL and keeps intermediate results in variables. It can call itself on sub-problems. Temperature 1, up to 100 turns and 80 sub-calls maximum.
- **Tool-calling agents:** openai-agents with LiteLLM. They call the graph functions directly, with tool output truncated at 10,000 characters, up to 100 calls and $0.50 per question maximum.

### Results

| System | Single-entity | Multi-hop | Overall | Cost / question | Tokens / question (median) | Seconds / question (median) |
| :---- | ----: | ----: | ----: | ----: | ----: | ----: |
| Qwen3.8-27B + RLM | 46/50 | 28/50 | 74/100 | $0.04 ¹ | 35,754 | 25 ² |
| gpt-6-luna + RLM | 42/50 | 19/50 | 61/100 | $0.013 | 24,942 | 18 |
| glm-5.3-flash agent | 37/50 | 13/50 | 50/100 | $0.067 | 30,478 | 33 |
| gpt-6-luna agent | 40/50 | 9/50 | 49/100 | $0.005 | 24,924 | 16 |
| gemini-3.1-flash-lite agent | 31/50 | 10/50 | 41/100 | $0.19 | 460,426 | 104 |

For the API systems, cost is input tokens × input price + output tokens × output price, with prices from LiteLLM's model price table. For every model, reasoning tokens are counted as output tokens, in both the token counts and the costs.

¹ Estimated cost of batched serving on an H100 in Google Colab, about 40 compute units in total.
² Processing one question at a time.

## Repository

| Path | Contents |
| :---- | :---- |
| `scripts/rlm_loop.py` | The harness: runs one question and archives the trajectory |
| `scripts/wd_graph_env.py` | The graph environment: the functions the model calls |
| `scripts/bench/` | Batch runner, scorer and summary |
| `tasks/eval/` | The 100 evaluation questions with their reference answers |
| `config/runs/reference.env` | The settings used for the evaluation |
| `docs/running.md` | How to set up and run the harness |
| `docs/graph-format.md` | The graph files the environment reads |

## Running the harness

The harness needs Python 3.13, a Wikidata graph in the format described in [docs/graph-format.md](docs/graph-format.md), and any OpenAI-compatible endpoint: a local server such as vLLM, or a hosted API. [docs/running.md](docs/running.md) covers installation, serving a model, the settings, and how to run and score the evaluation.

```bash
uv sync
set -a; source config/runs/reference.env; set +a
export WIKIDATA_GRAPH_DIR=/path/to/graph
export BASE_URL=http://127.0.0.1:8000/v1 API_KEY=EMPTY MODEL_NAME=<served model name>

.venv/bin/python scripts/bench/run_batch.py --families eval --workers 1 --tag eval
.venv/bin/python scripts/bench/summary.py results/runs/<batch>
```
