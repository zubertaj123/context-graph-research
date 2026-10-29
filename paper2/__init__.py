"""paper2: structural information value (SIV) as an LLM-free admission and consolidation
signal for agent memory graphs. See docs/00-preregistration.md for the design this code serves.

Modules
  data      loaders for LoCoMo and LongMemEval (cleaned)
  extract   candidate triples: oracle observations (LoCoMo), heuristic, LLM adapter
  graph     the memory graph (cgraph.ContextGraph) with entity resolution and consolidation
  measures  Trust / anchoring / soft agreement / Impact / novelty / SIVa, 2010 and redesigned
  policies  admission policies and baselines
  retrieve  BM25 + 1-hop retrieval, identical across policies
  llm       OpenAI-compatible and Anthropic chat adapters (optional; offline mode needs none)
  evaluate  LoCoMo F1/EM, evidence recall, admission precision/recall, bootstrap intervals
  run       the experiment loop and CLI
"""
__version__ = "0.1.0"
