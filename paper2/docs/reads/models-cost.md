# Reader: models:cost

MODEL STACK (solo researcher, one 24 GB GPU + small API budget, as of 2026-10-18)

(a) Local open-weight models for triple extraction and QA. Primary: Qwen3-8B (Apache-2.0, 8.2B, 32k native / 128k YaRN; BF16 ~16.4 GB fits 24 GB with ~16k max-model-len) run with enable_thinking=False, or Qwen3-14B-AWQ (official 4-bit, Apache-2.0, ~9.3 GB per Ollama's qwen3:14b; vllm>=0.8.5). 14B in BF16 (~29.6 GB) does NOT fit 24 GB; Phi-4 and Qwen2.5-14B also need AWQ/GPTQ. Cheap "weak extractor" arm: Qwen3-4B-Instruct-2507 (non-thinking only, 262k ctx, Apache-2.0). Newest alternatives: Qwen3.5-9B (Feb 2026, Apache-2.0, 262k ctx, vision-language, thinking default), Gemma 4 12B (Apr 2026, Apache 2.0, 256k, ungated; int4 needed on 24 GB) or Gemma 4 E4B (8B with embeddings, 128k); Qwen3.8-27B (Aug 2026, Apache-2.0) only at 4-bit. Llama-3.1-8B-Instruct remains Meta's latest 8B text model (3.2 = 1B/3B, 3.3 = 70B, Llama 4 = 109B+ MoE) but is gated under the Llama 3.1 Community License ("Built with Llama", Llama-prefix naming). Phi-4 (MIT, 14B) has only 16k context; Phi-4-mini-instruct (3.8B, MIT, 128k) is the usable Phi; no Phi-5 exists. Prefer Gemma 4 over Gemma 3 (Gemma Terms of Use, gated). Hold ONE extraction model and ONE QA model fixed across all six policies. Always disable thinking (Qwen3/3.5/3.8 and Gemma 4 think by default) or output tokens inflate 10-50x.

(b) API model for the Mem0-style ADD/UPDATE/DELETE/NOOP gate and judging: Claude Haiku 5.5, id claude-haiku-5-5 ($0.10 in / $0.50 out per MTok for prompts <=100k tokens; cache reads $0.01; Batch API 50% off; 1M context). Same list price as OpenAI gpt-6-luna ($0.10/$0.01 cached/$0.50) and Gemini 2.5 Flash-Lite ($0.10/$0.40), so provider choice is not a cost decision. For comparability run LongMemEval's canonical judge gpt-4o-2024-08-06 ($2.50/$10; >97% human agreement per the paper) on at least the LME-S judgments, and use a different model family for judge than for gate to avoid self-preference. Mem0 used GPT-4o-mini ($0.15/$0.60) for its gate with m=10 recent messages, s=10 similar memories.

(c) Embeddings: all-MiniLM-L6-v2 (Apache-2.0, 22.7M, 384-d, 256-token cap; ~90 MB fp32) for entity-resolution of triple endpoints; bge-base-en-v1.5 (MIT, ~109M, 768-d, 512) or nomic-embed-text-v1.5 (Apache-2.0, 137M, 768-d Matryoshka down to 64, 8192 ctx, requires search_document:/search_query: prefixes) for turn/fact retrieval; bge-small-en-v1.5 and gte-small (both MIT, 33.4M, 384-d, 512) as lighter options; gte-base-en-v1.5 (Apache-2.0, 137M, 768-d, 8192). Fix the retriever; vary only admission.

(d) LLM-free extraction arm: GLiNER2 (Apache-2.0, pip gliner2, DeBERTa-v3 backbones; gliner2.5-small 74M / base 194M / gliner2-large 340M; schema-based entity+relation+JSON extraction) is the primary choice; spaCy (MIT; en_core_web_sm 3.8.0 is MIT, 12 MB) DependencyMatcher SVO rules as the second; Stanford CoreNLP OpenIE (GPL-3.0, Java, ~100 sentences/s/core, Angeli et al. ACL 2015; Stanza Python client is Apache-2.0 but the server is GPL) as the classic baseline. REBEL (Babelscape/rebel-large) is CC-BY-NC-SA-4.0 (0.4B BART seq2seq, 200+ Wikidata relations): fine for a paper, ShareAlike taints derived artifacts. OpenIE6 is GPL-3.0 and pinned to Python 3.6 + transformers==2.6.0 with 6.6 GB of Zenodo models: treat as optional/brittle.

(e) MEASURED DATA VOLUMES (I downloaded both datasets). LongMemEval-S (MIT): 500 questions (single-session-user 70, -assistant 56, -preference 30, multi-session 133, knowledge-update 78, temporal-reasoning 133; 30 abstention), 50.2 sessions/question (39-66), 25,112 session instances but only 19,829 UNIQUE sessions (dedupe saves 21% of extraction), 244.9M chars of history (~61M tokens at 4 chars/token; README's ~115k/question is Llama-3 tokens), unique-session text ~197.8M chars (~49.4M tokens), 1.9 evidence sessions/question. The LLM gate must still run per question-history (25,112 instances) because decisions depend on the existing graph. LoCoMo (CC BY-NC 4.0): 10 conversations, 1,986 QA (category counts 1:282, 2:321, 3:96, 4:841, 5:446; Mem0 drops category 5 adversarial), 27.2 sessions and 588 turns per conversation, 133,772 words of utterance text; Mem0 reports ~26k tokens/conversation.

COST MODEL (Haiku 5.5 standard prices; cached-prefix instruction per call; output = 15% of input for extraction). Extraction per seed: 65.8M input (16.1M cached) + 7.4M output = $8.85; 3 seeds = $27 ($13 Batch). Gate baseline per seed (25,384 session-level calls, ~3.7k in / 600 out, 600-token cached prefix): $15.6; 3 seeds = $47 ($24 Batch). Answering 6 policies x 2,486 questions x 3 seeds = 44,748 calls at ~1.85k in / 60 out: $8.4 ($4.2 Batch). Judging 44,748 at ~410 in / 30 out: $1.5 on Haiku; $9 for gpt-4o on the 9,000 LME-S judgments ($45 for all). Retrieved-set ablation (k=20 facts x 2,486 questions, 1 policy, 1 seed) = 49,720 re-answers ~ $11; full leave-one-fact-out (~750 facts/question) is infeasible (~1.9M re-answers). Total ~ $85 standard / ~$45 Batch for the core grid; ~ $105 / $55 with canonical gpt-4o judge and ablation. On Sonnet 5.5 ($2/$10) the same grid costs ~20x (~$1,700). GPU-HOURS (local 7-9B, assumed 5,000 tok/s prefill and 1,500 tok/s batched decode on an RTX 4090 with vLLM; halve for 14B-AWQ): extraction 5.0 h/seed -> 15 h (14B: 30 h); answering 5.1 h (14B: 10 h); ablation 5.7 h; embeddings <0.5 h; total ~26 GPU-h (7-9B) or ~52 GPU-h (14B). RunPod RTX 4090 at $0.34/h community or $0.89/h secure -> $9-46. Running the LLM gate locally instead adds ~8 GPU-h/seed. Memory-R1's learned gate was trained with PPO/GRPO on 4x H100 80GB (8 for 14B) on 152 LoCoMo QA pairs: on 24 GB do LoRA-GRPO on Qwen3-4B-Instruct-2507, or rent 4x H100 (RunPod H100 SXM $2.69/h community). Throughput anchors: RTX 4090 vLLM FP16 7B-class 4,173 total tok/s (databasemart, 100-in/600-out); A40 FP16 Qwen2.5-14B 635-698 tok/s at 50-100 concurrency; measure with vllm bench before committing.

## Facts (each with source and verification flag)

- [verified] Claude Haiku 5.5: $0.10/MTok input, $0.01/MTok cache read, $0.50/MTok output for prompts up to 100,000 tokens; $0.50/$0.05/$2.50 for prompts over 100k; Batch API gives 50% off ($0.05 batch input).  
  source: https://platform.claude.com/docs/en/about-claude/pricing
- [verified] Claude Sonnet 5.5 is $2 input / $0.10 cache read / $10 output per MTok; Claude Opus 5.5 is $4 / $0.20 / $20; batch input $1 and $2 respectively.  
  source: https://platform.claude.com/docs/en/about-claude/pricing
- [verified] Claude API model IDs are claude-haiku-5-5 and claude-sonnet-5-5 (also claude-opus-5-5, claude-fable-5-1); all have 1M context and 128K max output; Haiku 5.5 default effort is medium.  
  source: https://platform.claude.com/docs/en/models/overview
- [verified] OpenAI Standard pricing per 1M tokens (input/cached/output): gpt-6-luna $0.10/$0.01/$0.50; gpt-6.1-sol $2.00/$0.10/$10.00; gpt-6-astra $10/$1/$50; gpt-5-mini $0.25/$0.025/$2.00; gpt-4.1-mini $0.40/$0.10/$1.60; gpt-4o-mini $0.15/$0.075/$0.60; gpt-4o $2.50/$1.25/$10.00. Batch rates are half of Standard.  
  source: https://developers.openai.com/api/docs/pricing
- [verified] OpenAI models page lists gpt-6-astra, gpt-6.1-sol and gpt-6-luna as the current text models, each with a 1.05M-token context window and 128K max output.  
  source: https://developers.openai.com/api/docs/models
- [verified] Gemini paid-tier pricing per 1M tokens (input/output): gemini-2.5-flash-lite $0.10/$0.40; gemini-3.1-flash-lite $0.25/$1.50; gemini-3.5-flash-lite $0.30/$2.50; gemini-3.8-flash $0.75/$3.75 through Dec 31 2026 ($1.50/$7.50 after); gemini-2.5-flash $0.30/$2.50; Batch API is 50% of Standard.  
  source: https://ai.google.dev/gemini-api/docs/pricing
- [verified] Qwen2.5-7B-Instruct: Apache-2.0, 7.61B params (6.53B non-embedding), 131,072 context with YaRN (config default 32,768), September 2024 release.  
  source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- [verified] Qwen2.5-14B-Instruct: Apache-2.0, 14.7B params (13.1B non-embedding), 131,072 context with YaRN; official Qwen2.5-14B-Instruct-AWQ (4-bit) exists and serves with `vllm serve Qwen/Qwen2.5-14B-Instruct-AWQ`.  
  source: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ
- [verified] Qwen3-8B: Apache-2.0, 8.2B params (6.95B non-embedding), 32,768 native context / 131,072 with YaRN; thinking on by default, disable with enable_thinking=False; recommended non-thinking sampling T=0.7, TopP=0.8, TopK=20.  
  source: https://huggingface.co/Qwen/Qwen3-8B
- [verified] Qwen3-14B: Apache-2.0, 14.8B params (13.2B non-embedding), 32,768 native / 131,072 YaRN; same thinking/non-thinking switch.  
  source: https://huggingface.co/Qwen/Qwen3-14B
- [verified] Qwen3-14B-AWQ is an official 4-bit AWQ checkpoint (Apache-2.0) requiring vllm>=0.8.5 (`vllm serve Qwen/Qwen3-14B-AWQ --enable-reasoning --reasoning-parser deepseek_r1`), transformers>=4.51.0; card recommends presence_penalty 1.5 for quantized models; AWQ scores slightly below bf16 (AIME24 77.0 vs 79.3).  
  source: https://huggingface.co/Qwen/Qwen3-14B-AWQ
- [verified] Qwen3-4B-Instruct-2507: Apache-2.0, 4.0B params, 262,144 native context, non-thinking only (no <think> blocks).  
  source: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- [verified] Qwen3-30B-A3B-Instruct-2507: Apache-2.0, 30.5B total / 3.3B activated, 262,144 context, non-thinking only.  
  source: https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507
- [verified] Qwen3.5-9B: Apache-2.0, 9B params (Hub lists 10B), 262,144 native context extensible to ~1.01M with YaRN, vision-language (image/video input), thinking mode default with enable_thinking=False available; /think and /nothink soft switches not officially supported; citation dated February 2026.  
  source: https://huggingface.co/Qwen/Qwen3.5-9B
- [verified] Qwen3.5-4B: Apache-2.0, 4B params, 262,144 context, thinking default; Qwen3.5 blog at qwen.ai/blog?id=qwen3.5; collection updated March 2026.  
  source: https://huggingface.co/Qwen/Qwen3.5-4B
- [verified] Qwen3.8-27B (Aug 2026): Apache-2.0, 27B params (28B listed), 262,144 context extensible to 1M, image-text-to-text, thinking default with reasoning_effort xhigh/medium/low; Qwen org page shows Qwen3.8 only in 27B, 180B-Flash-Next and 2.4T-A95B sizes.  
  source: https://huggingface.co/Qwen/Qwen3.8-27B
- [verified] Ollama qwen3 tag download sizes: 4b 2.5 GB, 8b 5.2 GB, 14b 9.3 GB, 30b 19 GB, 32b 20 GB (default quantization not stated).  
  source: https://ollama.com/library/qwen3
- [verified] meta-llama/Llama-3.1-8B-Instruct: Llama 3.1 Community License, 8B params, 128k context, released July 23 2024, gated (must share contact info).  
  source: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
- [verified] Llama 3.1 Community License requires 'Built with Llama' display, a Notice file, derivative model names beginning with 'Llama', and a separate licence if >700M MAU as of July 23 2024; no research-specific restrictions.  
  source: https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE
- [verified] Meta's HF org lists Llama 4 (Scout 109B, Maverick 402B MoE), Llama 3.3 70B, Llama 3.2 1B/3B text and 11B/90B vision, Llama 3.1 8B-405B; nothing newer than Llama 4, so 3.1-8B is the latest ~8B text instruct model.  
  source: https://huggingface.co/meta-llama
- [verified] google/gemma-3-12b-it: licence 'gemma' (Gemma Terms of Use), 12B BF16, 128K input context, 8,192 output cap, gated; family sizes 1B/4B/12B/27B.  
  source: https://huggingface.co/google/gemma-3-12b-it
- [verified] Gemma Terms of Use require passing use restrictions downstream, a Notice file ('Gemma is provided under and subject to the Gemma Terms of Use...'), prominent modification notices, and state that Gemma 4 is covered by a separate licence.  
  source: https://ai.google.dev/gemma/terms
- [verified] Google's Gemma 4 model card states 'License: Apache 2.0'; sizes E2B (2.3B effective/5.1B), E4B (4.5B/8B), 12B Unified (11.95B dense), 26B A4B MoE (25.2B total/3.8B active), 31B dense (30.7B); 128K context for E2B/E4B, 256K for 12B/26B/31B.  
  source: https://ai.google.dev/gemma/docs/core/model_card_4
- [verified] Gemma 4 was released April 2, 2026 under Apache 2.0 per the Hugging Face launch blog.  
  source: https://huggingface.co/blog/gemma4
- [verified] google/gemma-4-12B-it: Apache 2.0, 12B BF16, 256K context, text/image/audio/video input, no gating notice shown.  
  source: https://huggingface.co/google/gemma-4-12B-it
- [verified] google/gemma-4-E4B-it: Apache 2.0, 8B params with embeddings (4.5B effective), 128K context, no gating notice shown.  
  source: https://huggingface.co/google/gemma-4-E4B-it
- [verified] microsoft/phi-4: MIT, 14B params, 16K context, released December 12 2024.  
  source: https://huggingface.co/microsoft/phi-4
- [verified] microsoft/Phi-4-mini-instruct: MIT, 3.8B params, 128K context, February 2025.  
  source: https://huggingface.co/microsoft/Phi-4-mini-instruct
- [verified] microsoft/Phi-4-reasoning-vision-15B: MIT, ~15B, 16,384 context, text+image input, released March 4 2026; web search found no Phi-5 release as of 2026.  
  source: https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
- [verified] sentence-transformers/all-MiniLM-L6-v2: Apache-2.0, 22.7M params, 384-d, input truncated at 256 word pieces.  
  source: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- [verified] BAAI/bge-small-en-v1.5: MIT, ~33.4M params, 384-d, 512 max tokens.  
  source: https://huggingface.co/BAAI/bge-small-en-v1.5
- [verified] BAAI/bge-base-en-v1.5: MIT (free for commercial use), ~0.1B params, 768-d, 512 max tokens; bge-large is 1024-d.  
  source: https://huggingface.co/BAAI/bge-base-en-v1.5
- [verified] Alibaba-NLP/gte-base-en-v1.5: Apache-2.0, 137M params, 768-d, 8192 max tokens.  
  source: https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5
- [verified] thenlper/gte-small: MIT, ~33.4M params, 384-d, 512 max tokens.  
  source: https://huggingface.co/thenlper/gte-small
- [verified] nomic-ai/nomic-embed-text-v1.5: Apache-2.0, ~0.1B params, 768-d with Matryoshka 512/256/128/64, 8192 tokens, requires task prefixes search_document:/search_query:/clustering:/classification:.  
  source: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
- [verified] spaCy is MIT licensed (Copyright ExplosionAI GmbH).  
  source: https://raw.githubusercontent.com/explosion/spaCy/master/LICENSE
- [verified] en_core_web_sm 3.8.0: MIT, 12 MB, components tok2vec/tagger/parser/attribute_ruler/lemmatizer/ner (+senter disabled), spaCy >=3.8.0,<3.9.0.  
  source: https://raw.githubusercontent.com/explosion/spacy-models/master/meta/en_core_web_sm-3.8.0.json
- [verified] GLiNER (urchade/GLiNER): Apache-2.0, pip gliner, NAACL 2024 (arXiv 2311.08526); checkpoints gliner_small-v2.5, gliner_mediumv2.1, gliner-relex-large-v1.0 etc.  
  source: https://github.com/urchade/GLiNER
- [verified] GLiNER2 (fastino-ai/GLiNER2): Apache-2.0, pip gliner2 (gliner2[local] for local inference), DeBERTa-v3 backbones, schema-based entity/relation/structured JSON extraction and classification; sizes gliner2.5-small-v1 74M, gliner2.5-base-v1 194M, gliner2-base-v1 205M, gliner2.5-multi-v1 287M, gliner2-large-v1 340M; EMNLP 2025 System Demos (aclanthology.org/2025.emnlp-demos.10).  
  source: https://github.com/fastino-ai/GLiNER2
- [verified] Babelscape/rebel-large: CC-BY-NC-SA-4.0, 0.4B BART-based seq2seq relation extraction emitting <triplet><subj><obj> linearized triples over 200+ relation types; Huguet Cabot & Navigli, Findings of EMNLP 2021.  
  source: https://huggingface.co/Babelscape/rebel-large
- [verified] OpenIE6 (dair-iitd/openie6): GPL-3.0; Python 3.6 conda env, transformers==2.6.0 pinned (newer versions silently wrong), models ~6.6 GB via zenodo_get 4055395, data via 4054476; three models (OIE, conjunction, rescoring); Kolluru et al. EMNLP 2020.  
  source: https://github.com/dair-iitd/openie6
- [verified] Stanford CoreNLP OpenIE annotator requires natlog; ~100 sentences/s/core; invoked via annotators tokenize,pos,lemma,depparse,natlog,openie or Simple API Sentence.openieTriples(); options openie.resolve_coref, openie.triple.strict, openie.max_entailments_per_clause (default 1000).  
  source: https://stanfordnlp.github.io/CoreNLP/openie.html
- [verified] Stanford CoreNLP is licensed under GNU GPL version 3 (29 June 2007).  
  source: https://raw.githubusercontent.com/stanfordnlp/CoreNLP/main/LICENSE.txt
- [verified] Stanza (Python client for CoreNLP) is Apache License 2.0, Copyright 2019 Stanford.  
  source: https://raw.githubusercontent.com/stanfordnlp/stanza/main/LICENSE
- [verified] Stanford OpenIE paper: Angeli, Johnson Premkumar, Manning, 'Leveraging Linguistic Structure For Open Domain Information Extraction', ACL-IJCNLP 2015, pp. 344-354.  
  source: https://aclanthology.org/P15-1034/
- [verified] vLLM quantization support on Ada (SM 8.9, e.g. RTX 4090): AWQ, GPTQ, Marlin (GPTQ/AWQ/FP8/FP4), LLM Compressor INT8 W8A8 and FP8 W8A8, BitsAndBytes, GGUF all supported.  
  source: https://docs.vllm.ai/en/latest/features/quantization/
- [verified] LongMemEval README: 500 questions; LongMemEval_S history ~115k tokens (~40 sessions, Llama 3 tokenizer); LongMemEval_M ~500 sessions; six question types plus _abs abstention suffix; evaluate_qa.py uses gpt-4o; MIT licence.  
  source: https://github.com/xiaowu0162/LongMemEval
- [verified] LongMemEval paper (ICLR 2025): LongMemEval_M is ~1.5M tokens; judge is gpt-4o-2024-08-06 with >97% agreement with human experts; 30 abstention questions.  
  source: https://arxiv.org/html/2410.10813
- [verified] HF dataset xiaowu0162/longmemeval (MIT) files: longmemeval_s 278 MB, longmemeval_m 2.75 GB, longmemeval_oracle 15.4 MB (no .json extension).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval/tree/main
- [verified] Measured from downloaded longmemeval_s: 500 questions (single-session-user 70, single-session-assistant 56, single-session-preference 30, multi-session 133, knowledge-update 78, temporal-reasoning 133; 30 abstention); 50.2 sessions/question (39-66); 25,112 session instances; 19,829 unique session ids; 246,930 turn instances; 244.9M chars total (~61M tokens at 4 chars/token, ~122k/question); unique-session text 197.8M chars (~49.4M tokens); 948 evidence sessions (1.9/question).  
  source: https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s
- [verified] Measured from downloaded locomo10.json: 10 conversations; 1,986 QA pairs; category counts {1:282, 2:321, 3:96, 4:841, 5:446}; sessions per conversation 19-32 (mean 27.2); 5,882 turns (mean 588/conv); 133,772 words of utterance text (mean 13,377/conv).  
  source: https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
- [verified] LoCoMo released data is licensed CC BY-NC 4.0 (non-commercial, attribution).  
  source: https://raw.githubusercontent.com/snap-research/locomo/main/LICENSE.txt
- [verified] LoCoMo paper (arXiv 2402.17753): 50 conversations collected, avg 19.3 sessions, 304.9 turns, 9,209 tokens per conversation; 7,512 QA (single-hop 2,705, multi-hop 1,104, temporal 1,547, open-domain 285, adversarial 1,871); README says 10 conversations are released.  
  source: https://arxiv.org/html/2402.17753
- [verified] Mem0 paper (arXiv 2504.19413): LoCoMo = 10 conversations, ~600 dialogues and ~26,000 tokens each, ~200 questions/conversation, four categories used (adversarial excluded); all LLM ops used GPT-4o-mini with ADD/UPDATE/DELETE/NOOP tool calls, m=10 recent messages, s=10 similar memories; Mem0 uses 1,764 tokens/query vs 26,031 full-context; memory store ~7k tokens/conversation (Mem0), 14k (Mem0g).  
  source: https://arxiv.org/html/2504.19413
- [verified] Memory-R1 (arXiv 2508.19828): LLaMA-3.1-8B-Instruct and Qwen-2.5 3B/7B/14B backbones, PPO and GRPO in VERL, 152 train / 81 val / 1,307 test LoCoMo QA pairs (adversarial excluded), ADD/UPDATE/DELETE/NOOP operations, trained on 4x NVIDIA H100 80GB (8 for 14B), batch 128; EM reward.  
  source: https://arxiv.org/html/2508.19828
- [verified] RunPod GPU pricing: RTX 4090 $0.34/h community / $0.89/h secure; RTX A5000 $0.16/$0.27; A40 $0.35/$0.59; L40S $0.79/$1.09; A100 80GB PCIe $1.19/$1.79; H100 SXM $2.69/$3.99.  
  source: https://www.runpod.io/pricing
- [verified] RTX 4090 vLLM FP16 benchmark (100 input / 600 output tokens, 300 requests, offline): DeepSeek-R1-Distill-Qwen-7B 4,172.8 total tok/s; DeepSeek-R1-Distill-Llama-8B 2,769 tok/s; Qwen2.5-3B 7,214 tok/s; gemma-2-9b-it only 473 tok/s (VRAM-limited).  
  source: https://databasemart.com/blog/vllm-gpu-benchmark-rtx4090
- [verified] A40 48GB vLLM FP16 (100 in / 600 out): Qwen2.5-7B-Instruct 1,307-1,357 tok/s; Llama-3.1-8B 1,246-2,081 tok/s; Qwen2.5-14B-Instruct 635-698 tok/s at 50-100 concurrent requests; 14B needed --max-model-len 4096.  
  source: https://databasemart.com/blog/vllm-gpu-benchmark-a40
- [unverified] Derived (not fetched): Qwen3-14B BF16 weights ~29.6 GB exceed 24 GB; Qwen3-8B BF16 ~16.4 GB fits; all-MiniLM-L6-v2 ~90 MB fp32 (22.7M x 4 bytes); cost/GPU-hour totals in the summary use assumed 5,000 tok/s prefill and 1,500 tok/s batched decode for a 7-9B model on a 24 GB Ada GPU.  
  source: https://databasemart.com/blog/vllm-gpu-benchmark-rtx4090

## reusable_assets

[
  {
    "what": "Qwen3-8B (primary local extractor/QA model; Apache-2.0, 8.2B, 32k/128k ctx)",
    "url": "https://huggingface.co/Qwen/Qwen3-8B",
    "how_to_use": "vllm serve Qwen/Qwen3-8B --max-model-len 16384 --gpu-memory-utilization 0.92; pass chat_template_kwargs {\"enable_thinking\": false}; sampling T=0.7, top_p=0.8, top_k=20 (card's non-thinking defaults) or greedy for deterministic extraction; fits 24 GB in BF16.",
    "license": "Apache-2.0"
  },
  {
    "what": "Qwen3-14B-AWQ (official 4-bit; the only way to run a 14B on 24 GB with vLLM)",
    "url": "https://huggingface.co/Qwen/Qwen3-14B-AWQ",
    "how_to_use": "vllm>=0.8.5: vllm serve Qwen/Qwen3-14B-AWQ (drop --enable-reasoning and set enable_thinking=false for extraction); add presence_penalty=1.5 per card; ~9.3 GB weights leaves ~12 GB KV cache.",
    "license": "Apache-2.0"
  },
  {
    "what": "Qwen3-4B-Instruct-2507 (non-thinking-only small instruct; weak-extractor arm or LoRA-GRPO base for Memory-R1-style gate)",
    "url": "https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507",
    "how_to_use": "Drop-in replacement in vLLM/Ollama; 262k native context; no enable_thinking flag needed.",
    "license": "Apache-2.0"
  },
  {
    "what": "Gemma 4 12B-it / E4B-it (Apache-2.0 replacement for Gemma 3; 256K/128K ctx)",
    "url": "https://huggingface.co/google/gemma-4-12B-it",
    "how_to_use": "12B BF16 (~24 GB) will not fit a 24 GB card: use an int4/GGUF build (Ollama) or E4B-it (8B with embeddings) in BF16; disable thinking in the chat template for extraction.",
    "license": "Apache 2.0"
  },
  {
    "what": "Claude Haiku 5.5 (API gate + judge-at-scale)",
    "url": "https://platform.claude.com/docs/en/about-claude/pricing",
    "how_to_use": "model=\"claude-haiku-5-5\"; put the fixed gate/judge instruction first with cache_control ephemeral (cache reads $0.01/MTok); submit via Message Batches for 50% off; keep prompts <100k tokens to stay on the $0.10/$0.50 tier; use structured outputs (output_config.format) for ADD/UPDATE/DELETE/NOOP decisions.",
    "license": "Commercial API (Anthropic terms)"
  },
  {
    "what": "OpenAI gpt-4o (canonical LongMemEval judge) and gpt-6-luna (price-matched alternative gate/judge)",
    "url": "https://developers.openai.com/api/docs/pricing",
    "how_to_use": "Run LongMemEval's evaluate_qa.py with gpt-4o(-2024-08-06) on the 9,000 LME-S judgments (~$9) for comparability with published numbers; gpt-6-luna at $0.10/$0.50 can serve as a second-family judge.",
    "license": "Commercial API (OpenAI terms)"
  },
  {
    "what": "all-MiniLM-L6-v2 (entity-resolution embeddings for triple endpoints)",
    "url": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    "how_to_use": "SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); 384-d, 256-token cap is fine for entity strings/triples; cosine threshold + string normalization for node merging.",
    "license": "Apache-2.0"
  },
  {
    "what": "bge-base-en-v1.5 / nomic-embed-text-v1.5 (turn and fact retrieval, held fixed across policies)",
    "url": "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5",
    "how_to_use": "nomic requires trust_remote_code=True and prefixes 'search_document: ' / 'search_query: '; 8192-token input lets you embed whole sessions; bge-base (512 tokens) for per-turn/per-fact indexing.",
    "license": "Apache-2.0 (nomic) / MIT (bge)"
  },
  {
    "what": "GLiNER2 (primary LLM-free entity+relation+schema extractor)",
    "url": "https://github.com/fastino-ai/GLiNER2",
    "how_to_use": "pip install gliner2[local]; define a schema with entity types and relation types (speaker, attribute, event) and call extract(); gliner2.5-base-v1 (194M) runs on CPU/GPU at thousands of sentences/min.",
    "license": "Apache-2.0"
  },
  {
    "what": "spaCy + en_core_web_sm/trf (rule-based SVO triple arm)",
    "url": "https://raw.githubusercontent.com/explosion/spacy-models/master/meta/en_core_web_sm-3.8.0.json",
    "how_to_use": "python -m spacy download en_core_web_sm (12 MB, MIT); use DependencyMatcher patterns (nsubj-VERB-dobj/pobj, copula attributes) plus coref-free speaker substitution for first-person turns.",
    "license": "MIT"
  },
  {
    "what": "Stanford CoreNLP OpenIE (classic Open IE baseline)",
    "url": "https://stanfordnlp.github.io/CoreNLP/openie.html",
    "how_to_use": "Run CoreNLP server (Java) with annotators tokenize,pos,lemma,depparse,natlog,openie and openie.resolve_coref=true; call from Python via stanza.server.CoreNLPClient (Stanza is Apache-2.0, server is GPL-3.0; keep it as a separate process, do not vendor).",
    "license": "GPL-3.0"
  },
  {
    "what": "REBEL rebel-large (seq2seq relation extraction, research-only)",
    "url": "https://huggingface.co/Babelscape/rebel-large",
    "how_to_use": "transformers pipeline('text2text-generation'); decode <triplet>/<subj>/<obj> markers with the card's extract_triplets helper; relation vocabulary is Wikidata-centric so expect low recall on conversational attributes.",
    "license": "CC-BY-NC-SA-4.0"
  },
  {
    "what": "LongMemEval-S data (500 q, 278 MB)",
    "url": "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s",
    "how_to_use": "Extract once per unique haystack_session_id (19,829 vs 25,112 instances), then build each question's graph from its 50 sessions; judge with evaluate_qa.py.",
    "license": "MIT"
  },
  {
    "what": "LoCoMo locomo10.json (10 conversations, 1,986 QA)",
    "url": "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    "how_to_use": "Iterate conversation['session_N'] keyed turns with speaker and dia_id; qa[].category 1-5 (map per community eval code: 1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial; verify against the repo's evaluation script); drop category 5 to match Mem0/Memory-R1.",
    "license": "CC BY-NC 4.0"
  },
  {
    "what": "vLLM quantization/hardware matrix",
    "url": "https://docs.vllm.ai/en/latest/features/quantization/",
    "how_to_use": "Confirms AWQ/GPTQ/Marlin/FP8/BitsAndBytes/GGUF all work on Ada (RTX 4090); use Marlin kernels for AWQ throughput.",
    "license": "Apache-2.0 (vLLM)"
  },
  {
    "what": "RunPod GPU pricing (for converting GPU-hours to USD)",
    "url": "https://www.runpod.io/pricing",
    "how_to_use": "RTX 4090 $0.34/h community, $0.89/h secure; ~26 GPU-h (7-9B) or ~52 GPU-h (14B) for the full grid = $9-46.",
    "license": "n/a"
  }
]

## risks

[
  "Cost/GPU-hour totals rest on unverified throughput assumptions (5,000 tok/s prefill, 1,500 tok/s batched decode for 7-9B on a 24 GB Ada GPU) and on assumed per-call token shapes (800-token cached extraction prefix, 15% output ratio, 3.7k-token gate calls, 1.85k-token QA context). Measure with `vllm bench` and a 50-session pilot before committing; a 2x error is plausible.",
  "Thinking-by-default models (Qwen3, Qwen3.5, Qwen3.8, Gemma 4) will inflate output tokens 10-50x if enable_thinking is not disabled, blowing both USD and GPU-hour budgets; use -Instruct-2507 non-thinking variants or set the flag explicitly in every call.",
  "The LLM gate must run per question-history (25,112 LME-S session instances), not per unique session, because ADD/UPDATE/DELETE/NOOP depends on the current graph; only extraction benefits from the 21% unique-session dedup.",
  "Claude Haiku 5.5 jumps to $0.50/$2.50 for prompts over 100k tokens; any full-context baseline on LME-S (~115-122k tokens) must use Sonnet 5.5, gpt-6-luna or Gemini Flash-Lite instead, or be budgeted separately.",
  "OpenAI pricing page and models page disagree on what is 'current' (GPT-6 family on both, but gpt-5.6-* rows appear only on pricing); OpenAI and Google quote promotional prices with expiry dates (Gemini 3.8 Flash doubles after Dec 31 2026) and the WebFetch summaries were produced by a small model, so re-check exact rows before quoting in the paper.",
  "Judge choice confounds results: LongMemEval's published numbers use gpt-4o-2024-08-06; using Haiku 5.5 for both the Mem0-style gate and the judge risks same-family self-preference. Budget ~$9 for a gpt-4o judge pass on LME-S and report agreement between judges.",
  "3 extraction 'seeds' are meaningless at temperature 0 with a deterministic local model; define seeds as sampling temperature >0 or as extractor variants (LLM vs GLiNER2 vs spaCy), otherwise reviewers will flag it.",
  "Full leave-one-fact-out for the structural-change/accuracy correlation is infeasible (~750 admitted facts per LME-S question -> ~1.9M re-answers); restrict ablation to the retrieved top-k facts or to grouped ablations and state this as a design limit.",
  "Licence traps: Llama 3.1 is gated with naming/attribution duties; Gemma 3 is under the restrictive Gemma Terms (Gemma 4 is Apache-2.0, so prefer it); REBEL is CC-BY-NC-SA (ShareAlike on derived outputs); OpenIE6 and CoreNLP are GPL-3.0 (run as separate processes, do not link into an MIT/Apache release); LoCoMo is CC BY-NC 4.0 (no commercial use of derived memory stores).",
  "OpenIE6 is effectively unmaintained (Python 3.6, transformers==2.6.0, 6.6 GB Zenodo models) and may not install on a 2026 CUDA stack; treat it as optional.",
  "Memory-R1-style learned gate replication needs multi-GPU RL (paper: 4x H100 80GB, 8 for 14B); a 24 GB card can only do LoRA-GRPO on a 3-4B model, which changes the baseline and must be disclosed.",
  "Phi-4 (14B) has only a 16K context and no official AWQ/GPTQ checkpoint; Qwen3.5/3.8 small-model availability is thin (Qwen3.5 tops out at 9B dense; Qwen3.8 has no 7-14B dense), so Qwen3-8B/14B-AWQ remain the pragmatic picks despite being 2025 models."
]

## confidence_notes

High confidence (fetched primary pages): all per-token prices (Anthropic platform docs, OpenAI developers pricing page, Google ai.google.dev pricing), model licences and parameter counts (Hugging Face model cards, Google Gemma 4 model card, Meta licence text, spaCy/CoreNLP/Stanza LICENSE files, GLiNER2/OpenIE6 READMEs), dataset licences, and the dataset statistics I computed directly from the downloaded longmemeval_s (278 MB) and locomo10.json files. Medium confidence: WebFetch summaries are produced by a small model, so individual table rows (especially OpenAI's many GPT-5.x/6 rows and Gemini promotional dates) should be re-read before quoting in the paper; the LoCoMo category-number-to-name mapping (1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial) comes from my knowledge of community eval code, not from a fetched page. Low confidence (explicitly derived/assumed, verified=false): throughput assumptions (5,000 tok/s prefill, 1,500 tok/s decode on a 24 GB Ada GPU for 7-9B; half for 14B-AWQ), per-call token shapes, the 15% output ratio for extraction, BF16 weight-size arithmetic, and the resulting USD and GPU-hour totals (~$85-105 standard / ~$45-55 Batch on Haiku 5.5; ~26 GPU-h for 7-9B or ~52 GPU-h for 14B). Token counts from the datasets use a 4-chars-per-token heuristic (gives ~122k/question vs the README's ~115k Llama-3 tokens, a 7% gap). Web search could not find Qwen3.5 or Qwen3.8 dense models between 9B and 27B, nor any Phi-5; absence claims are only as good as the pages fetched (Qwen org page, Meta org page, Phi search results).