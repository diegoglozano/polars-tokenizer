# Provider capability matrix

Last reviewed: 2026-09-24.

This matrix describes `polars-tokenizer` support, not everything a provider's
SDK or remote API can do. All three columns refer to raw UTF-8 text in a Polars
column. Chat wrappers, tools, images, audio, request serialization, and other
provider-added tokens remain outside this layer.

Status labels:

- **Available**: exposed through the supported Python/Polars API.
- **Prototype**: implemented and tested in Rust, but intentionally not exposed
  through the supported Python/Polars API yet.
- **Planned**: architecture and correctness requirements are understood, but
  implementation has not landed.
- **Research**: exact artifacts, behavior, licensing, or provider contracts
  still need investigation.
- **Not local-exact**: the provider does not currently expose a reproducible
  local definition that satisfies this project's exactness contract.

| Provider / model family | Exact counting | Estimate counting | Cost estimation |
|---|---|---|---|
| **OpenAI** | **Available:** exact `o200k_base` and `cl100k_base` raw-text counts, plus a dated registry of exact model aliases. | **Planned:** separately trained and evaluated per tokenizer family. | **Planned:** model aliases will resolve tokenizer metadata independently from dated input, cached-input, and output prices. |
| **Google Gemini / Gemma** | **Prototype:** pinned Gemma 3 `SentencePiece` BPE kernel has exact local parity on the initial corpus. Artifact distribution, wider parity, Polars integration, and Gemma 4 remain gates. | **Planned:** Gemma 3 and Gemma 4 require separate error models. Google's remote `countTokens` is a reference/request-counting path, not this local estimator. | **Planned:** keyed by Gemini model and dated Google pricing snapshot; multimodal and request-level charges remain separate. |
| **Anthropic Claude** | **Not local-exact:** current tokenizer definitions are not published as a stable local contract. Anthropic documents its message token-count result as an estimate, and its archived local tokenizer is inaccurate for Claude 3 and later. | **Research:** a local raw-text estimator is possible, but must be labeled independently from Anthropic's request-level count endpoint and evaluated per tokenizer generation. | **Planned:** exact-count mode will be unavailable until an exact definition exists; estimated-token cost may be supported with an explicit mode and dated model prices. |
| **Mistral AI** | **Planned:** treat `SentencePiece` V1/V2/V3 and `tiktoken`-based Tekken as distinct, versioned tokenizer definitions. No Mistral family is implemented today. | **Planned:** separate estimators for `SentencePiece` and Tekken families. | **Planned:** keyed by Mistral API model and dated price snapshot; open-weight deployments require their serving provider. |
| **Cohere** | **Research:** Cohere exposes model-specific tokenization and downloadable `tokenizer_url` metadata, but no definition is pinned or implemented here yet. | **Planned:** only after representative corpora and model-family stability are established. | **Planned:** keyed by Cohere model and dated price snapshot. |
| **Meta Llama** | **Research:** versioned tokenizer artifacts exist, but family/version behavior and redistribution terms must be audited before implementation. | **Planned:** per tokenizer generation, not one estimator for every Llama release. | **Caller/host dependent:** self-hosted models have no universal per-token API price; hosted costs must be keyed by serving provider plus model. |
| **Alibaba Qwen** | **Research:** official families use byte-level BPE/tokenizer artifacts, but versions, special-token behavior, and licenses must be pinned independently. | **Planned:** per compatible tokenizer generation and language mix. | **Provider/region dependent:** registry entries must identify the serving API, region/currency where relevant, model, and effective date. |
| **DeepSeek** | **Research:** local tokenizer artifacts exist for open models, but API-model equivalence, revisions, and special-token behavior require proof before exact support. | **Planned:** per tokenizer generation after exact/reference corpus work. | **Planned for direct API snapshots;** third-party hosting is keyed separately by serving provider and model. |

## Registry dimensions

Provider names do not belong in low-level tokenizer kernels. The higher-level
registry needs separate identities for:

```text
model owner + model identifier + serving provider
    -> tokenizer definition/version
    -> optional dated pricing snapshot
```

This distinction matters for open-weight models and cloud marketplaces. For
example, one Llama model can have one tokenizer but different prices on
self-hosting, Amazon Bedrock, Azure, Vertex AI, or another host. Likewise,
Azure-hosted OpenAI pricing is not interchangeable with direct OpenAI pricing.

Cost results must always include the model, serving provider, billing category,
currency, unit, effective date, pricing-registry version, and token-count mode
(`exact` or `estimate`). DataFrame expressions must not fetch live pricing.

## Primary references

- [OpenAI `tiktoken`](https://github.com/openai/tiktoken)
- [Google token counting](https://ai.google.dev/gemini-api/docs/tokens) and the
  [pinned Gemma 3 prototype notes](gemma3-prototype.md)
- [Anthropic token counting](https://platform.claude.com/docs/en/build-with-claude/token-counting)
  and its [archived local tokenizer warning](https://github.com/anthropics/anthropic-tokenizer-typescript)
- [Mistral tokenizer families](https://docs.mistral.ai/resources/cookbooks/concept-deep-dive-tokenization-templates)
  and [`mistral-common`](https://github.com/mistralai/mistral-common)
- [Cohere tokens and tokenizers](https://docs.cohere.com/v1/docs/tokens-and-tokenizers)
- [Meta Llama tokenizer artifacts](https://github.com/meta-llama/llama-models)
- [Qwen 3 tokenization concepts](https://github.com/QwenLM/Qwen3/blob/main/docs/source/getting_started/concepts.md)
- [DeepSeek model/tokenizer reference implementation](https://github.com/deepseek-ai/DeepSeek-V2)

These references establish feasibility and provider semantics. Exact support
still requires a pinned artifact, checksum, license audit, local/reference
parity corpus, and immutable tokenizer-version identifier.
