# MyAI --- From-Scratch Foundation LLM

## No pretrained LLM weights

**Project decision**

This version deliberately does **not** use Qwen, Llama, Mistral, Gemma,
Claude, GPT, or any other pretrained LLM as the model foundation.

We will write the model architecture and training system ourselves.

We may use public/open-source:

-   PyTorch
-   CUDA
-   tokenizer libraries or our own tokenizer implementation
-   distributed-training libraries
-   public datasets
-   public research papers
-   public evaluation datasets
-   public data-processing tools

The distinction is:

``` text
NOT allowed as the model starting point:
Qwen weights
Llama weights
Mistral weights
Gemma weights
GPT weights
Claude weights
any other pretrained LLM checkpoint

Allowed:
PyTorch
CUDA
datasets
data-processing libraries
training infrastructure
research papers
public benchmark data
```

------------------------------------------------------------------------

# 1. What We Are Actually Building

The target is a real decoder-only Transformer foundation model:

``` text
                         MyAI Foundation Model
                                  |
                    +-------------+-------------+
                    |                           |
               Tokenizer                    Dataset
                    |                           |
                    +-------------+-------------+
                                  |
                           Transformer
                                  |
                    +-------------+-------------+
                    |                           |
                Pretraining                Evaluation
                    |
                    v
              MyAI-Base
                    |
                    v
             Instruction tuning
                    |
                    v
             MyAI-Instruct
                    |
          +---------+---------+
          |                   |
     Coding training    Reasoning training
          |                   |
          v                   v
     MyAI-Coder          MyAI-Reasoner
          \                   /
           +--------+--------+
                    |
               Agent Runtime
                    |
       +------------+-------------+
       |            |             |
      Tools       Memory          RAG
       |            |             |
      Web        Context       Documents
      Files      Long-term     Knowledge
      Shell
                    |
                    v
                 MyAI
```

The model itself is ours.

------------------------------------------------------------------------

# 2. What "From Scratch" Means

From scratch means:

``` text
Randomly initialized model
        ↓
Our tokenizer
        ↓
Our training data pipeline
        ↓
Our training loop
        ↓
Our checkpoints
        ↓
Our model weights
```

We are NOT doing:

``` text
Qwen weights
   ↓
fine-tuning
   ↓
MyAI
```

That would be adaptation, not foundation-model pretraining from scratch.

------------------------------------------------------------------------

# 3. What We Can Reuse

We do not need to reinvent GPU mathematics.

Use:

``` text
PyTorch
CUDA
NCCL
NumPy
Parquet
Hugging Face Datasets
Datatrove
SentencePiece / tokenizers if desired
FSDP / DeepSpeed for scaling
FlashAttention if desired
```

These are infrastructure.

The neural network itself can be implemented in our repository.

------------------------------------------------------------------------

# 4. Model Architecture

Initial architecture:

``` text
Token IDs
    |
Embedding
    |
Transformer Block × N
    |
Final RMSNorm
    |
LM Head
    |
Logits
```

Each Transformer block:

``` text
Input
  |
RMSNorm
  |
Self Attention
  |
Residual
  |
RMSNorm
  |
SwiGLU MLP
  |
Residual
  |
Output
```

Core components:

-   Token embedding
-   RMSNorm
-   Causal self-attention
-   RoPE
-   Grouped-query attention if selected
-   SwiGLU
-   Residual connections
-   Linear layers
-   Final normalization
-   Language-model head
-   KV cache for inference

------------------------------------------------------------------------

# 5. Model Code We Write

Example structure:

``` text
myai/
  foundation/
    config.py
    model.py
    layers/
      embeddings.py
      rmsnorm.py
      rope.py
      attention.py
      mlp.py
      transformer_block.py
    generation.py
    cache.py
```

We should understand and implement each component rather than copying an
existing LLM implementation.

------------------------------------------------------------------------

# 6. First Model Should Be Small

Do not start by trying to train an 8B model.

Development sequence:

``` text
MyAI-20M
    ↓
MyAI-100M
    ↓
MyAI-300M
    ↓
MyAI-1B
    ↓
MyAI-3B
    ↓
larger model
```

The first objective is proving that:

``` text
data
→ tokenizer
→ model
→ training
→ checkpoint
→ generation
→ evaluation
```

works correctly.

------------------------------------------------------------------------

# 7. Initial Model Configuration

Example research configuration:

``` yaml
model_name: MyAI-100M

vocab_size: 32000
hidden_size: 768
num_layers: 12
num_heads: 12
max_sequence_length: 2048

normalization: rmsnorm
activation: swiglu
position_encoding: rope

tie_embeddings: true
```

These values are an experimental starting point, not a claim that they
are optimal.

After the pipeline works, run scaling experiments.

------------------------------------------------------------------------

# 8. Tokenizer

Because we are training from scratch, we can train our own tokenizer.

Pipeline:

``` text
Public training corpus
       ↓
Normalization
       ↓
Tokenizer training corpus
       ↓
BPE / Unigram tokenizer
       ↓
Vocabulary
       ↓
Tokenizer
```

Start around:

``` text
32K vocabulary
```

Later test:

``` text
32K
50K
64K
100K
```

Evaluate:

-   tokens per byte
-   compression
-   multilingual coverage
-   code representation
-   memory
-   downstream loss

------------------------------------------------------------------------

# 9. Dataset Strategy

We use public datasets, but "public" does not mean "license-free".

Every dataset gets a manifest:

``` json
{
  "name": "dataset-name",
  "version": "version",
  "source": "source-url",
  "license": "license",
  "download_date": "YYYY-MM-DD",
  "allowed_use": "research/commercial/etc",
  "provenance": "source information",
  "filters": []
}
```

Do not mix datasets before recording their licenses.

------------------------------------------------------------------------

# 10. General Text Dataset

## FineWeb

FineWeb is a large English web corpus produced from Common Crawl and
processed/deduplicated with the Datatrove pipeline.

The current Hugging Face dataset card reports more than 18.5T tokens.

License:

``` text
ODC-By 1.0
```

It is also subject to Common Crawl terms.

Source:

https://huggingface.co/datasets/HuggingFaceFW/fineweb

Important:

The underlying web pages can have their own copyright/terms. We must
retain provenance and review the intended use.

FineWeb is therefore a possible large-scale pretraining source, not a
blanket statement that every underlying webpage is unrestricted.

------------------------------------------------------------------------

# 11. FineWeb Development Strategy

Do NOT download the complete corpus first.

Use a small subset:

``` text
10M tokens
        ↓
pipeline test

50M tokens
        ↓
training test

100M-1B tokens
        ↓
research experiment

larger corpus
        ↓
scaling
```

The goal is to validate the pipeline before spending large amounts of
storage/GPU time.

------------------------------------------------------------------------

# 12. Code Dataset

## The Stack v2

The Stack v2 contains source code from more than 600 programming
languages.

However, it is NOT a simple unrestricted dataset.

Its terms state:

-   bulk download requires an agreement with Software Heritage and INRIA
-   training use must follow Software Heritage principles
-   source repositories have different licenses
-   original repository licenses and attribution requirements apply
-   provenance is provided
-   validated removal requests are incorporated into updates

Therefore:

``` text
The Stack v2
      ↓
license/provenance filter
      ↓
allowed code
      ↓
deduplication
      ↓
MyAI code corpus
```

Source:

https://huggingface.co/datasets/bigcode/the-stack-v2

Do not blindly copy all repositories into the training set.

------------------------------------------------------------------------

# 13. Other Public Data Sources

Potential sources:

``` text
FineWeb
FineWeb-Edu
RedPajama datasets
The Stack / The Stack v2
Wikipedia
open-access scientific corpora
public-domain books
permissively licensed documentation
permissively licensed source code
your own generated/curated instruction data
```

Every source must be individually checked.

For example, RedPajama-Data-V2 uses Common Crawl-derived material and
its code is Apache-2.0, but the underlying data has its own terms.

------------------------------------------------------------------------

# 14. Dataset Mixture

Do not assume one universal mixture.

A research starting point could be:

``` text
General text       60%
Code               15%
Math/science       10%
Educational        10%
Other high-quality  5%
```

Then run experiments.

For a coding-heavy model:

``` text
General text       50%
Code               30%
Math/science       10%
Technical docs      10%
```

These are experimental starting points.

The correct mixture should be determined by evaluation.

------------------------------------------------------------------------

# 15. Data Processing Pipeline

``` text
Raw datasets
     ↓
Provenance
     ↓
License filtering
     ↓
Language identification
     ↓
HTML/boilerplate cleanup
     ↓
Quality filtering
     ↓
PII filtering
     ↓
Sensitive-content filtering
     ↓
Exact deduplication
     ↓
Near deduplication
     ↓
Document scoring
     ↓
Mixture weighting
     ↓
Train/validation split
     ↓
Tokenization
     ↓
Packing
     ↓
Sharded training files
```

------------------------------------------------------------------------

# 16. PII Filtering

The web contains personal information.

Build filters for:

-   email addresses
-   phone numbers
-   physical addresses where identifiable
-   authentication secrets
-   API keys
-   private credentials
-   access tokens
-   financial identifiers

Do not assume a public webpage is safe to put directly into training.

------------------------------------------------------------------------

# 17. Secret Detection

Code data needs secret scanning.

Detect patterns such as:

``` text
API keys
private keys
tokens
passwords
cloud credentials
database credentials
```

Potential tools:

``` text
Gitleaks
TruffleHog
custom regex/rule engine
```

Run them before the code enters the training corpus.

------------------------------------------------------------------------

# 18. Deduplication

Implement:

## Exact deduplication

``` text
SHA-256(document)
```

Same hash:

``` text
remove duplicate
```

## Near deduplication

Use:

``` text
MinHash
SimHash
LSH
n-gram similarity
```

This prevents the same content appearing thousands of times.

------------------------------------------------------------------------

# 19. Quality Scoring

Every document can receive:

``` json
{
  "quality_score": 0.87,
  "language_score": 0.99,
  "length": 12345,
  "duplicate_score": 0.01
}
```

Quality signals may include:

-   repetition
-   sentence quality
-   markup ratio
-   length
-   language confidence
-   code validity
-   source quality
-   boilerplate ratio

------------------------------------------------------------------------

# 20. Training Objective

For a decoder-only language model:

Given:

``` text
tokens = [t1, t2, t3, ... tn]
```

train:

``` text
P(t2 | t1)
P(t3 | t1,t2)
P(t4 | t1,t2,t3)
...
```

The standard loss is next-token cross entropy:

``` text
L = -Σ log P(target_token | previous_tokens)
```

This is the core pretraining objective.

------------------------------------------------------------------------

# 21. Training Loop

Our code should look conceptually like:

``` python
for batch in dataloader:

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    logits = model(input_ids)

    loss = cross_entropy(
        logits,
        labels
    )

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()
```

Production training adds:

-   mixed precision
-   gradient accumulation
-   gradient clipping
-   checkpointing
-   distributed training
-   logging
-   validation
-   resume support
-   fault tolerance

------------------------------------------------------------------------

# 22. Optimizer

Start with:

``` text
AdamW
```

Track:

``` text
learning rate
weight decay
gradient norm
training loss
validation loss
tokens/second
GPU utilization
```

------------------------------------------------------------------------

# 23. Learning Rate

Do not choose a single value forever.

Use:

``` text
warmup
    ↓
peak learning rate
    ↓
decay
```

For example:

``` text
0%
 ↓
warmup
 ↓
100%
 ↓
cosine decay
 ↓
final LR
```

The exact LR requires experiments based on model size, batch size and
dataset.

------------------------------------------------------------------------

# 24. Compute Scaling

The major limitation is compute.

A serious foundation model requires:

``` text
GPU cluster
+
large dataset
+
distributed training
+
high-speed storage
+
high-speed interconnect
+
checkpoint infrastructure
```

Do not confuse:

``` text
"I can run an 8B model"
```

with:

``` text
"I can pretrain an 8B model from zero."
```

Training is dramatically more expensive.

------------------------------------------------------------------------

# 25. Distributed Training

When the model becomes large:

``` text
GPU 0
GPU 1
GPU 2
...
GPU N
```

Use:

``` text
PyTorch Distributed
FSDP
DeepSpeed
NCCL
```

For the first small model, use a single GPU.

------------------------------------------------------------------------

# 26. Checkpointing

Every checkpoint should include:

``` text
model weights
optimizer state
scheduler state
step
tokens_seen
random states
configuration
git commit
dataset version
```

Example:

``` text
checkpoints/
  step-0001000/
  step-0002000/
  step-0003000/
```

------------------------------------------------------------------------

# 27. Evaluation During Pretraining

Track:

``` text
training loss
validation loss
perplexity
tokens processed
```

But do not stop at perplexity.

Create downstream evaluations for:

``` text
knowledge
math
reasoning
code
instruction following
multilingual
```

------------------------------------------------------------------------

# 28. Instruction Training After Base Pretraining

After the foundation model learns language:

``` text
MyAI-Base
      ↓
instruction dataset
      ↓
SFT
      ↓
MyAI-Instruct
```

The SFT data can be:

-   human-written
-   properly licensed
-   internally generated
-   synthetic data generated by a permitted process
-   carefully reviewed

If synthetic data is produced by another model, keep its provenance and
terms documented.

------------------------------------------------------------------------

# 29. Coding Specialization

Create:

``` text
MyAI-Coder
```

Training data:

``` text
licensed code
+
documentation
+
tests
+
bug/fix examples
+
code explanations
+
repository tasks
```

But training alone is insufficient.

The coding agent needs tools:

``` text
read_file()
write_file()
search_code()
run_tests()
run_command()
git_diff()
git_status()
```

------------------------------------------------------------------------

# 30. Reasoning Specialization

Create:

``` text
MyAI-Reasoner
```

Use high-quality:

``` text
math
logic
science
programming
planning
verification
```

Evaluate final correctness.

Do not equate longer generated reasoning with better reasoning.

------------------------------------------------------------------------

# 31. Preference Training

After SFT:

``` text
Prompt
  ↓
Answer A
Answer B
  ↓
Preference label
  ↓
DPO / other preference optimization
```

This improves:

-   helpfulness
-   instruction following
-   style
-   response quality

------------------------------------------------------------------------

# 32. Tool-Use Training

Teach structured calls:

``` json
{
  "tool": "search_web",
  "arguments": {
    "query": "..."
  }
}
```

Training example:

``` text
User
 ↓
Model decides tool
 ↓
Tool call
 ↓
Tool result
 ↓
Model
 ↓
Final response
```

------------------------------------------------------------------------

# 33. Agent System

The model is only one component.

``` text
MyAI Model
     ↓
Planner
     ↓
Tool selection
     ↓
Execution
     ↓
Observation
     ↓
Verification
     ↓
Correction
     ↓
Final result
```

------------------------------------------------------------------------

# 34. RAG

Use RAG for external/private knowledge:

``` text
Documents
 ↓
Chunking
 ↓
Embedding
 ↓
Vector DB
 ↓
Retriever
 ↓
Reranker
 ↓
MyAI
```

Do not retrain the foundation model every time a document changes.

------------------------------------------------------------------------

# 35. Memory

Implement:

``` text
working context
episodic memory
semantic memory
user-approved persistent memory
```

Store provenance:

``` text
memory
source
timestamp
confidence
```

------------------------------------------------------------------------

# 36. Security

A coding/computer agent must be sandboxed.

``` text
LLM
 ↓
Policy
 ↓
Permission
 ↓
Sandbox
 ↓
Tool
```

Never allow an unrestricted production agent to execute arbitrary
commands with host-level privileges.

------------------------------------------------------------------------

# 37. Model API

Expose our model through:

``` text
/v1/chat/completions
/v1/completions
/v1/embeddings
/v1/models
```

This allows applications to use MyAI without knowing its internal
implementation.

------------------------------------------------------------------------

# 38. Model Versioning

Use:

``` text
MyAI-100M-Base-v0.1
MyAI-100M-Instruct-v0.1
MyAI-300M-Base-v0.1
MyAI-1B-Base-v0.1
```

Every release records:

``` text
dataset version
data mixture
token count
architecture
hyperparameters
training compute
evaluation
git commit
```

------------------------------------------------------------------------

# 39. Reproducibility

A training run should be reproducible from:

``` text
Git commit
+
dataset manifest
+
dataset hashes
+
config
+
environment
+
random seed
+
checkpoint
```

Use:

``` text
Docker
lockfiles
Git
experiment tracking
dataset manifests
```

------------------------------------------------------------------------

# 40. Project Phases

## Phase 0 --- Research Model

Build:

``` text
MyAI-20M
```

Goal:

-   verify architecture
-   verify tokenizer
-   verify training loop
-   generate text

## Phase 1 --- Small Foundation

``` text
MyAI-100M
```

Goal:

-   real pretraining experiment
-   data pipeline
-   evaluation
-   checkpointing

## Phase 2 --- Scaling

``` text
300M
→
1B
→
3B
```

Only scale if evaluation improves predictably.

## Phase 3 --- Instruction

``` text
SFT
→
preference
→
tool use
```

## Phase 4 --- Specialists

``` text
MyAI-General
MyAI-Coder
MyAI-Reasoner
```

## Phase 5 --- Agent

``` text
Tools
Memory
RAG
Planning
Verification
Security
```

------------------------------------------------------------------------

# 41. What We Are NOT Doing

We are NOT:

``` text
Downloading Qwen weights
Downloading Llama weights
Fine-tuning Claude
Fine-tuning GPT
Calling an existing LLM "our foundation model"
```

We ARE:

``` text
Writing our own architecture
Training random initialization
Creating our own tokenizer
Building our data pipeline
Training on public/licensed data
Creating our own checkpoints
Creating our own post-training
Building our own inference/API layer
```

------------------------------------------------------------------------

# 42. Recommended Technology

## Model

``` text
Python
PyTorch
CUDA
```

## Data

``` text
Hugging Face Datasets
Datatrove
Parquet
Arrow
Polars
```

## Training

``` text
PyTorch Distributed
FSDP
DeepSpeed
NCCL
```

## Tokenizer

``` text
SentencePiece
or
Hugging Face Tokenizers
```

We can implement the tokenizer-training pipeline ourselves while using a
proven tokenizer implementation.

## Inference

For the first implementation:

``` text
our own PyTorch inference
```

Later optimize with:

``` text
CUDA kernels
FlashAttention
custom batching
KV cache
quantization
```

------------------------------------------------------------------------

# 43. Minimum First Working System

Before trying to train billions of parameters, the repository must
successfully do:

``` text
1. Download a permitted dataset sample
2. Record provenance/license
3. Clean data
4. Deduplicate data
5. Train tokenizer
6. Tokenize corpus
7. Initialize random Transformer
8. Train on GPU
9. Save checkpoint
10. Resume checkpoint
11. Evaluate validation loss
12. Generate text
13. Export model
14. Run inference
```

Only then scale.

------------------------------------------------------------------------

# 44. First Practical Dataset Experiment

Use:

``` text
FineWeb sample
```

but start extremely small.

Example:

``` text
10M tokens
```

Pipeline:

``` text
FineWeb
  ↓
stream
  ↓
filter
  ↓
deduplicate
  ↓
tokenize
  ↓
pack 2048-token sequences
  ↓
MyAI-20M
  ↓
train
  ↓
validation
  ↓
generate sample
```

This is a **pipeline test**, not an attempt to create a competitive LLM.

------------------------------------------------------------------------

# 45. Serious Scaling Experiment

Once the pipeline is proven:

``` text
100M tokens
        ↓
1B tokens
        ↓
10B tokens
        ↓
larger
```

For each step:

``` text
training loss
validation loss
tokens/sec
GPU utilization
compute cost
downstream benchmarks
```

must be recorded.

------------------------------------------------------------------------

# 46. Foundation Model vs Product

The final system has two separate products:

## MyAI Foundation

``` text
Tokenizer
+
Transformer
+
Weights
+
Inference
```

## MyAI Platform

``` text
Foundation
+
Reasoning
+
Coding
+
Agents
+
Tools
+
Memory
+
RAG
+
Security
+
UI
+
API
```

Keep these repositories/modules separate.

------------------------------------------------------------------------

# 47. Final Architecture

``` text
                    MYAI PLATFORM
                         |
                  AI ORCHESTRATOR
                         |
                    MODEL ROUTER
                         |
             +-----------+-----------+
             |                       |
       MyAI Foundation         Other Models
             |
      +------+------+
      |             |
   General        Specialists
      |          /          \
   Instruct    Coder      Reasoner
      \          |           /
       +---------+----------+
                 |
            Agent Runtime
                 |
       +---------+---------+
       |         |         |
     Tools     Memory      RAG
       |         |         |
     Web       Context   Knowledge
     Files     Episodes  Documents
     Shell
                 |
            Verification
                 |
              Security
                 |
              API / UI
```

------------------------------------------------------------------------

# 48. Golden Rule

The project should be judged by:

``` text
Data quality
+
training stability
+
scaling efficiency
+
benchmark performance
+
real task performance
+
reproducibility
```

not by:

``` text
number of files
number of parameters
number of features
```

A smaller well-trained model is more useful than a huge badly-trained
model.

------------------------------------------------------------------------

# 49. References

FineWeb: https://huggingface.co/datasets/HuggingFaceFW/fineweb

FineWeb documentation and license:
https://huggingface.co/datasets/HuggingFaceFW/fineweb/blob/main/README.md

The Stack v2: https://huggingface.co/datasets/bigcode/the-stack-v2

Datatrove: https://github.com/huggingface/datatrove

PyTorch: https://pytorch.org/

Hugging Face Datasets: https://huggingface.co/docs/datasets

Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers

DeepSpeed: https://github.com/deepspeedai/DeepSpeed

------------------------------------------------------------------------

# 50. First Build Order

The implementation order is:

``` text
STEP 1
Create MyAI repository
        ↓
STEP 2
Docker + CUDA/PyTorch environment
        ↓
STEP 3
Implement tokenizer
        ↓
STEP 4
Implement RMSNorm
        ↓
STEP 5
Implement RoPE
        ↓
STEP 6
Implement causal attention
        ↓
STEP 7
Implement SwiGLU
        ↓
STEP 8
Implement Transformer block
        ↓
STEP 9
Implement MyAI model
        ↓
STEP 10
Implement dataset pipeline
        ↓
STEP 11
Download small permitted public dataset sample
        ↓
STEP 12
Tokenize + pack
        ↓
STEP 13
Train MyAI-20M
        ↓
STEP 14
Evaluation
        ↓
STEP 15
Checkpoint/resume
        ↓
STEP 16
Scale to 100M
        ↓
STEP 17
Scale experiments
        ↓
STEP 18
Instruction tuning
        ↓
STEP 19
Reasoning/coding
        ↓
STEP 20
Agents/tools/memory/RAG
        ↓
STEP 21
Production API
```

**This is the correct interpretation of "my own LLM from coding": the
model starts from random initialization, and public datasets provide the
learning material. No pretrained LLM checkpoint is the foundation.**
