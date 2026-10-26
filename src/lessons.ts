export type Question = {
  prompt: string;
  options: { id: string; label: string }[];
  correct: string;
  reply: string;
  miss: string;
};

export type Lesson = {
  id: string;
  group: string;
  label: string;
  flow: string | null;
  /** Which stage of the L→R pipeline to highlight in the center viz. */
  pipe: string;
  title: string;
  blurb: string;
  concept: string;
  narration: string;
  question: Question;
};

export const LESSONS: Lesson[] = [
  {
    id: "problem",
    group: "Intro",
    label: "The problem",
    flow: null,
    pipe: "tokens",
    title: "Autoregressive next-token prediction",
    blurb:
      "You already know neural nets map vectors → vectors. Language models cast “finish the sentence” as: given tokens x₁…xₜ, return a distribution p(xₜ₊₁ | x₁…xₜ). Transformers are one architecture for that conditional.",
    concept: "p(next | context) via a stack of residual blocks — not a hand-written grammar.",
    narration:
      "Assume you know layers, activations, and training by gradient descent. The new question is how to encode a variable-length sequence so each position can depend on earlier ones. Autoregressive LMs answer that by predicting one next token at a time. Type a guess for “the cat sat on ____”; we will compare it to the model’s softmax at the end.",
    question: {
      prompt: "An autoregressive LM’s training target is…",
      options: [
        { id: "next", label: "p(xₜ₊₁ | x₁…xₜ)" },
        { id: "bag", label: "A bag-of-words class label" },
        { id: "parse", label: "A full parse tree" },
      ],
      correct: "next",
      reply: "Next-token conditionals. Everything else is architecture for that distribution.",
      miss: "It is a next-token conditional, not a single sentence-level class.",
    },
  },
  {
    id: "map",
    group: "Intro",
    label: "The whole model",
    flow: null,
    pipe: "attn",
    title: "Decoder block, left → right",
    blurb:
      "A GPT-style decoder is: embed → (LayerNorm → causal self-attention → residual → LayerNorm → MLP → residual)×N → unembed → softmax. This toy has N=1, d_model=4, one attention head. Same equations as a larger model.",
    concept: "Attention mixes across positions; the MLP mixes across channels at one position.",
    narration:
      "Read the center strip left to right. Tokens become vectors. LayerNorm stabilizes them. Self-attention builds Q, K, V and a causal attention matrix, then mixes Values. A residual adds the pre-attention representation back. The MLP is a position-wise feed-forward net with another residual. Finally the last position’s vector is dotted with the vocabulary and softmaxed. Click a stage on the left map to zoom that column.",
    question: {
      prompt: "In a residual decoder block, attention’s job is primarily…",
      options: [
        { id: "across", label: "Mixing information across sequence positions" },
        { id: "within", label: "Mixing channels inside one position only" },
        { id: "loss", label: "Computing the training loss" },
      ],
      correct: "across",
      reply: "Across positions. The MLP is the within-position nonlinear rewrite.",
      miss: "Attention is the cross-position mixer; the MLP is position-wise.",
    },
  },
  {
    id: "tokens",
    group: "Components",
    label: "Tokens",
    flow: "Tokens",
    pipe: "tokens",
    title: "Tokenization → discrete IDs",
    blurb:
      "Text is not the network’s input. A tokenizer maps a string to integer IDs. Subword schemes (BPE, WordPiece) split rare words so the vocabulary stays finite. Each ID indexes one row of the embedding matrix.",
    concept: "Sequence of IDs ∈ {1…|V|}^T — the only discrete step before continuous vectors.",
    narration:
      "“the cat sat on” becomes four IDs. “unbelievable” might be three subword IDs. From here on, every tensor is indexed by position in that ID sequence. No string operations remain inside the block.",
    question: {
      prompt: "Why use subword tokens instead of only whole words?",
      options: [
        { id: "finite", label: "Keep |V| finite while covering rare / new words" },
        { id: "faster", label: "They make softmax run in O(1)" },
        { id: "grammar", label: "They encode syntax trees" },
      ],
      correct: "finite",
      reply: "Finite vocab + open vocabulary via composition of pieces.",
      miss: "Subwords mainly solve open-vocabulary coverage under a fixed |V|.",
    },
  },
  {
    id: "embed",
    group: "Components",
    label: "Embeddings",
    flow: "Embed",
    pipe: "embed",
    title: "Token + positional embeddings",
    blurb:
      "Lookup E[id] ∈ ℝ^d gives a token embedding. Add a position vector P[t] (learned here; transformers also use sinusoids or RoPE). The sum is the starting residual stream x⁰_t. Attention has no recurrence — position must be injected.",
    concept: "x⁰_t = E[token_t] + P[t]   then LayerNorm before attention.",
    narration:
      "Click a word to highlight its column in E. Then we add the positional row for that index. Without P, the same word in two places would be identical vectors, and attention could not tell order. Real models use d ≈ 768–4096 and |V| ≈ 50k; we keep d=4 so you can see the arithmetic.",
    question: {
      prompt: "Why add positional encodings at all?",
      options: [
        { id: "order", label: "Self-attention is permutation-equivariant without them" },
        { id: "softmax", label: "Softmax needs them numerically" },
        { id: "gpu", label: "GPUs require them" },
      ],
      correct: "order",
      reply: "Without position, reordering tokens would not change attention’s view of the set.",
      miss: "Attention on a bag of vectors has no inherent order — P supplies it.",
    },
  },
  {
    id: "norm",
    group: "Components",
    label: "Layer norm",
    flow: "Norm",
    pipe: "norm",
    title: "LayerNorm on the residual stream",
    blurb:
      "LayerNorm(x) = γ ⊙ (x − μ) / √(σ²+ε) + β, computed over the feature axis of one token. Unlike batch norm, it does not depend on other batch elements — required for variable-length sequences. Pre-norm (LN before attention/MLP) is the GPT-2/3 style.",
    concept: "Per-token mean/variance over d_model — stabilizes the residual stream.",
    narration:
      "Follow the worked vector 2,4,4,6. Subtract mean 4, divide by √variance. γ and β are affine parameters the model can learn per channel; here one shared pair is enough to see the effect. In the full forward pass, LN is applied to E+P before QKV projections.",
    question: {
      prompt: "LayerNorm’s statistics are computed…",
      options: [
        { id: "features", label: "Across features of one token" },
        { id: "batch", label: "Across the batch only" },
        { id: "time", label: "Across the time axis only" },
      ],
      correct: "features",
      reply: "Across the d_model features of that position.",
      miss: "It is feature-wise for one position — that is why it works for any T.",
    },
  },
  {
    id: "qkv",
    group: "Components",
    label: "Self-attention",
    flow: "Attention",
    pipe: "qkv",
    title: "Q, K, V linear maps",
    blurb:
      "From the LN’d stream X ∈ ℝ^{T×d}, three learned matrices produce Q=XW_Q, K=XW_K, V=XW_V. Queries ask “what am I looking for?”, keys advertise “what do I contain?”, values carry the content that will be mixed. Multi-head attention splits d into h heads; we show h=1 with d_k=2.",
    concept: "Q,K ∈ ℝ^{T×d_k}, V ∈ ℝ^{T×d_v} — same X, different projections.",
    narration:
      "Select a token. Its Q, K, and V appear under the left-to-right strip. In Transformer Explainer you would see twelve heads; the algebra per head is identical to this one. Causal masking will zero out future keys in the next step so position t cannot attend to t+1…T.",
    question: {
      prompt: "Q, K, and V are…",
      options: [
        { id: "proj", label: "Linear projections of the same residual stream X" },
        { id: "raw", label: "Raw one-hot token IDs" },
        { id: "loss", label: "Three copies of the loss gradient" },
      ],
      correct: "proj",
      reply: "Three learned linear maps of X (after LayerNorm in pre-norm).",
      miss: "They are XW_Q, XW_K, XW_V — not the discrete IDs themselves.",
    },
  },
  {
    id: "scores",
    group: "Components",
    label: "Attention scores",
    flow: "Attention",
    pipe: "attn",
    title: "Scaled dot-product attention",
    blurb:
      "Score matrix S = QKᵀ / √d_k. Scale by √d_k so dots stay O(1) as d_k grows — otherwise softmax saturates. Causal mask sets S_{ij}=−∞ for j>i. Softmax over j yields A; output is A V. That is the entire attention op.",
    concept: "Attention(Q,K,V) = softmax(QKᵀ/√d_k + mask) V",
    narration:
      "For the selected query row, each visible key contributes a scaled dot product. Cat and sat score high for “on”, so their value vectors dominate the mix. The heatmap in the center is the causal A matrix — darker means more weight. Residual connection: we will add this mix back to X next.",
    question: {
      prompt: "We divide QKᵀ by √d_k mainly to…",
      options: [
        { id: "scale", label: "Keep logits from growing with d_k before softmax" },
        { id: "causal", label: "Implement the causal mask" },
        { id: "norm", label: "Replace LayerNorm" },
      ],
      correct: "scale",
      reply: "Variance of dots grows with d_k; scaling keeps softmax in a useful range.",
      miss: "It is a variance / saturation fix, not the mask itself.",
    },
  },
  {
    id: "softmax",
    group: "Components",
    label: "Softmax",
    flow: "Attention",
    pipe: "attn",
    title: "Softmax → attention weights",
    blurb:
      "Row-wise softmax turns scores into a distribution over keys. Every visible position keeps mass ≥ 0 summing to 1 — a soft lookup, not an argmax. Temperature in generation is a later softmax over vocabulary logits; here temperature is fixed at 1.",
    concept: "A_{i:} = softmax(S_{i:}) · then attn_out_i = Σⱼ A_{ij} Vⱼ",
    narration:
      "Watch the last row of A. Softmax concentrates mass on cat and sat. The mixed value is the weighted sum of V rows. Bump the cat score with the slider to steal mass — same operator as the final vocab softmax, different axis.",
    question: {
      prompt: "Attention weights A_{i:} always…",
      options: [
        { id: "dist", label: "Form a probability distribution over keys j ≤ i" },
        { id: "neg", label: "Can be negative if the score is" },
        { id: "one", label: "Pick exactly one key (hard attention)" },
      ],
      correct: "dist",
      reply: "Nonnegative and sum to 1 over allowed keys — soft attention.",
      miss: "Softmax yields a distribution; hard argmax is a different (rarer) design.",
    },
  },
  {
    id: "mlp",
    group: "Components",
    label: "The MLP",
    flow: "MLP",
    pipe: "mlp",
    title: "Position-wise feed-forward + residual",
    blurb:
      "After attention: x ← x + Attn(…). Then LN, then MLP(x) = W₂ σ(W₁x + b₁) + b₂ (GELU/ReLU). Applied independently at each t — no mixing across time. Second residual: x ← x + MLP(x). Attention = communication; MLP = computation.",
    concept: "x + MLP(LN(x)) — expand d → d_ff → d, then residual add.",
    narration:
      "The strip shows residual₁ = X + AttnOut, then a ReLU expansion, then residual₂. In GPT-2, d_ff is usually 4×d_model. Here the MLP is a tiny affine-ReLU so you can see channels light up without a wall of weights. No other token’s vector enters this stage.",
    question: {
      prompt: "The MLP in a transformer block…",
      options: [
        { id: "pos", label: "Is shared across positions but does not mix them" },
        { id: "mix", label: "Attends over the full sequence again" },
        { id: "tok", label: "Looks up the next token ID directly" },
      ],
      correct: "pos",
      reply: "Same W₁,W₂ at every t; each t is transformed alone.",
      miss: "It is position-wise. Cross-position mixing already happened in attention.",
    },
  },
  {
    id: "output",
    group: "Output",
    label: "The next word",
    flow: "Output",
    pipe: "out",
    title: "Unembed → logits → softmax",
    blurb:
      "Take the residual stream at the last position. Multiply by the unembedding U ∈ ℝ^{|V|×d} (often tied with Eᵀ). Softmax gives p(·|context). Sampling, top-k, and temperature act on these logits — we show the raw distribution.",
    concept: "logits = U h_T ;   p = softmax(logits)",
    narration:
      "The final vector for “on” is scored against every vocab row. Mat aligns with the direction the block carved out, so it wins. Compare your opening guess. Real models stack dozens of identical blocks before this unembed — algebra per block matches what you just walked.",
    question: {
      prompt: "Vocabulary probabilities come from…",
      options: [
        { id: "dot", label: "Softmax of U h_T (unembedding · last hidden)" },
        { id: "attn", label: "The attention matrix A alone" },
        { id: "pos", label: "The positional encoding row" },
      ],
      correct: "dot",
      reply: "Linear unembed of the last residual state, then softmax.",
      miss: "A mixes context earlier; the distribution is always over U h_T.",
    },
  },
];

export const FLOW = [
  { key: "Tokens", lesson: "tokens" },
  { key: "Embed", lesson: "embed" },
  { key: "Norm", lesson: "norm" },
  { key: "Attention", lesson: "qkv" },
  { key: "MLP", lesson: "mlp" },
  { key: "Output", lesson: "output" },
];
