export type Question = {
  prompt: string;
  options: { id: string; label: string }[];
  correct: string;
  reply: string;
  miss: string;
};

/** What the explainer box shows; `focus` and `pick` tie a step to the visual. */
export type GuideStep = { text: string; focus?: string; pick?: number };
export type Guide = { prompt?: string; steps: GuideStep[] };

export type Lesson = {
  id: string;
  group: string;
  label: string;
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
    label: "Tokens & embeddings",
    pipe: "tokens",
    title: "Tokens → IDs → embeddings",
    blurb:
      "Text is not the network’s input. A tokenizer maps a string to integer IDs, and subword schemes (BPE, WordPiece) split rare words so the vocabulary stays finite. Each ID then picks one row of the embedding table. That row of numbers is the token’s vector, and it is what the network actually receives.",
    concept: "text → token IDs → rows of the embedding table (one vector per token)",
    narration:
      "“the cat sat on” becomes four IDs. “unbelievable” becomes four subword IDs: un, bel, iev, able. Type your own sentence to see how GPT-2 splits it. Each ID then picks one row of the embedding table, and that row of numbers is what the network receives. From here on, every tensor is indexed by position in that ID sequence. No string operations remain inside the block.",
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
    id: "norm",
    group: "Components",
    label: "Layer norm",
    pipe: "norm",
    title: "LayerNorm on the residual stream",
    blurb:
      "Layer norm tidies up each token’s numbers before attention: subtract their average, divide by their spread, then let two learned dials (γ and β) stretch and shift the result. It works on one token at a time, so it handles any sentence length. GPT-2 applies it before attention and before the MLP.",
    concept: "Per-token mean/variance over d_model — stabilizes the residual stream.",
    narration:
      "Layer norm tidies one token’s numbers in a few small steps. Find the average, subtract it, measure the spread, divide by it, then stretch and shift with γ and β.",
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
