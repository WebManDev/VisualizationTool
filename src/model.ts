/** Tiny decoder block: one head, d_model=4, d_k=2. Numbers are small enough to read. */

export const CONTEXT = ["the", "cat", "sat", "on"] as const;
export const VOCAB = ["the", "cat", "sat", "on", "mat", "floor", "hat", "run"] as const;
export type Word = (typeof VOCAB)[number];

export const D_MODEL = 4;
export const D_K = 2;
export const SCALE = Math.sqrt(D_K);

/** Token embeddings E ∈ ℝ^{|V|×d}. Context rows are the first four. */
export const EMBEDDINGS: Record<Word, number[]> = {
  the: [0.9, -0.4, 0.3, 0.1],
  cat: [0.2, 1.0, -0.2, 0.5],
  sat: [-0.4, 0.6, 0.9, -0.1],
  on: [0.3, -0.2, 0.4, 0.8],
  mat: [-0.8, -1.0, -0.55, 0.3],
  floor: [0.3, 0.2, 0.2, 0.9],
  hat: [0.9, 0.3, 0.1, 0.1],
  run: [0.1, 0.4, 0.8, -0.1],
};

/** Learned positional encodings P ∈ ℝ^{T×d} for the four context positions. */
export const POSITIONAL: number[][] = [
  [0.5, 0.0, -0.2, 0.1],
  [0.1, 0.4, 0.0, -0.2],
  [-0.2, 0.1, 0.4, 0.0],
  [0.0, -0.3, 0.1, 0.5],
];

/** Projection matrices: Q = X W_Q, K = X W_K, V = X W_V. */
export const WQ: number[][] = [
  [1.09, -0.69],
  [0.05, -0.71],
  [0.95, -0.24],
  [-0.86, 1.08],
];
export const WK: number[][] = [
  [-0.62, -1.03],
  [-0.03, 1.08],
  [0.21, 0.38],
  [0.66, 0.94],
];
export const WV: number[][] = [
  [0.36, 0.38, -0.53, 0.78],
  [-0.61, 0.58, -0.79, -0.74],
  [0.16, -0.65, -0.09, -0.13],
  [-0.64, -0.33, 0.71, -0.33],
];

/** Unembedding rows ≈ embedding columns for this toy vocab. */
export const UNEMBED: Record<Word, number[]> = {
  the: [1, 0, 0, 0],
  cat: [0, 1, 0, 0],
  sat: [0, 0, 1, 0],
  on: [0, 0, 0, 1],
  mat: [-0.91, -1.17, -0.64, 0.34],
  floor: [0.3, 0.2, 0.2, 0.9],
  hat: [0.9, 0.3, 0.1, 0.1],
  run: [0.1, 0.4, 0.8, -0.1],
};

export function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

export function matMul(A: number[][], B: number[][]) {
  const out = Array.from({ length: A.length }, () => Array(B[0].length).fill(0));
  for (let i = 0; i < A.length; i++) {
    for (let k = 0; k < B.length; k++) {
      for (let j = 0; j < B[0].length; j++) out[i][j] += A[i][k] * B[k][j];
    }
  }
  return out;
}

export function softmax(xs: number[]) {
  const finite = xs.filter((x) => x > -1e8);
  const m = finite.length ? Math.max(...finite) : 0;
  const exps = xs.map((x) => (x < -1e8 ? 0 : Math.exp(x - m)));
  const sum = exps.reduce((a, b) => a + b, 0) || 1;
  return exps.map((e) => e / sum);
}

export function layerNorm(x: number[], gamma = 1, beta = 0, eps = 1e-5) {
  const n = x.length || 1;
  const mean = x.reduce((a, b) => a + b, 0) / n;
  const variance = x.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance + eps);
  const centered = x.map((v) => v - mean);
  const normalized = centered.map((v) => v / std);
  const scaled = normalized.map((v) => v * gamma + beta);
  return { mean, variance, std, centered, normalized, scaled };
}

function roundVec(v: number[], digits = 2) {
  const f = 10 ** digits;
  return v.map((x) => Math.round(x * f) / f);
}

export type Forward = {
  tokens: readonly string[];
  embeddings: number[][];
  positional: number[][];
  summed: number[][];
  x: number[][];
  Q: number[][];
  K: number[][];
  V: number[][];
  /** Causal attention weights A[i][j], null if j > i. */
  weights: (number | null)[][];
  scores: (number | null)[][];
  dots: (number | null)[][];
  attnOut: number[][];
  residual1: number[][];
  mlpHidden: number[][];
  mlpOut: number[][];
  residual2: number[][];
  logits: number[];
  probs: number[];
  best: Word;
};

/** One causal decoder block → next-token distribution over VOCAB. */
export function forward(focus = CONTEXT.length - 1): Forward {
  const tokens = CONTEXT;
  const embeddings = tokens.map((t) => EMBEDDINGS[t]);
  const positional = POSITIONAL.map((r) => [...r]);
  const summed = embeddings.map((e, i) => e.map((v, d) => v + positional[i][d]));
  const x = summed.map((row) => layerNorm(row).scaled);

  const Q = matMul(x, WQ);
  const K = matMul(x, WK);
  const V = matMul(x, WV);

  const dots: (number | null)[][] = [];
  const scores: (number | null)[][] = [];
  const weights: (number | null)[][] = [];
  const attnOut: number[][] = [];

  for (let i = 0; i < tokens.length; i++) {
    const rowDots = K.map((k, j) => (j > i ? null : dot(Q[i], k)));
    const rowScores = rowDots.map((d) => (d === null ? null : Math.round((d / SCALE) * 100) / 100));
    const soft = softmax(rowScores.map((s) => (s === null ? -1e9 : s)));
    dots.push(rowDots);
    scores.push(rowScores);
    weights.push(soft.map((w, j) => (j > i ? null : Math.round(w * 1000) / 1000)));
    const mixed = Array(D_MODEL).fill(0);
    soft.forEach((w, j) => {
      if (j > i) return;
      for (let d = 0; d < D_MODEL; d++) mixed[d] += w * V[j][d];
    });
    attnOut.push(roundVec(mixed));
  }

  const residual1 = x.map((row, i) => roundVec(row.map((v, d) => v + attnOut[i][d])));
  // Tiny MLP: h = ReLU(x + 0.2), out = 0.35 h  (same idea as W₂ ReLU(W₁x+b))
  const mlpHidden = residual1.map((row) => roundVec(row.map((v) => Math.max(0, v + 0.2))));
  const mlpOut = mlpHidden.map((h) => roundVec(h.map((v) => 0.35 * v)));
  const residual2 = residual1.map((row, i) => roundVec(row.map((v, d) => v + mlpOut[i][d])));

  const final = residual2[focus];
  const logits = VOCAB.map((w) => Math.round(dot(final, UNEMBED[w]) * 100) / 100);
  const probs = softmax(logits);
  let best = 0;
  probs.forEach((p, i) => {
    if (p > probs[best]) best = i;
  });

  return {
    tokens,
    embeddings,
    positional,
    summed,
    x: x.map((r) => roundVec(r)),
    Q: Q.map((r) => roundVec(r)),
    K: K.map((r) => roundVec(r)),
    V: V.map((r) => roundVec(r)),
    weights,
    scores,
    dots,
    attnOut,
    residual1,
    mlpHidden,
    mlpOut,
    residual2,
    logits,
    probs,
    best: VOCAB[best],
  };
}

/** @deprecated Prefer forward(). Kept for call sites that only need last-token attention. */
export function attend(query: number) {
  const f = forward(query);
  return {
    dots: f.dots[query],
    scores: f.scores[query],
    exps: f.scores[query].map((s) => (s === null ? 0 : Math.exp(s))),
    weights: f.weights[query].map((w) => w ?? 0),
    mixed: f.attnOut[query],
  };
}

export function applyMlp(x: number[]) {
  const hidden = x.map((v, i) => ({
    label: `dim ${i + 1}`,
    formula: `ReLU(x_${i + 1} + 0.2)`,
    raw: v + 0.2,
    out: Math.max(0, v + 0.2),
  }));
  const y = hidden.map((h) => Math.round(0.35 * h.out * 100) / 100);
  return { hidden, y };
}

export function predict() {
  const f = forward();
  return {
    attn: attend(CONTEXT.length - 1),
    mlp: applyMlp(f.residual1[CONTEXT.length - 1]),
    logits: f.logits,
    probs: f.probs,
    best: f.best,
    forward: f,
  };
}

export function fmt(n: number, digits = 2) {
  if (!Number.isFinite(n)) return "—";
  const v = Math.abs(n) < 5 * 10 ** -(digits + 1) ? 0 : n;
  return v.toFixed(digits);
}

export function fmtSmart(n: number) {
  if (!Number.isFinite(n)) return "—";
  if (Math.abs(n - Math.round(n)) < 1e-6) return String(Math.round(n));
  return fmt(n, 2);
}

export const PIPE_STAGES = [
  "tokens",
  "embed",
  "norm",
  "qkv",
  "attn",
  "mlp",
  "out",
] as const;

export type PipeStage = (typeof PIPE_STAGES)[number];
