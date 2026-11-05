/**
 * GPT-2's real token embedding table (wte), prepared offline into public/embeddings/:
 * - first4: the first 4 of each token's 768 numbers, exact to 5 decimals, for display.
 * - vectors: the whole table mean-centred and compressed 768 → 128 with PCA, stored as int8.
 *   Closeness and vector maths use these, because the full table is about 150 MB.
 */
export type Embeddings = {
  first4: (id: number) => number[];
  vector: (id: number) => Float32Array;
  /** Token IDs whose vectors point most nearly the same way as `target` (cosine similarity). */
  nearest: (target: Float32Array, exclude: Set<number>, count: number) => number[];
};

type Meta = { tokens: number; dims: number; scales: number[]; first4Scale: number };

let loading: Promise<Embeddings> | null = null;

async function fetchBytes(path: string) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.arrayBuffer();
}

async function load(): Promise<Embeddings> {
  const base = `${import.meta.env.BASE_URL}embeddings/`;
  const [meta, packed, first] = await Promise.all([
    fetch(`${base}gpt2-wte.json`).then((response) => {
      if (!response.ok) throw new Error(`gpt2-wte.json: ${response.status}`);
      return response.json() as Promise<Meta>;
    }),
    fetchBytes(`${base}gpt2-wte-128.i8`),
    fetchBytes(`${base}gpt2-wte-first4.i16`),
  ]);
  const { tokens, dims, scales, first4Scale } = meta;
  const table = new Int8Array(packed);
  const firstFour = new Int16Array(first);

  // Fold the per-dimension scales in once, so every lookup is a plain slice.
  const values = new Float32Array(tokens * dims);
  const norms = new Float32Array(tokens);
  for (let token = 0; token < tokens; token += 1) {
    let sum = 0;
    for (let d = 0; d < dims; d += 1) {
      const value = table[token * dims + d] * scales[d];
      values[token * dims + d] = value;
      sum += value * value;
    }
    norms[token] = Math.sqrt(sum) || 1;
  }

  return {
    first4: (id) => Array.from(firstFour.subarray(id * 4, id * 4 + 4), (value) => value * first4Scale),
    vector: (id) => values.slice(id * dims, (id + 1) * dims),
    nearest(target, exclude, count) {
      let length = 0;
      for (const value of target) length += value * value;
      length = Math.sqrt(length) || 1;
      const best: { id: number; score: number }[] = [];
      for (let token = 0; token < tokens; token += 1) {
        if (exclude.has(token)) continue;
        let dot = 0;
        const offset = token * dims;
        for (let d = 0; d < dims; d += 1) dot += values[offset + d] * target[d];
        const score = dot / (norms[token] * length);
        if (best.length < count || score > best[best.length - 1].score) {
          best.push({ id: token, score });
          best.sort((a, b) => b.score - a.score);
          if (best.length > count) best.pop();
        }
      }
      return best.map((item) => item.id);
    },
  };
}

export function loadEmbeddings() {
  loading ??= load().catch((error) => {
    loading = null;
    throw error;
  });
  return loading;
}
