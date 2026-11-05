import { useMemo, type ReactElement, type ReactNode } from "react";
import {
  CONTEXT,
  UNEMBED,
  VOCAB,
  D_K,
  SCALE,
  forward,
  fmt,
  fmtSmart,
} from "./model";
import type { Guide } from "./lessons";
import { NormSteps } from "./NormSteps";
import { TokenPlayground } from "./TokenPlayground";

export type LessonProps = {
  token: number;
  setToken: (index: number) => void;
  guess: string;
  setGuess: (value: string) => void;
  go: (id: string) => void;
  pipe?: string;
  guideStep: number;
  setGuide: (guide: Guide | null, keepStep?: boolean) => void;
  setGuideStep: (step: number) => void;
};

const PEN = ["#e23b3b", "#2f6bff", "#1aa36a", "#7a3ff0"];

export const STAGES: Record<string, (props: LessonProps) => ReactElement> = {
  problem: Problem,
  map: Map,
  tokens: TokenPlayground,
  norm: NormSteps,
  qkv: Qkv,
  scores: Scores,
  softmax: SoftmaxStage,
  mlp: Mlp,
  output: Output,
};

function Vec({ values, tone }: { values: number[]; tone?: string }) {
  return (
    <span className={tone ? `vec ${tone}` : "vec"}>
      {values.map((value, index) => (
        <span key={index} className={value < -0.001 ? "neg" : value > 0.001 ? "pos" : ""}>
          {fmtSmart(value)}
        </span>
      ))}
    </span>
  );
}

function WordPicker({
  token,
  setToken,
}: {
  token: number;
  setToken: (index: number) => void;
}) {
  return (
    <div className="picker" role="group" aria-label="Focus token">
      {CONTEXT.map((word, index) => (
        <button
          key={word}
          type="button"
          className="chip button"
          aria-pressed={token === index}
          onClick={() => setToken(index)}
        >
          {word}
        </button>
      ))}
    </div>
  );
}

/** Shared left→right decoder strip; `active` lights the stage under discussion. */
function Pipeline({
  active,
  token = CONTEXT.length - 1,
  children,
}: {
  active: string;
  token?: number;
  children?: ReactNode;
}) {
  const f = useMemo(() => forward(token), [token]);
  const stages = [
    { id: "tokens", label: "Tokens" },
    { id: "embed", label: "Embed" },
    { id: "norm", label: "LN" },
    { id: "qkv", label: "QKV" },
    { id: "attn", label: "Attn" },
    { id: "mlp", label: "MLP" },
    { id: "out", label: "Softmax" },
  ];

  return (
    <div className="ltr">
      <div className="ltr-rail" aria-label="Decoder path left to right">
        {stages.map((stage, index) => (
          <span key={stage.id} className="ltr-step">
            {index > 0 && <span className="ltr-arrow" aria-hidden>
              →
            </span>}
            <span className={stage.id === active ? "ltr-pill on" : "ltr-pill"}>{stage.label}</span>
          </span>
        ))}
      </div>
      <div className="ltr-flow">
        <div className={active === "tokens" || active === "embed" ? "col on" : "col"}>
          {CONTEXT.map((word, i) => (
            <span key={word} className="tok" style={{ borderColor: PEN[i] }}>
              {word}
            </span>
          ))}
        </div>
        <svg className="ribbons" viewBox="0 0 120 160" preserveAspectRatio="none" aria-hidden>
          {CONTEXT.map((_, i) => {
            const y = 20 + i * 40;
            const w = active === "attn" ? 1.5 + (f.weights[token][i] ?? 0) * 6 : 2.2;
            return (
              <path
                key={i}
                d={`M 0 ${y} C 40 ${y}, 80 80, 120 80`}
                fill="none"
                stroke={PEN[i]}
                strokeWidth={w}
                opacity={i > token && active === "attn" ? 0.2 : 0.9}
              />
            );
          })}
        </svg>
        <div className={active === "qkv" || active === "attn" ? "col mid on" : "col mid"}>
          <span className="tag q">Q</span>
          <span className="tag k">K</span>
          <span className="tag v">V</span>
          <div className="heat" title="Causal attention matrix A">
            {CONTEXT.map((_, i) =>
              CONTEXT.map((_, j) => {
                const w = f.weights[i][j];
                const blocked = w === null;
                const heat = blocked ? 0 : w;
                return (
                  <span
                    key={`${i}-${j}`}
                    className={blocked ? "cell blocked" : "cell"}
                    style={{
                      background: blocked
                        ? "#ddd"
                        : `rgba(122, 63, 240, ${0.15 + heat * 0.85})`,
                    }}
                  />
                );
              }),
            )}
          </div>
        </div>
        <svg className="ribbons short" viewBox="0 0 60 160" preserveAspectRatio="none" aria-hidden>
          <path d="M 0 80 C 20 80, 40 80, 60 80" stroke="#7a3ff0" strokeWidth="5" fill="none" />
          <path
            d="M 5 40 C 25 40, 25 80, 55 80"
            stroke="#999"
            strokeWidth="1.5"
            fill="none"
            strokeDasharray="3 2"
          />
        </svg>
        <div className={active === "mlp" ? "col on" : "col"}>
          <span className="block-label">MLP</span>
          <span className="residual">+ residual</span>
        </div>
        <svg className="ribbons short" viewBox="0 0 40 160" preserveAspectRatio="none" aria-hidden>
          <path d="M 0 80 C 15 80, 25 80, 40 80" stroke="#7a3ff0" strokeWidth="4" fill="none" />
        </svg>
        <div className={active === "out" || active === "mlp" ? "col on out" : "col out"}>
          <span className="pred">{f.best}</span>
          <span className="pct">{(f.probs[VOCAB.indexOf(f.best)] * 100).toFixed(0)}%</span>
        </div>
      </div>
      {children}
    </div>
  );
}

function Problem({ guess, setGuess, pipe = "tokens" }: LessonProps) {
  return (
    <Pipeline active={pipe}>
      <div className="sentence">
        {CONTEXT.map((word) => (
          <span key={word} className="chip">
            {word}
          </span>
        ))}
        <input
          className="guess"
          value={guess}
          placeholder="?"
          aria-label="Your guess for the next word"
          autoComplete="off"
          onChange={(event) => setGuess(event.target.value)}
        />
      </div>
      <p className="eq">maximize log p(xₜ₊₁ | x₁…xₜ) over a neural parameterization</p>
    </Pipeline>
  );
}

function Map({ pipe = "attn" }: LessonProps) {
  return (
    <Pipeline active={pipe}>
      <p className="eq">
        block: x ← x + Attn(LN(x)); &nbsp; x ← x + MLP(LN(x)); &nbsp; then Ux → softmax
      </p>
    </Pipeline>
  );
}

function Qkv({ token, setToken, pipe = "qkv" }: LessonProps) {
  const f = useMemo(() => forward(token), [token]);
  return (
    <Pipeline active={pipe} token={token}>
      <WordPicker token={token} setToken={setToken} />
      <p className="eq">
        Q = XW_Q ∈ ℝ<sup>T×{D_K}</sup>, K = XW_K, V = XW_V (h=1 head)
      </p>
      <div className="qkv-row">
        <span>
          Q<sub>{CONTEXT[token]}</sub> <Vec values={f.Q[token]} tone="query" />
        </span>
        <span>
          K <Vec values={f.K[token]} tone="key" />
        </span>
        <span>
          V <Vec values={f.V[token]} tone="value" />
        </span>
      </div>
    </Pipeline>
  );
}

function Scores({ token, setToken, pipe = "attn" }: LessonProps) {
  const f = useMemo(() => forward(token), [token]);
  const query = f.Q[token];
  return (
    <Pipeline active={pipe} token={token}>
      <WordPicker token={token} setToken={setToken} />
      <p className="eq">
        S = QKᵀ / √d_k &nbsp;·&nbsp; √d_k = √{D_K} ≈ {fmt(SCALE)} &nbsp;·&nbsp; mask j&gt;i
      </p>
      <p className="readout">
        Query <Vec values={query} tone="query" />
      </p>
      <ul className="score-list">
        {CONTEXT.map((word, index) => {
          const hidden = index > token;
          return (
            <li key={word} className={hidden ? "dim" : ""}>
              <b>{word}</b>
              {hidden ? (
                <span>masked (j &gt; i)</span>
              ) : (
                <span className="arith">
                  ⟨Q,K⟩={fmtSmart(f.dots[token][index] ?? 0)}
                  <span> / √d_k </span>
                  <strong>{fmt(f.scores[token][index] ?? 0)}</strong>
                  <span className="wtag"> A={fmt(f.weights[token][index] ?? 0, 3)}</span>
                </span>
              )}
            </li>
          );
        })}
      </ul>
    </Pipeline>
  );
}

function SoftmaxStage({ token, setToken, pipe = "attn" }: LessonProps) {
  const f = useMemo(() => forward(token), [token]);
  return (
    <Pipeline active={pipe} token={token}>
      <WordPicker token={token} setToken={setToken} />
      <p className="eq">A = softmax(S) over keys &nbsp;→&nbsp; AttnOut = A V</p>
      <table className="matrix">
        <thead>
          <tr>
            <th>key</th>
            <th>S</th>
            <th>A</th>
          </tr>
        </thead>
        <tbody>
          {CONTEXT.map((word, index) => {
            const hidden = index > token;
            return (
              <tr key={word} className={hidden ? "dim" : ""}>
                <th>{word}</th>
                <td>{hidden ? "−∞" : fmt(f.scores[token][index] ?? 0)}</td>
                <td>{hidden ? "—" : fmt(f.weights[token][index] ?? 0, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="readout">
        AttnOut[{CONTEXT[token]}] <Vec values={f.attnOut[token]} />
        {" · "}
        residual X+Attn <Vec values={f.residual1[token]} />
      </p>
    </Pipeline>
  );
}

function Mlp({ token, setToken, pipe = "mlp" }: LessonProps) {
  const f = useMemo(() => forward(token), [token]);
  return (
    <Pipeline active={pipe} token={token}>
      <WordPicker token={token} setToken={setToken} />
      <p className="eq">h = ReLU(x + b); &nbsp; y = x + 0.35 h &nbsp;(toy stand-in for W₂σ(W₁x))</p>
      <p className="readout">
        residual₁ <Vec values={f.residual1[token]} />
      </p>
      <ul className="neurons">
        {f.mlpHidden[token].map((h, i) => (
          <li key={i} className={h > 0 ? "lit" : "blocked"}>
            <span>
              channel {i + 1}
              <em>ReLU(x+0.2)</em>
            </span>
            <b>
              {fmt(f.residual1[token][i] + 0.2)} → {fmt(h)}
            </b>
          </li>
        ))}
      </ul>
      <p className="readout">
        residual₂ <Vec values={f.residual2[token]} />
      </p>
    </Pipeline>
  );
}

function Output({ pipe = "out" }: LessonProps) {
  const f = useMemo(() => forward(), []);
  return (
    <Pipeline active={pipe}>
      <p className="eq">logits = U h_T &nbsp;·&nbsp; p = softmax(logits)</p>
      <p className="readout">
        h_T <Vec values={f.residual2[3]} /> · U[mat] <Vec values={UNEMBED.mat} />
      </p>
      <ul className="probs">
        {VOCAB.map((word, index) => (
          <li key={word} className={word === f.best ? "winner" : ""}>
            <span>{word}</span>
            <span className="bar">
              <span style={{ width: `${f.probs[index] * 100}%` }} />
            </span>
            <span>{(f.probs[index] * 100).toFixed(1)}%</span>
          </li>
        ))}
      </ul>
    </Pipeline>
  );
}
