import { useEffect, useMemo, useState, type ReactElement } from "react";
import { Calculator } from "./Calculator";
import type { GuideStep } from "./lessons";
import type { LessonProps } from "./Stages";

const FALLBACK = ["the", "cat", "sat", "on"];
const COUNT = 4;
const ROOT = 1.41; // √2, rounded the way students will type it
const CHUNKS = ["Setup", "1. Dot product", "2. Scale", "3. eˣ", "4. Normalise", "Result"];

const r2 = (x: number) => Math.round(x * 100) / 100;

function fmt(x: number, digits = 2) {
  const v = Math.abs(x) < 0.5 * 10 ** -digits ? 0 : x;
  return (v < 0 ? "−" : "") + Math.abs(v).toFixed(digits);
}

/** Practice numbers: a word always gets the same Q and K, one decimal each so the maths works by hand. */
function practice(word: string, salt: number) {
  let h = 2166136261 ^ Math.imul(salt, 0x9e3779b1);
  for (const c of word.toLowerCase()) h = Math.imul(h ^ c.charCodeAt(0), 16777619);
  // Finalise so short words that differ by one letter still get unrelated numbers.
  const mix = (x: number) => {
    x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
    x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
    return (x ^ (x >>> 16)) >>> 0;
  };
  const a = mix(h);
  const b = mix(h ^ 0x5bd1e995);
  // Each number is ±0.5 to ±2.0, so dot products differ enough for the weights to be interesting.
  const pick = (x: number) => ((x & 1 ? -1 : 1) * (5 + ((x >>> 1) % 16))) / 10;
  return [pick(a), pick(b)];
}

/** Negative numbers in brackets, so "0.8 × (−0.9)" can't be misread. */
function term(x: number, digits = 1) {
  return x < 0 ? `(${fmt(x, digits)})` : fmt(x, digits);
}

function close(answer: string, truth: number) {
  const value = Number(answer.replace("−", "-"));
  if (answer.trim() === "" || !Number.isFinite(value)) return null;
  return Math.abs(value - truth) <= Math.max(0.02, Math.abs(truth) * 0.015);
}

function texts(words: string[], query: number, weights: number[], visible: number): GuideStep[] {
  const top = weights.slice(0, visible).reduce((best, w, i) => (w > weights[best] ? i : best), 0);
  return [
    {
      text: `Attention lets one word look back at earlier words and decide which matter. The query here is “${words[query]}”. Click a different word to make it the query. Each word has a Query (Q) and a Key (K).`,
    },
    {
      text: `Step 1: multiply the query’s Q by each word’s K, number by number, and add. A big result means a good match. Words after the query are masked: a word can’t look ahead.`,
    },
    { text: "Step 2: divide each score by √2 ≈ 1.41, because each vector has 2 numbers. Real models divide by √64 = 8 to keep scores from getting too big." },
    { text: "Step 3: raise e to each scaled score (the eˣ key). This turns every score positive and makes big scores stand out even more." },
    { text: "Step 4: add up the eˣ values, then divide each one by that total. The results are the attention weights, and they always add up to 1." },
    {
      text: `“${words[query]}” pays the most attention to “${words[top]}” (${Math.round(weights[top] * 100)}%). These are practice numbers: GPT-2 does the same maths with 64 numbers per word in each of its 12 attention heads.`,
    },
  ];
}

export function AttentionCalc({ guideStep, setGuide, setGuideStep, sentenceTokens }: LessonProps) {
  const words = useMemo(() => {
    const picked = (sentenceTokens ?? []).map((t) => t.trim()).filter(Boolean).slice(0, COUNT);
    return picked.length >= 2 ? picked : FALLBACK;
  }, [sentenceTokens]);
  const [query, setQuery] = useState(words.length - 1);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [focused, setFocused] = useState<string | null>(null);
  const step = Math.min(guideStep, CHUNKS.length - 1);

  useEffect(() => {
    setQuery(words.length - 1);
    setAnswers({});
  }, [words]);

  const maths = useMemo(() => {
    const q = practice(words[query], 1);
    // Repeated words share a Key (same token, same numbers); different words never do.
    const keys: number[][] = [];
    words.forEach((word, i) => {
      const same = words.findIndex((other) => other.toLowerCase() === word.toLowerCase());
      if (same < i) return keys.push(keys[same]);
      let salt = 2;
      let key = practice(word, salt);
      while (keys.some((k) => k[0] === key[0] && k[1] === key[1])) key = practice(word, ++salt);
      keys.push(key);
    });
    const visible = query + 1;
    const dots = keys.map((k) => r2(q[0] * k[0] + q[1] * k[1]));
    const scaled = dots.map((d) => r2(d / ROOT));
    const exps = scaled.map((s) => r2(Math.exp(s)));
    const sum = r2(exps.slice(0, visible).reduce((a, b) => a + b, 0));
    const weights = exps.map((e, i) => (i < visible ? r2(e / sum) : 0));
    return { q, keys, visible, dots, scaled, exps, sum, weights };
  }, [words, query]);

  useEffect(() => {
    setGuide({ steps: texts(words, query, maths.weights, maths.visible) }, true);
  }, [words, query, maths, setGuide]);
  useEffect(() => () => setGuide(null), [setGuide]);

  function chooseQuery(i: number) {
    setQuery(i);
    setAnswers({});
  }

  function answerBox(id: string, truth: number, label: string) {
    const value = answers[id] ?? "";
    const ok = close(value, truth);
    return (
      <span className="ac-answer">
        <input
          inputMode="decimal"
          value={value}
          aria-label={label}
          className={ok === null ? "" : ok ? "right" : "wrong"}
          onFocus={() => setFocused(id)}
          onChange={(event) => setAnswers({ ...answers, [id]: event.target.value })}
        />
        <b aria-hidden>{ok === null ? "" : ok ? "✓" : "✗"}</b>
      </span>
    );
  }

  function reveal(entries: [string, number][]) {
    const next = { ...answers };
    for (const [id, value] of entries) next[id] = fmt(value).replace("−", "-");
    setAnswers(next);
  }

  const vec = (v: number[]) => `[${v.map((x) => fmt(x, 1)).join(", ")}]`;
  const rows = words.map((word, i) => ({ word, i, masked: i >= maths.visible }));
  const live = rows.filter((row) => !row.masked);

  let body: ReactElement;
  if (step === 0) {
    body = (
      <>
        <div className="ac-query" role="group" aria-label="Choose the query word">
          <span>Query:</span>
          {words.map((word, i) => (
            <button key={i} type="button" aria-pressed={i === query} onClick={() => chooseQuery(i)}>
              {word}
            </button>
          ))}
        </div>
        <table className="ac-table">
          <thead>
            <tr>
              <th>word</th>
              <th>Key K</th>
              <th>Query Q</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.i} className={row.i === query ? "is-query" : ""}>
                <td>{row.word}</td>
                <td>{vec(maths.keys[row.i])}</td>
                <td>{row.i === query ? vec(maths.q) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </>
    );
  } else if (step === 5) {
    body = (
      <div className="ac-result">
        {rows.map((row) => (
          <div key={row.i} className={row.masked ? "ac-bar masked" : "ac-bar"}>
            <span>{row.word}</span>
            <span className="ac-track">
              <span style={{ width: `${maths.weights[row.i] * 100}%` }} />
            </span>
            <b>{row.masked ? "masked" : `${Math.round(maths.weights[row.i] * 100)}%`}</b>
          </div>
        ))}
      </div>
    );
  } else {
    const q = maths.q;
    const lines = rows.map((row) => {
      const k = maths.keys[row.i];
      if (row.masked) return { row, work: "masked: comes after the query", id: "", truth: 0 };
      if (step === 1)
        return { row, work: `${term(q[0])} × ${term(k[0])} + ${term(q[1])} × ${term(k[1])} =`, id: `d${row.i}`, truth: maths.dots[row.i] };
      if (step === 2) return { row, work: `${fmt(maths.dots[row.i])} ÷ ${ROOT} =`, id: `s${row.i}`, truth: maths.scaled[row.i] };
      if (step === 3) return { row, work: `e^${term(maths.scaled[row.i], 2)} =`, id: `e${row.i}`, truth: maths.exps[row.i] };
      return { row, work: `${fmt(maths.exps[row.i])} ÷ ${fmt(maths.sum)} =`, id: `w${row.i}`, truth: maths.weights[row.i] };
    });
    const entries: [string, number][] = lines.filter((l) => l.id).map((l) => [l.id, l.truth]);
    if (step === 4) entries.unshift(["sum", maths.sum]);
    body = (
      <>
        {step === 1 && (
          <p className="ac-given">
            Query “{words[query]}”: Q = {vec(q)}
          </p>
        )}
        {step === 4 && (
          <p className="ac-sum">
            Total: {live.map((row) => fmt(maths.exps[row.i])).join(" + ")} = {answerBox("sum", maths.sum, "Sum of e to the x values")}
          </p>
        )}
        <div className="ac-lines">
          {lines.map(({ row, work, id, truth }) => (
            <div key={row.i} className={row.masked ? "ac-line masked" : "ac-line"}>
              <span className="ac-word">{row.word}</span>
              <span className="ac-work">{work}</span>
              {id && answerBox(id, truth, `${CHUNKS[step]} for ${row.word}`)}
            </div>
          ))}
        </div>
        <button type="button" className="ac-reveal" onClick={() => reveal(entries)}>
          Show me
        </button>
      </>
    );
  }

  return (
    <div className="ac">
      <div className="ac-chunks" role="tablist" aria-label="Steps">
        {CHUNKS.map((chunk, i) => (
          <button key={chunk} type="button" role="tab" aria-selected={i === step} onClick={() => setGuideStep(i)}>
            {chunk}
          </button>
        ))}
      </div>
      <div className="ac-main">
        <div className="ac-work-area" key={step}>
          {body}
          <p className="ac-note">
            Words from your sentence; the Q and K numbers are practice values so you can do the maths by hand.
          </p>
        </div>
        <Calculator
          target={step >= 1 && step <= 4 ? focused : null}
          onInsert={(value) => focused && setAnswers({ ...answers, [focused]: value })}
        />
      </div>
    </div>
  );
}
