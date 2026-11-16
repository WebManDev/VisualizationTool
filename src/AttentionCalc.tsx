import { useEffect, useMemo, useState, type PointerEvent, type ReactElement } from "react";
import { Calculator } from "./Calculator";
import type { GuideStep } from "./lessons";
import type { LessonProps } from "./Stages";

const FALLBACK = ["the", "cat", "sat", "on"];
const COUNT = 4;
const ROOT = 1.41; // √2, rounded the way students will type it
const CHUNKS = ["Setup", "1. Dot product", "2. Scale", "3. eˣ", "4. Normalise", "Result"];
/** Practice position numbers: each slot in the sentence adds its own small nudge to Q and K. */
const POSITION = [
  [0, 0],
  [0.2, -0.3],
  [-0.3, 0.2],
  [0.3, 0.3],
];

const r1 = (x: number) => Math.round(x * 10) / 10;
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

function parse(answer: string) {
  const value = Number(answer.replace("−", "-"));
  return answer.trim() === "" || !Number.isFinite(value) ? null : value;
}

function near(a: number, b: number, rel = 0.015) {
  return Math.abs(a - b) <= Math.max(0.02, Math.abs(b) * rel);
}

type Line = { pos: number; id: string; truth: number; work: string; hint: (value: number) => string };

export function AttentionCalc({ guideStep, setGuide, setGuideStep, sentenceTokens }: LessonProps) {
  const words = useMemo(() => {
    const picked = (sentenceTokens ?? []).map((t) => t.trim()).filter(Boolean).slice(0, COUNT);
    return picked.length >= 2 ? picked : FALLBACK;
  }, [sentenceTokens]);
  const start = useMemo(() => words.map((_, i) => i), [words]);
  const [order, setOrder] = useState(start);
  const [queryWord, setQueryWord] = useState(words.length - 1);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [focused, setFocused] = useState<string | null>(null);
  const [doneSteps, setDoneSteps] = useState<Set<number>>(new Set());
  // Steps where "Show me" filled the boxes; those don't count as the student's own work.
  const [revealed, setRevealed] = useState<Set<number>>(new Set());
  const [dragging, setDragging] = useState<number | null>(null);
  const step = Math.min(guideStep, CHUNKS.length - 1);

  useEffect(() => {
    setOrder(words.map((_, i) => i));
    setQueryWord(words.length - 1);
    setAnswers({});
    setDoneSteps(new Set());
    setRevealed(new Set());
  }, [words]);

  // Word part (fixed per word) plus position part (fixed per slot), rounded to one decimal.
  const computeMaths = (sequence: number[], query: number) => {
    const wordKeys: number[][] = [];
    words.forEach((word, i) => {
      const same = words.findIndex((other) => other.toLowerCase() === word.toLowerCase());
      if (same < i) return wordKeys.push(wordKeys[same]);
      let salt = 2;
      let key = practice(word, salt);
      while (wordKeys.some((k) => k[0] === key[0] && k[1] === key[1])) key = practice(word, ++salt);
      wordKeys.push(key);
    });
    const qpos = sequence.indexOf(query);
    const qWord = practice(words[query], 1);
    const q = qWord.map((v, d) => r1(v + POSITION[qpos][d]));
    const keys = sequence.map((w, pos) => wordKeys[w].map((v, d) => r1(v + POSITION[pos][d])));
    const visible = qpos + 1;
    const dots = keys.map((k) => r2(q[0] * k[0] + q[1] * k[1]));
    const scaled = dots.map((x) => r2(x / ROOT));
    const exps = scaled.map((x) => r2(Math.exp(x)));
    const sum = r2(exps.slice(0, visible).reduce((a, b) => a + b, 0));
    const weights = exps.map((e, i) => (i < visible ? r2(e / sum) : 0));
    return { wordKeys, qWord, q, keys, qpos, visible, dots, scaled, exps, sum, weights };
  };

  const maths = useMemo(() => computeMaths(order, queryWord), [words, order, queryWord]);
  const startMaths = useMemo(() => computeMaths(start, queryWord), [words, start, queryWord]);
  const seq = order.map((w) => words[w]);
  const live = seq.map((_, pos) => pos).filter((pos) => pos < maths.visible);
  const topPos = live.reduce((best, pos) => (maths.weights[pos] > maths.weights[best] ? pos : best), live[0]);

  // Lines for the four calculation steps, each with a hint for the usual slips.
  const lines = (s: number): Line[] =>
    live.map((pos) => {
      const k = maths.keys[pos];
      const q = maths.q;
      if (s === 1)
        return {
          pos,
          id: `d${pos}`,
          truth: maths.dots[pos],
          work: `${term(q[0])} × ${term(k[0])} + ${term(q[1])} × ${term(k[1])} =`,
          hint: (v) => {
            if (near(v, -maths.dots[pos])) return "Right size, wrong sign. Remember: negative × negative is positive.";
            if (near(v, q[0] * k[0])) return "That’s only the first pair. Add the second pair’s product too.";
            if (near(v, q[0] + k[0] + q[1] + k[1])) return "Multiply each pair first, then add the two products.";
            if (near(v, q[0] * k[0] * q[1] * k[1])) return "Add the two products together, don’t multiply them.";
            return "Multiply first × first and second × second, then add the two results.";
          },
        };
      if (s === 2)
        return {
          pos,
          id: `s${pos}`,
          truth: maths.scaled[pos],
          work: `${fmt(maths.dots[pos])} ÷ ${ROOT} =`,
          hint: (v) => {
            if (near(v, maths.dots[pos] * ROOT)) return "That’s multiplied. Divide by 1.41 instead.";
            if (near(v, maths.dots[pos] / 2)) return "Divide by √2 ≈ 1.41, not by 2.";
            if (near(v, -maths.scaled[pos])) return "Keep the sign: dividing doesn’t change it.";
            return "Divide the score by 1.41.";
          },
        };
      if (s === 3)
        return {
          pos,
          id: `e${pos}`,
          truth: maths.exps[pos],
          work: `e^${term(maths.scaled[pos], 2)} =`,
          hint: (v) => {
            if (near(v, Math.exp(-maths.scaled[pos]))) return "Check the sign of the power before pressing eˣ.";
            if (near(v, maths.scaled[pos] * Math.E)) return "That’s e × the number. Type the number, then press eˣ (e to the power).";
            if (v < 0) return "eˣ is never negative. Type the number (with its sign), then press eˣ.";
            return "Type the scaled score, then press the eˣ key.";
          },
        };
      return {
        pos,
        id: `w${pos}`,
        truth: maths.weights[pos],
        work: `${fmt(maths.exps[pos])} ÷ ${fmt(maths.sum)} =`,
        hint: (v) => {
          if (near(v, maths.sum / maths.exps[pos])) return "Flip it: this word’s eˣ divided by the total.";
          if (v > 1) return "A weight can’t be more than 1. Divide this word’s eˣ by the total.";
          return "Divide this word’s eˣ by the total.";
        },
      };
    });

  const stepLines = step >= 1 && step <= 4 ? lines(step) : [];
  const fields: { id: string; truth: number; hint: (v: number) => string }[] = [...stepLines];
  if (step === 4)
    fields.unshift({
      id: "sum",
      truth: maths.sum,
      hint: (v) =>
        near(v, maths.exps.reduce((a, b) => a + b, 0)) && maths.visible < seq.length
          ? "Only add the words that aren’t masked."
          : "Add up the eˣ values of the words that aren’t masked.",
    });
  const status = (id: string, truth: number) => {
    const v = parse(answers[id] ?? "");
    return v === null ? null : near(v, truth);
  };
  const allRight = fields.length > 0 && fields.every((f) => status(f.id, f.truth) === true);

  useEffect(() => {
    if (allRight && !revealed.has(step) && !doneSteps.has(step)) setDoneSteps(new Set([...doneSteps, step]));
  }, [allRight, step, doneSteps, revealed]);

  const queryChanged = queryWord !== words.length - 1;
  const reordered = order.some((w, i) => w !== i);

  const steps = useMemo((): GuideStep[] => {
    const q = words[queryWord];
    const top = seq[topPos];
    const startLive = start.filter((pos) => pos < startMaths.visible);
    const startTop = startLive.reduce((best, pos) => (startMaths.weights[pos] > startMaths.weights[best] ? pos : best), startLive[0]);
    const best = live.reduce((b, pos) => (maths.dots[pos] > maths.dots[b] ? pos : b), live[0]);
    const low = live.reduce((b, pos) => (maths.exps[pos] < maths.exps[b] ? pos : b), live[0]);
    const fillTask = (s: number, feedback: string) => ({
      text: "Fill in every box. The calculator is on the right; click a box, work it out, then press Insert.",
      done: doneSteps.has(s),
      feedback,
    });
    return [
      {
        text: `Attention lets one word look back at earlier words and decide which matter. The query here is “${q}”. Each word has a Query (Q) and a Key (K), and each position in the sentence adds its own small numbers to them.`,
        task: {
          text: "Click a different word to make it the query.",
          done: queryChanged,
          feedback: `Now “${q}” is asking. It can only look at itself and the ${maths.qpos} ${maths.qpos === 1 ? "word" : "words"} before it.`,
        },
      },
      {
        text: "Step 1: multiply the query’s Q by each word’s K, number by number, and add. A big result means a good match. Words after the query are masked: a word can’t look ahead.",
        task: fillTask(1, `Best match: “${seq[best]}” scored ${fmt(maths.dots[best])}. Higher score, more attention.`),
      },
      {
        text: "Step 2: divide each score by √2 ≈ 1.41, because each vector has 2 numbers. Real models divide by √64 = 8 to keep scores from getting too big.",
        task: fillTask(2, "Same order as before, just smaller numbers. Scaling stops any one score from dominating too early."),
      },
      {
        text: "Step 3: raise e to each scaled score (the eˣ key). This turns every score positive and makes big scores stand out even more.",
        task: fillTask(3, `Look at the gap now: “${seq[best]}” is ${fmt(maths.exps[best])} while “${seq[low]}” is only ${fmt(maths.exps[low])}.`),
      },
      {
        text: "Step 4: add up the eˣ values, then divide each one by that total. The results are the attention weights, and they always add up to 1.",
        task: fillTask(4, `Those are the attention weights. Check: they add up to about 1, and “${top}” gets the biggest share.`),
      },
      {
        text: `“${q}” pays the most attention to “${top}” (${Math.round(maths.weights[topPos] * 100)}%). The thicker the line, the more attention. These are practice numbers: GPT-2 does the same maths with 64 numbers per word in each of its 12 attention heads.`,
        task: {
          text: "Drag a word to a new spot in the row below the lines and watch the connections change.",
          done: reordered,
          feedback:
            order[topPos] === startTop
              ? `Same favourite, “${top}”, but now ${Math.round(maths.weights[topPos] * 100)}% instead of ${Math.round(startMaths.weights[startTop] * 100)}%. Moving a word changes its position numbers, so every score shifts.`
              : `Now “${q}” favours “${top}” (${Math.round(maths.weights[topPos] * 100)}%) instead of “${words[startTop]}”. Same words, new positions, new scores.`,
        },
      },
    ];
  }, [words, queryWord, seq.join(), topPos, maths, startMaths, doneSteps, queryChanged, reordered]);

  useEffect(() => {
    setGuide({ steps }, true);
  }, [steps, setGuide]);
  useEffect(() => () => setGuide(null), [setGuide]);

  function chooseQuery(w: number) {
    setQueryWord(w);
    setAnswers({});
  }

  function move(word: number, toPos: number) {
    const from = order.indexOf(word);
    const to = Math.max(0, Math.min(order.length - 1, toPos));
    if (from === to) return;
    const next = [...order];
    next.splice(from, 1);
    next.splice(to, 0, word);
    setOrder(next);
    setAnswers({});
  }

  function onPointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (dragging === null) return;
    const slot = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>("[data-slot]");
    if (slot) move(dragging, Number(slot.dataset.slot));
  }

  function answerBox(id: string, truth: number, label: string) {
    const value = answers[id] ?? "";
    const ok = status(id, truth);
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

  const focusedField = fields.find((f) => f.id === focused);
  const focusedValue = focusedField ? parse(answers[focusedField.id] ?? "") : null;
  const hint =
    focusedField && focusedValue !== null && !near(focusedValue, focusedField.truth)
      ? near(focusedValue, focusedField.truth, 0.06)
        ? "Very close: check your rounding (2 decimal places)."
        : focusedField.hint(focusedValue)
      : null;

  const vec = (v: number[]) => `[${v.map((x) => fmt(x, 1)).join(", ")}]`;

  let body: ReactElement;
  if (step === 0) {
    body = (
      <>
        <div className="ac-query" role="group" aria-label="Choose the query word">
          <span>Query:</span>
          {order.map((w) => (
            <button key={w} type="button" aria-pressed={w === queryWord} onClick={() => chooseQuery(w)}>
              {words[w]}
            </button>
          ))}
        </div>
        <table className="ac-table">
          <thead>
            <tr>
              <th>word</th>
              <th>word’s K</th>
              <th>+ position</th>
              <th>= K</th>
            </tr>
          </thead>
          <tbody>
            {order.map((w, pos) => (
              <tr key={w} className={w === queryWord ? "is-query" : ""}>
                <td>{words[w]}</td>
                <td>{vec(maths.wordKeys[w])}</td>
                <td>{vec(POSITION[pos])}</td>
                <td>
                  <b>{vec(maths.keys[pos])}</b>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="ac-given">
          Query “{words[queryWord]}”: Q = {vec(maths.qWord)} + {vec(POSITION[maths.qpos])} = <b>{vec(maths.q)}</b>
        </p>
      </>
    );
  } else if (step === 5) {
    const n = seq.length;
    const x = (pos: number) => ((pos + 0.5) / n) * 100;
    body = (
      <div className="ac-result">
        <svg className="ac-links" viewBox="0 0 100 46" preserveAspectRatio="none" aria-hidden>
          {live.map((pos) => {
            const w = maths.weights[pos];
            const from = x(maths.qpos);
            const to = x(pos);
            const lift = pos === maths.qpos ? 18 : 10 + Math.abs(from - to) * 0.45;
            const d =
              pos === maths.qpos
                ? `M ${from - 3} 44 C ${from - 7} ${44 - lift}, ${from + 7} ${44 - lift}, ${from + 3} 44`
                : `M ${from} 44 C ${from} ${44 - lift}, ${to} ${44 - lift}, ${to} 44`;
            return (
              <path
                key={pos}
                d={d}
                fill="none"
                stroke="var(--purple)"
                strokeOpacity={0.25 + w * 0.75}
                strokeWidth={1 + w * 9}
                strokeLinecap="round"
                vectorEffect="non-scaling-stroke"
              />
            );
          })}
        </svg>
        <div className="ac-slots" style={{ gridTemplateColumns: `repeat(${n}, 1fr)` }}>
          {order.map((w, pos) => (
            <div key={pos} className="ac-slot" data-slot={pos}>
              <button
                type="button"
                className={[
                  w === queryWord && "is-query",
                  pos >= maths.visible && "masked",
                  dragging === w && "dragging",
                ]
                  .filter(Boolean)
                  .join(" ")}
                aria-label={`${words[w]}, position ${pos + 1}. Use left and right arrows to move.`}
                onPointerDown={(event) => {
                  event.currentTarget.setPointerCapture(event.pointerId);
                  setDragging(w);
                }}
                onPointerMove={onPointerMove}
                onPointerUp={() => setDragging(null)}
                onPointerCancel={() => setDragging(null)}
                onKeyDown={(event) => {
                  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
                  event.preventDefault();
                  move(w, pos + (event.key === "ArrowLeft" ? -1 : 1));
                }}
              >
                {words[w]}
              </button>
              <b>{pos >= maths.visible ? "masked" : `${Math.round(maths.weights[pos] * 100)}%`}</b>
            </div>
          ))}
        </div>
        <p className="ac-slot-help">
          Lines start at the query “{words[queryWord]}”. Drag a word (or select it and press ← →) to move it.{" "}
          {reordered && (
            <button type="button" onClick={() => setOrder(start)}>
              Reset order
            </button>
          )}
        </p>
      </div>
    );
  } else {
    const masked = seq.map((_, pos) => pos).filter((pos) => pos >= maths.visible);
    const reveal = () => {
      const next = { ...answers };
      for (const f of fields) next[f.id] = fmt(f.truth).replace("−", "-");
      setAnswers(next);
      setRevealed(new Set([...revealed, step]));
    };
    body = (
      <>
        {step === 1 && (
          <p className="ac-given">
            Query “{words[queryWord]}”: Q = {vec(maths.q)}
          </p>
        )}
        {step === 4 && (
          <p className="ac-sum">
            Total: {live.map((pos) => fmt(maths.exps[pos])).join(" + ")} = {answerBox("sum", maths.sum, "Sum of e to the x values")}
          </p>
        )}
        <div className="ac-lines">
          {stepLines.map((line) => (
            <div key={line.pos} className="ac-line">
              <span className="ac-word">{seq[line.pos]}</span>
              <span className="ac-work">{line.work}</span>
              {answerBox(line.id, line.truth, `${CHUNKS[step]} for ${seq[line.pos]}`)}
            </div>
          ))}
          {masked.map((pos) => (
            <div key={pos} className="ac-line masked">
              <span className="ac-word">{seq[pos]}</span>
              <span className="ac-work">masked: comes after the query</span>
            </div>
          ))}
        </div>
        <p className={allRight ? "ac-hint good" : hint ? "ac-hint" : "ac-hint idle"} aria-live="polite">
          {allRight
            ? revealed.has(step) && !doneSteps.has(step)
              ? "Answers shown. Try the next step yourself: the task ticks when you get them all."
              : "✓ All correct."
            : hint ?? "Type each answer to 2 decimal places. You’ll get a hint if one’s off."}
        </p>
        <button type="button" className="ac-reveal" onClick={reveal}>
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
            {doneSteps.has(i) ? "✓ " : ""}
            {chunk}
          </button>
        ))}
      </div>
      <div className={step >= 1 && step <= 4 ? "ac-main" : "ac-main solo"}>
        <div className="ac-work-area" key={step}>
          {body}
          <p className="ac-note">
            Words from your sentence; the Q, K and position numbers are practice values so you can do the maths by hand.
          </p>
        </div>
        {step >= 1 && step <= 4 && (
          <Calculator target={focused} onInsert={(value) => focused && setAnswers({ ...answers, [focused]: value })} />
        )}
      </div>
    </div>
  );
}
