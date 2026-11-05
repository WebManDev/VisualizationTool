import { useEffect, useMemo, useState } from "react";
import type { GuideStep } from "./lessons";
import { layerNorm } from "./model";
import type { LessonProps } from "./Stages";

const START = [2, 4, 4, 6];
const CHUNKS = ["Numbers", "Average", "Subtract", "Spread", "Divide", "Stretch & shift", "Recap"];

function num(value: number) {
  const rounded = Math.abs(value) < 0.005 ? 0 : value;
  return (rounded < 0 ? "−" : "") + Math.abs(rounded).toFixed(2);
}

/** σ can be tiny (only ε) when all numbers match, so show enough digits to see it isn't 0. */
function sigma(value: number) {
  return value < 0.01 ? value.toFixed(4) : num(value);
}

/** Short explainer text per chunk, using the student's own numbers. */
function texts(x: number[], ln: ReturnType<typeof layerNorm>): GuideStep[] {
  const flat = ln.variance < 1e-9;
  return [
    {
      text: "These are one token’s numbers. Change any of them in the boxes above: layer norm will tidy them up one small step at a time.",
    },
    {
      text: `Step 1: find the average of the numbers. Yours average to ${num(ln.mean)}, shown as the dashed line.`,
    },
    {
      text: "Step 2: subtract the average from every number. Now the numbers are centred: they balance around 0.",
    },
    {
      text: flat
        ? "Step 3: measure how spread out they are. Your numbers are all equal, so there is no spread at all: the standard deviation would be 0."
        : `Step 3: measure how spread out they are. This is the standard deviation, σ = ${num(ln.std)}. The shaded band shows one σ either side of 0.`,
    },
    {
      text: flat
        ? `Step 4: divide by σ. Dividing by 0 is impossible, so layer norm adds a tiny ε first and divides by ${sigma(ln.std)} instead. Everything stays 0. Change one number to see the usual case.`
        : "Step 4: divide every number by σ. Now the spread is exactly 1, however big or small your numbers were. Try the ×10 button to check.",
    },
    {
      text: "Step 5: the model then stretches the result with γ and shifts it with β. It learns these two dials in training. Drag them to see what they do.",
    },
    {
      text: `That’s all of layer norm, and every token gets it separately. GPT-2 does the same with 768 numbers per token instead of ${x.length}.`,
    },
  ];
}

function Bars({
  values,
  ghost,
  meanLine,
  band,
}: {
  values: number[];
  ghost?: number[];
  meanLine?: number;
  band?: number;
}) {
  const reach = Math.max(1, ...values.map(Math.abs), ...(ghost ?? []).map(Math.abs), Math.abs(meanLine ?? 0), band ?? 0) * 1.15;
  const y = (value: number) => 70 - (value / reach) * 62;
  const slot = 360 / values.length;
  return (
    <svg className="ln-bars" viewBox="0 0 480 150" role="img" aria-label={`Bars: ${values.map(num).join(", ")}`}>
      {band !== undefined && <rect className="ln-band" x="30" width="360" y={y(band)} height={y(-band) - y(band)} />}
      <line className="ln-zero" x1="30" x2="390" y1={y(0)} y2={y(0)} />
      <text className="ln-axis" x="24" y={y(0) + 4}>
        0
      </text>
      {values.map((value, i) => {
        const cx = 30 + slot * (i + 0.5);
        const top = Math.min(y(0), y(value));
        return (
          <g key={i}>
            {ghost && (
              <rect
                className="ln-ghost"
                x={cx - 18}
                width="36"
                y={Math.min(y(0), y(ghost[i]))}
                height={Math.max(1, Math.abs(y(ghost[i]) - y(0)))}
              />
            )}
            <rect
              className={value < 0 ? "ln-bar neg" : "ln-bar"}
              x={cx - 14}
              width="28"
              y={top}
              height={Math.max(1.5, Math.abs(y(value) - y(0)))}
            />
            <text className="ln-value" x={cx} y="146">
              {num(value)}
            </text>
          </g>
        );
      })}
      {meanLine !== undefined && (
        <>
          <line className="ln-mean" x1="30" x2="390" y1={y(meanLine)} y2={y(meanLine)} />
          <text className="ln-mean-label" x="396" y={y(meanLine) + 4}>
            average {num(meanLine)}
          </text>
        </>
      )}
    </svg>
  );
}

export function NormSteps({ guideStep, setGuide, setGuideStep }: LessonProps) {
  const [x, setX] = useState(START);
  const [drafts, setDrafts] = useState(START.map(String));
  const [gamma, setGamma] = useState(1);
  const [beta, setBeta] = useState(0);
  const ln = useMemo(() => layerNorm(x, gamma, beta), [x, gamma, beta]);
  const step = Math.min(guideStep, CHUNKS.length - 1);

  useEffect(() => {
    setGuide({ steps: texts(x, ln) }, true);
  }, [x, ln, setGuide]);

  useEffect(() => () => setGuide(null), [setGuide]);

  function setValue(index: number, text: string) {
    setDrafts(drafts.map((draft, i) => (i === index ? text : draft)));
    const value = Number(text);
    if (text.trim() !== "" && Number.isFinite(value)) setX(x.map((item, i) => (i === index ? value : item)));
  }

  function replace(next: number[]) {
    const rounded = next.map((value) => Math.round(value * 100) / 100);
    setX(rounded);
    setDrafts(rounded.map(String));
  }

  const squares = ln.centered.map((value) => value * value);

  return (
    <div className="ln">
      <div className="ln-inputs">
        <span>Your token’s numbers</span>
        {drafts.map((draft, i) => (
          <input
            key={i}
            type="number"
            step="any"
            value={draft}
            aria-label={`Number ${i + 1}`}
            onChange={(event) => setValue(i, event.target.value)}
            onBlur={() => setDrafts(x.map(String))}
          />
        ))}
        <button type="button" onClick={() => replace(START)}>
          Reset
        </button>
      </div>

      <div className="ln-chunks" role="tablist" aria-label="Steps">
        {CHUNKS.map((chunk, i) => (
          <button key={chunk} type="button" role="tab" aria-selected={i === step} onClick={() => setGuideStep(i)}>
            {i > 0 && i < CHUNKS.length - 1 ? `${i}. ` : ""}
            {chunk}
          </button>
        ))}
      </div>

      <div className="ln-stage" key={step}>
        {step === 0 && <Bars values={x} />}

        {step === 1 && (
          <>
            <Bars values={x} meanLine={ln.mean} />
            <p className="ln-calc">
              ({x.map(num).join(" + ")}) ÷ {x.length} = <b>{num(ln.mean)}</b>
            </p>
          </>
        )}

        {step === 2 && (
          <>
            <Bars values={ln.centered} ghost={x} />
            <p className="ln-calc">
              {x.map((value, i) => (
                <span key={i}>
                  {num(value)} − {num(ln.mean)} = <b>{num(ln.centered[i])}</b>
                </span>
              ))}
            </p>
          </>
        )}

        {step === 3 && (
          <>
            <Bars values={ln.centered} band={ln.std} />
            <p className="ln-calc">
              <span>square each: {squares.map(num).join(", ")}</span>
              <span>average: {num(ln.variance)}</span>
              <span>
                square root: σ = <b>{sigma(ln.std)}</b>
              </span>
            </p>
          </>
        )}

        {step === 4 && (
          <>
            <Bars values={ln.normalized} band={1} ghost={ln.centered} />
            <p className="ln-calc">
              {ln.centered.map((value, i) => (
                <span key={i}>
                  {num(value)} ÷ {sigma(ln.std)} = <b>{num(ln.normalized[i])}</b>
                </span>
              ))}
            </p>
            <div className="ln-try">
              <button type="button" onClick={() => replace(x.map((value) => value * 10))}>
                ×10 my numbers
              </button>
              <button type="button" onClick={() => replace(x.map((value) => value / 10))}>
                ÷10 my numbers
              </button>
              <span>The bars here won’t move.</span>
            </div>
          </>
        )}

        {step === 5 && (
          <>
            <Bars values={ln.scaled} ghost={ln.normalized} />
            <div className="ln-dials">
              <label>
                <span>
                  γ stretch <b>{num(gamma)}</b>
                </span>
                <input type="range" min={0} max={3} step={0.1} value={gamma} onChange={(e) => setGamma(Number(e.target.value))} />
              </label>
              <label>
                <span>
                  β shift <b>{num(beta)}</b>
                </span>
                <input type="range" min={-2} max={2} step={0.1} value={beta} onChange={(e) => setBeta(Number(e.target.value))} />
              </label>
            </div>
            <p className="ln-calc">
              <span>
                each number × {num(gamma)} + {num(beta)}
              </span>
            </p>
          </>
        )}

        {step === 6 && (
          <div className="ln-recap">
            <p className="ln-flow">
              <span>your numbers</span>→<span className="c1">− average</span>→<span className="c2">÷ spread σ</span>→
              <span className="c3">× γ + β</span>
            </p>
            <p className="ln-formula">
              LN(x) = <span className="c3">γ ×</span> <span className="c1">(x − μ)</span> <span className="c2">÷ √(σ² + ε)</span>{" "}
              <span className="c3">+ β</span>
            </p>
            <p className="ln-small">
              ε is a tiny number (0.00001) so the maths never divides by zero, which would happen if all of a token’s numbers
              were equal.
            </p>
            <Bars values={ln.scaled} />
          </div>
        )}
      </div>
    </div>
  );
}
