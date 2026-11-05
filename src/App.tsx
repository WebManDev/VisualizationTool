import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import { LESSONS, type Guide } from "./lessons";
import { VOCAB, predict } from "./model";
import { STAGES } from "./Stages";

type Panel = "toc" | "quiz" | null;

/** Mini-map node for each lesson, in TOC order. Both columns share this order. */
const MINI: Record<string, string> = {
  problem: "x₁…xₜ → ?",
  map: "decoder block",
  tokens: "Tokens → E",
  norm: "LayerNorm",
  qkv: "Q · K · V",
  scores: "QKᵀ / √dₖ",
  softmax: "softmax → A",
  mlp: "MLP",
  output: "Output",
};
const ATTENTION = new Set(["qkv", "scores", "softmax"]);
const MIN_ZOOM = 0.7;

function sentences(text: string) {
  return text.match(/[^.?!]+[.?!]+["”]?\s*/g)?.map((line) => line.trim()) ?? [text];
}

export function App() {
  const [step, setStep] = useState(0);
  const [token, setToken] = useState(3);
  const [guess, setGuess] = useState("");
  const [panel, setPanel] = useState<Panel>(null);
  const [popup, setPopup] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const [guide, setGuide] = useState<Guide | null>(null);
  const [guideStep, setGuideStep] = useState(0);

  const lesson = LESSONS[step];
  const View = STAGES[lesson.id];
  // Authored with the correct answer first; rotate so position gives nothing away.
  const shift = (step + 2) % lesson.question.options.length;
  const options = [...lesson.question.options.slice(shift), ...lesson.question.options.slice(0, shift)];

  const visual = useRef<HTMLElement>(null);
  const fit = useRef<HTMLDivElement>(null);

  // Shrink tall stages to the Visual box; below MIN_ZOOM it scrolls to stay readable.
  useLayoutEffect(() => {
    const box = visual.current;
    const content = fit.current;
    if (!box || !content) return;
    const resize = () => {
      const previous = content.style.zoom;
      content.style.zoom = "1";
      const style = getComputedStyle(box);
      const room = box.clientHeight - parseFloat(style.paddingTop) - parseFloat(style.paddingBottom);
      const zoom = Math.max(MIN_ZOOM, Math.min(1, (room - 8) / content.offsetHeight));
      // Leaving a near-identical zoom alone stops the observer re-triggering itself.
      content.style.zoom = Math.abs(zoom - Number(previous)) < 0.01 ? previous : String(zoom);
    };
    resize();
    // Watches the content too, because a stage can change height from its own state.
    const observer = new ResizeObserver(resize);
    observer.observe(box);
    observer.observe(content);
    return () => observer.disconnect();
  }, [step]);

  // keepStep: the stage only updated its text (e.g. new numbers), so stay on the same step.
  const showGuide = useCallback((next: Guide | null, keepStep = false) => {
    setGuide(next);
    if (!keepStep) setGuideStep(0);
  }, []);

  function goTo(index: number) {
    if (index < 0 || index >= LESSONS.length || index === step) return;
    setPicked(null);
    setGuideStep(0);
    setStep(index);
  }

  function goId(id: string) {
    goTo(LESSONS.findIndex((item) => item.id === id));
  }

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setPanel(null);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
    };
  }, []);

  return (
    <main className="frame">
      <button
        type="button"
        className="tab tab-left"
        aria-label="Open table of contents"
        aria-expanded={panel === "toc"}
        hidden={panel === "toc"}
        onClick={() => setPanel("toc")}
      >
        <svg viewBox="0 0 85 85" aria-hidden>
          <path className="tab-shape" d="M0 1 C45 11 81 27 81 44 C81 62 45 76 5 84" />
          <path className="tab-lines" d="M22 34 L44 28 M22 44 L44 39 M23 54 L45 49" />
        </svg>
      </button>

      <button
        type="button"
        className="tab tab-right"
        aria-label="Open question"
        aria-expanded={panel === "quiz"}
        hidden={panel !== null}
        onClick={() => setPanel("quiz")}
      >
        <svg viewBox="0 0 80 78" aria-hidden>
          <path className="tab-shape" d="M66 1 C30 15 2 30 2 45 C2 60 40 72 78 76" />
          <text x="47" y="56">Q</text>
        </svg>
      </button>

      <section ref={visual} className="visual" aria-label={lesson.title}>
        <div ref={fit}>
          <View
            key={lesson.id}
            token={token}
            setToken={setToken}
            guess={guess}
            setGuess={setGuess}
            go={goId}
            pipe={lesson.pipe}
            guideStep={guideStep}
            setGuide={showGuide}
            setGuideStep={setGuideStep}
          />
        </div>
      </section>

      <Explainer
        prompt={guide?.prompt}
        steps={guide?.steps ?? sentences(lesson.narration).map((text) => ({ text }))}
        step={guideStep}
        setStep={setGuideStep}
      />

      <button
        type="button"
        className="arrow arrow-back"
        aria-label="Previous section"
        disabled={step === 0}
        onClick={() => goTo(step - 1)}
      >
        <svg viewBox="0 0 32 48" aria-hidden>
          <path d="M12 1 L1 34 L30 45 L17 1" />
        </svg>
      </button>

      <button
        type="button"
        className="arrow arrow-forward"
        aria-label="Next section"
        disabled={step === LESSONS.length - 1}
        onClick={() => goTo(step + 1)}
      >
        <svg viewBox="0 0 32 42" aria-hidden>
          <path d="M3 1 C2 13 2 26 1 40 M3 4 L30 18 L2 33" />
        </svg>
      </button>

      <div className={panel === "toc" ? "toc-drawer open" : "toc-drawer"} inert={panel !== "toc"}>
        <button
          type="button"
          className={popup ? "popup out" : "popup"}
          aria-expanded={popup}
          onClick={() => setPopup((open) => !open)}
        >
          <span className="popup-label">Key idea</span>
          {popup && <span className="popup-text">{lesson.concept}</span>}
        </button>

        <aside className="drawer-left" aria-label="Table of contents">
          <button type="button" className="close" aria-label="Close table of contents" onClick={() => setPanel(null)}>
            ×
          </button>

          <div className="mini" role="group" aria-label="Model map">
            {LESSONS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={[item.id === lesson.id && "on", ATTENTION.has(item.id) && "attn"].filter(Boolean).join(" ")}
                onClick={() => goId(item.id)}
              >
                {MINI[item.id]}
              </button>
            ))}
          </div>

          <svg className="links" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden>
            {LESSONS.map((item, index) => {
              const from = 13.4 + (57.7 * (index + 0.5)) / LESSONS.length;
              const to = 20.2 + (43.3 * (index + 0.5)) / LESSONS.length;
              return (
                <path
                  key={item.id}
                  className={item.id === lesson.id ? "on" : ""}
                  d={`M52.3 ${from} H57 L61 ${to} H66.7`}
                />
              );
            })}
          </svg>

          <nav className="toc" aria-label="Sections">
            {LESSONS.map((item) => (
              <button
                key={item.id}
                type="button"
                className={item.id === lesson.id ? "on" : ""}
                aria-current={item.id === lesson.id ? "step" : undefined}
                onClick={() => goId(item.id)}
              >
                {item.label}
              </button>
            ))}
          </nav>

          <p className="overview">{lesson.blurb}</p>
        </aside>
      </div>

      <aside className={panel === "quiz" ? "drawer-right open" : "drawer-right"} aria-label="Question" inert={panel !== "quiz"}>
        <button type="button" className="close" aria-label="Close question" onClick={() => setPanel(null)}>
          ×
        </button>
        <div className="quiz-row">
          <span className="quiz-letter">Q:</span>
          <p className="quiz-prompt">{lesson.question.prompt}</p>
        </div>
        <span className="quiz-letter">A:</span>
        <div className="quiz-answers" role="group" aria-label="Answers">
          {options.map((option) => {
            const right = option.id === lesson.question.correct;
            return (
              <button
                key={option.id}
                type="button"
                disabled={picked !== null}
                className={option.id === picked ? "picked" : ""}
                onClick={() => setPicked(option.id)}
              >
                <span>{option.label}</span>
                {picked && (
                  <b className={right ? "mark right" : "mark wrong"} aria-label={right ? "correct" : "incorrect"}>
                    {right ? "✓" : "✗"}
                  </b>
                )}
              </button>
            );
          })}
        </div>
        <div className="quiz-why" aria-live="polite">
          {picked && <p>{picked === lesson.question.correct ? lesson.question.reply : lesson.question.miss}</p>}
          {picked && lesson.id === "problem" && guess.trim() && (
            <p>Saved “{guess.trim()}”. The model makes its own guess at the end.</p>
          )}
          {picked && lesson.id === "output" && <GuessNote guess={guess} />}
        </div>
      </aside>
    </main>
  );
}

function Explainer({
  prompt,
  steps,
  step,
  setStep,
}: {
  prompt?: string;
  steps: Guide["steps"];
  step: number;
  setStep: (step: number) => void;
}) {
  const last = steps.length - 1;
  const current = Math.min(step, last);
  return (
    <section className="explainer" aria-label="Explanation">
      <div className={prompt ? "flip" : "flip over"}>
        <div className="face front" inert={!prompt}>
          <p>{prompt}</p>
        </div>
        <div className="face back" inert={Boolean(prompt)}>
          <p className="explainer-text" aria-live="polite">
            {steps[current].text}
          </p>
          <div className="explainer-nav">
            <button type="button" disabled={current === 0} onClick={() => setStep(current - 1)}>
              Back
            </button>
            <span>
              {current + 1} of {steps.length}
            </span>
            <button type="button" disabled={current === last} onClick={() => setStep(current + 1)}>
              Next
            </button>
          </div>
        </div>
      </div>
    </section>
  );
}

function GuessNote({ guess }: { guess: string }) {
  const typed = guess.trim().toLowerCase();
  const result = predict();
  if (!typed) return <p>No opening guess yet. The model’s top word is still {result.best}.</p>;
  if (typed === result.best) return <p>You guessed {typed}. That is the model’s top word too.</p>;
  const known = VOCAB.find((word) => word === typed);
  if (known) {
    const share = (result.probs[VOCAB.indexOf(known)] * 100).toFixed(1);
    return (
      <p>
        You guessed {typed}. The model gives it {share}% and prefers {result.best}.
      </p>
    );
  }
  return (
    <p>
      “{guess.trim()}” is outside this 8-word vocabulary. The model can only score {VOCAB.join(", ")}, and{" "}
      {result.best} wins.
    </p>
  );
}
