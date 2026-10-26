import { useEffect, useRef, useState } from "react";
import { FLOW, LESSONS } from "./lessons";
import { VOCAB, predict } from "./model";
import { STAGES } from "./Stages";

export function App() {
  const [step, setStep] = useState(0);
  const [token, setToken] = useState(3);
  const [guess, setGuess] = useState("");
  const [playing, setPlaying] = useState(false);
  const [transcript, setTranscript] = useState(false);
  const [picked, setPicked] = useState<string | null>(null);
  const playId = useRef(0);

  const lesson = LESSONS[step];
  const View = STAGES[lesson.id];
  const groups = [...new Set(LESSONS.map((item) => item.group))];

  function goId(id: string) {
    const index = LESSONS.findIndex((item) => item.id === id);
    if (index < 0 || index === step) return;
    window.speechSynthesis?.cancel();
    setPlaying(false);
    setTranscript(false);
    setPicked(null);
    setStep(index);
  }

  function toggleAudio() {
    const synth = window.speechSynthesis;
    if (!synth) return;
    playId.current += 1;
    synth.cancel();
    if (playing) {
      setPlaying(false);
      return;
    }
    const id = playId.current;
    const utterance = new SpeechSynthesisUtterance(lesson.narration);
    utterance.rate = 0.96;
    const voice = synth.getVoices().find((item) => item.lang.startsWith("en"));
    if (voice) utterance.voice = voice;
    utterance.onend = () => {
      if (playId.current === id) setPlaying(false);
    };
    utterance.onerror = () => {
      if (playId.current === id) setPlaying(false);
    };
    setPlaying(true);
    synth.speak(utterance);
  }

  useEffect(() => {
    window.speechSynthesis?.getVoices();
    return () => window.speechSynthesis?.cancel();
  }, []);

  const question = lesson.question;

  return (
    <div className="board">
      <div className="left">
        <section className="ink flow-box" aria-label="Model map">
          <div className="flow">
            {FLOW.map((item, index) => (
              <span key={item.key} className="flow-step">
                {index > 0 && <span className="stem" aria-hidden />}
                <button
                  type="button"
                  className={lesson.flow === item.key ? "on" : ""}
                  onClick={() => goId(item.lesson)}
                >
                  {item.key}
                </button>
              </span>
            ))}
          </div>
          <div className="toc">
            <p className="hand-label">Table of Contents</p>
            {groups.map((group) => (
              <div key={group}>
                <p className="toc-group">{group}</p>
                {LESSONS.filter((item) => item.group === group).map((item) => (
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
              </div>
            ))}
          </div>
        </section>

        <section className="ink explain" aria-label="Short explanation">
          <h1>{lesson.title}</h1>
          <p>{lesson.blurb}</p>
        </section>
      </div>

      <div className="center">
        <section className="blue" aria-label="Visualization">
          <div className="viz">
            <p className="hand-label">visualization</p>
            <div className="viz-body">
              <View
                token={token}
                setToken={setToken}
                guess={guess}
                setGuess={setGuess}
                go={goId}
                pipe={lesson.pipe}
              />
            </div>
          </div>
          <div className="concept">
            <p className="hand-label">visualize complex concepts</p>
            <p className="concept-eq">{lesson.concept}</p>
          </div>
        </section>

        <section className="ink audio" aria-label="Audio">
          <button type="button" className="audio-play" onClick={toggleAudio} aria-pressed={playing}>
            Audio
          </button>
          <button
            type="button"
            className={playing ? "wave on" : "wave"}
            onClick={toggleAudio}
            aria-label={playing ? "Pause narration" : "Play narration"}
          >
            {[12, 26, 34, 16, 28, 14, 36, 20, 24, 12, 30, 18, 22, 32, 14, 26, 16, 28].map((height, index) => (
              <span key={index} style={{ height, animationDelay: `${index * 0.06}s` }} />
            ))}
          </button>
          <button
            type="button"
            className="transcript-toggle"
            aria-expanded={transcript}
            onClick={() => setTranscript((open) => !open)}
          >
            Transcription <span aria-hidden>{transcript ? "▴" : "▾"}</span>
          </button>
          {transcript && <p className="transcript">{lesson.narration}</p>}
        </section>
      </div>

      <aside className="ink workbook" aria-label="Workbook">
        <h2>Workbook / Discussion</h2>
        <div key={lesson.id} className="pop">
          <p className="prompt">{question.prompt}</p>
          <div className="options" role="group">
            {question.options.map((option) => {
              let state = "";
              if (picked) state = option.id === question.correct ? "is-right" : option.id === picked ? "is-wrong" : "";
              return (
                <button
                  key={option.id}
                  type="button"
                  className={state}
                  onClick={() => setPicked(option.id)}
                >
                  <span className="dot" aria-hidden />
                  {option.label}
                </button>
              );
            })}
          </div>
          {picked && <p className="reply">{picked === question.correct ? question.reply : question.miss}</p>}
          {lesson.id === "problem" && guess.trim() && (
            <p className="reply">Saved “{guess.trim()}”. The model makes its own guess at the end.</p>
          )}
          {lesson.id === "output" && <GuessNote guess={guess} />}
        </div>
      </aside>
    </div>
  );
}

function GuessNote({ guess }: { guess: string }) {
  const typed = guess.trim().toLowerCase();
  if (!typed) return <p className="reply">No opening guess yet. The model’s top word is still mat.</p>;
  const result = predict();
  if (typed === result.best) {
    return <p className="reply">You guessed {typed}. That is the model’s top word too.</p>;
  }
  const known = VOCAB.find((word) => word === typed);
  if (known) {
    const share = (result.probs[VOCAB.indexOf(known)] * 100).toFixed(1);
    return (
      <p className="reply">
        You guessed {typed}. The model gives it {share}% and prefers {result.best}.
      </p>
    );
  }
  return (
    <p className="reply">
      “{guess.trim()}” is outside this 8-word vocabulary. The model can only score {VOCAB.join(", ")}, and{" "}
      {result.best} wins.
    </p>
  );
}
