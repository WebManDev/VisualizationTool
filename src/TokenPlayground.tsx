import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import type { Tiktoken } from "js-tiktoken/lite";
import { loadEmbeddings, type Embeddings } from "./embeddings";
import type { Guide, GuideStep } from "./lessons";
import type { LessonProps } from "./Stages";
import { WordReorder } from "./WordReorder";

const SLIDE_SENTENCE = "I love purple cats. Purple cats are great.";
const PENS = ["#e23b3b", "#2f6bff", "#7a3ff0", "#1aa36a", "#a16207", "#555555"];
const MAX_TERMS = 6;
const PRESETS = [
  { label: "king − man + woman", words: [" king", " man", " woman"], signs: [1, -1, 1] },
  { label: "cats − cat + dog", words: [" cats", " cat", " dog"], signs: [1, -1, 1] },
  { label: "bigger − big + small", words: [" bigger", " big", " small"], signs: [1, -1, 1] },
];

type Term = { id: number; sign: 1 | -1 };

function fmt(value: number) {
  return (value < 0 ? "−" : "") + Math.abs(value).toFixed(3);
}

/** Make a token's leading space visible, since " cats" and "cats" are different tokens. */
function showPiece(piece: string) {
  return piece.replaceAll(" ", "␣");
}

/** The walkthrough, written from the reader's own sentence. */
function walkthrough(tokenizer: Tiktoken, ids: number[], rows: number[]): GuideStep[] {
  const pieces = ids.map((id) => tokenizer.decode([id]));
  const show = (id: number) => `“${showPiece(tokenizer.decode([id]))}”`;
  const steps: GuideStep[] = [
    {
      text: `Your sentence became ${ids.length} ${ids.length === 1 ? "token" : "tokens"}. A token is a chunk of text the model treats as one unit, and each coloured underline marks one.`,
      focus: "sentence",
    },
  ];

  // A letters-only piece glued to the previous piece means one word was split.
  const split = pieces.findIndex((piece, i) => i > 0 && /^\p{L}+$/u.test(piece) && /\p{L}$/u.test(pieces[i - 1]));
  if (split > 0) {
    let start = split - 1;
    while (start > 0 && !pieces[start].startsWith(" ") && /^\p{L}+$/u.test(pieces[start])) start -= 1;
    let end = split;
    while (end + 1 < pieces.length && /^\p{L}+$/u.test(pieces[end + 1])) end += 1;
    const parts = pieces.slice(start, end + 1);
    steps.push({
      text: `“${parts.join("").trim()}” isn’t one entry in GPT-2’s vocabulary, so it was split into ${parts.length} pieces: ${parts.map((part) => part.trim()).join(" + ")}.`,
      focus: "sentence",
    });
  }

  const spaced = ids.find((_, i) => pieces[i].startsWith(" "));
  if (spaced !== undefined) {
    steps.push({
      text: `Look at ${show(spaced)} in the table. The ␣ is a space, and it belongs to the token, so a word with a space and the same word without one are different tokens.`,
      focus: "rows",
      pick: spaced,
    });
  }

  if (pieces.filter((piece) => /\p{L}/u.test(piece)).length > 1) {
    steps.push({
      text: "Does a word’s position change its token? Press Reorder words and drag a word to the very front, then move the first word into the middle. Watch which IDs change, and which stay the same.",
      focus: "reorder",
    });
  }

  steps.push({
    text: `Each token has an ID, which is its position in GPT-2’s vocabulary of 50,257 tokens. ${show(ids[0])} is ID ${ids[0].toLocaleString()}. The IDs are where the words stop: the model never sees the text itself.`,
    focus: "ids",
    pick: ids[0],
  });

  const repeated = rows.find((id) => ids.filter((item) => item === id).length > 1);
  steps.push(
    repeated !== undefined
      ? {
          text: `${show(repeated)} appears ${ids.filter((id) => id === repeated).length} times but has one row. A token always looks up the same row, so ${ids.length} tokens need only ${rows.length} rows here.`,
          focus: "rows",
          pick: repeated,
        }
      : {
          text: `The table has one row for each different token. Nothing repeats in your sentence, so ${ids.length} ${ids.length === 1 ? "token needs 1 row" : `tokens need ${rows.length} rows`}. A repeated word would share its row.`,
          focus: "rows",
        },
  );

  steps.push(
    {
      text: "The green numbers are the start of that token’s vector, also called its embedding. These are GPT-2’s real numbers: each token has 768, and the table shows the first 4. This vector, not the word or the ID, is what goes into the network.",
      focus: "cells",
    },
    {
      text: "So what does each number mean? Nothing you could name on its own: there is no “plural” number or “animal” number. Meaning is spread across all 768 and shows up in where the whole vector points. Tokens used in similar ways point in similar directions, so check “Closest in meaning” in the card.",
      focus: "detail",
    },
    {
      text: "Because meaning lives in directions, you can do maths with whole vectors. Open Vector maths and try king − man + woman: the result points closest to “queen”.",
      focus: "tabs",
    },
    {
      text: "Click any token to see which tokens sit closest to it. One gap remains: the same word gets the same vector wherever it sits, so real models also add a position vector to tell word order apart.",
      focus: "detail",
    },
  );
  return steps;
}

const PROMPT: Guide = {
  prompt: "Type a sentence in the box above, then press See its embeddings.",
  steps: [{ text: "" }],
};

export function TokenPlayground({ guideStep, setGuide, setSentenceTokens }: LessonProps) {
  const [tokenizer, setTokenizer] = useState<Tiktoken | null>(null);
  const [failed, setFailed] = useState(false);
  const [draft, setDraft] = useState("");
  const [sentence, setSentence] = useState<string | null>(null);
  const [original, setOriginal] = useState("");
  const [reorder, setReorder] = useState(false);
  // Reordering rewrites the sentence; keep the reader on the same walkthrough step when it does.
  const keepStep = useRef(false);
  const [selected, setSelected] = useState<number | null>(null);
  const [embeddings, setEmbeddings] = useState<Embeddings | null>(null);
  const [embedFailed, setEmbedFailed] = useState(false);
  const [view, setView] = useState<"table" | "maths">("table");
  const [terms, setTerms] = useState<Term[]>([]);
  const [extra, setExtra] = useState("");
  const [extraError, setExtraError] = useState<string | null>(null);

  // The vocabulary is about 500 KB, so it only downloads when this section opens.
  useEffect(() => {
    let live = true;
    Promise.all([import("js-tiktoken/lite"), import("js-tiktoken/ranks/gpt2")])
      .then(([{ Tiktoken }, ranks]) => {
        if (live) setTokenizer(new Tiktoken(ranks.default));
      })
      .catch(() => {
        if (live) setFailed(true);
      });
    // The embedding data is about 7 MB, so it starts downloading while people type.
    loadEmbeddings()
      .then((loaded) => {
        if (live) setEmbeddings(loaded);
      })
      .catch(() => {
        if (live) setEmbedFailed(true);
      });
    return () => {
      live = false;
    };
  }, []);

  // Empty lists: treat text like "<|endoftext|>" as ordinary characters instead of throwing.
  const ids = useMemo(
    () => (tokenizer && sentence ? tokenizer.encode(sentence, [], []) : []),
    [tokenizer, sentence],
  );
  const rows = useMemo(() => [...new Set(ids)], [ids]);

  const steps = useMemo(
    () => (tokenizer && ids.length ? walkthrough(tokenizer, ids, rows) : null),
    [tokenizer, ids, rows],
  );
  const step = steps?.[Math.min(guideStep, steps.length - 1)];

  useEffect(() => {
    setGuide(steps ? { steps } : PROMPT, keepStep.current);
    keepStep.current = false;
  }, [steps, setGuide]);

  useEffect(() => () => setGuide(null), [setGuide]);

  useEffect(() => {
    if (step?.pick !== undefined) setSelected(step.pick);
  }, [step]);

  // Share the sentence with later sections (attention works on its first few tokens).
  useEffect(() => {
    if (tokenizer && ids.length) setSentenceTokens(ids.map((id) => tokenizer.decode([id])));
  }, [tokenizer, ids, setSentenceTokens]);

  // Reordering can remove the selected token (e.g. " cat" becomes "cat"); follow the sentence instead.
  useEffect(() => {
    if (selected !== null && ids.length && !ids.includes(selected)) setSelected(ids[0]);
  }, [ids, selected]);

  const neighbours = useMemo(
    () => (embeddings && selected !== null ? embeddings.nearest(embeddings.vector(selected), new Set([selected]), 5) : []),
    [embeddings, selected],
  );

  const result = useMemo(() => {
    if (!embeddings || terms.length === 0) return null;
    const total = new Float32Array(embeddings.vector(terms[0].id).length);
    const shown = [0, 0, 0, 0];
    for (const term of terms) {
      embeddings.vector(term.id).forEach((value, d) => (total[d] += term.sign * value));
      embeddings.first4(term.id).forEach((value, d) => (shown[d] += term.sign * value));
    }
    // Leaving out the inputs is the usual rule; otherwise "king" is often closest to king − man + woman.
    const closest = embeddings.nearest(total, new Set(terms.map((term) => term.id)), 5);
    return { shown, closest };
  }, [embeddings, terms]);

  if (failed) return <p className="tp-status">The tokenizer didn’t load. Refresh the page to try again.</p>;

  function start(text: string) {
    const clean = text.trim();
    if (!tokenizer || !clean) return;
    setDraft(clean);
    setSentence(clean);
    setOriginal(clean);
    setReorder(false);
    setSelected(tokenizer.encode(clean, [], [])[0]);
    setTerms([]);
    setView("table");
  }

  function addTerm(id: number, sign: 1 | -1 = 1) {
    setTerms((current) => (current.length >= MAX_TERMS ? current : [...current, { id, sign }]));
  }

  function addWord(event: FormEvent) {
    event.preventDefault();
    const word = extra.trim();
    if (!tokenizer || !word) return;
    // Mid-sentence words carry a leading space, and those forms have the most meaningful vectors.
    const pieces = tokenizer.encode(` ${word}`, [], []);
    if (pieces.length !== 1) {
      setExtraError(`“${word}” is ${pieces.length} tokens. Vector maths here uses single tokens, so try a shorter or more common word.`);
      return;
    }
    addTerm(pieces[0]);
    setExtra("");
    setExtraError(null);
  }

  function applyPreset(words: string[], signs: number[]) {
    if (!tokenizer) return;
    setTerms(words.map((word, i) => ({ id: tokenizer.encode(word, [], [])[0], sign: signs[i] as 1 | -1 })));
    setExtraError(null);
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    start(draft);
  }

  if (!tokenizer || sentence === null) {
    return (
      <form className="tp-start" onSubmit={submit}>
        <label htmlFor="tp-sentence">Type a sentence</label>
        <input
          id="tp-sentence"
          className="tp-input"
          value={draft}
          maxLength={160}
          autoComplete="off"
          placeholder="Anything you like…"
          onChange={(event) => setDraft(event.target.value)}
        />
        <div className="tp-actions">
          <button type="submit" className="tp-go" disabled={!tokenizer || !draft.trim()}>
            {tokenizer ? "See its embeddings" : "Loading tokenizer…"}
          </button>
          <button type="button" className="tp-link" disabled={!tokenizer} onClick={() => start(SLIDE_SENTENCE)}>
            or use “{SLIDE_SENTENCE}”
          </button>
        </div>
      </form>
    );
  }

  const label = (id: number) => tokenizer.decode([id]);
  const pen = (id: number) => PENS[rows.indexOf(id) % PENS.length];
  const count = ids.filter((id) => id === selected).length;

  return (
    <div className="tp" data-focus={step?.focus}>
      <div className="tp-sentence">
        <div className="tp-tokens" role="group" aria-label="Tokens">
          {ids.map((id, index) => (
            <button
              key={index}
              type="button"
              className={id === selected ? "on" : ""}
              style={{ borderBottomColor: pen(id) }}
              aria-pressed={id === selected}
              onClick={() => (view === "maths" ? addTerm(id) : setSelected(id))}
            >
              {label(id)}
            </button>
          ))}
        </div>
        <div className="tp-sentence-actions">
          <button
            type="button"
            className="tp-edit tp-reorder-toggle"
            aria-pressed={reorder}
            onClick={() => setReorder(!reorder)}
          >
            {reorder ? "Done reordering" : "Reorder words"}
          </button>
          <button type="button" className="tp-edit" onClick={() => setSentence(null)}>
            New sentence
          </button>
        </div>
      </div>

      {reorder && (
        <WordReorder
          key={original}
          tokenizer={tokenizer}
          original={original}
          onChange={(text) => {
            keepStep.current = true;
            setSentence(text);
          }}
        />
      )}

      <div className="tp-tabs" role="tablist" aria-label="View">
        <button type="button" role="tab" aria-selected={view === "table"} onClick={() => setView("table")}>
          Embedding table
        </button>
        <button type="button" role="tab" aria-selected={view === "maths"} onClick={() => setView("maths")}>
          Vector maths
        </button>
      </div>

      {embedFailed && <p className="tp-status">The embedding numbers didn’t load. Refresh the page to try again.</p>}

      {view === "table" ? (
        <div className="tp-body">
          <div className="tp-matrix">
            <p className="tp-title">
              {rows.length} {rows.length === 1 ? "row" : "rows"} · first 4 of 768 numbers per token
            </p>
            <div className="tp-rows">
              {rows.map((id) => (
                <button
                  key={id}
                  type="button"
                  className={id === selected ? "tp-row on" : "tp-row"}
                  aria-pressed={id === selected}
                  onClick={() => setSelected(id)}
                >
                  <span className="tp-word" title={showPiece(label(id))}>
                    {showPiece(label(id))}
                  </span>
                  <span className="tp-id">{id}</span>
                  {(embeddings ? embeddings.first4(id) : [null, null, null, null]).map((value, index) => (
                    <span key={index} className="tp-cell">
                      {value === null ? "…" : fmt(value)}
                    </span>
                  ))}
                </button>
              ))}
            </div>
          </div>

          {selected !== null && (
            <aside className="tp-detail" aria-live="polite">
              <p className="tp-picked">“{showPiece(label(selected))}”</p>
              <p>
                token ID <b>{selected.toLocaleString()}</b>
                {count > 1 ? ` · appears ${count} times, one shared vector` : ""}
              </p>
              <p className="tp-near-title">Closest in meaning</p>
              <p className="tp-near">
                {embeddings
                  ? neighbours.map((id) => <span key={id}>{showPiece(label(id))}</span>)
                  : "Loading…"}
              </p>
            </aside>
          )}
        </div>
      ) : (
        <div className="tp-maths">
          <p className="tp-hint-line">
            Click tokens in your sentence to add them, then click a sign to flip it between + and −.
          </p>
          <div className="tp-maths-tools">
            <form className="tp-add" onSubmit={addWord}>
              <input
                value={extra}
                maxLength={30}
                autoComplete="off"
                placeholder="Add any word…"
                aria-label="Add a word"
                onChange={(event) => setExtra(event.target.value)}
              />
              <button type="submit" disabled={!extra.trim() || terms.length >= MAX_TERMS}>
                Add
              </button>
            </form>
            <div className="tp-presets" role="group" aria-label="Examples">
              {PRESETS.map((preset) => (
                <button key={preset.label} type="button" onClick={() => applyPreset(preset.words, preset.signs)}>
                  {preset.label}
                </button>
              ))}
            </div>
          </div>
          {extraError && <p className="tp-error">{extraError}</p>}

          {terms.length === 0 ? (
            <p className="tp-empty">No vectors yet. Click a token above or try an example.</p>
          ) : (
            <div className="tp-sum" role="table" aria-label="Vector sum">
              {terms.map((term, index) => (
                <div key={index} className="tp-sum-row" role="row">
                  <button
                    type="button"
                    className="tp-sign"
                    aria-label={`${term.sign > 0 ? "Plus" : "Minus"}; click to flip`}
                    onClick={() =>
                      setTerms(terms.map((item, i) => (i === index ? { ...item, sign: item.sign > 0 ? -1 : 1 } : item)))
                    }
                  >
                    {term.sign > 0 ? "+" : "−"}
                  </button>
                  <span className="tp-word">{showPiece(label(term.id))}</span>
                  {(embeddings ? embeddings.first4(term.id) : [null, null, null, null]).map((value, d) => (
                    <span key={d} className="tp-cell">
                      {value === null ? "…" : fmt(value)}
                    </span>
                  ))}
                  <button
                    type="button"
                    className="tp-remove"
                    aria-label={`Remove ${showPiece(label(term.id))}`}
                    onClick={() => setTerms(terms.filter((_, i) => i !== index))}
                  >
                    ×
                  </button>
                </div>
              ))}
              {result && (
                <div className="tp-sum-row total" role="row">
                  <span className="tp-sign">=</span>
                  <span className="tp-word">result</span>
                  {result.shown.map((value, d) => (
                    <span key={d} className="tp-cell">
                      {fmt(value)}
                    </span>
                  ))}
                  <span />
                </div>
              )}
              <p className="tp-closest">
                <span>Closest token to the result:</span>{" "}
                {result
                  ? result.closest.map((id, i) => (
                      <b key={id} className={i === 0 ? "top" : ""}>
                        {showPiece(label(id))}
                      </b>
                    ))
                  : "loading…"}
              </p>
            </div>
          )}
        </div>
      )}

      <p className="tp-note">
        Real GPT-2 tokens, IDs and embeddings. No single number means anything on its own; meaning is in where the whole
        vector points. “Closest” and the maths use a compressed copy of each vector (128 of 768 numbers’ worth).
      </p>
    </div>
  );
}
