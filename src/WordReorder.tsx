import { useEffect, useRef, useState, type KeyboardEvent, type PointerEvent } from "react";
import type { Tiktoken } from "js-tiktoken/lite";

type Unit = { key: number; text: string; word: boolean };
type Change = { from: number[]; to: number[] };

const FLASH_MS = 2600;

function split(sentence: string): Unit[] {
  return (sentence.match(/[\p{L}\p{N}'’]+|[^\s\p{L}\p{N}'’]+/gu) ?? []).map((text, key) => ({
    key,
    text,
    word: /[\p{L}\p{N}]/u.test(text),
  }));
}

/** Words get a space before them; punctuation hugs the previous word, and nothing follows an opening bracket or quote. */
function spaced(units: Unit[]) {
  return units.map((unit, i) => {
    const after = i > 0 ? units[i - 1].text : "";
    const space = i > 0 && unit.word && !/[([{“‘"]$/.test(after);
    return { ...unit, prefix: space ? " " : "" };
  });
}

function build(units: Unit[]) {
  return spaced(units)
    .map((unit) => unit.prefix + unit.text)
    .join("");
}

export function WordReorder({
  tokenizer,
  original,
  onChange,
}: {
  tokenizer: Tiktoken;
  original: string;
  onChange: (sentence: string) => void;
}) {
  const [units, setUnits] = useState(() => split(original));
  const [changes, setChanges] = useState<Record<number, Change>>({});
  const [dragging, setDragging] = useState<number | null>(null);
  const timers = useRef<number[]>([]);

  useEffect(() => () => timers.current.forEach((id) => window.clearTimeout(id)), []);

  // GPT-2 tokenizes each word together with the space in front of it, so this matches the full sentence's tokens.
  const tokensOf = (list: Unit[]) =>
    Object.fromEntries(spaced(list).map((unit) => [unit.key, tokenizer.encode(unit.prefix + unit.text, [], [])]));

  function commit(next: Unit[]) {
    const before = tokensOf(units);
    const after = tokensOf(next);
    const changed: Record<number, Change> = {};
    for (const unit of next) {
      if (unit.word && before[unit.key].join() !== after[unit.key].join()) {
        changed[unit.key] = { from: before[unit.key], to: after[unit.key] };
      }
    }
    if (Object.keys(changed).length) {
      setChanges((current) => ({ ...current, ...changed }));
      const timer = window.setTimeout(() => {
        setChanges((current) => {
          const rest = { ...current };
          for (const key of Object.keys(changed)) delete rest[Number(key)];
          return rest;
        });
      }, FLASH_MS);
      timers.current.push(timer);
    }
    setUnits(next);
    onChange(build(next));
  }

  /** Move a word into another word's slot; punctuation never moves. */
  function move(key: number, toSlot: number) {
    const slots = units.flatMap((unit, i) => (unit.word ? [i] : []));
    const order = slots.map((i) => units[i]);
    const from = order.findIndex((unit) => unit.key === key);
    const to = Math.max(0, Math.min(order.length - 1, toSlot));
    if (from < 0 || from === to) return;
    const [picked] = order.splice(from, 1);
    order.splice(to, 0, picked);
    const next = [...units];
    slots.forEach((slot, i) => (next[slot] = order[i]));
    commit(next);
  }

  const wordIndex = (key: number) => units.filter((unit) => unit.word).findIndex((unit) => unit.key === key);

  function onPointerDown(event: PointerEvent<HTMLButtonElement>, key: number) {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(key);
  }

  function onPointerMove(event: PointerEvent<HTMLButtonElement>) {
    if (dragging === null) return;
    const target = document.elementFromPoint(event.clientX, event.clientY)?.closest<HTMLElement>("[data-word]");
    if (target) move(dragging, Number(target.dataset.word));
  }

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, key: number) {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    move(key, wordIndex(key) + (event.key === "ArrowLeft" ? -1 : 1));
    // Keep focus on the moved word so it can keep travelling.
    requestAnimationFrame(() => document.querySelector<HTMLElement>(`[data-key="${key}"]`)?.focus());
  }

  const tokens = tokensOf(units);
  let words = -1;

  return (
    <div className="wr">
      <p className="wr-hint">
        Drag a word to a new spot (or select it and press ← →). Punctuation stays put. Try moving a word to the very front.
      </p>
      <div className="wr-row" role="list" aria-label="Words in your sentence">
        {units.map((unit) => {
          if (!unit.word) {
            return (
              <span key={unit.key} className="wr-punct" role="listitem">
                {unit.text}
              </span>
            );
          }
          words += 1;
          const change = changes[unit.key];
          return (
            <button
              key={unit.key}
              type="button"
              role="listitem"
              data-word={words}
              data-key={unit.key}
              className={[dragging === unit.key && "dragging", change && "changed"].filter(Boolean).join(" ")}
              aria-label={`${unit.text}, word ${words + 1}. Use left and right arrows to move.`}
              onPointerDown={(event) => onPointerDown(event, unit.key)}
              onPointerMove={onPointerMove}
              onPointerUp={() => setDragging(null)}
              onPointerCancel={() => setDragging(null)}
              onKeyDown={(event) => onKeyDown(event, unit.key)}
            >
              <span className="wr-word">{unit.text}</span>
              <small aria-live="polite">
                {change ? `${change.from.join("·")} → ${change.to.join("·")}` : tokens[unit.key].join("·")}
              </small>
            </button>
          );
        })}
      </div>
      <div className="wr-actions">
        <button
          type="button"
          onClick={() => {
            const reset = split(original);
            if (build(reset) !== build(units)) commit(reset);
          }}
        >
          Reset order
        </button>
        <span>Numbers under each word are its token IDs. A flash means moving it changed its token.</span>
      </div>
    </div>
  );
}
