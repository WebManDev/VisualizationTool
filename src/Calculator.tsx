import { useState } from "react";

type Op = "+" | "−" | "×" | "÷";

function apply(a: number, op: Op, b: number) {
  if (op === "+") return a + b;
  if (op === "−") return a - b;
  if (op === "×") return a * b;
  return a / b;
}

function show(value: number) {
  if (!Number.isFinite(value)) return "Error";
  return String(parseFloat(value.toPrecision(10)));
}

/** Evaluates a flat expression with × and ÷ before + and −, like a scientific calculator. */
function evaluate(numbers: number[], ops: Op[]) {
  const terms = [numbers[0]];
  const signs: Op[] = [];
  ops.forEach((op, i) => {
    const next = numbers[i + 1];
    if (op === "×" || op === "÷") terms[terms.length - 1] = apply(terms[terms.length - 1], op, next);
    else {
      terms.push(next);
      signs.push(op);
    }
  });
  return signs.reduce((total, op, i) => apply(total, op, terms[i + 1]), terms[0]);
}

/** A small calculator with proper order of operations. `onInsert` copies the display into the answer box in use. */
export function Calculator({ onInsert, target }: { onInsert: (value: string) => void; target: string | null }) {
  const [display, setDisplay] = useState("0");
  const [numbers, setNumbers] = useState<number[]>([]);
  const [ops, setOps] = useState<Op[]>([]);
  const [fresh, setFresh] = useState(true);
  // True only right after an operator key, so a second operator replaces it instead of adding a step.
  const [afterOp, setAfterOp] = useState(false);

  const current = () => (display === "Error" ? 0 : Number(display));

  function digit(d: string) {
    if (fresh || display === "0" || display === "Error") setDisplay(d === "." ? "0." : d);
    else if (!(d === "." && display.includes(".")) && display.length < 14) setDisplay(display + d);
    setFresh(false);
    setAfterOp(false);
  }

  function operator(next: Op) {
    if (afterOp) {
      setOps([...ops.slice(0, -1), next]);
      return;
    }
    setNumbers([...numbers, current()]);
    setOps([...ops, next]);
    setFresh(true);
    setAfterOp(true);
  }

  function equals() {
    if (!ops.length) return;
    setDisplay(show(evaluate([...numbers, current()], ops)));
    setNumbers([]);
    setOps([]);
    setFresh(true);
    setAfterOp(false);
  }

  function unary(fn: (x: number) => number) {
    setDisplay(show(fn(current())));
    setFresh(true);
    setAfterOp(false);
  }

  function clear() {
    setDisplay("0");
    setNumbers([]);
    setOps([]);
    setFresh(true);
    setAfterOp(false);
  }

  const pending = numbers.map((n, i) => `${show(n)} ${ops[i] ?? ""}`).join(" ");

  const keys: [string, () => void, string?][] = [
    ["C", clear, "fn"],
    ["±", () => { setDisplay(show(-current())); setAfterOp(false); }, "fn"],
    ["eˣ", () => unary(Math.exp), "fn"],
    ["÷", () => operator("÷"), "op"],
    ["7", () => digit("7")],
    ["8", () => digit("8")],
    ["9", () => digit("9")],
    ["×", () => operator("×"), "op"],
    ["4", () => digit("4")],
    ["5", () => digit("5")],
    ["6", () => digit("6")],
    ["−", () => operator("−"), "op"],
    ["1", () => digit("1")],
    ["2", () => digit("2")],
    ["3", () => digit("3")],
    ["+", () => operator("+"), "op"],
    ["0", () => digit("0")],
    [".", () => digit(".")],
    ["√", () => unary(Math.sqrt), "fn"],
    ["=", equals, "eq"],
  ];

  return (
    <div className="calc" aria-label="Calculator">
      <div className="calc-screen">
        <small>{pending || " "}</small>
        <output aria-live="polite">{display}</output>
      </div>
      <div className="calc-keys">
        {keys.map(([label, run, kind]) => (
          <button key={label} type="button" className={kind ?? ""} onClick={run} aria-label={label === "eˣ" ? "e to the power" : label}>
            {label}
          </button>
        ))}
      </div>
      <button
        type="button"
        className="calc-insert"
        disabled={!target || display === "Error"}
        onClick={() => onInsert(display)}
        title={target ? "Put this number in the answer box you last clicked" : "Click an answer box first"}
      >
        Insert into answer ↵
      </button>
    </div>
  );
}
