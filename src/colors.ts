import type { CSSProperties } from "react";

/** Colour scales from the dataviz reference palette: diverging blue ↔ red with a grey midpoint, and a one-hue orange ramp. */
const NEUTRAL = [240, 239, 236]; // #f0efec
const POSITIVE = [37, 106, 191]; // #256abf, blue
const NEGATIVE = [184, 49, 47]; // #b8312f, red
const ID_LOW = [253, 227, 211]; // #fde3d3, light orange
const ID_HIGH = [154, 58, 18]; // #9a3a12, dark orange

/** GPT-2 embedding numbers mostly fall within ±0.3, so that is where colour reaches full strength. */
export const VALUE_RANGE = 0.3;
const VOCAB = 50257;

function mix(from: number[], to: number[], t: number) {
  return from.map((channel, i) => Math.round(channel + (to[i] - channel) * t));
}

function paint(rgb: number[]): CSSProperties {
  const [r, g, b] = rgb.map((c) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  const light = 0.2126 * r + 0.7152 * g + 0.0722 * b;
  // Dark ink on light fills, white on dark fills, whichever has more contrast.
  return { background: `rgb(${rgb.join(",")})`, color: light > 0.18 ? "#1a1a1a" : "#fff" };
}

/** Blue for positive, red for negative; deeper colour means a bigger number. */
export function valueStyle(value: number): CSSProperties {
  const t = Math.min(1, Math.abs(value) / VALUE_RANGE);
  return paint(mix(NEUTRAL, value >= 0 ? POSITIVE : NEGATIVE, t));
}

/** Light orange for low token IDs, dark for high ones (log scale, since IDs run 0 to 50,256). */
export function idStyle(id: number): CSSProperties {
  return paint(mix(ID_LOW, ID_HIGH, Math.log1p(id) / Math.log1p(VOCAB - 1)));
}

export const VALUE_GRADIENT = `linear-gradient(90deg, rgb(${NEGATIVE}), rgb(${NEUTRAL}), rgb(${POSITIVE}))`;
export const ID_GRADIENT = `linear-gradient(90deg, rgb(${ID_LOW}), rgb(${ID_HIGH}))`;
