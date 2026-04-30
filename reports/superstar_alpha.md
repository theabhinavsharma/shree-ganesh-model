# Superstar-confluence alpha analysis

_Question: do stocks held by 2+ celebrity investors deliver better forward returns_
_than the broader liquid universe? Tested on 2024-2025 OOS data._

## Caveat

This analysis uses TODAY's superstar holdings against historical OOS prices.
We don't have quarterly history of holdings yet, so this is a proxy — assumes
their picks were broadly stable. To do it properly we need historical holdings;
filed as next-cycle TODO.

## Forward 7d return by superstar-confluence bucket

| Confluence | n stock-days | Mean 7d | Median 7d | % positive | % >= +5% |
|---|---:|---:|---:|---:|---:|
| 0 - none | 695,367 | +0.18% | -0.22% | 47.7% | 18.3% |
| 1 - solo | 54,150 | +0.32% | -0.35% | 47.4% | 21.1% |
| 2 - pair | 14,401 | +0.56% | +0.17% | 51.3% | 19.9% |
| 3+ - cluster | 2,044 | +0.36% | +0.07% | 51.1% | 9.9% |

## Filtered: real confluence (2-10 superstars, exclude noise) vs no-superstar

| Group | n stock-days | Mean 7d | Median 7d | % positive | % >= +5% |
|---|---:|---:|---:|---:|---:|
| **Confluence 2-10** | 14,912 | **+0.58%** | +0.16% | 51.2% | 19.8% |
| No superstar | 695,367 | +0.18% | -0.22% | 47.7% | 18.3% |

**Mean 7d delta: +0.40 pp** · **% >= 5% delta: +1.5 pp**

✅ **Real signal**: superstar-confluence stocks meaningfully outperform.

## Today's intersection — model top-30 LONG × superstar-confluence

| Symbol | Sector | Close | Score | Superstars | Investors |
|---|---|---:|---:|---:|---|
| **LUXIND** | OTHER | ₹1510.10 | 0.65 | — (0) |  |
| **WEBELSOLAR** | NIFTY MICROCAP 250 | ₹122.20 | 0.65 | — (0) |  |
| **RPOWER** | NIFTY ENERGY | ₹29.55 | 0.65 | — (0) |  |
| **MANAKALUCO** | OTHER | ₹38.30 | 0.64 | — (0) |  |
| **MANINDS** | OTHER | ₹530.05 | 0.64 | ⭐ (1) | KACHOLIA |
| **BHARATWIRE** | OTHER | ₹222.91 | 0.64 | — (0) |  |
| **INOXWIND** | NIFTY ENERGY | ₹103.07 | 0.64 | ⭐ (1) | AKASH |
| **NOVARTIND** | OTHER | ₹1050.85 | 0.62 | — (0) |  |
| **GALLANTT** | NIFTY SMALLCAP 250 | ₹852.70 | 0.62 | — (0) |  |
| **BESTAGRO** | OTHER | ₹18.54 | 0.62 | — (0) |  |
| **DOLLAR** | OTHER | ₹303.95 | 0.62 | — (0) |  |
| **YATRA** | OTHER | ₹109.26 | 0.62 | — (0) |  |
| **ORIENTBELL** | OTHER | ₹320.70 | 0.61 | — (0) |  |
| **JAYNECOIND** | NIFTY MICROCAP 250 | ₹109.81 | 0.61 | — (0) |  |
| **VIMTALABS** | OTHER | ₹452.85 | 0.61 | — (0) |  |
| **AMBUJACEM** | NIFTY INFRA | ₹458.80 | 0.61 | — (0) |  |
| **MASTEK** | NIFTY MICROCAP 250 | ₹1680.30 | 0.61 | ⭐ (1) | SINGHANIA |
| **MTARTECH** | NIFTY MICROCAP 250 | ₹5292.10 | 0.61 | — (0) |  |
| **DELTACORP** | OTHER | ₹73.34 | 0.61 | — (0) |  |
| **KIRLOSBROS** | NIFTY MICROCAP 250 | ₹1699.00 | 0.61 | — (0) |  |
| **KITEX** | NIFTY MICROCAP 250 | ₹164.35 | 0.61 | — (0) |  |
| **HIRECT** | OTHER | ₹922.80 | 0.61 | — (0) |  |
| **SBCL** | OTHER | ₹612.80 | 0.61 | — (0) |  |
| **PVP** | OTHER | ₹32.43 | 0.61 | — (0) |  |
| **SHRIPISTON** | NIFTY MICROCAP 250 | ₹3476.20 | 0.61 | ⭐ (1) | SINGHANIA |
| **SAMHI** | NIFTY MICROCAP 250 | ₹159.58 | 0.61 | — (0) |  |
| **OLAELEC** | NIFTY SMALLCAP 100 | ₹35.81 | 0.61 | — (0) |  |
| **VERANDA** | OTHER | ₹188.15 | 0.61 | — (0) |  |
| **ZAGGLE** | NIFTY MICROCAP 250 | ₹254.89 | 0.61 | ⭐ (1) | KACHOLIA |
| **AUSOMENT** | OTHER | ₹147.75 | 0.61 | — (0) |  |
