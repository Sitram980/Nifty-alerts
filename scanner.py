"""Daily NIFTY 500 scan: multi-kernel regression slope-flip BUY/SELL signals -> Telegram."""
import os, math, csv, io
import numpy as np, pandas as pd, requests, yfinance as yf

BW = int(os.getenv("BANDWIDTH", "14"))
KERNEL = os.getenv("KERNEL", "laplace").lower()
TOP_N = int(os.getenv("TOP_N", "15"))
UA = {"User-Agent": "Mozilla/5.0"}
URLS = [
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://huggingface.co/spaces/riteshcp/Market_Analysis_Tool/resolve/main/ind_nifty500list.csv",
]
K = {
    "laplace": lambda x, b: np.exp(-abs(x / b)) / (2 * b),
    "gaussian": lambda x, b: np.exp(-(x * x / (b * b)) / 2) / math.sqrt(2 * math.pi),
    "triangular": lambda x, b: max(0.0, 1 - abs(x / b)),
    "epanechnikov": lambda x, b: .75 * (1 - x * x / (b * b)) if abs(x / b) <= 1 else 0.0,
    "cauchy": lambda x, b: 1 / (math.pi * b * (1 + x * x / (b * b))),
}

def universe():
    for u in URLS:
        try:
            r = requests.get(u, headers=UA, timeout=30); r.raise_for_status()
            rows = list(csv.DictReader(io.StringIO(r.text)))
            syms = [x["Symbol"].strip() for x in rows if x.get("Symbol")]
            if len(syms) >= 450: return syms
        except Exception as e:
            print("universe source failed:", u, e)
    raise SystemExit("Could not load NIFTY 500 list")

def download(tickers):
    out = {}
    for i in range(0, len(tickers), 100):
        chunk = tickers[i:i + 100]
        d = yf.download(chunk, period="1y", interval="1d", group_by="ticker",
                        auto_adjust=False, threads=True, progress=False)
        for t in chunk:
            try:
                df = d[t].dropna(subset=["Close"]) if len(chunk) > 1 else d.dropna(subset=["Close"])
                if len(df) >= BW + 2: out[t] = df
            except Exception:
                pass
    return out

def signal(close):
    fn = K[KERNEL]
    w = np.array([fn(i * i / (BW * BW), 1.0) for i in range(BW)])
    k = lambda t: float(np.dot(close[t - BW + 1:t + 1][::-1], w) / w.sum())
    n = len(close) - 1
    d, pd_ = k(n) - k(n - 1), k(n - 1) - k(n - 2)
    if d > 0 and pd_ <= 0: return "BUY"
    if d < 0 and pd_ >= 0: return "SELL"
    return "NEUTRAL"

def ret20(close):
    return 100 * (close[-1] / close[-21] - 1) if len(close) > 21 else 0.0

def send(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    print(text)
    if not (tok and chat):
        print("(Telegram not configured)"); return
    for i in range(0, len(text), 3800):
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      data={"chat_id": chat, "text": text[i:i + 3800]}, timeout=30)

def main():
    syms = universe()
    data = download([s + ".NS" for s in syms] + ["^CRSLDX"])
    bench = ret20(data["^CRSLDX"]["Close"].values) if "^CRSLDX" in data else 0.0
    buys, sells, date = [], [], ""
    for t, df in data.items():
        if t == "^CRSLDX": continue
        c = df["Close"].values.astype(float)
        date = str(df.index[-1].date())
        s = signal(c)
        rs = ret20(c) - bench
        if s == "BUY": buys.append((rs, t[:-3], c[-1]))
        elif s == "SELL": sells.append((rs, t[:-3], c[-1]))
    buys.sort(reverse=True); sells.sort()
    line = lambda x: f"{x[1]}  ₹{x[2]:.2f}  RS {x[0]:+.1f}%"
    msg = [f"NIFTY 500 scan - {date}", f"Kernel {KERNEL} / {BW}",
           f"BUY signals: {len(buys)} | SELL signals: {len(sells)}", "",
           "BUY (strongest RS first):"] + [line(x) for x in buys[:TOP_N]] + \
          ["", "SELL (weakest RS first):"] + [line(x) for x in sells[:TOP_N]] + \
          ["", "Not investment advice."]
    send("\n".join(msg))

if __name__ == "__main__":
    main()
