import asyncio, json, time, requests, threading, os, sys
from collections import defaultdict, deque
from websocket import WebSocketApp
from telegram import Bot
from http.server import SimpleHTTPRequestHandler, HTTPServer

# ==================== 环境变量配置 ====================
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

PRICE_PCT = float(os.getenv('PRICE_PCT', 3))
VOL_MULT = float(os.getenv('VOL_MULT', 2))
COOLDOWN = int(os.getenv('COOLDOWN', 180))
TIME_WINDOW = int(os.getenv('TIME_WINDOW', 180))
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True') == 'True'

USE_PROXY = os.getenv('USE_PROXY', 'False') == 'True'
PROXY_HOST = os.getenv('PROXY_HOST', '127.0.0.1')
PROXY_PORT = int(os.getenv('PROXY_PORT', 7890))
PROXY_HTTP = f"http://{PROXY_HOST}:{PROXY_PORT}"
PROXIES = {"http": PROXY_HTTP, "https": PROXY_HTTP} if USE_PROXY else None

# ==================== 全局变量 ====================
bot = Bot(token=BOT_TOKEN)
realtime, vol_sum = {}, defaultdict(float)
vol_180, benchmark, previous_data = defaultdict(deque), {}, {}
stop_signal = False
SYMBOLS_ALL, SYMBOLS_TOP20 = [], []

# ==================== 获取交易对 ====================
def get_symbols():
    try:
        data = requests.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=10, proxies=PROXIES).json()
        return [s["symbol"] for s in data["symbols"] if s["contractType"] == "PERPETUAL" and s["status"] == "TRADING"]
    except:
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

def get_top20():
    try:
        data = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=10, proxies=PROXIES).json()
        data = sorted(data, key=lambda x: float(x["quoteVolume"]), reverse=True)
        return [d["symbol"] for d in data[:20]]
    except:
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]

SYMBOLS_ALL = get_symbols()
SYMBOLS_TOP20 = get_top20()
print(f"✅ Top20: {SYMBOLS_TOP20[:5]}... 共 {len(SYMBOLS_TOP20)} 个")
print(f"✅ 全市场币种: {len(SYMBOLS_ALL)} 个")

# ==================== 发送 Telegram ====================
async def send(msg: str):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML", disable_web_page_preview=True)
        print(f"[✅ 已发送] {msg[:60]}...")
    except Exception as e:
        print(f"[❌ Telegram发送失败] {e}")

# ==================== WebSocket ====================
def on_message(ws, raw):
    try:
        data = json.loads(raw)
        if "data" in data:
            data = [data["data"]]
        for d in data:
            s = d.get("s")
            if not s or s not in SYMBOLS_ALL:
                continue
            if "c" in d:
                realtime[s] = {"p": float(d["c"]), "t": time.time()}
            if "q" in d:
                ts, qty = d.get("T", time.time() * 1000) / 1000, float(d["q"])
                vol_180[s].append((ts, qty))
                vol_sum[s] += qty
                while vol_180[s] and ts - vol_180[s][0][0] > TIME_WINDOW:
                    _, old = vol_180[s].popleft()
                    vol_sum[s] -= old
    except Exception:
        pass

def ws_worker(symbols):
    global stop_signal
    streams = "/".join([f"{s.lower()}@miniTicker" for s in symbols])
    url = f"wss://fstream.binance.com/stream?streams={streams}"
    while not stop_signal:
        try:
            ws = WebSocketApp(url, on_message=on_message)
            ws.run_forever(ping_interval=20, ping_timeout=10)
        except:
            time.sleep(5)

# ==================== Binance 数据函数 ====================
def get_klines(symbol, interval="1h", limit=2):
    try:
        url = "https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        return requests.get(url, params=params, timeout=5, proxies=PROXIES).json()
    except:
        return []

def get_ticker_info(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        return requests.get(url, params={"symbol": symbol}, timeout=5, proxies=PROXIES).json()
    except:
        return {}

def get_funding_rate(symbol):
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}
        data = requests.get(url, params=params, timeout=5, proxies=PROXIES).json()
        return float(data[0]["fundingRate"]) * 100
    except:
        return 0

# ==================== 核心逻辑 ====================
async def main():
    await send(f"✅ Binance 异动监控启动 🚀 调试模式: {DEBUG_MODE}")
    timeout = time.time() + 20
    while not realtime and time.time() < timeout:
        await asyncio.sleep(0.5)

    for s in SYMBOLS_TOP20:
        if s in realtime:
            benchmark[s] = {"p": realtime[s]["p"], "v": max(vol_sum[s], 1), "t": time.time()}
            previous_data[s] = {"p": realtime[s]["p"], "v": max(vol_sum[s], 1), "oi": 0}
            print(f"[INIT] {s} 初始价 {realtime[s]['p']}")
        else:
            print(f"⚠️ {s} 暂无初始数据")

    print("✅ 初始化完成，开始监控...\n")

    while not stop_signal:
        now = time.time()
        for s in SYMBOLS_TOP20:
            if s not in realtime:
                continue
            rt = realtime[s]
            p_now = rt["p"]
            v_now = max(vol_sum[s], 1)
            base = benchmark.get(s)
            prev = previous_data.get(s, {"p": p_now, "v": v_now, "oi": 0})
            if not base:
                benchmark[s] = {"p": p_now, "v": v_now, "t": now}
                continue

            pct_3m = (p_now - base["p"]) / base["p"] * 100
            vol_mul = v_now / base["v"]

            if abs(pct_3m) < PRICE_PCT or vol_mul < VOL_MULT:
                continue

            kl_1h = get_klines(s, interval="1h", limit=2)
            kl_4h = get_klines(s, interval="4h", limit=2)
            price_1h = (float(kl_1h[-1][4]) - float(kl_1h[-2][4])) / float(kl_1h[-2][4]) * 100 if len(kl_1h) == 2 else 0
            price_4h = (float(kl_4h[-1][4]) - float(kl_4h[-2][4])) / float(kl_4h[-2][4]) * 100 if len(kl_4h) == 2 else 0

            ticker = get_ticker_info(s)
            open_interest = float(ticker.get("openInterest", 0))
            oi_pct = (open_interest - prev["oi"]) / prev["oi"] * 100 if prev["oi"] > 0 else 0
            funding_rate = get_funding_rate(s)

            arrow = "📈" if pct_3m > 0 else "📉"
            msg = (
                f"{arrow} {s} 永续\n"
                f"当前价 ${p_now:,.6f} {pct_3m:+.2f}% (3m)  3m成交增量 +{vol_mul:.2f}x\n\n"
                f"1h {price_1h:+.2f}% 4h {price_4h:+.2f}%\n\n"
                f"OI 3m增 {oi_pct:+.2f}% 持仓人数 3m {oi_pct:+.2f}%\n"
                f"当前总未平仓 ${open_interest:.2f}B\n"
                f"7d高 ${ticker.get('highPrice', 0)} 7d低 ${ticker.get('lowPrice', 0)}\n"
                f"资金费率 {funding_rate:+.4f}%\n"
                f"https://www.binance.com/en/futures/{s}"
            )
            await send(msg)

            benchmark[s] = {"p": p_now, "v": v_now, "t": now}
            previous_data[s] = {"p": p_now, "v": v_now, "oi": open_interest}

        await asyncio.sleep(5)

# ==================== Render 防休眠 Web 服务 ====================
class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(b"<h2>Binance Monitor Running ✅</h2>")

def keep_alive():
    port = int(os.environ.get("PORT", "10000"))
    print(f"🌐 Web Service Running on Port {port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

if __name__ == "__main__":
    threading.Thread(target=keep_alive, daemon=True).start()
    threading.Thread(target=lambda: ws_worker(SYMBOLS_ALL[:400]), daemon=True).start()
    threading.Thread(target=lambda: ws_worker(SYMBOLS_TOP20), daemon=True).start()
    asyncio.run(main())