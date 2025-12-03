import asyncio
import json
import time
import requests
import threading
import sys
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from collections import defaultdict, deque
from websocket import WebSocketApp
from telegram import Bot

# ==================== 环境变量配置 ====================
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

PRICE_PCT = float(os.getenv('PRICE_PCT', 3))       # 价格异动阈值（百分比）
VOL_MULT = float(os.getenv('VOL_MULT', 2))         # 成交量倍数阈值
COOLDOWN = int(os.getenv('COOLDOWN', 180))         # 报警冷却时间
TIME_WINDOW = int(os.getenv('TIME_WINDOW', 180))   # 成交量统计窗口（秒）
DEBUG_MODE = os.getenv('DEBUG_MODE', 'True') == 'True'
USE_PROXY = os.getenv('USE_PROXY', 'False') == 'True'

PROXY_HOST = os.getenv('PROXY_HOST', '127.0.0.1')
PROXY_PORT = int(os.getenv('PROXY_PORT', 7890))
PROXY_HTTP = f"http://{PROXY_HOST}:{PROXY_PORT}"
PROXIES = {"http": PROXY_HTTP, "https": PROXY_HTTP} if USE_PROXY else None

# ==================== 初始化全局变量 ====================
bot = Bot(token=BOT_TOKEN)
realtime = {}
vol_180 = defaultdict(deque)
vol_sum = defaultdict(float)
benchmark = {}
last_alert = {}
previous_data = {}
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
print(f"✅ 获取市值前20币: {SYMBOLS_TOP20[:5]}... 共 {len(SYMBOLS_TOP20)} 个")
print(f"✅ 全市场币种: {len(SYMBOLS_ALL)} 个")

# ==================== 发送 Telegram 消息 ====================
async def send(msg: str):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode="HTML", disable_web_page_preview=True)
        print(f"[✅ 已发送] {msg[:60]}...")
    except Exception as e:
        print(f"[❌ Telegram发送失败] {e}")

# ==================== WebSocket 数据处理 ====================
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
                p = float(d.get("c", 0))
                if p > 0:
                    realtime[s] = {"p": p, "t": time.time()}
            if "q" in d:
                ts, qty = d.get("T", time.time()*1000)/1000, float(d["q"])
                q = vol_180[s]
                q.append((ts, qty))
                vol_sum[s] += qty
                while q and ts - q[0][0] > TIME_WINDOW:
                    _, old = q.popleft()
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
        except Exception as e:
            print(f"[WS异常] {e}, 5秒后重连")
            time.sleep(5)

# ==================== 获取币安数据函数 ====================
def get_klines(symbol, interval="1h", limit=2):
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        response = requests.get(url, params=params, proxies=PROXIES)
        return response.json()
    except Exception as e:
        print(f"[❌ 获取K线数据失败] {e}")
        return []

def get_ticker_info(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/ticker/24hr"
        params = {"symbol": symbol}
        response = requests.get(url, params=params, proxies=PROXIES)
        return response.json()
    except Exception as e:
        print(f"[❌ 获取市场数据失败] {e}")
        return {}

def get_funding_rate(symbol):
    try:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate"
        params = {"symbol": symbol, "limit": 1}
        response = requests.get(url, params=params, proxies=PROXIES)
        data = response.json()
        return float(data[0]["fundingRate"]) * 100  # 转为百分比
    except Exception as e:
        print(f"[❌ 获取资金费率失败] {e}")
        return 0

# ==================== 主监控逻辑 ====================
async def main():
    await send(f"✅ Binance 异动监控已启动（重点币+全币）🚀 调试模式: {DEBUG_MODE}")
    print("等待 WebSocket 数据稳定中...")

    timeout = time.time() + 20
    while not realtime and time.time() < timeout:
        await asyncio.sleep(0.5)

    for s in SYMBOLS_TOP20:
        if s in realtime:
            benchmark[s] = {"p": realtime[s]["p"], "v": max(vol_sum[s], 1)}
            previous_data[s] = {"p": realtime[s]["p"], "v": max(vol_sum[s], 1), "oi": 0}
            print(f"[INIT] {s} 初始价 {realtime[s]['p']}")
        else:
            print(f"⚠️ {s} 暂无初始数据")

    print("✅ 初始化完成，开始实时监控...\n")

    while not stop_signal:
        now = time.time()
        for s in SYMBOLS_TOP20:
            if s not in realtime:
                continue

            p_now = realtime[s]["p"]
            v_now = max(vol_sum[s], 1)
            base = benchmark.get(s)
            if not base:
                benchmark[s] = {"p": p_now, "v": v_now}
                continue

            pct = (p_now - base["p"]) / base["p"] * 100
            mul = v_now / base["v"]

            # 仅在触发阈值时才请求额外数据
            if abs(pct) >= PRICE_PCT and mul >= VOL_MULT and now - last_alert.get(s, 0) >= COOLDOWN:
                klines_1h = get_klines(s, "1h", 2)
                klines_4h = get_klines(s, "4h", 2)
                price_1h = (float(klines_1h[1][4]) - float(klines_1h[0][4])) / float(klines_1h[0][4]) * 100 if len(klines_1h) == 2 else 0
                price_4h = (float(klines_4h[1][4]) - float(klines_4h[0][4])) / float(klines_4h[0][4]) * 100 if len(klines_4h) == 2 else 0

                ticker_info = get_ticker_info(s)
                open_interest = float(ticker_info.get("openInterest", 0))
                oi_change = 0
                if previous_data[s]["oi"] > 0:
                    oi_change = (open_interest - previous_data[s]["oi"]) / previous_data[s]["oi"] * 100

                funding_rate = get_funding_rate(s)
                arrow = "📈" if pct > 0 else "📉"

                msg = (
                    f"{arrow} {s} 永续\n"
                    f"当前价 ${p_now:,.6f}　{pct:+.2f}% (3m)  3m成交增量 +{mul:.2f}x\n\n"
                    f"1h {price_1h:+.2f}%　4h {price_4h:+.2f}%\n\n"
                    f"OI 3m增 {oi_change:+.2f}%　持仓人数 3m {oi_change:+.2f}%\n"
                    f"当前总未平仓 ${open_interest:.2f}B\n"
                    f"7d高 ${ticker_info.get('highPrice', 0)}　7d低 ${ticker_info.get('lowPrice', 0)}\n\n"
                    f"资金费率 {funding_rate:+.4f}%\n"
                    f"https://www.binance.com/en/futures/{s}"
                )
                await send(msg)
                last_alert[s] = now
                benchmark[s] = {"p": p_now, "v": v_now}
                previous_data[s] = {"p": p_now, "v": v_now, "oi": open_interest}
        await asyncio.sleep(3 if DEBUG_MODE else 1)

# ==================== HTTP服务防休眠 ====================
class KeepAliveHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write("<h2>Binance 监控器 正在运行 ✅</h2>".encode("utf-8"))

def start_server():
    port = int(os.getenv("PORT", 10000))
    server = HTTPServer(("0.0.0.0", port), KeepAliveHandler)
    print(f"🌐 Web Service Running on Port {port}")
    server.serve_forever()

# ==================== 自动KeepAlive Ping ====================
def keep_alive_ping():
    url = f"https://{os.getenv('RENDER_EXTERNAL_HOSTNAME', 'your-service-name.onrender.com')}"
    while True:
        try:
            r = requests.get(url, timeout=10)
            print(f"[KeepAlive] Ping {url} - {r.status_code}")
        except Exception as e:
            print(f"[KeepAlive Error] {e}")
        time.sleep(300)

# ==================== 启动 ====================
if __name__ == "__main__":
    threading.Thread(target=start_server, daemon=True).start()
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=lambda: ws_worker(SYMBOLS_ALL[:400]), daemon=True).start()
    threading.Thread(target=lambda: ws_worker(SYMBOLS_TOP20), daemon=True).start()
    asyncio.run(main())
