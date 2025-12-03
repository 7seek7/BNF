# 🚀 Binance 异动监控 Web Service（Render 版本）

## ✅ 功能
- 监控币安所有永续合约
- 异动触发条件：
  - 3分钟涨幅 ≥ PRICE_PCT
  - 3分钟成交量倍数 ≥ VOL_MULT
- 每个警报消息包含：
  - 3m / 1h / 4h 涨幅
  - 成交量增量倍数
  - 持仓人数与资金费率变化
- Telegram 实时推送
- Web 保活接口防 Render 休眠

---

## ⚙️ 环境变量设置
| 变量 | 示例值 | 说明 |
|------|----------|------|
| BOT_TOKEN | 你的Bot Token |
| CHAT_ID | 你的Telegram用户ID |
| PRICE_PCT | 3 |
| VOL_MULT | 2 |
| COOLDOWN | 180 |
| TIME_WINDOW | 180 |
| DEBUG_MODE | False |
| USE_PROXY | False |
| PROXY_HOST | 127.0.0.1 |
| PROXY_PORT | 7890 |

---

## 🧩 Render 部署步骤
1. 上传本目录到 GitHub。
2. 在 Render 新建 **Web Service**。
3. 环境：`Python 3.10+`
4. Build Command: *(留空)*
5. Start Command:
   ```bash
   python BNF_render.py
