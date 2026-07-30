# pyright: reportCallIssue=false
# pyright: reportUndefinedVariable=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportAttributeAccessIssue=false
from __future__ import annotations
import asyncio
import threading
import time
import telegram
from telegram.error import TimedOut
from framework.core.config import get_main_config
from config.settings import Settings
from utils.logger import Logger

settings = Settings()

logger = Logger.get_logger('telegram')

class TelegramBot:
    """Telegram机器人类"""
    
    # pyright: reportAttributeAccessIssue=false
    main_config = None  # 类属性初始化
    
    def __init__(self, bot_type='alert'):
        """
        初始化
        :param bot_type: 机器人类型 'alert' 或 'trade'
        """
        from framework.core.config import get_config_manager
        
        self.bot_type = bot_type
        
        # 加载配置
        config_manager = get_config_manager()
        if not config_manager.load_main_config():
            logger.error("主配置加载失败")
        
        main_config = config_manager.main_config
        
        # 存储到实例属性
        self.main_config = main_config
        self.chat_id = int(main_config.telegram_chat_id)
        
        if bot_type == 'alert':
            token = main_config.telegram_bot_token_alert
        else:
            token = main_config.telegram_bot_token_trade
        
        self.bot: telegram.Bot = telegram.Bot(token=token)

        # 创建事件循环
        self._loop_started = False
        self.loop = asyncio.new_event_loop()
        self.loop_thread = None
        self._start_loop()

        # 用于图表缓存的目录
        self._chart_dir = None

        # 命令接收（polling）
        self.command_handler = None
        self._polling = False
        self._polling_thread = None
        self._last_update_id = 0

        # 交易按钮回调
        self.trade_executor = None
    
    def _start_loop(self):
        """在单独的线程中启动事件循环"""
        if self._loop_started:
            return
            
        def run_loop():
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
        self._loop_started = True
        logger.info(f"[{self.bot_type}] 事件循环已启动")
    
    async def _send_message_async(self, message, parse_mode='HTML'):
        """内部异步发送消息"""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=parse_mode,
                disable_web_page_preview=True
            )
            logger.info(f"[{self.bot_type}] 消息发送成功")
            return True
        except TimedOut as e:
            logger.error(f"[{self.bot_type}] 发送超时: {str(e)}")
            raise e from e
        except Exception as e:
            logger.error(f"[{self.bot_type}] 发送消息失败: {str(e)}")
            raise e from e
    
    def send_message_sync(self, message, parse_mode='HTML'):
        """同步发送消息的封装，带指数退避重试"""
        max_retries = 5
        base_delay = 1  # 基础延迟1秒
        
        for attempt in range(1, max_retries + 1):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_message_async(message, parse_mode),
                    self.loop
                )
                result = future.result(timeout=30)
                return result
            except asyncio.TimeoutError:
                logger.error(f"[{self.bot_type}] 发送超时（{attempt}/{max_retries}）")
                if attempt < max_retries:
                    # 指数退避: 1s, 2s, 4s, 8s, 16s
                    delay = base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
            except Exception as e:
                logger.error(f"[{self.bot_type}] 调用发送失败，尝试 {attempt}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    # 指数退避
                    delay = base_delay * (2 ** (attempt - 1))
                    time.sleep(delay)
        return False
    
    def send_message(self, message, parse_mode='HTML'):
        """同步发送消息（对外接口）"""
        try:
            return self.send_message_sync(message)
        except Exception as e:
            logger.error(f"[{self.bot_type}] 消息发送失败: {str(e)}")
            return False

    async def _send_photo_async(self, photo_bytes, caption='', parse_mode='HTML', reply_markup=None):
        """内部异步发送图片（photo_bytes为已读入内存的bytes）"""
        try:
            from io import BytesIO
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=BytesIO(photo_bytes),
                caption=caption,
                parse_mode=parse_mode,
                reply_markup=reply_markup,
            )
            return True
        except TimedOut as e:
            logger.error(f"[{self.bot_type}] 图片发送超时: {str(e)}")
            raise e from e
        except Exception as e:
            logger.error(f"[{self.bot_type}] 图片发送失败: {str(e)}")
            raise e from e

    async def _edit_caption_async(self, chat_id, message_id, caption, reply_markup=None):
        """编辑消息标题（用于按钮响应后更新）"""
        try:
            await self.bot.edit_message_caption(
                chat_id=chat_id, message_id=message_id,
                caption=caption, reply_markup=reply_markup,
            )
        except Exception as e:
            logger.error(f"[{self.bot_type}] 编辑消息失败: {str(e)}")

    def send_photo_sync(self, photo_input, caption='', parse_mode='HTML', reply_markup=None):
        """同步发送图片，带指数退避重试

        Args:
            photo_input: 文件路径(str) 或 BytesIO 或 bytes
        """
        max_retries = 3
        base_delay = 1

        # 支持多种输入类型：文件路径 / BytesIO / bytes
        try:
            if isinstance(photo_input, (bytes, bytearray)):
                photo_bytes = photo_input
            elif hasattr(photo_input, 'read'):
                photo_bytes = photo_input.read()
            else:
                with open(str(photo_input), 'rb') as f:
                    photo_bytes = f.read()
        except Exception as e:
            logger.error(f"[{self.bot_type}] 读取图片失败: {str(e)}")
            return False

        for attempt in range(1, max_retries + 1):
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._send_photo_async(photo_bytes, caption, parse_mode, reply_markup),
                    self.loop
                )
                result = future.result(timeout=60)
                return result
            except asyncio.TimeoutError:
                logger.error(f"[{self.bot_type}] 图片发送超时（{attempt}/{max_retries}）")
                if attempt < max_retries:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
            except Exception as e:
                logger.error(f"[{self.bot_type}] 图片发送失败，尝试 {attempt}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(base_delay * (2 ** (attempt - 1)))
        return False

    def send_photo(self, photo_path, caption='', parse_mode='HTML'):
        """同步发送图片（对外接口）"""
        try:
            return self.send_photo_sync(photo_path, caption, parse_mode)
        except Exception as e:
            logger.error(f"[{self.bot_type}] 图片发送失败: {str(e)}")
            return False

    def send_signal_chart(self, df, symbol, entry_price, side='LONG',
                          signal_type='', trend_type='', stop_loss=None,
                          leverage=15, entry_sup=None, entry_res=None,
                          tl_slope=None, tl_base_idx=None, tl_base_val=None,
                          triangle_up=None, triangle_lo=None,
                          caption_extra='', signal_id='',
                          with_trade_button=False, strategy_name=''):
        """生成信号K线图并发送至Telegram

        Args:
            df: K线DataFrame（含open/high/low/close），末尾为信号K线
            symbol: 币种
            entry_price: 入场价
            side: LONG/SHORT
            signal_type: 信号类型
            trend_type: 趋势类型
            stop_loss: 止损价
            leverage: 杠杆
            entry_sup: 支撑位
            entry_res: 阻力位
            tl_slope: 趋势线斜率
            tl_base_idx: 趋势线起始bar索引
            tl_base_val: 趋势线起始价格
            triangle_up: 三角上边界
            triangle_lo: 三角下边界
            caption_extra: 附加消息文字
            signal_id: 信号唯一ID（用于按钮回调）
            with_trade_button: 是否显示交易按钮
        """
        try:
            from framework.shared.chart_utils import plot_signal_chart
            chart_path = plot_signal_chart(
                df, symbol, entry_price, side=side,
                signal_type=signal_type, trend_type=trend_type,
                stop_loss=stop_loss, leverage=leverage,
                entry_sup=entry_sup, entry_res=entry_res,
                tl_slope=tl_slope, tl_base_idx=tl_base_idx,
                tl_base_val=tl_base_val,
                triangle_up=triangle_up, triangle_lo=triangle_lo,
                strategy_name=strategy_name,
            )
            if chart_path:
                _name_tag = f'[{strategy_name}] ' if strategy_name else ''
                side_str = '做多' if side == 'LONG' else '做空'
                caption = f'{_name_tag}<b>【{symbol}】{signal_type} {side_str}</b>'
                if caption_extra:
                    caption += f'\n{caption_extra}'
                reply_markup = None
                if with_trade_button and signal_id:
                    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
                    btn = InlineKeyboardButton(
                        '✅ 交易', callback_data=f'trade_{signal_id}'
                    )
                    reply_markup = InlineKeyboardMarkup([[btn]])
                return self.send_photo_sync(chart_path, caption=caption, reply_markup=reply_markup)
        except Exception as e:
            logger.error(f"[{self.bot_type}] 信号图表发送失败: {str(e)}")
            return False

    def edit_signal_caption(self, chat_id, message_id, new_caption, remove_keyboard=False):
        """编辑信号消息标题（标记已执行/已过期）"""
        reply_markup = None
        if remove_keyboard:
            from telegram import InlineKeyboardMarkup
            reply_markup = InlineKeyboardMarkup([])
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._edit_caption_async(chat_id, message_id, new_caption, reply_markup),
                self.loop
            )
            future.result(timeout=15)
            return True
        except Exception as e:
            logger.error(f"[{self.bot_type}] 编辑信号消息失败: {str(e)}")
            return False

    def set_trade_executor(self, executor):
        """设置交易执行回调

        Args:
            executor: 回调函数 executor(signal_data: dict) -> bool
        """
        self.trade_executor = executor

    def send_alert(self, symbol, alert_data):
        """发送警报消息（符合需求文档模板）"""
        try:
            # 确保数值类型正确转换
            price_change = float(alert_data.get('price_change', 0)) if alert_data.get('price_change', 0) != 0 else 0
            direction = "上涨" if price_change > 0 else "下跌"
            direction_emoji = "📈" if price_change > 0 else "📉"
            price_change_sign = "+" if price_change >= 0 else ""

            # 统计24小时内该方向的次数
            alert_count = int(alert_data.get('alert_count', 1))

            # 资金费率
            funding_rate = alert_data.get('funding_rate', 'N/A')
            if funding_rate != 'N/A' and funding_rate is not None:
                funding_rate_float = float(str(funding_rate).replace('%', ''))
                funding_emoji = "📈" if funding_rate_float >= 0 else "📉"
            else:
                funding_emoji = ""

            # 1小时和4小时价格变化
            price_change_1h = float(alert_data.get('price_change_1h', 0)) if alert_data.get('price_change_1h', 0) != 0 else 0
            price_change_4h = float(alert_data.get('price_change_4h', 0)) if alert_data.get('price_change_4h', 0) != 0 else 0
            price_1h_emoji = "📈" if price_change_1h >= 0 else "📉"
            price_4h_emoji = "📈" if price_change_4h >= 0 else "📉"

            # 持仓增量
            oi_change_pct = float(alert_data.get('oi_change_pct', 0)) if alert_data.get('oi_change_pct', 0) != 0 else 0
            oi_emoji = "📈" if oi_change_pct >= 0 else "📉"

            # 格式化成交量（K、M、B）
            volume_usdt = float(alert_data.get('volume_usdt', 0))
            if volume_usdt >= 1e9:
                volume_str = f"{volume_usdt / 1e9:.2f}B"
            elif volume_usdt >= 1e6:
                volume_str = f"{volume_usdt / 1e6:.2f}M"
            elif volume_usdt >= 1e3:
                volume_str = f"{volume_usdt / 1e3:.2f}K"
            else:
                volume_str = f"{volume_usdt:.2f}"

            # 格式化价格（根据价格大小决定小数位数）
            current_price = float(alert_data.get('current_price', 0))
            if current_price >= 1000:
                current_price_str = f"{current_price:,.2f}"
            elif current_price >= 1:
                current_price_str = f"{current_price:.2f}"
            else:
                current_price_str = f"{current_price:.8f}"

# 按开发文档格式输出消息
            message = f"""<b>{symbol} （{direction_emoji}{direction}24h第{alert_count}次）</b>

<b>当前价格</b>: ${current_price_str}
<b>价格涨跌</b>: {price_change_sign}{price_change:.2f}% {direction_emoji} ({self.main_config.monitor_interval if hasattr(self.main_config, 'monitor_interval') else 120}分钟)
<b>成交量涨幅</b>: 📊 {float(alert_data.get('volume_ratio', 0)):.2f}x ({self.main_config.monitor_interval if hasattr(self.main_config, 'monitor_interval') else 120}分钟)

<b>当前成交量</b>: {volume_str} USDT
<b>持仓量</b>: {alert_data.get('open_interest', 'N/A')}（{oi_change_pct:+.2f}% {oi_emoji}）
<b>资金费率</b>: {funding_rate} {funding_emoji}
<b>1小时价格</b>: {price_1h_emoji} {price_change_1h:+.2f}% {price_1h_emoji}
<b>4小时价格</b>: {price_4h_emoji} {price_change_4h:+.2f}% {price_4h_emoji}

时间戳: {alert_data.get('timestamp', '')}
<a href="https://www.binance.com/zh-CN/futures/{symbol}">链接</a>"""

            if not self.send_message(message):
                logger.warning(f"警报消息可能未送达: {symbol}")
            else:
                logger.info(f"警报消息已发送: {symbol}")
        except Exception as e:
            logger.error(f"发送警报消息失败: {str(e)}")
    
    def send_trade_message(self, symbol, trade_data):
        """
        发送交易消息（符合需求文档模板）
        """
        try:
            # 确保数值类型正确
            side = trade_data.get('side', 'LONG')
            side_str = "做多" if side == 'LONG' else "做空"
            side_emoji = "🚀" if side == 'LONG' else "⚡"

            pnl = float(trade_data.get('pnl', 0)) if trade_data.get('pnl', 0) != 0 else 0
            pnl_percent = float(trade_data.get('pnl_percent', 0)) if trade_data.get('pnl_percent', 0) != 0 else 0
            pnl_emoji = "🚀" if pnl >= 0 else "⚡"
            pnl_sign = "+" if pnl >= 0 else ""

            # 计算仓位比例（方向+动作）
            status = trade_data.get('status', '持仓中')
            
            # ✅ 修复：在分支前初始化 avg_entry_price，避免变量未定义错误
            avg_entry_price = float(trade_data.get('avg_entry_price', 0))
            
            if '建仓' in status:
                # 建仓消息：方向+动作
                action_str = f"{side_str}建仓"
                position_ratio = float(trade_data.get('position_ratio', 0))
                position_ratio_str = f"建仓{position_ratio:.0f}%"
            elif '加仓' in status:
                # 加仓消息：方向+动作
                action_str = f"{side_str}加仓"
                position_ratio = float(trade_data.get('position_ratio', 0))
                position_ratio_str = f"加仓至{position_ratio:.0f}%"
            elif '平仓' in status:
                # 平仓消息：不显示方向
                action_str = "平仓"
                position_ratio = float(trade_data.get('position_ratio', 0))
                position_ratio_str = f"平仓至{position_ratio:.0f}%"
            else:
                # 持仓中
                action_str = side_str
                position_ratio = float(trade_data.get('position_ratio', 100))
                if position_ratio < 100:
                    position_ratio_str = f"剩余{position_ratio:.0f}%仓位"
                else:
                    position_ratio_str = "100%"
            current_price = float(trade_data.get('current_price', 0))

            # 根据价格决定小数位数
            if avg_entry_price >= 1000:
                avg_entry_str = f"{avg_entry_price:,.2f}"
            elif avg_entry_price >= 1:
                avg_entry_str = f"{avg_entry_price:.2f}"
            else:
                avg_entry_str = f"{avg_entry_price:.8f}"

            if current_price >= 1000:
                current_str = f"{current_price:,.2f}"
            elif current_price >= 1:
                current_str = f"{current_price:.2f}"
            else:
                current_str = f"{current_price:.8f}"

            # 按开发文档格式输出消息
            message = f"""<b>【{symbol}】</b>: {trade_data.get('leverage', 20)}X
<b>【方向】</b>: {action_str}
<b>【仓位】</b>: {float(trade_data.get('position_usdt', 0)):,.2f} USDT（{position_ratio_str}）
<b>【开仓均价】</b>: {avg_entry_str}
<b>【当前价】</b>: {current_str}
<b>【保证金】</b>: {float(trade_data.get('margin', 0)):,.2f} USDT
<b>【收益额】</b>: {pnl_sign}{pnl:,.2f} USDT({pnl_sign}{pnl_percent:.2f}%)"""

            if not self.send_message(message):
                logger.warning(f"交易消息可能未送达: {symbol}")
            else:
                logger.info(f"交易消息已发送: {symbol}")
        except Exception as e:
            logger.error(f"发送交易消息失败: {str(e)}")

    def start_command_polling(self, handler):
        """启动命令轮询（接收Telegram消息作为命令）

        Args:
            handler: 回调函数 handler(chat_id, text) -> str(回复消息)
        """
        self.command_handler = handler
        if self._polling:
            return
        self._polling = True
        self._last_update_id = 0

        def poll_loop():
            while self._polling:
                try:
                    future = asyncio.run_coroutine_threadsafe(
                        self._poll_updates(), self.loop
                    )
                    future.result(timeout=30)
                except asyncio.TimeoutError:
                    pass
                except Exception as e:
                    logger.error(f"[{self.bot_type}] 命令轮询异常: {e}")
                time.sleep(2)

        self._polling_thread = threading.Thread(target=poll_loop, daemon=True)
        self._polling_thread.start()
        logger.info(f"[{self.bot_type}] 命令轮询已启动")

    async def _poll_updates(self):
        """轮询新消息和按钮回调"""
        try:
            updates = await self.bot.get_updates(
                offset=self._last_update_id,
                timeout=10,
                allowed_updates=['message', 'callback_query']
            )
            for update in updates:
                if update.update_id >= self._last_update_id:
                    self._last_update_id = update.update_id + 1

                # 消息命令
                if update.message and update.message.text:
                    text = update.message.text.strip()
                    chat = update.message.chat_id
                    if chat != self.chat_id:
                        logger.info(f"[{self.bot_type}] 忽略非授权chat_id: {chat}")
                        continue
                    logger.info(f"[{self.bot_type}] 收到命令: {text}")
                    if self.command_handler:
                        reply = self.command_handler(chat, text)
                        if reply:
                            await self._send_message_async(reply)

                # 按钮回调
                if update.callback_query:
                    cq = update.callback_query
                    data = cq.data or ''
                    chat_id = cq.message.chat_id if cq.message else 0
                    msg_id = cq.message.message_id if cq.message else 0

                    if data.startswith('trade_'):
                        signal_id = data[len('trade_'):]
                        await self._handle_trade_callback(chat_id, msg_id, signal_id, cq)
                    else:
                        await cq.answer('未知操作')
        except TimedOut:
            pass
        except Exception as e:
            logger.error(f"[{self.bot_type}] 轮询消息失败: {e}")

    async def _handle_trade_callback(self, chat_id, msg_id, signal_id, cq):
        """处理交易按钮回调"""
        from framework.shared.signal_store import pop_signal
        signal_data = pop_signal(signal_id)
        if signal_data is None:
            await cq.answer('信号已过期', show_alert=True)
            try:
                await self.bot.edit_message_caption(
                    chat_id=chat_id, message_id=msg_id,
                    caption=f'{cq.message.caption}\n\n⏰ 已过期',
                )
            except Exception:
                logger.debug(f"编辑过期消息失败: {msg_id}")
            return

        await cq.answer('交易执行中...')

        if self.trade_executor:
            success = self.trade_executor(signal_data)
            if success:
                new_caption = f'{cq.message.caption}\n\n✅ 已执行'
                try:
                    await self.bot.edit_message_caption(
                        chat_id=chat_id, message_id=msg_id,
                        caption=new_caption,
                    )
                except Exception:
                    logger.debug(f"编辑成功消息失败: {msg_id}")
                logger.info(f"[{self.bot_type}] 交易已执行: {signal_id}")
            else:
                new_caption = f'{cq.message.caption}\n\n❌ 执行失败'
                try:
                    await self.bot.edit_message_caption(
                        chat_id=chat_id, message_id=msg_id,
                        caption=new_caption,
                    )
                except Exception:
                    logger.debug(f"编辑失败消息异常: {msg_id}")
                logger.error(f"[{self.bot_type}] 交易执行失败: {signal_id}")
        else:
            await cq.answer('未设置交易执行器', show_alert=True)

    def stop_command_polling(self):
        """停止命令轮询"""
        self._polling = False
        if self._polling_thread and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=3)
        self._polling_thread = None
        logger.info(f"[{self.bot_type}] 命令轮询已停止")

    def close(self):
        """关闭TelegramBot，清理bot HTTP会话和事件循环资源"""
        self.stop_command_polling()
        if self._loop_started and self.loop.is_running():
            logger.info(f"[{self.bot_type}] 正在关闭事件循环...")

            # 关闭 bot 的 HTTP session
            if hasattr(self.bot, 'shutdown'):
                try:
                    future = asyncio.run_coroutine_threadsafe(self.bot.shutdown(), self.loop)
                    future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"[{self.bot_type}] bot.shutdown() 失败: {e}")
            elif hasattr(self.bot, 'close'):
                try:
                    future = asyncio.run_coroutine_threadsafe(self.bot.close(), self.loop)
                    future.result(timeout=5)
                except Exception as e:
                    logger.warning(f"[{self.bot_type}] bot.close() 失败: {e}")

            # 停止事件循环
            self.loop.call_soon_threadsafe(self.loop.stop)

            # 等待线程结束
            if self.loop_thread and self.loop_thread.is_alive():
                self.loop_thread.join(timeout=5.0)
                if self.loop_thread.is_alive():
                    logger.warning(f"[{self.bot_type}] 事件循环线程未能及时停止")

            self._loop_started = False
            self.loop_thread = None
            logger.info(f"[{self.bot_type}] 资源已清理")

    def __del__(self):
        """析构函数，确保资源被清理"""
        try:
            if self._loop_started:
                self.close()
        except Exception as e:
            pass