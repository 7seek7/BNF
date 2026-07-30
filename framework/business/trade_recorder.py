"""
交易记录管理模块 - 记录每个持仓的完整生命周期，并与币安核对

功能：
- 开仓时创建记录，平仓时更新记录
- 开仓/平仓后自动查询币安API核对实际数据
- 15字段全填充：币种、策略、方向、杠杆、数量、保证金、占比、开仓价、平仓价、平仓原因、开仓时间、平仓时间、盈亏、盈亏率、状态
"""

import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from utils.logger import Logger

logger = Logger.get_logger('trade_recorder')


class TradeRecord:
    """单条交易记录（一条记录 = 一次完整持仓生命周期）"""

    def __init__(self, symbol: str, strategy: str, direction: str, leverage: int,
                 quantity: float, margin: float, capital_ratio: float, entry_price: float):
        self.symbol = symbol
        self.strategy = strategy
        self.direction = direction
        self.leverage = leverage
        self.quantity = quantity
        self.margin = margin
        self.capital_ratio = capital_ratio
        self.entry_price = entry_price
        self.open_time = datetime.now()
        # 平仓时填充
        self.close_price: float = 0.0
        self.close_reason: str = ''
        self.close_time: Optional[datetime] = None
        self.profit: float = 0.0
        self.profit_pct: float = 0.0
        self.status: str = 'open'
        # 币安核对标记
        self.binance_verified: bool = False
        self.binance_entry_price: float = 0.0
        self.binance_quantity: float = 0.0
        self.binance_leverage: int = 0
        self.binance_margin: float = 0.0

    def close(self, close_price: float, reason: str):
        """平仓时更新记录"""
        self.close_price = close_price
        self.close_reason = reason
        self.close_time = datetime.now()
        self.status = 'closed'
        if self.direction == 'LONG':
            self.profit_pct = (close_price - self.entry_price) / self.entry_price * self.leverage * 100
        else:
            self.profit_pct = (self.entry_price - close_price) / self.entry_price * self.leverage * 100
        self.profit = self.margin * self.profit_pct / 100.0

    def verify_from_binance(self, bn_price: float, bn_qty: float, bn_leverage: int, bn_margin: float):
        """用币安实际数据核对并修正"""
        self.binance_verified = True
        self.binance_entry_price = bn_price
        self.binance_quantity = bn_qty
        self.binance_leverage = bn_leverage
        self.binance_margin = bn_margin
        # 以币安数据为准修正本地记录
        if bn_price > 0:
            self.entry_price = bn_price
        if bn_qty > 0:
            self.quantity = bn_qty
        if bn_leverage > 0:
            self.leverage = bn_leverage
        if bn_margin > 0:
            self.margin = bn_margin

    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'strategy': self.strategy,
            'direction': self.direction,
            'leverage': self.leverage,
            'quantity': self.quantity,
            'margin': self.margin,
            'capital_ratio': round(self.capital_ratio, 4),
            'entry_price': self.entry_price,
            'close_price': self.close_price,
            'close_reason': self.close_reason,
            'open_time': self.open_time.strftime('%Y-%m-%d %H:%M:%S'),
            'close_time': self.close_time.strftime('%Y-%m-%d %H:%M:%S') if self.close_time else '',
            'profit': round(self.profit, 4),
            'profit_pct': round(self.profit_pct, 4),
            'status': self.status,
            'binance_verified': self.binance_verified,
        }

    def __str__(self):
        verified = ' [已核对]' if self.binance_verified else ''
        close_info = f" → {self.close_price:.6f} ({self.close_reason})" if self.status == 'closed' else ''
        return (f"{self.open_time.strftime('%m-%d %H:%M')} | {self.symbol} | {self.strategy} | "
                f"{self.direction} {self.leverage}x | {self.quantity:.4f} | "
                f"{self.margin:.1f}USDT({self.capital_ratio:.1%}) | "
                f"@{self.entry_price:.6f}{close_info}{verified}")


class TradeRecorder:
    """交易记录管理器（带币安核对）"""

    _instance: Optional['TradeRecorder'] = None

    @classmethod
    def get_instance(cls, data_dir: str = "data/trades") -> 'TradeRecorder':
        """获取共享实例（所有调用方共用，避免多实例写同一文件相互覆盖）"""
        if cls._instance is None:
            cls._instance = cls(data_dir)
        return cls._instance

    def __init__(self, data_dir: str = "data/trades"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._data_file = self.data_dir / "trades_data.json"
        self.records: Dict[str, List[TradeRecord]] = {}
        self._lock = threading.RLock()
        self._client = None  # BinanceClient引用，用于核对
        self._load()
        logger.info(f"交易记录器初始化完成，已加载 {self._total_records()} 条记录")

    def set_client(self, client):
        """设置币安客户端（由position_monitor注入）"""
        self._client = client
        logger.info("交易记录器已绑定币安客户端，开仓/平仓将自动核对")

    def _total_records(self) -> int:
        return sum(len(records) for records in self.records.values())

    def _load(self):
        if not self._data_file.exists():
            return
        try:
            with open(self._data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for symbol, trades in data.get('symbols', {}).items():
                self.records[symbol] = []
                for t in trades:
                    r = TradeRecord(
                        symbol=t['symbol'], strategy=t['strategy'], direction=t['direction'],
                        leverage=t['leverage'], quantity=t['quantity'], margin=t['margin'],
                        capital_ratio=t.get('capital_ratio', 0), entry_price=t['entry_price'],
                    )
                    r.close_price = t.get('close_price', 0)
                    r.close_reason = t.get('close_reason', '')
                    r.profit = t.get('profit', 0)
                    r.profit_pct = t.get('profit_pct', 0)
                    r.status = t.get('status', 'closed')
                    r.binance_verified = t.get('binance_verified', False)
                    r.open_time = datetime.strptime(t['open_time'], '%Y-%m-%d %H:%M:%S')
                    if t.get('close_time'):
                        r.close_time = datetime.strptime(t['close_time'], '%Y-%m-%d %H:%M:%S')
                    self.records[symbol].append(r)
            logger.info(f"已加载 {self._total_records()} 条历史交易记录")
        except Exception as e:
            logger.warning(f"加载历史记录失败: {e}")

    def _save(self):
        try:
            data = {
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbols': {s: [r.to_dict() for r in rs] for s, rs in self.records.items()}
            }
            with open(self._data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存交易记录失败: {e}")

    def _query_binance_position(self, symbol: str) -> Optional[Dict]:
        """查询币安实际持仓数据"""
        if not self._client:
            return None
        try:
            return self._client.get_position(symbol)
        except Exception as e:
            logger.warning(f"查询币安持仓失败 {symbol}: {e}")
            return None

    def _query_binance_recent_trade(self, symbol: str, direction: str = '') -> Optional[Dict]:
        """查询币安最近平仓成交（只有realizedPnl!=0才是真正的平仓成交）"""
        if not self._client:
            return None
        try:
            trades = self._client.get_account_trades(symbol=symbol, limit=50)
            if not trades:
                return None
            # 找有 realizedPnl 的成交（平仓成交的唯一可靠标志）
            for t in reversed(trades):
                pnl = float(t.get('realizedPnl', 0))
                if pnl != 0:
                    return t
            # 所有成交pnl=0（极端情况：平仓盈亏恰好为0）
            return None
        except Exception as e:
            logger.warning(f"查询币安成交失败 {symbol}: {e}")
        return None

    def record_open(self, symbol: str, strategy: str, direction: str, leverage: int,
                    quantity: float, margin: float, capital_ratio: float, entry_price: float) -> TradeRecord:
        """开仓时创建记录，并查询币安核对实际数据"""
        record = TradeRecord(symbol, strategy, direction, leverage, quantity, margin, capital_ratio, entry_price)
        with self._lock:
            if symbol not in self.records:
                self.records[symbol] = []
            self.records[symbol].append(record)
            self._save()
            logger.info(f"【开仓记录】{record}")

        # 币安核对：查询实际持仓数据修正本地记录
        bn_pos = self._query_binance_position(symbol)
        if bn_pos and bn_pos.get('entry_price', 0) > 0:
            record.verify_from_binance(
                bn_price=float(bn_pos.get('entry_price', 0)),
                bn_qty=abs(float(bn_pos.get('position_amt', 0))),
                bn_leverage=int(bn_pos.get('leverage', 0)),
                bn_margin=float(bn_pos.get('isolated_wallet', 0)),
            )
            with self._lock:
                self._save()
            logger.info(f"【币安核对】{symbol} 实际: 价格={record.entry_price:.6f} 数量={record.quantity:.4f} 杠杆={record.leverage}x 保证金={record.margin:.2f}")
        else:
            logger.warning(f"【币安核对】{symbol} 无法获取持仓数据，使用本地估算值")

        return record

    def record_close(self, symbol: str, close_price: float, reason: str) -> bool:
        """平仓时更新最近一条open记录，并查询币安核对"""
        with self._lock:
            trades = self.records.get(symbol, [])
            target = None
            for r in reversed(trades):
                if r.status == 'open':
                    target = r
                    break
            if not target:
                logger.warning(f"未找到{symbol}的open记录，无法更新平仓")
                return False

            # 先用本地数据更新
            target.close(close_price, reason)
            self._save()
            self._append_to_csv(target)

        # 币安核对：查询最近成交获取实际平仓价和盈亏
        bn_trade = self._query_binance_recent_trade(symbol, target.direction)
        if bn_trade:
            actual_price = float(bn_trade.get('price', 0))
            actual_qty = abs(float(bn_trade.get('qty', 0)))
            actual_pnl = float(bn_trade.get('realizedPnl', 0))
            if actual_price > 0:
                target.close_price = actual_price
            if actual_pnl != 0:
                target.profit = actual_pnl
                target.profit_pct = actual_pnl / target.margin * 100 if target.margin > 0 else 0
            with self._lock:
                self._save()
            logger.info(f"【币安核对】{symbol} 实际平仓价={target.close_price:.6f} 实际盈亏={target.profit:.4f}USDT({target.profit_pct:.2f}%)")
        else:
            logger.warning(f"【币安核对】{symbol} 无法获取成交数据，使用本地估算值")

        logger.info(f"【平仓记录】{target}")
        return True

    def reconcile_all_open(self) -> int:
        """核对所有未平仓记录与币安实际数据（限速1s/次，防止API限流）"""
        count = 0
        with self._lock:
            open_trades = [(s, r) for s, rs in self.records.items() for r in rs if r.status == 'open']
        if not open_trades:
            return 0
        logger.info(f"【批量核对】{len(open_trades)}个持仓需核对，限速执行中...")
        for i, (symbol, record) in enumerate(open_trades):
            time.sleep(1.0)  # 每秒1次，防止限流
            bn_pos = self._query_binance_position(symbol)
            if bn_pos:
                record.verify_from_binance(
                    bn_price=float(bn_pos.get('entry_price', 0)),
                    bn_qty=abs(float(bn_pos.get('position_amt', 0))),
                    bn_leverage=int(bn_pos.get('leverage', 0)),
                    bn_margin=float(bn_pos.get('isolated_wallet', 0)),
                )
                count += 1
            else:
                logger.warning(f"【核对】{symbol} 币安无持仓")
        if count > 0:
            with self._lock:
                self._save()
            logger.info(f"【批量核对】完成 {count}/{len(open_trades)} 个持仓核对")
        return count

    def reconcile_zero_profit_closed(self) -> int:
        """重新核对profit=0的已平仓记录（限速1s/次，防止API限流）"""
        if not self._client:
            logger.warning("币安客户端未注入，跳过批量核对")
            return 0
        count = 0
        fixed = 0
        symbols_fixed = []
        with self._lock:
            for symbol, trades in self.records.items():
                for t in trades:
                    if t.status == 'closed' and t.profit == 0:
                        count += 1
                        symbols_fixed.append(symbol)
                        break  # 每symbol只需修复最旧的一条
        if count == 0:
            return 0

        logger.info(f"【批量修复】需核对 {count} 个symbol，每秒1次限速执行中...")
        for i, symbol in enumerate(symbols_fixed):
            if i > 0:
                time.sleep(1.0)  # 每秒1次，防止限流
            bn_trade = self._query_binance_recent_trade(symbol)
            if bn_trade:
                actual_price = float(bn_trade.get('price', 0))
                actual_pnl = float(bn_trade.get('realizedPnl', 0))
                if actual_price > 0 and actual_pnl != 0:
                    with self._lock:
                        for t in self.records.get(symbol, []):
                            if t.status == 'closed' and t.profit == 0:
                                t.close_price = actual_price
                                t.profit = actual_pnl
                                t.profit_pct = actual_pnl / t.margin * 100 if t.margin > 0 else 0
                                t.binance_verified = True
                                break
                    fixed += 1
                    logger.info(f"【修复】{symbol} 平仓价={actual_price:.6f} 盈亏={actual_pnl:.4f}USDT")
        if fixed > 0:
            with self._lock:
                self._save()
            logger.info(f"【批量修复】完成 {fixed}/{count} 条profit=0记录修复")
        return fixed

    def _append_to_csv(self, record: TradeRecord):
        csv_file = self.data_dir / "trades_all.csv"
        file_exists = csv_file.exists()
        try:
            with open(csv_file, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        '币种', '策略', '方向', '杠杆', '数量', '保证金(USDT)',
                        '占比', '开仓价', '平仓价', '平仓原因',
                        '开仓时间', '平仓时间', '盈亏(USDT)', '盈亏率(%)', '状态', '币安核对'
                    ])
                d = record.to_dict()
                writer.writerow([
                    d['symbol'], d['strategy'], d['direction'], f"{d['leverage']}x",
                    f"{d['quantity']:.6f}", f"{d['margin']:.2f}",
                    f"{d['capital_ratio']:.2%}", f"{d['entry_price']:.6f}",
                    f"{d['close_price']:.6f}" if d['close_price'] else '',
                    d['close_reason'], d['open_time'], d['close_time'],
                    f"{d['profit']:.4f}", f"{d['profit_pct']:.4f}", d['status'],
                    '✓' if d['binance_verified'] else ''
                ])
        except Exception as e:
            logger.error(f"追加CSV失败: {e}")

    def get_symbol_trades(self, symbol: str) -> List[TradeRecord]:
        return self.records.get(symbol, [])

    def get_profit_summary(self) -> Dict[str, Any]:
        total_trades = self._total_records()
        total_margin = sum(r.margin for rs in self.records.values() for r in rs)
        closed = [r for rs in self.records.values() for r in rs if r.status == 'closed']
        total_profit = sum(r.profit for r in closed)
        win = [r for r in closed if r.profit > 0]
        lose = [r for r in closed if r.profit <= 0]
        verified = sum(1 for rs in self.records.values() for r in rs if r.binance_verified)
        strategy_stats = {}
        for r in [r for rs in self.records.values() for r in rs]:
            if r.strategy not in strategy_stats:
                strategy_stats[r.strategy] = {'trades': 0, 'wins': 0, 'total_profit': 0.0, 'total_margin': 0.0}
            strategy_stats[r.strategy]['trades'] += 1
            strategy_stats[r.strategy]['total_margin'] += r.margin
            if r.status == 'closed':
                strategy_stats[r.strategy]['total_profit'] += r.profit
                if r.profit > 0:
                    strategy_stats[r.strategy]['wins'] += 1
        return {
            'total_trades': total_trades,
            'total_margin': total_margin,
            'closed_trades': len(closed),
            'open_trades': total_trades - len(closed),
            'total_profit': total_profit,
            'win_rate': len(win) / len(closed) * 100 if closed else 0,
            'wins': len(win),
            'losses': len(lose),
            'binance_verified': verified,
            'strategy_stats': strategy_stats,
        }

    def export_csv(self, symbol: Optional[str] = None) -> str:
        symbols = [symbol] if symbol and symbol in self.records else list(self.records.keys()) if not symbol else []
        if not symbols:
            return ""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = self.data_dir / f"{'all' if len(symbols) > 1 else symbols[0]}_trades_{timestamp}.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow([
                    '币种', '策略', '方向', '杠杆', '数量', '保证金(USDT)',
                    '占比', '开仓价', '平仓价', '平仓原因',
                    '开仓时间', '平仓时间', '盈亏(USDT)', '盈亏率(%)', '状态', '币安核对'
                ])
                for s in symbols:
                    for r in self.records[s]:
                        d = r.to_dict()
                        writer.writerow([
                            d['symbol'], d['strategy'], d['direction'], f"{d['leverage']}x",
                            f"{d['quantity']:.6f}", f"{d['margin']:.2f}",
                            f"{d['capital_ratio']:.2%}", f"{d['entry_price']:.6f}",
                            f"{d['close_price']:.6f}" if d['close_price'] else '',
                            d['close_reason'], d['open_time'], d['close_time'],
                            f"{d['profit']:.4f}", f"{d['profit_pct']:.4f}", d['status'],
                            '✓' if d['binance_verified'] else ''
                        ])
            return str(csv_file)
        except Exception as e:
            logger.error(f"导出CSV失败: {e}")
            return ""

    def print_summary(self):
        s = self.get_profit_summary()
        logger.info("=" * 60)
        logger.info(f"【交易统计】总持仓={s['total_trades']} 已平={s['closed_trades']} 持仓中={s['open_trades']}")
        logger.info(f"  总保证金={s['total_margin']:.1f}U 总盈亏={s['total_profit']:.2f}U 胜率={s['win_rate']:.1f}%")
        logger.info(f"  币安核对={s['binance_verified']}/{s['total_trades']}")
        for name, st in s['strategy_stats'].items():
            wr = st['wins'] / st['trades'] * 100 if st['trades'] else 0
            logger.info(f"  [{name}] 交易={st['trades']} 胜率={wr:.0f}% 盈亏={st['total_profit']:.2f}U")
        logger.info("=" * 60)
