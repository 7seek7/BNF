"""
交易记录管理模块 - 记录每个币种的交易信息

功能：
- 记录每个币种的交易信息（开仓、加仓、止盈、止损）
- 生成交易统计表格
- 导出交易记录为CSV或JSON
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from utils.logger import Logger

logger = Logger.get_logger('trade_recorder')


class TradeRecord:
    """单条交易记录"""
    
    def __init__(self, symbol: str, action: str, direction: str, price: float, 
                 quantity: float, amount: float, leverage: int, ratio: str = ""):
        """
        Args:
            symbol: 币种代码（如BTCUSDT）
            action: 操作类型（OPEN/ADD/CLOSE/TP/SL）
            direction: 方向（LONG/SHORT）
            price: 成交价格
            quantity: 成交数量
            amount: 投入金额（USDT）
            leverage: 杠杆倍数
            ratio: 仓位比例（如 30/100）
        """
        self.symbol = symbol
        self.action = action
        self.direction = direction
        self.price = price
        self.quantity = quantity
        self.amount = amount
        self.leverage = leverage
        self.ratio = ratio
        self.timestamp = datetime.now()
        self.status = 'success'
        self.profit = 0.0
        self.profit_pct = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'action': self.action,
            'direction': self.direction,
            'price': self.price,
            'quantity': self.quantity,
            'amount': self.amount,
            'leverage': self.leverage,
            'ratio': self.ratio,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'status': self.status,
            'profit': self.profit,
            'profit_pct': self.profit_pct
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        action_text = {
            'OPEN': '开仓',
            'ADD': '加仓',
            'CLOSE': '平仓',
            'TP': '止盈',
            'SL': '止损'
        }.get(self.action, self.action)
        
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {self.symbol} | {action_text} | {self.direction} | {self.ratio} | {self.amount:.2f}USDT @ {self.price:.6f} ({self.leverage}x)"


class TradeRecorder:
    """交易记录管理器"""
    
    def __init__(self, data_dir: str = "data/trades"):
        """
        初始化交易记录器
        
        Args:
            data_dir: 数据保存目录
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # {symbol: [TradeRecord, ...]}
        self.records: Dict[str, List[TradeRecord]] = {}
        
        # 统计数据：{symbol: {action: count, ...}}
        self.statistics: Dict[str, Dict[str, int]] = {}
        
        logger.info(f"交易记录器初始化完成，数据目录: {self.data_dir}")
    
    def record_trade(self, symbol: str, action: str, direction: str, price: float,
                    quantity: float, amount: float, leverage: int, ratio: str = "") -> TradeRecord:
        """
        记录一笔交易
        
        Args:
            symbol: 币种
            action: 操作类型
            direction: 方向
            price: 价格
            quantity: 数量
            amount: 金额
            leverage: 杠杆
            ratio: 仓位比例
            
        Returns:
            TradeRecord: 交易记录
        """
        record = TradeRecord(symbol, action, direction, price, quantity, amount, leverage, ratio)
        
        if symbol not in self.records:
            self.records[symbol] = []
            self.statistics[symbol] = {}
        
        self.records[symbol].append(record)
        
        # 更新统计
        action_count = self.statistics[symbol].get(action, 0)
        self.statistics[symbol][action] = action_count + 1
        
        logger.info(f"交易记录已保存: {record}")
        return record
    
    def get_symbol_trades(self, symbol: str) -> List[TradeRecord]:
        """获取指定币种的所有交易记录"""
        return self.records.get(symbol, [])
    
    def get_symbol_summary(self, symbol: str) -> Dict[str, Any]:
        """获取指定币种的交易摘要"""
        trades = self.get_symbol_trades(symbol)
        
        if not trades:
            return {
                'symbol': symbol,
                'total_trades': 0,
                'trade_count': {},
                'last_trade': None,
                'total_investment': 0.0
            }
        
        # 计算统计数据
        total_investment = sum(t.amount for t in trades if t.action in ['OPEN', 'ADD'])
        
        return {
            'symbol': symbol,
            'total_trades': len(trades),
            'trade_count': self.statistics.get(symbol, {}),
            'last_trade': trades[-1].timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'total_investment': total_investment,
            'trades': [t.to_dict() for t in trades]
        }
    
    def export_csv(self, symbol: Optional[str] = None) -> str:
        """
        导出交易记录为CSV
        
        Args:
            symbol: 若指定，只导出该币种，否则导出所有
            
        Returns:
            str: CSV文件路径
        """
        if symbol:
            symbols = [symbol] if symbol in self.records else []
        else:
            symbols = list(self.records.keys())
        
        if not symbols:
            logger.warning("没有交易记录可导出")
            return ""
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if len(symbols) == 1:
            csv_file = self.data_dir / f"{symbols[0]}_trades_{timestamp}.csv"
        else:
            csv_file = self.data_dir / f"all_trades_{timestamp}.csv"
        
        # 写入CSV
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['币种', '操作', '方向', '价格', '数量', '金额(USDT)', '杠杆', '仓位比例', '时间', '状态'])
                
                for symbol in symbols:
                    for record in self.records.get(symbol, []):
                        writer.writerow([
                            record.symbol,
                            record.action,
                            record.direction,
                            f"{record.price:.6f}",
                            f"{record.quantity:.6f}",
                            f"{record.amount:.2f}",
                            f"{record.leverage}x",
                            record.ratio,
                            record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                            record.status
                        ])
            
            logger.info(f"交易记录已导出到: {csv_file}")
            return str(csv_file)
        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            return ""
    
    def export_json(self, symbol: Optional[str] = None) -> str:
        """
        导出交易记录为JSON
        
        Args:
            symbol: 若指定，只导出该币种，否则导出所有
            
        Returns:
            str: JSON文件路径
        """
        if symbol:
            symbols = [symbol] if symbol in self.records else []
        else:
            symbols = list(self.records.keys())
        
        if not symbols:
            logger.warning("没有交易记录可导出")
            return ""
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if len(symbols) == 1:
            json_file = self.data_dir / f"{symbols[0]}_trades_{timestamp}.json"
        else:
            json_file = self.data_dir / f"all_trades_{timestamp}.json"
        
        # 构建数据
        data = {
            'export_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'symbols': {}
        }
        
        for symbol in symbols:
            data['symbols'][symbol] = self.get_symbol_summary(symbol)
        
        # 写入JSON
        try:
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"交易记录已导出到: {json_file}")
            return str(json_file)
        except Exception as e:
            logger.error(f"导出JSON失败: {str(e)}")
            return ""
    
    def print_summary(self):
        """打印交易摘要"""
        if not self.records:
            logger.info("暂无交易记录")
            return
        
        logger.info("=" * 80)
        logger.info("【交易统计摘要】")
        logger.info("=" * 80)
        
        for symbol in sorted(self.records.keys()):
            summary = self.get_symbol_summary(symbol)
            logger.info(f"\n{symbol}:")
            logger.info(f"  总交易数: {summary['total_trades']}")
            logger.info(f"  操作分类: {summary['trade_count']}")
            logger.info(f"  总投入: {summary['total_investment']:.2f} USDT")
            logger.info(f"  最后操作: {summary['last_trade']}")
        
        logger.info("\n" + "=" * 80)
