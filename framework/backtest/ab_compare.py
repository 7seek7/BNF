# -*- coding: utf-8 -*-
"""
A/B compare: old tupo_core vs new tupo_core
Runs backtest twice with different tupo_core versions, outputs per-signal comparison.
"""
import sys, os, subprocess, json, re, time
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')
TC_PATH = os.path.join(ROOT, 'strategies', '15mTupo', 'private', 'tupo_core.py')

def run_backtest(label, env_extra=None, output_file=None):
    """Run backtest and parse per-signal output."""
    env = os.environ.copy()
    env['BASE_MARGIN_PCT'] = '0.20'
    env['DD_T1'] = '0.35'
    env['DD_T2'] = '0.55'
    if env_extra:
        env.update(env_extra)
    print(f'\n{"="*60}')
    print(f'Running: {label}')
    print(f'{"="*60}')
    t0 = time.time()
    if output_file:
        with open(output_file, 'w', encoding='utf-8', errors='replace') as f:
            proc = subprocess.run(
                [sys.executable, os.path.join(ROOT, 'framework', 'backtest', 'run_final.py')],
                cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1800
            )
        with open(output_file, 'r', encoding='utf-8', errors='replace') as f:
            output = f.read()
    else:
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, 'framework', 'backtest', 'run_final.py')],
            cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800
        )
        output = proc.stdout + proc.stderr
    elapsed = time.time() - t0
    
    # Parse per-signal stats
    signals = {}
    # Pattern: BO_LONG_S1: 14笔 PnL=325% 胜=11/14=79%
    sig_pattern = re.compile(r'^(\w+): (\d+)笔 PnL=([-\d.]+)% 胜=(\d+)/(\d+)=(\d+)%', re.MULTILINE)
    for m in sig_pattern.finditer(output):
        name, count, pnl, wins, total, wr = m.groups()
        signals[name] = {
            'count': int(count),
            'pnl': float(pnl),
            'wins': int(wins),
            'total': int(total),
            'wr': int(wr),
        }
    
    # Parse detailed stats: 胜: 杠杆... and 亏: 杠杆...
    detail_pattern = re.compile(
        r'^(\w+) \((\d+)笔\): 胜=(\d+)\((\d+)%\) 亏=(\d+)$.*?'
        r'  胜: 杠杆([\d.]+)x ADX([\d.]+) RSI([\d.]+) ATR([\d.]+) 持仓([\d.]+)根5m 均收益([-\d.]+)%.*?'
        r'  亏: 杠杆([\d.]+)x ADX([\d.]+) RSI([\d.]+) ATR([\d.]+) 持仓([\d.]+)根5m 均收益([-\d.]+)%',
        re.MULTILINE | re.DOTALL
    )
    for m in detail_pattern.finditer(output):
        name = m.group(1)
        if name in signals:
            signals[name]['w_lev'] = float(m.group(6))
            signals[name]['w_adx'] = float(m.group(7))
            signals[name]['w_rsi'] = float(m.group(8))
            signals[name]['w_atr'] = float(m.group(9))
            signals[name]['w_hb'] = float(m.group(10))
            signals[name]['w_avg_pnl'] = float(m.group(11))
            signals[name]['l_lev'] = float(m.group(12))
            signals[name]['l_adx'] = float(m.group(13))
            signals[name]['l_rsi'] = float(m.group(14))
            signals[name]['l_atr'] = float(m.group(15))
            signals[name]['l_hb'] = float(m.group(16))
            signals[name]['l_avg_pnl'] = float(m.group(17))
    
    # Parse total
    total_match = re.search(r'Total: (\d+) trades, PnL=([-\d.]+)%, Win=(\d+)/(\d+)=(\d+)%', output)
    if total_match:
        signals['__TOTAL__'] = {
            'count': int(total_match.group(1)),
            'pnl': float(total_match.group(2)),
            'wins': int(total_match.group(3)),
            'total': int(total_match.group(4)),
            'wr': int(total_match.group(5)),
        }
    
    print(f'Elapsed: {elapsed:.0f}s, signals found: {len(signals)}')
    return signals

def compare(a, b):
    """Compare two signal dicts."""
    all_keys = sorted(set(list(a.keys()) + list(b.keys())) - {'__TOTAL__'})
    
    # Group by base type
    groups = defaultdict(list)
    for k in all_keys:
        base = k.rsplit('_', 1)[0] if '_' in k else k
        groups[base].append(k)
    
    print(f'\n{"="*80}')
    print(f'{"Signal":<20} {"Old":>5} {"New":>5} {"Δ":>5} | {"Old PnL":>8} {"New PnL":>8} {"Δ":>8} | {"Old WR":>6} {"New WR":>6} {"Δ":>4} | {"Old AvgLoss":>10} {"New AvgLoss":>10}')
    print(f'{"="*80}')
    
    for base in sorted(groups.keys()):
        keys = groups[base]
        for k in keys:
            oa = a.get(k, {})
            ob = b.get(k, {})
            oc = oa.get('count', 0)
            nc = ob.get('count', 0)
            op = oa.get('pnl', 0)
            np_ = ob.get('pnl', 0)
            ow = oa.get('wr', 0)
            nw = ob.get('wr', 0)
            ol = oa.get('l_avg_pnl', 0)
            nl = ob.get('l_avg_pnl', 0)
            dc = nc - oc
            dp = np_ - op
            dw = nw - ow
            print(f'{k:<20} {oc:>5} {nc:>5} {dc:>+5} | {op:>+8.0f} {np_:>+8.0f} {dp:>+8.0f} | {ow:>5}% {nw:>5}% {dw:>+4}% | {ol:>+10.1f} {nl:>+10.1f}')
    
    # Totals
    if '__TOTAL__' in a and '__TOTAL__' in b:
        ta, tb = a['__TOTAL__'], b['__TOTAL__']
        print(f'\n{"TOTAL":<20} {ta["count"]:>5} {tb["count"]:>5} {tb["count"]-ta["count"]:>+5} | {ta["pnl"]:>+8.0f} {tb["pnl"]:>+8.0f} {tb["pnl"]-ta["pnl"]:>+8.0f} | {ta["wr"]:>5}% {tb["wr"]:>5}% {tb["wr"]-ta["wr"]:>+4}%')

if __name__ == '__main__':
    out_a = os.path.join(ROOT, 'bt_A_output.txt')
    out_b = os.path.join(ROOT, 'bt_B_output.txt')
    
    # Run A (current = new code with fallback + best-cluster + triangle 0.2)
    a = run_backtest('A: current (best-cluster + fallback + tri=0.2)', output_file=out_a)
    
    # Save current, restore old
    import shutil
    shutil.copy2(TC_PATH, TC_PATH + '.new')
    OLD_TC = os.path.join(ROOT, '新建文件夹', 'strategies', '15mTupo', 'private', 'tupo_core.py')
    shutil.copy2(OLD_TC, TC_PATH)
    
    # Run B (old code)
    b = run_backtest('B: old (first-found cluster + fallback + tri=0.6)', output_file=out_b)
    
    # Restore new
    shutil.copy2(TC_PATH + '.new', TC_PATH)
    os.remove(TC_PATH + '.new')
    
    # Compare
    compare(b, a)  # old vs new
