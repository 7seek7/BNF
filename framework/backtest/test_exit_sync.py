# -*- coding: utf-8 -*-
"""
退出逻辑同步测试: 验证 run_final.py inline 与 tupo_core.py check_exit() 行为一致
运行: python framework/backtest/test_exit_sync.py
"""
import os, sys, numpy as np

os.environ.setdefault('BASE_LEV', '15')
os.environ.setdefault('MAX_LOSS', '30')
os.environ.setdefault('TREND_BE_MX', '0')
os.environ.setdefault('BE_HARD_MX', '0')
os.environ.setdefault('BE_WEAK_MX', '0')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import importlib
_tupo_core = importlib.import_module('strategies.15mTupo.private.tupo_core')
check_exit = _tupo_core.check_exit
ExitState = _tupo_core.ExitState
PositionSide = _tupo_core.PositionSide
TrendType = _tupo_core.TrendType
BacktestSettings = _tupo_core.BacktestSettings

_s = BacktestSettings()

def _es(prev_pnl=0):
    e = ExitState()
    e.prev_pnl = prev_pnl
    return e

def _base(**kw):
    d = dict(
        pos=PositionSide.LONG, ep=100.0, sl=98.0, lev=15,
        cp=100.5, hb=5, hb15=1, mx=3.0, mn=-1.0,
        st='TR_LONG', tt=TrendType.UPTREND,
        eidx=0, eidx15=0, eres=102.0, esup=98.0,
        ci=50, i=65, avgv_ci=300.0, v15s=300.0, v5_i=100.0,
        c5_i=100.5, c5_prev=100.3, l5_i=99.5, h5_i=101.0, c15s=100.5,
        pr=1.0, sr_room=1.5,
    )
    d.update(kw)
    return d

def _inline(p, es=None, prev_pnl=0):
    """run_final.py inline exit logic"""
    pos=p['pos']; ep=p['ep']; sl=p['sl']; lev=p['lev']; cp=p['cp']
    hb=p['hb']; hb15=p['hb15']; mx=p['mx']; mn=p['mn']
    st=p['st']; tt=p['tt']; ci=p['ci']; i=p['i']
    avgv_ci=p['avgv_ci']; v15s=p['v15s']; v5_i=p['v5_i']
    c5_i=p['c5_i']; c5_prev=p['c5_prev']
    l5_i=p['l5_i']; h5_i=p['h5_i']; c15s=p['c15s']; pr=p['pr']
    sr_room=p.get('sr_room', 99.0)

    FUND=float(os.environ.get('FUNDING','0.0001'))
    SLIP=float(os.environ.get('SLIPPAGE','0.03'))
    BE_MX=float(os.environ.get('BE_MX_PCT','5.0'))
    BE_BUF=float(os.environ.get('BE_SL_BUFFER_PCT','0'))
    BE_HARD=float(os.environ.get('BE_HARD_MX','0'))
    BE_WK=float(os.environ.get('BE_WEAK_MX','0'))
    BE_WKV=float(os.environ.get('BE_WEAK_VOL','5.0'))
    TR_BE=float(os.environ.get('TREND_BE_MX','0'))
    VOL_TH=float(os.environ.get('VOL_CONFIRM_THRESH','5.0'))
    LIQ_MM=float(os.environ.get('LIQ_MM_RATE','0.5'))
    LIQ_TH=float(os.environ.get('LIQ_THRESHOLD','-90'))
    _ROP=bool(int(os.environ.get('REBOUND_OPP','0')))
    _RBE=bool(int(os.environ.get('REBOUND_BE','0')))
    _LVS=bool(int(os.environ.get('LADDER_VS','0')))
    VF=float(os.environ.get('VEL_FLOOR','11'))
    VMX=float(os.environ.get('VEL_MIN_MX','8'))
    VFT=float(os.environ.get('VEL_FLOOR_TR','0'))
    VTC=float(os.environ.get('VEL_TR_MX_CAP','20'))
    VW=int(os.environ.get('VEL_WINDOW','3'))
    VFM=float(os.environ.get('VEL_FLOOR_MULTI','15'))
    TA=float(os.environ.get('TRAIL_ACT','5.0'))
    TD=float(os.environ.get('TRAIL_DIST','1.5'))
    TTA=float(os.environ.get('TRAIL_TIGHT_ACT','0'))
    TTD=float(os.environ.get('TRAIL_TIGHT_DIST','0'))

    sl_hit=False; sc2=False; er=''; cr=1.0; pnl=0.0; new_sl=sl; exit_p=cp
    if es is None: es = _es(prev_pnl)

    _5m_vr = v5_i/max(avgv_ci/3,0.001) if avgv_ci>0 else 0

    # SL
    if (pos==PositionSide.LONG and l5_i<=sl) or (pos==PositionSide.SHORT and h5_i>=sl):
        sl_hit=True
        pnl=abs(ep-sl)/ep*100*lev if pos==PositionSide.LONG else abs(sl-ep)/ep*100*lev
        pnl=-pnl; pnl-=FUND*lev; pnl=max(pnl,-100.0); pnl-=SLIP*2*lev
        sc2,er=True,"trigger_stop"; cr=1.0; exit_p=sl
        return sl_hit,er,cr,new_sl,mx,mn,pnl,exit_p

    # REBOUND_OPP
    if not sl_hit and _ROP and p.get('eres',0)>0 and p.get('esup',0)>0 and i>=2 and sl!=ep:
        if pos==PositionSide.LONG and h5_i>=p['eres']*(1-0.005) and c5_i<c5_prev:
            cr=min(0.5/pr,1.0) if pr>0 else 0.5; new_sl=ep; sc2,er=True,"res_stop"
        elif pos==PositionSide.SHORT and l5_i<=p['esup']*(1+0.005) and c5_i>c5_prev:
            cr=min(0.5/pr,1.0) if pr>0 else 0.5; new_sl=ep; sc2,er=True,"sup_stop"

    # BE_MX
    if not sl_hit:
        if pos==PositionSide.LONG:
            if mx>=BE_MX and new_sl<ep and es.peak_vol_ratio>=VOL_TH: new_sl=ep*(1-BE_BUF/100)
            elif hb>=6 and max(mx,(h5_i-ep)/ep*100*lev)>=BE_MX and new_sl<ep:
                if _5m_vr>=VOL_TH: new_sl=ep*(1-BE_BUF/100)
        elif pos==PositionSide.SHORT:
            if mx>=BE_MX and new_sl>ep and es.peak_vol_ratio>=VOL_TH: new_sl=ep*(1+BE_BUF/100)
            elif hb>=6 and max(mx,(ep-l5_i)/ep*100*lev)>=BE_MX and new_sl>ep:
                if _5m_vr>=VOL_TH: new_sl=ep*(1+BE_BUF/100)

    # BE_HARD
    if not sl_hit and BE_HARD>0 and mx>=BE_HARD:
        if pos==PositionSide.LONG and new_sl<ep: new_sl=ep*(1-BE_BUF/100)
        elif pos==PositionSide.SHORT and new_sl>ep: new_sl=ep*(1+BE_BUF/100)

    # BE_WEAK
    if not sl_hit and BE_WK>0 and mx>=BE_WK:
        if _5m_vr<BE_WKV:
            if pos==PositionSide.LONG and new_sl<ep: new_sl=ep*(1-BE_BUF/100)
            elif pos==PositionSide.SHORT and new_sl>ep: new_sl=ep*(1+BE_BUF/100)

    # TREND_BE + Liquidation (elif chain)
    if not sl_hit and TR_BE>0 and mx>=TR_BE:
        if tt not in (TrendType.CONSOLIDATION, TrendType.UNKNOWN):
            if pos==PositionSide.LONG and new_sl<ep: new_sl=ep*(1-BE_BUF/100)
            elif pos==PositionSide.SHORT and new_sl>ep: new_sl=ep*(1+BE_BUF/100)
    elif (pos==PositionSide.LONG and l5_i<=ep*(1-1.0/lev+LIQ_MM/100)) or \
         (pos==PositionSide.SHORT and h5_i>=ep*(1+1.0/lev-LIQ_MM/100)):
        sl_hit=True
        lpct=(1.0/lev-LIQ_MM/100)*100*lev; pnl=-lpct; pnl-=FUND*lev
        pnl=max(pnl,-100.0); pnl-=SLIP*2*lev; pnl=max(pnl,LIQ_TH)
        sc2,er=True,"liq"; cr=1.0

    # 15m close
    is_15m = (i>=0 and i%3==2)
    if not sl_hit and is_15m:
        pnl_c=(c15s-ep)/ep*100*lev if pos==PositionSide.LONG else (ep-c15s)/ep*100*lev
        pnl=pnl_c; pnl-=FUND*lev*3; pnl=max(pnl,-100.0)
        cur_vr=v15s/max(avgv_ci,0.001) if avgv_ci>0 else 0
        if pnl_c>mx: es.peak_vol_ratio=max(es.peak_vol_ratio,cur_vr)
        mx=max(mx,pnl_c); mn=min(mn,pnl_c); sc2=False; er=''; cr=1.0; exit_p=c15s
        # Velocity
        _vf=VF
        if VFT>0 and st.startswith('TR_') and mx<VTC: _vf=VFT
        if _vf>0 and mx>=VMX and es.prev_pnl!=0 and es.prev_pnl-pnl_c>=_vf: sc2,er=True,"velocity"
        if not sc2 and VW>0 and VFM>0 and mx>=VMX and len(es.ph)>=VW:
            if max(es.ph[-VW:])-es.ph[-1]>=VFM: sc2,er=True,"velocity"
        es.prev_pnl=pnl_c
        # Trailing
        if True and es.trail_active:
            _td=TTD if (TTD>0 and mx>=TTA) else TD
            if pos==PositionSide.LONG:
                es.trail_high=max(es.trail_high,c15s)
                if c15s<=es.trail_high*(1-_td/100): sc2,er=True,"trailing"
            else:
                es.trail_high=min(es.trail_high,c15s)
                if c15s>=es.trail_high*(1+_td/100): sc2,er=True,"trailing"
        if True and not es.trail_active:
            if pos==PositionSide.LONG and pnl_c>=TA: es.trail_high=c15s;es.trail_active=True
            elif pos==PositionSide.SHORT and pnl_c>=TA: es.trail_high=c15s;es.trail_active=True
        # Ladder
        if not sc2 and es.tc<3:
            if pnl_c>es.tsp: es.tsp=pnl_c
            ir=st.startswith('RB_') or 'REBOUND' in st
            _csr=(not ir and tt==TrendType.CONSOLIDATION and sr_room<2.0)
            if ir or _csr:
                lp=_s.ladder_peak_rebound
                ddt=[_s.ladder_dd_t1_rebound,_s.ladder_dd_t2_rebound,_s.ladder_dd_t3_rebound]
            else:
                lp=_s.ladder_peak_uptrend
                ddt=[_s.ladder_dd_t1_uptrend,_s.ladder_dd_t2_uptrend,_s.ladder_dd_t3_uptrend]
            if es.tsp>=lp:
                dd_t=es.tsp-pnl_c>=ddt[es.tc]
                if _LVS and dd_t and ci>=0 and avgv_ci>0:
                    if v15s>=avgv_ci*0.7: dd_t=False
                if dd_t:
                    ratios=[_s.ladder_close_t1/100,_s.ladder_close_t2/100,_s.ladder_close_t3/100]
                    cr=max(0,min(ratios[es.tc]/pr if pr>0 else ratios[es.tc],1.0))
                    sc2,er=True,"ladder";es.tc+=1;es.hrt=True
        # REBOUND BE
        if not sc2 and _RBE and (st.startswith('RB_') or 'REBOUND' in st) and mx>=10.0 and pnl_c<=3.0:
            sc2,er=True,"rebound_be"
        # Entry stop
        if not sc2 and hb15>=_s.entry_stop_bars//3 and mx<=0:
            sc2,er=True,"entry_stop"
        if sc2: pnl-=SLIP*2*lev

    should_close = sl_hit or (is_15m and sc2)
    # hrt override
    if es.hrt and should_close and er != "ladder": er = "ladder"
    return should_close, er, cr, new_sl, mx, mn, pnl, exit_p


def _chk(name, p, prev_pnl=0, es_over=None):
    e1 = _es(prev_pnl)
    r1 = check_exit(**p, es=e1)
    e2 = _es(prev_pnl)
    r2 = _inline(p, es=e2, prev_pnl=prev_pnl)
    ok = True
    for idx, (a, b) in enumerate(zip(r1, r2)):
        if isinstance(a, float):
            if abs(a - b) > 0.001:
                print(f"  FAIL [{name}] field{idx}: {a} vs {b}"); ok = False
        elif a != b:
            print(f"  FAIL [{name}] field{idx}: {a!r} vs {b!r}"); ok = False
    if ok:
        print(f"  PASS: {name}")
    return ok


if __name__ == '__main__':
    ok = True
    ok &= _chk("no_exit", _base())
    ok &= _chk("sl_long", _base(l5_i=97.5))
    ok &= _chk("sl_short", _base(pos=PositionSide.SHORT, h5_i=102.5, sl=102.0, ep=100.0, l5_i=99.5))
    ok &= _chk("be_mx", _base(mx=6.0, hb=8, h5_i=106.0, v5_i=500.0, avgv_ci=300.0))
    ok &= _chk("velocity", _base(mx=12.0, c15s=100.0), prev_pnl=20.0)
    ok &= _chk("ladder", _base(mx=30.0, c15s=102.0), es_over={'tsp': 25.0, 'tc': 0})
    ok &= _chk("entry_stop", _base(hb15=12, mx=0.0, c15s=99.5))
    ok &= _chk("rebound_be", _base(mx=12.0, c15s=100.5, st='RB_LONG'))
    ok &= _chk("partial_cr", _base(pr=0.5, mx=30.0, c15s=102.0))
    ok &= _chk("liq_long", _base(mx=0.0, l5_i=80.0))
    ok &= _chk("be_hard", _base(mx=11.0, hb=8, h5_i=106.0, v5_i=500.0, avgv_ci=300.0))
    ok &= _chk("be_weak", _base(mx=8.0, hb=8, h5_i=106.0, v5_i=10.0, avgv_ci=300.0))

    print(f"\n{'ALL PASSED' if ok else 'SOME FAILED'}")
    sys.exit(0 if ok else 1)
