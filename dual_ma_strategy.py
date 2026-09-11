# -*- coding: utf-8 -*-
"""
双均线策略回测：沪深300 vs 贵州茅台
策略：MA5上穿MA20买入，下穿卖出
回测期：2023-01-01 至 2026-08-31
"""

import akshare as ak
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ==================== 参数 ====================
START_DATE = "2023-01-01"
END_DATE = "2026-08-31"
SHORT_WINDOW = 5
LONG_WINDOW = 20
COMMISSION = 0.0003   # 手续费 万分之三
SLIPPAGE = 0.001      # 滑点 千分之一


# ==================== 数据获取 ====================
def get_data(symbol, name, start_date, end_date):
    """获取数据，优先腾讯接口，失败则用备用接口"""
    print(f"正在获取 {name}（{symbol}）数据...")

    try:
        df = ak.stock_zh_a_hist_tx(
            symbol=symbol,
            start_date=start_date.replace("-", ""),
            end_date=end_date.replace("-", "")
        )
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        print(f"  腾讯接口成功，{len(df)} 条")
        return df[['close']]
    except Exception as e:
        print(f"  腾讯接口失败：{e}")

    # 备用方案
    try:
        code = symbol[2:]
        if symbol.startswith("sh000") or symbol.startswith("sz399"):
            df = ak.index_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", "")
            )
            df = df.rename(columns={'日期': 'date', '收盘': 'close'})
        else:
            df = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                adjust="qfq"
            )
            df = df.rename(columns={'日期': 'date', '收盘': 'close'})

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        print(f"  备用接口成功，{len(df)} 条")
        return df[['close']]
    except Exception as e2:
        print(f"  备用接口也失败：{e2}")
        raise


# ==================== 策略回测 ====================
def backtest_dual_ma(symbol, name, start_date, end_date,
                     short_window=SHORT_WINDOW, long_window=LONG_WINDOW,
                     commission=COMMISSION, slippage=SLIPPAGE):
    """双均线策略回测"""
    df = get_data(symbol, name, start_date, end_date)
    df = df[(df.index >= start_date) & (df.index <= end_date)].copy()

    # 均线
    df['MA_short'] = df['close'].rolling(short_window).mean()
    df['MA_long'] = df['close'].rolling(long_window).mean()

    # 信号：1=持仓，0=空仓
    df['signal'] = np.where(df['MA_short'] > df['MA_long'], 1, 0)
    df['trade'] = df['signal'].diff()

    # 逐日回测
    nav = 1.0
    position = 0
    entry_price = 0
    entry_date = None
    trades = []
    nav_list = []

    for i in range(len(df)):
        date = df.index[i]
        price = df['close'].iloc[i]

        # 当日净值变动（基于持仓）
        if i > 0 and position == 1:
            daily_ret = (price - df['close'].iloc[i - 1]) / df['close'].iloc[i - 1]
            nav *= (1 + daily_ret)

        # 交易信号
        trade_signal = df['trade'].iloc[i] if not pd.isna(df['trade'].iloc[i]) else 0

        if trade_signal == 1 and position == 0:
            entry_price = price * (1 + slippage)
            entry_date = date
            position = 1
            nav *= (1 - commission)
        elif trade_signal == -1 and position == 1:
            exit_price = price * (1 - slippage)
            ret = (exit_price - entry_price) / entry_price
            nav *= (1 - commission)
            trades.append({
                '买入日期': entry_date.strftime('%Y-%m-%d'),
                '买入价格': round(entry_price, 2),
                '卖出日期': date.strftime('%Y-%m-%d'),
                '卖出价格': round(exit_price, 2),
                '收益率(%)': round(ret * 100, 2)
            })
            position = 0
            entry_price = 0
            entry_date = None

        nav_list.append({'date': date, 'nav': nav})

    # 最后还持仓，平仓
    if position == 1:
        exit_price = df['close'].iloc[-1] * (1 - slippage)
        ret = (exit_price - entry_price) / entry_price
        nav *= (1 - commission)
        trades.append({
            '买入日期': entry_date.strftime('%Y-%m-%d'),
            '买入价格': round(entry_price, 2),
            '卖出日期': df.index[-1].strftime('%Y-%m-%d'),
            '卖出价格': round(exit_price, 2),
            '收益率(%)': round(ret * 100, 2)
        })
        nav_list[-1]['nav'] = nav

    nav_df = pd.DataFrame(nav_list).set_index('date')
    trades_df = pd.DataFrame(trades)

    # 绩效指标
    total_ret = nav - 1
    days = (df.index[-1] - df.index[0]).days
    annual_ret = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0

    daily_rets = nav_df['nav'].pct_change().dropna()
    annual_vol = daily_rets.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

    cummax = nav_df['nav'].cummax()
    drawdown = nav_df['nav'] / cummax - 1
    max_dd = drawdown.min()

    win_rate = (trades_df['收益率(%)'] > 0).sum() / len(trades_df) * 100 if len(trades_df) > 0 else 0

    result = {
        'name': name, 'symbol': symbol,
        'total_ret': total_ret, 'annual_ret': annual_ret,
        'annual_vol': annual_vol, 'sharpe': sharpe,
        'max_dd': max_dd, 'win_rate': win_rate,
        'n_trades': len(trades_df), 'final_nav': nav
    }

    print(f"\n{'=' * 50}")
    print(f"{name}（{symbol}）回测结果")
    print(f"{'=' * 50}")
    print(f"回测期：{df.index[0].date()} 至 {df.index[-1].date()}")
    print(f"总收益率：    {total_ret:.2%}")
    print(f"年化收益率：  {annual_ret:.2%}")
    print(f"年化波动率：  {annual_vol:.2%}")
    print(f"最大回撤：    {max_dd:.2%}")
    print(f"夏普比率：    {sharpe:.2f}")
    print(f"交易次数：    {len(trades_df)}")
    print(f"胜率：        {win_rate:.1f}%")
    print(f"期末净值：    {nav:.4f}")
    print(f"{'=' * 50}")

    return result, trades_df, nav_df


# ==================== 运行 ====================
hs300_result, hs300_trades, hs300_nav = backtest_dual_ma(
    "sh000300", "沪深300", START_DATE, END_DATE
)
mt_result, mt_trades, mt_nav = backtest_dual_ma(
    "sh600519", "贵州茅台", START_DATE, END_DATE
)

# 保存交易记录
if len(hs300_trades) > 0:
    hs300_trades.to_csv("hs300_trades.csv", index=False, encoding="utf-8-sig")
if len(mt_trades) > 0:
    mt_trades.to_csv("mt_trades.csv", index=False, encoding="utf-8-sig")

# ==================== 可视化 ====================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

axes[0, 0].plot(hs300_nav.index, hs300_nav['nav'], color='steelblue', lw=1.5)
axes[0, 0].axhline(1.0, color='gray', ls=':', lw=1)
axes[0, 0].set_title('沪深300 双均线策略净值')
axes[0, 0].grid(True, alpha=0.3)

dd_hs300 = hs300_nav['nav'] / hs300_nav['nav'].cummax() - 1
axes[0, 1].fill_between(hs300_nav.index, dd_hs300, 0, color='crimson', alpha=0.5)
axes[0, 1].set_title('沪深300 回撤曲线')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(mt_nav.index, mt_nav['nav'], color='darkred', lw=1.5)
axes[1, 0].axhline(1.0, color='gray', ls=':', lw=1)
axes[1, 0].set_title('贵州茅台 双均线策略净值')
axes[1, 0].grid(True, alpha=0.3)

dd_mt = mt_nav['nav'] / mt_nav['nav'].cummax() - 1
axes[1, 1].fill_between(mt_nav.index, dd_mt, 0, color='crimson', alpha=0.5)
axes[1, 1].set_title('贵州茅台 回撤曲线')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dual_ma_result.png', dpi=150)
plt.show()

# 对比图
plt.figure(figsize=(14, 7))
plt.plot(hs300_nav.index, hs300_nav['nav'], label='沪深300', color='blue', lw=2)
plt.plot(mt_nav.index, mt_nav['nav'], label='贵州茅台', color='red', lw=2)
plt.axhline(1.0, color='black', ls='--', alpha=0.5)
plt.title('双均线策略净值对比：沪深300 vs 贵州茅台', fontsize=16)
plt.xlabel('日期')
plt.ylabel('净值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dual_ma_comparison.png', dpi=150)
plt.show()

# ==================== 汇总 ====================
print("\n" + "=" * 50)
print("汇总对比")
print("=" * 50)
summary = pd.DataFrame([
    {'标的': hs300_result['name'], '总收益': f"{hs300_result['total_ret']:.2%}",
     '年化收益': f"{hs300_result['annual_ret']:.2%}", '最大回撤': f"{hs300_result['max_dd']:.2%}",
     '夏普': f"{hs300_result['sharpe']:.2f}", '交易次数': hs300_result['n_trades'],
     '胜率': f"{hs300_result['win_rate']:.1f}%"},
    {'标的': mt_result['name'], '总收益': f"{mt_result['total_ret']:.2%}",
     '年化收益': f"{mt_result['annual_ret']:.2%}", '最大回撤': f"{mt_result['max_dd']:.2%}",
     '夏普': f"{mt_result['sharpe']:.2f}", '交易次数': mt_result['n_trades'],
     '胜率': f"{mt_result['win_rate']:.1f}%"},
])
print(summary.to_string(index=False))