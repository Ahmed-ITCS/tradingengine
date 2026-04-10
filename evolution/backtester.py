"""
EvoTrade AI - Backtesting Engine
Fast vectorized backtesting using numpy/pandas (vectorbt-style).
Supports multiple strategy types and produces rich performance reports.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BacktestResult:
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return_pct: float
    annualized_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    avg_trade_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    avg_holding_bars: float
    equity_curve: List[float]
    trade_returns: List[float]

    @property
    def composite_score(self) -> float:
        """Weighted composite score for strategy ranking."""
        sharpe = max(min(self.sharpe_ratio, 5), -3)
        ret = self.total_return_pct
        dd = abs(self.max_drawdown_pct)
        wr = self.win_rate_pct / 100
        pf = min(self.profit_factor, 5)

        dd_penalty = max(0, dd - 15) * 2
        score = (sharpe * 25) + (ret * 0.4) + (wr * 20) + (pf * 5) - dd_penalty
        return round(score, 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": round(self.final_capital, 2),
            "total_return_pct": round(self.total_return_pct, 2),
            "annualized_return_pct": round(self.annualized_return_pct, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown_pct, 2),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "win_rate_pct": round(self.win_rate_pct, 1),
            "profit_factor": round(self.profit_factor, 2),
            "total_trades": self.total_trades,
            "avg_trade_return_pct": round(self.avg_trade_return_pct, 3),
            "avg_win_pct": round(self.avg_win_pct, 3),
            "avg_loss_pct": round(self.avg_loss_pct, 3),
            "best_trade_pct": round(self.best_trade_pct, 2),
            "worst_trade_pct": round(self.worst_trade_pct, 2),
            "avg_holding_bars": round(self.avg_holding_bars, 1),
            "composite_score": self.composite_score,
        }


class VectorizedBacktester:
    """
    Fast vectorized backtester — no loops for signal generation.
    Supports: EMA cross, RSI mean-reversion, Bollinger Band squeeze,
    MACD momentum, and combined multi-signal strategies.
    """

    def __init__(self, initial_capital: float = 10000.0, commission_pct: float = 0.001):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct

    def run(self, df: pd.DataFrame, strategy_config: Dict[str, Any],
            symbol: str = "BTC/USDT", timeframe: str = "1h") -> BacktestResult:
        """Run a full backtest and return detailed results."""
        df = df.copy()
        df = self._compute_all_indicators(df)

        strategy_type = strategy_config.get("strategy_type", "ema_cross")
        signals = self._generate_signals(df, strategy_type, strategy_config)

        trades, equity_curve = self._simulate_trades(df, signals, strategy_config)

        return self._compute_metrics(
            trades=trades,
            equity_curve=equity_curve,
            df=df,
            strategy_name=strategy_config.get("name", strategy_type),
            symbol=symbol,
            timeframe=timeframe,
        )

    def _compute_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed for signal generation."""
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # EMAs
        for p in [9, 20, 50, 200]:
            df[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()

        # RSI
        delta = close.diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        df["rsi_14"] = 100 - (100 / (1 + up.rolling(14).mean() / down.rolling(14).mean().replace(0, 1e-10)))
        df["rsi_7"] = 100 - (100 / (1 + up.rolling(7).mean() / down.rolling(7).mean().replace(0, 1e-10)))

        # MACD
        df["macd"] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
        df["macd_signal"] = df["macd"].ewm(span=9).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Bollinger Bands
        df["bb_mid"] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df["bb_upper"] = df["bb_mid"] + 2 * bb_std
        df["bb_lower"] = df["bb_mid"] - 2 * bb_std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
        df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, 1)

        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(span=14).mean()

        # ADX
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        plus_dm[plus_dm < (-low.diff()).clip(lower=0)] = 0
        minus_dm[minus_dm < high.diff().clip(lower=0)] = 0
        plus_di = 100 * plus_dm.ewm(span=14).mean() / df["atr_14"].replace(0, 1)
        minus_di = 100 * minus_dm.ewm(span=14).mean() / df["atr_14"].replace(0, 1)
        dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 1))
        df["adx_14"] = dx.ewm(span=14).mean()

        # Volume MA
        df["vol_ma_20"] = volume.rolling(20).mean()
        df["vol_ratio"] = volume / df["vol_ma_20"].replace(0, 1)

        # VWAP (rolling daily proxy)
        tp = (high + low + close) / 3
        df["vwap"] = (tp * volume).rolling(20).sum() / volume.rolling(20).sum()

        # Momentum
        df["mom_10"] = close.pct_change(10)
        df["mom_20"] = close.pct_change(20)

        return df.fillna(method="ffill").fillna(0)

    def _generate_signals(self, df: pd.DataFrame, strategy_type: str,
                           config: Dict) -> pd.Series:
        """Generate entry signals (1=long, -1=short, 0=flat) vectorized."""
        n = len(df)
        signals = pd.Series(0, index=df.index)

        if strategy_type == "ema_cross":
            fast = config.get("ema_fast", 20)
            slow = config.get("ema_slow", 50)
            fast_key = f"ema_{fast}"
            slow_key = f"ema_{slow}"
            if fast_key in df.columns and slow_key in df.columns:
                cross_up = (df[fast_key] > df[slow_key]) & (df[fast_key].shift(1) <= df[slow_key].shift(1))
                cross_dn = (df[fast_key] < df[slow_key]) & (df[fast_key].shift(1) >= df[slow_key].shift(1))
                signals[cross_up] = 1
                signals[cross_dn] = -1

        elif strategy_type == "rsi_reversal":
            oversold = config.get("rsi_oversold", 30)
            overbought = config.get("rsi_overbought", 70)
            rsi = df["rsi_14"]
            signals[(rsi < oversold) & (rsi.shift(1) >= oversold)] = 1
            signals[(rsi > overbought) & (rsi.shift(1) <= overbought)] = -1

        elif strategy_type == "bb_squeeze":
            # Enter when price breaks out from squeeze
            squeeze = df["bb_width"] < df["bb_width"].rolling(20).quantile(0.2)
            breakout_up = (df["close"] > df["bb_upper"]) & squeeze.shift(1)
            breakout_dn = (df["close"] < df["bb_lower"]) & squeeze.shift(1)
            signals[breakout_up] = 1
            signals[breakout_dn] = -1

        elif strategy_type == "macd_momentum":
            macd_cross_up = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
            macd_cross_dn = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)
            # Confirm with trend
            trend_up = df["close"] > df["ema_50"]
            trend_dn = df["close"] < df["ema_50"]
            signals[macd_cross_up & trend_up] = 1
            signals[macd_cross_dn & trend_dn] = -1

        elif strategy_type == "multi_signal":
            # Combine EMA + RSI + MACD + Volume
            ema_bull = df["ema_20"] > df["ema_50"]
            rsi_ok = (df["rsi_14"] > 45) & (df["rsi_14"] < 75)
            macd_bull = df["macd_hist"] > 0
            vol_ok = df["vol_ratio"] > 1.0

            ema_bear = df["ema_20"] < df["ema_50"]
            rsi_bear_ok = (df["rsi_14"] < 55) & (df["rsi_14"] > 25)
            macd_bear = df["macd_hist"] < 0

            long_cond = ema_bull & rsi_ok & macd_bull & vol_ok
            short_cond = ema_bear & rsi_bear_ok & macd_bear

            long_entry = long_cond & ~long_cond.shift(1).fillna(False)
            short_entry = short_cond & ~short_cond.shift(1).fillna(False)

            signals[long_entry] = 1
            signals[short_entry] = -1

        elif strategy_type == "vwap_reversion":
            price = df["close"]
            vwap = df["vwap"]
            rsi = df["rsi_14"]
            dev = (price - vwap) / vwap
            signals[(dev < -0.01) & (rsi < 40)] = 1
            signals[(dev > 0.01) & (rsi > 60)] = -1

        elif strategy_type == "momentum_breakout":
            high_20 = df["high"].rolling(20).max()
            low_20 = df["low"].rolling(20).min()
            vol_ok = df["vol_ratio"] > 1.5
            adx_trend = df["adx_14"] > 25
            signals[(df["close"] > high_20.shift(1)) & vol_ok & adx_trend] = 1
            signals[(df["close"] < low_20.shift(1)) & vol_ok & adx_trend] = -1

        return signals

    def _simulate_trades(self, df: pd.DataFrame, signals: pd.Series,
                          config: Dict) -> Tuple[List[Dict], List[float]]:
        """
        Simulate trades from signals with SL/TP management.
        Returns (trades list, equity curve).
        """
        risk_pct = config.get("risk_per_trade", 0.02)
        rr_ratio = config.get("take_profit_ratio", 2.0)
        sl_atr_mult = config.get("sl_atr_mult", 1.5)
        long_only = config.get("long_only", True)

        capital = self.initial_capital
        equity_curve = [capital]
        trades = []

        in_position = False
        entry_price = 0.0
        entry_idx = 0
        position_side = 0  # 1=long, -1=short
        sl_price = 0.0
        tp_price = 0.0
        position_size = 0.0

        close = df["close"].values
        atr = df["atr_14"].values
        sig = signals.values

        for i in range(1, len(df)):
            price = close[i]

            if not in_position:
                if sig[i] == 1 or (sig[i] == -1 and not long_only):
                    side = sig[i]
                    atr_val = atr[i] if atr[i] > 0 else price * 0.01
                    sl_dist = atr_val * sl_atr_mult

                    if side == 1:
                        sl_price = price - sl_dist
                        tp_price = price + sl_dist * rr_ratio
                    else:
                        sl_price = price + sl_dist
                        tp_price = price - sl_dist * rr_ratio

                    # Position sizing (fixed risk %)
                    risk_amount = capital * risk_pct
                    position_size = risk_amount / sl_dist if sl_dist > 0 else 0

                    if position_size > 0 and price > 0:
                        cost = position_size * price * self.commission_pct
                        capital -= cost
                        in_position = True
                        entry_price = price
                        entry_idx = i
                        position_side = side

            else:
                # Check exit conditions
                exit_price = None
                exit_reason = None

                if position_side == 1:
                    if price <= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                    elif price >= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                    elif sig[i] == -1:
                        exit_price = price
                        exit_reason = "signal"
                else:
                    if price >= sl_price:
                        exit_price = sl_price
                        exit_reason = "SL"
                    elif price <= tp_price:
                        exit_price = tp_price
                        exit_reason = "TP"
                    elif sig[i] == 1:
                        exit_price = price
                        exit_reason = "signal"

                if exit_price is not None:
                    if position_side == 1:
                        pnl = (exit_price - entry_price) * position_size
                    else:
                        pnl = (entry_price - exit_price) * position_size

                    commission = abs(pnl) * self.commission_pct
                    net_pnl = pnl - commission
                    capital += net_pnl

                    ret_pct = (exit_price - entry_price) / entry_price * position_side * 100

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "side": position_side,
                        "size": position_size,
                        "pnl": net_pnl,
                        "ret_pct": ret_pct,
                        "exit_reason": exit_reason,
                        "holding_bars": i - entry_idx,
                    })

                    in_position = False
                    position_size = 0.0

            equity_curve.append(capital)

        # Close any open position at end
        if in_position:
            price = close[-1]
            if position_side == 1:
                pnl = (price - entry_price) * position_size
            else:
                pnl = (entry_price - price) * position_size
            capital += pnl - abs(pnl) * self.commission_pct
            equity_curve[-1] = capital

        return trades, equity_curve

    def _compute_metrics(self, trades: List[Dict], equity_curve: List[float],
                          df: pd.DataFrame, strategy_name: str,
                          symbol: str, timeframe: str) -> BacktestResult:
        """Compute comprehensive performance metrics from trades."""
        initial = self.initial_capital
        final = equity_curve[-1] if equity_curve else initial
        eq = np.array(equity_curve)

        # Returns
        total_return = (final / initial - 1) * 100
        n_bars = len(df)
        bars_per_year = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760,
                          "4h": 2190, "1d": 365, "1w": 52}.get(timeframe, 8760)
        years = n_bars / bars_per_year
        ann_return = ((final / initial) ** (1 / max(years, 0.01)) - 1) * 100 if final > 0 else -100

        # Drawdown
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1)
        max_dd = float(dd.min()) * 100

        # Sharpe / Sortino
        if len(eq) > 1:
            bar_returns = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1)
            ann_factor = np.sqrt(bars_per_year)
            mean_r = bar_returns.mean()
            std_r = bar_returns.std()
            sharpe = float(mean_r / std_r * ann_factor) if std_r > 0 else 0.0
            downside = bar_returns[bar_returns < 0]
            sortino_std = downside.std() if len(downside) > 0 else std_r
            sortino = float(mean_r / sortino_std * ann_factor) if sortino_std > 0 else 0.0
        else:
            sharpe = sortino = 0.0

        calmar = ann_return / abs(max_dd) if max_dd < 0 else 0.0

        # Trade stats
        if trades:
            rets = [t["ret_pct"] for t in trades]
            wins = [r for r in rets if r > 0]
            losses = [r for r in rets if r <= 0]
            win_rate = len(wins) / len(rets) * 100
            avg_win = np.mean(wins) if wins else 0.0
            avg_loss = np.mean(losses) if losses else 0.0
            best = max(rets)
            worst = min(rets)
            avg_ret = np.mean(rets)
            avg_hold = np.mean([t["holding_bars"] for t in trades])
            gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
            gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            trade_returns = rets
        else:
            win_rate = avg_win = avg_loss = best = worst = avg_ret = avg_hold = profit_factor = 0.0
            trade_returns = []

        start_date = str(df.index[0])[:10] if len(df) > 0 else ""
        end_date = str(df.index[-1])[:10] if len(df) > 0 else ""

        return BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial,
            final_capital=final,
            total_return_pct=total_return,
            annualized_return_pct=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown_pct=max_dd,
            calmar_ratio=calmar,
            win_rate_pct=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_return_pct=avg_ret,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            best_trade_pct=best,
            worst_trade_pct=worst,
            avg_holding_bars=avg_hold,
            equity_curve=equity_curve[-200:],  # last 200 points
            trade_returns=trade_returns,
        )


def run_backtest(
    df: pd.DataFrame,
    strategy_type: str,
    strategy_name: str,
    config: Dict[str, Any] = None,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    initial_capital: float = 10000.0,
) -> BacktestResult:
    """Convenience wrapper for running a single backtest."""
    bt = VectorizedBacktester(initial_capital=initial_capital)
    full_config = {"strategy_type": strategy_type, "name": strategy_name, **(config or {})}
    return bt.run(df, full_config, symbol, timeframe)


def run_strategy_comparison(
    df: pd.DataFrame,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> List[BacktestResult]:
    """Run all built-in strategies and return ranked results."""
    strategies = [
        ("ema_cross", "EMA Cross 20/50", {"ema_fast": 20, "ema_slow": 50, "take_profit_ratio": 2.0}),
        ("ema_cross", "EMA Cross 9/21", {"ema_fast": 9, "ema_slow": 20, "take_profit_ratio": 1.8}),
        ("rsi_reversal", "RSI Mean Reversion", {"rsi_oversold": 30, "rsi_overbought": 70, "take_profit_ratio": 1.5}),
        ("bb_squeeze", "Bollinger Squeeze Breakout", {"take_profit_ratio": 2.5}),
        ("macd_momentum", "MACD Momentum", {"take_profit_ratio": 2.0}),
        ("multi_signal", "Multi-Signal Alpha", {"take_profit_ratio": 2.2, "risk_per_trade": 0.015}),
        ("vwap_reversion", "VWAP Reversion", {"take_profit_ratio": 1.8}),
        ("momentum_breakout", "20-Period Breakout", {"take_profit_ratio": 3.0, "sl_atr_mult": 2.0}),
    ]

    results = []
    bt = VectorizedBacktester(initial_capital=10000.0)

    for strat_type, name, config in strategies:
        try:
            full_config = {"strategy_type": strat_type, "name": name, **config}
            result = bt.run(df, full_config, symbol, timeframe)
            results.append(result)
        except Exception as e:
            pass

    results.sort(key=lambda r: r.composite_score, reverse=True)
    return results
