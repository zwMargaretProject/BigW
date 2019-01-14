from matplotlib.ticker import FuncFormatter
from matplotlib import cm
from datetime import datetime

import quickBacktesting.quickPerformance as perf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
import os



class QuickStatistics(object):
    def __init__(self, symbol, returns, equity_s, title, periods=252, benchmark=None, rolling_sharpe=False, log_scale=False):
        self.symbol = symbol
        self.returns = returns
        self.benchmark = benchmark
        self.title = '\n'.join(title)
        self.equity_s = equity_s
        self.periods = periods
        self.rolling_sharpe = rolling_sharpe
        self.cum_returns = self.returns.cumsum()
        self.log_scale = log_scale

    def get_results(self):
        """
        Return a dict with all important results & stats.
        """
        # Returns
        returns_s = self.returns
        equity_s = self.equity_s
        # Rolling Annualised Sharpe
        rolling = returns_s.rolling(window=self.periods)
        rolling_sharpe_s = np.sqrt(self.periods) * (rolling.mean() / rolling.std())

        # Cummulative Returns
        cum_returns_s = np.exp(np.log(1 + returns_s).cumsum())

        # Drawdown, max drawdown, max drawdown duration
        dd_s, max_dd, dd_dur = perf.create_drawdowns(cum_returns_s)

        statistics = {}

        # Equity statistics
        statistics["sharpe"] = perf.create_sharpe_ratio(returns_s, self.periods)
        statistics["drawdowns"] = dd_s
        # TODO: need to have max_drawdown so it can be printed at end of test
        statistics["max_drawdown"] = max_dd
        statistics["max_drawdown_pct"] = max_dd
        statistics["max_drawdown_duration"] = dd_dur
        statistics["equity"] = equity_s
        statistics["returns"] = returns_s
        statistics["rolling_sharpe"] = rolling_sharpe_s
        statistics["cum_returns"] = cum_returns_s

        # Benchmark statistics if benchmark ticker specified
        if self.benchmark is not None:
            equity_b = pd.Series(self.equity_benchmark).sort_index()
            returns_b = equity_b.pct_change().fillna(0.0)
            rolling_b = returns_b.rolling(window=self.periods)
            rolling_sharpe_b = np.sqrt(self.periods) * (
                rolling_b.mean() / rolling_b.std()
            )
            cum_returns_b = np.exp(np.log(1 + returns_b).cumsum())
            dd_b, max_dd_b, dd_dur_b = perf.create_drawdowns(cum_returns_b)
            statistics["sharpe_b"] = perf.create_sharpe_ratio(returns_b)
            statistics["drawdowns_b"] = dd_b
            statistics["max_drawdown_pct_b"] = max_dd_b
            statistics["max_drawdown_duration_b"] = dd_dur_b
            statistics["equity_b"] = equity_b
            statistics["returns_b"] = returns_b
            statistics["rolling_sharpe_b"] = rolling_sharpe_b
            statistics["cum_returns_b"] = cum_returns_b

        return statistics


    def _plot_equity(self, stats, ax=None, **kwargs):
        """
        Plots cumulative rolling returns versus some benchmark.
        """
        def format_two_dec(x, pos):
            return '%.2f' % x

        equity = stats['cum_returns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if self.benchmark is not None:
            benchmark = stats['cum_returns_b']
            benchmark.plot(
                lw=2, color='gray', label=self.benchmark, alpha=0.60,
                ax=ax, **kwargs
            )

        equity.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axhline(1.0, linestyle='--', color='black', lw=1)
        ax.set_ylabel('Cumulative returns')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        if self.log_scale:
            ax.set_yscale('log')

        return ax


    def _plot_rolling_sharpe(self, stats, ax=None, **kwargs):
        """
        Plots the curve of rolling Sharpe ratio.
        """
        def format_two_dec(x, pos):
            return '%.2f' % x

        sharpe = stats['rolling_sharpe']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_two_dec)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.xaxis.set_tick_params(reset=True)
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        if self.benchmark is not None:
            benchmark = stats['rolling_sharpe_b']
            benchmark.plot(
                lw=2, color='gray', label=self.benchmark, alpha=0.60,
                ax=ax, **kwargs
            )

        sharpe.plot(lw=2, color='green', alpha=0.6, x_compat=False,
                    label='Backtest', ax=ax, **kwargs)

        ax.axvline(sharpe.index[252], linestyle="dashed", c="gray", lw=2)
        ax.set_ylabel('Rolling Annualised Sharpe')
        ax.legend(loc='best')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')

        return ax

    def _plot_drawdown(self, stats, ax=None, **kwargs):
        """
        Plots the underwater curve
        """
        def format_perc(x, pos):
            return '%.0f%%' % x

        drawdown = stats['drawdowns']

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
        ax.yaxis.grid(linestyle=':')
        ax.xaxis.set_tick_params(reset=True)
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.grid(linestyle=':')

        underwater = -100 * drawdown
        underwater.plot(ax=ax, lw=2, kind='area', color='red', alpha=0.3, **kwargs)
        ax.set_ylabel('')
        ax.set_xlabel('')
        plt.setp(ax.get_xticklabels(), visible=True, rotation=0, ha='center')
        ax.set_title('Drawdown (%)', fontweight='bold')
        return ax

    def _plot_txt_curve(self, stats, ax=None, **kwargs):
        """
        Outputs the statistics for the equity curve.
        """
        def format_perc(x, pos):
            return '%.0f%%' % x

        returns = stats["returns"]
        cum_returns = stats['cum_returns']

        if 'positions' not in stats:
            trd_yr = 0
        else:
            positions = stats['positions']
            trd_yr = positions.shape[0] / ((returns.index[-1] - returns.index[0]).days / 365.0)

        if ax is None:
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(format_perc)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        tot_ret = cum_returns[-1] - 1.0
        cagr = perf.create_cagr(cum_returns, self.periods)
        sharpe = perf.create_sharpe_ratio(returns, self.periods)
        sortino = perf.create_sortino_ratio(returns, self.periods)
        rsq = perf.rsquared(range(cum_returns.shape[0]), cum_returns)
        dd, dd_max, dd_dur = perf.create_drawdowns(cum_returns)

        ax.text(0.25, 8.9, 'Total Return', fontsize=8)
        ax.text(7.50, 8.9, '{:.0%}'.format(tot_ret), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 7.9, 'CAGR', fontsize=8)
        ax.text(7.50, 7.9, '{:.2%}'.format(cagr), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 6.9, 'Sharpe Ratio', fontsize=8)
        ax.text(7.50, 6.9, '{:.2f}'.format(sharpe), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 5.9, 'Sortino Ratio', fontsize=8)
        ax.text(7.50, 5.9, '{:.2f}'.format(sortino), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 4.9, 'Annual Volatility', fontsize=8)
        ax.text(7.50, 4.9, '{:.2%}'.format(returns.std() * np.sqrt(252)), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 3.9, 'R-Squared', fontsize=8)
        ax.text(7.50, 3.9, '{:.2f}'.format(rsq), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 2.9, 'Max Daily Drawdown', fontsize=8)
        ax.text(7.50, 2.9, '{:.2%}'.format(dd_max), color='red', fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 1.9, 'Max Drawdown Duration', fontsize=8)
        ax.text(7.50, 1.9, '{:.0f}'.format(dd_dur), fontweight='bold', horizontalalignment='right', fontsize=8)

        ax.text(0.25, 0.9, 'Trades per Year', fontsize=8)
        ax.text(7.50, 0.9, '{:.1f}'.format(trd_yr), fontweight='bold', horizontalalignment='right', fontsize=8)
        ax.set_title('Curve', fontweight='bold')

        if self.benchmark is not None:
            returns_b = stats['returns_b']
            equity_b = stats['cum_returns_b']
            tot_ret_b = equity_b[-1] - 1.0
            cagr_b = perf.create_cagr(equity_b)
            sharpe_b = perf.create_sharpe_ratio(returns_b)
            sortino_b = perf.create_sortino_ratio(returns_b)
            rsq_b = perf.rsquared(range(equity_b.shape[0]), equity_b)
            dd_b, dd_max_b, dd_dur_b = perf.create_drawdowns(equity_b)

            ax.text(9.75, 8.9, '{:.0%}'.format(tot_ret_b), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 7.9, '{:.2%}'.format(cagr_b), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 6.9, '{:.2f}'.format(sharpe_b), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 5.9, '{:.2f}'.format(sortino_b), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 4.9, '{:.2%}'.format(returns_b.std() * np.sqrt(252)), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 3.9, '{:.2f}'.format(rsq_b), fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 2.9, '{:.2%}'.format(dd_max_b), color='red', fontweight='bold', horizontalalignment='right', fontsize=8)
            ax.text(9.75, 1.9, '{:.0f}'.format(dd_dur_b), fontweight='bold', horizontalalignment='right', fontsize=8)

            ax.set_title('Curve vs. Benchmark', fontweight='bold')

        ax.grid(False)
        ax.spines['top'].set_linewidth(2.0)
        ax.spines['bottom'].set_linewidth(2.0)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_ylabel('')
        ax.set_xlabel('')

        ax.axis([0, 10, 0, 10])
        return ax


    def plot_results(self, filename=None):
        """
        Plot the Tearsheet
        """
        rc = {
            'lines.linewidth': 1.0,
            'axes.facecolor': '0.995',
            'figure.facecolor': '0.97',
            'font.family': 'serif',
            'font.serif': 'Ubuntu',
            'font.monospace': 'Ubuntu Mono',
            'font.size': 10,
            'axes.labelsize': 10,
            'axes.labelweight': 'bold',
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 10,
            'figure.titlesize': 12
        }
        sns.set_context(rc)
        sns.set_style("whitegrid")
        sns.set_palette("deep", desat=.6)

        if self.rolling_sharpe:
            offset_index = 1
        else:
            offset_index = 0
        vertical_sections = 5 + offset_index
        fig = plt.figure(figsize=(10, vertical_sections * 3.5))
        fig.suptitle(self.title, y=0.94, weight='bold')
        gs = gridspec.GridSpec(vertical_sections, 3, wspace=0.25, hspace=0.5)

        stats = self.get_results()
        ax_equity = plt.subplot(gs[:2, :])
        if self.rolling_sharpe:
            ax_sharpe = plt.subplot(gs[2, :])
        ax_drawdown = plt.subplot(gs[2 + offset_index, :])
        ax_txt_curve = plt.subplot(gs[3 + offset_index, 0:1])

        self._plot_equity(stats, ax=ax_equity)
        if self.rolling_sharpe:
            self._plot_rolling_sharpe(stats, ax=ax_sharpe)
        self._plot_drawdown(stats, ax=ax_drawdown)
        self._plot_txt_curve(stats, ax=ax_txt_curve)

        # Plot the figure
        plt.show(block=False)

