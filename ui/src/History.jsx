import { useState, useEffect } from 'react';
import { CheckCircle, XCircle, Filter, TrendingUp, Calendar, BarChart3 } from 'lucide-react';

function History() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterHighConf, setFilterHighConf] = useState(false);
  const [filterStat, setFilterStat] = useState('All');

  useEffect(() => {
    fetch('./api/history.json')
      .then(res => res.ok ? res.json() : [])
      .then(data => setHistory(Array.isArray(data) ? data : []))
      .catch(() => setHistory([]))
      .finally(() => setLoading(false));
  }, []);

  const filtered = history.filter(entry => {
    if (filterHighConf && entry.delta < 1.0) return false;
    if (filterStat !== 'All' && entry.stat !== filterStat) return false;
    return true;
  });

  // Group by date for daily summaries
  const dailySummaries = {};
  filtered.forEach(entry => {
    if (!dailySummaries[entry.date]) {
      dailySummaries[entry.date] = { wins: 0, losses: 0, total: 0 };
    }
    dailySummaries[entry.date].total++;
    if (entry.won) dailySummaries[entry.date].wins++;
    else dailySummaries[entry.date].losses++;
  });

  const sortedDates = Object.keys(dailySummaries).sort().reverse();

  const totalWins = filtered.filter(e => e.won).length;
  const totalLosses = filtered.length - totalWins;
  const winRate = filtered.length > 0 ? ((totalWins / filtered.length) * 100).toFixed(1) : '0.0';

  if (loading) {
    return (
      <div className="glass-panel">
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading performance history...</p>
        </div>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="glass-panel">
        <div className="error-message" style={{ color: 'var(--text-muted)', background: 'var(--card-bg)', borderColor: 'var(--card-border)' }}>
          <BarChart3 size={48} style={{ margin: '0 auto 1rem', opacity: 0.5 }} />
          <h3>No History Yet</h3>
          <p>Performance tracking will begin after the first full day of predictions. Check back tomorrow!</p>
        </div>
      </div>
    );
  }

  return (
    <div className="glass-panel">
      {/* Summary Cards */}
      <div className="history-summary">
        <div className="summary-card summary-wins">
          <CheckCircle size={24} />
          <div>
            <span className="summary-number">{totalWins}</span>
            <span className="summary-label">Wins</span>
          </div>
        </div>
        <div className="summary-card summary-losses">
          <XCircle size={24} />
          <div>
            <span className="summary-number">{totalLosses}</span>
            <span className="summary-label">Losses</span>
          </div>
        </div>
        <div className="summary-card summary-rate">
          <TrendingUp size={24} />
          <div>
            <span className="summary-number">{winRate}%</span>
            <span className="summary-label">Win Rate</span>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="history-filters">
        <button
          className={`filter-btn ${filterHighConf ? 'active' : ''}`}
          onClick={() => setFilterHighConf(!filterHighConf)}
        >
          <Filter size={14} style={{ marginRight: '6px', display: 'inline' }} />
          High Confidence Only (Δ≥1)
        </button>
        {['All', 'Points', 'Rebounds', 'Assists'].map(stat => (
          <button
            key={stat}
            className={`filter-btn ${filterStat === stat ? 'active' : ''}`}
            onClick={() => setFilterStat(stat)}
          >
            {stat}
          </button>
        ))}
      </div>

      {/* Daily Breakdown */}
      {sortedDates.map(date => {
        const summary = dailySummaries[date];
        const dayEntries = filtered.filter(e => e.date === date);
        const dayRate = ((summary.wins / summary.total) * 100).toFixed(0);
        return (
          <div key={date} className="history-day">
            <div className="day-header">
              <div className="day-date">
                <Calendar size={16} style={{ marginRight: '8px', display: 'inline', verticalAlign: 'middle' }} />
                {new Date(date + 'T12:00:00').toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
              </div>
              <div className="day-stats">
                <span className="day-record" style={{ color: 'var(--over-color)' }}>{summary.wins}W</span>
                <span style={{ color: 'var(--text-muted)', margin: '0 4px' }}>-</span>
                <span className="day-record" style={{ color: 'var(--under-color)' }}>{summary.losses}L</span>
                <span className="day-pct">({dayRate}%)</span>
              </div>
            </div>
            <div className="history-table-wrapper">
              <table className="history-table">
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Stat</th>
                    <th>Line</th>
                    <th>Predicted</th>
                    <th>Actual</th>
                    <th>Bet</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {dayEntries.map((entry, i) => (
                    <tr key={i} className={entry.won ? 'row-win' : 'row-loss'}>
                      <td className="cell-player">{entry.player}</td>
                      <td>{entry.stat}</td>
                      <td>{entry.line}</td>
                      <td style={{ color: entry.bet === 'Over' ? 'var(--over-color)' : 'var(--under-color)' }}>
                        {entry.predicted}
                      </td>
                      <td style={{ fontWeight: 600 }}>{entry.actual}</td>
                      <td>
                        <span className={`bet-badge ${entry.bet === 'Over' ? 'bet-over' : 'bet-under'}`}>
                          {entry.bet}
                        </span>
                      </td>
                      <td>
                        {entry.won
                          ? <CheckCircle size={18} color="var(--over-color)" />
                          : <XCircle size={18} color="var(--under-color)" />
                        }
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default History;
