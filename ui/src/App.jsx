import { useState, useEffect } from 'react';
import { Activity, Target, Trophy, AlertCircle, ChevronUp, ChevronDown } from 'lucide-react';
import './index.css';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeStat, setActiveStat] = useState('Points');

  const stats = [
    { id: 'Points', label: 'Points', icon: Target },
    { id: 'Rebounds', label: 'Rebounds', icon: Activity },
    { id: 'Assists', label: 'Assists', icon: Trophy }
  ];

  /* Fetch predictions based on active stat */
  const fetchPredictions = async (statCode) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`./api/latest_${statCode}.json`);
      if (!response.ok) throw new Error('Failed to fetch predictions');
      const data = await response.json();
      if (data.status === 'success') {
        const sortedData = data.data.sort((a, b) => Math.abs(b.Predicted - b.Line) - Math.abs(a.Predicted - a.Line));
        setPredictions(sortedData || []);
      } else {
        throw new Error(data.message || 'Error processing predictions');
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPredictions(activeStat);
  }, [activeStat]);

  return (
    <div className="app-container">
      <header className="app-header">
        <h1 className="app-title">BetHoopsXG</h1>
        <p className="app-subtitle">AI-Powered NBA Prop Predictions</p>
      </header>

      <main>
        <div className="controls-container">
          {stats.map((stat) => (
            <button
              key={stat.id}
              onClick={() => setActiveStat(stat.id)}
              className={`stat-btn ${activeStat === stat.id ? 'active' : ''}`}
            >
              <stat.icon size={18} style={{ marginRight: '8px', display: 'inline' }}/>
              {stat.label}
            </button>
          ))}
        </div>

        <div className="glass-panel">
          {loading ? (
            <div className="loading-container">
              <div className="spinner"></div>
              <p>Analyzing {activeStat} data & updating models... This may take a minute if fetching new boxscores.</p>
            </div>
          ) : error ? (
            <div className="error-message">
              <AlertCircle size={48} style={{ margin: '0 auto 1rem' }} />
              <h3>Failed to load predictions</h3>
              <p>{error}</p>
            </div>
          ) : predictions.length === 0 ? (
            <div className="error-message">
              <AlertCircle size={48} style={{ margin: '0 auto 1rem' }} />
              <h3>No props available right now.</h3>
              <p>Mabye there are no games today?</p>
            </div>
          ) : (
            <div className="predictions-grid">
              {predictions.map((pred, i) => {
                const diff = pred.Predicted - pred.Line;
                const isOver = diff > 0;
                // Exclude props that are too close to call
                if (Math.abs(diff) < 0.2) return null;
                
                return (
                  <div key={i} className="prediction-card">
                    <div className={`recommendation-badge ${isOver ? 'recommendation-over' : 'recommendation-under'}`}>
                      {isOver ? 'OVER' : 'UNDER'} {isOver ? <ChevronUp size={14} style={{display:'inline', verticalAlign:'middle'}}/> : <ChevronDown size={14} style={{display:'inline', verticalAlign:'middle'}}/>}
                    </div>
                    
                    <div className="card-header">
                      <h3 className="player-name">{pred.Player}</h3>
                      <span className="matchup-badge">{pred.MATCHUP || "vs OPP"}</span>
                    </div>

                    <div className="stats-row">
                      <div className="stat-block">
                        <span className="stat-label">Line ({pred.Stat || activeStat})</span>
                        <span className="stat-value">{pred.Line}</span>
                      </div>
                      <div className="stat-block" style={{ textAlign: 'right' }}>
                        <span className="stat-label">Model Projection</span>
                        <span className="stat-value" style={{ color: isOver ? 'var(--over-color)' : 'var(--under-color)' }}>
                          {pred.Predicted.toFixed(1)}
                        </span>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

export default App;
