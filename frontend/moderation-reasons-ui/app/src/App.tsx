import React, { useState } from 'react'
import './styles.css'
import { apiPredict, apiExplain } from './api'
import LabelChips from './components/LabelChips'
import HeatmapText, { SpansByLabel } from './components/HeatmapText'

export default function App(){
  const [text, setText] = useState<string>("Go back to your country or I will hurt you.")
  const [model, setModel] = useState<'lr'|'bert'>('lr')
  const [pred, setPred] = useState<any>(null)
  const [exp, setExp] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string|undefined>()

  const run = async () => {
    setError(undefined); setLoading(true)
    try {
      const p = await apiPredict(text, model)
      const e = await apiExplain(text, model)
      setPred(p); setExp(e)
    } catch (err:any){
      setError(err.message || 'Unknown error')
    } finally { setLoading(false) }
  }

  const spans: SpansByLabel = exp?.spans || {}

  return (
    <div className="container">
      <h2>Moderation with Reasons</h2>
      <div className="legend">
        <div className="box box-violence"></div> violence
        <div className="box box-sexual"></div> sexual
        <div className="box box-hate"></div> hate
      </div>
      <div className="card">
        <div style={{display:'flex', gap:12}}>
          <textarea className="textarea" value={text} onChange={e=>setText(e.target.value)} />
          <div style={{minWidth: 200}}>
            <label>Model</label><br/>
            <select value={model} onChange={e=>setModel(e.target.value as any)}>
              <option value="lr">LR (fast)</option>
              <option value="bert">BERT (slow)</option>
            </select>
            <br/><br/>
            <button className="button" onClick={run} disabled={loading}>
              {loading ? 'Running…' : 'Predict'}
            </button>
            {error && <div style={{color:'crimson', marginTop:8}}>{error}</div>}
          </div>
        </div>
      </div>

      {pred && (
        <div className="card">
          <h3>Predictions</h3>
          <LabelChips labels={pred.labels} probs={pred.probs} preds={pred.preds} />
          {exp?.reasons && (
            <div style={{marginTop:12}}>
              {Object.entries(exp.reasons).map(([k,v]) => (
                <div key={k} className="reason"><strong>{k}:</strong> {v}</div>
              ))}
            </div>
          )}
        </div>
      )}

      {exp && (
        <div className="card">
          <h3>Highlighted Evidence</h3>
          <HeatmapText text={text} spans={spans} />
        </div>
      )}
    </div>
  )
}
