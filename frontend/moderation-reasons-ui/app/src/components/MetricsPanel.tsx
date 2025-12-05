import React, { useEffect, useState } from 'react'

type PerLabel = { [k: string]: { ap: number, threshold: number } }

export default function MetricsPanel(){
  const [data, setData] = useState<{model:string, labels:string[], micro_f1:number, macro_f1:number, per_label:PerLabel} | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('http://localhost:8000/metrics')
      .then(r => r.json())
      .then(setData)
      .catch(e => setError(String(e)))
  }, [])

  return (
    <div className="card">
      <h3>Model Metrics</h3>
      {error && <div style={{color:'crimson'}}>{error}</div>}
      {!data ? <div>Loading…</div> : (
        <div>
          <div style={{display:'flex', gap:16, marginBottom:12}}>
            <div><strong>Micro-F1:</strong> {data.micro_f1.toFixed(3)}</div>
            <div><strong>Macro-F1:</strong> {data.macro_f1.toFixed(3)}</div>
          </div>
          <table style={{borderCollapse:'collapse', width:'100%'}}>
            <thead>
              <tr>
                <th style={{textAlign:'left', borderBottom:'1px solid #eee'}}>Label</th>
                <th style={{textAlign:'right', borderBottom:'1px solid #eee'}}>AP</th>
                <th style={{textAlign:'right', borderBottom:'1px solid #eee'}}>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {data.labels.map(l => (
                <tr key={l}>
                  <td>{l}</td>
                  <td style={{textAlign:'right'}}>{(data.per_label[l]?.ap ?? 0).toFixed(3)}</td>
                  <td style={{textAlign:'right'}}>{(data.per_label[l]?.threshold ?? 0.5).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
