import React, { useState } from 'react'

type Row = Record<string, any>

export default function UploadPanel(){
  const [file, setFile] = useState<File | null>(null)
  const [rows, setRows] = useState<Row[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [mode, setMode] = useState<'predict'|'eval'>('predict')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null
    setFile(f)
    setRows([]); setMetrics(null); setError(null)
  }

  const run = async () => {
    if (!file) return
    setLoading(true); setError(null); setRows([]); setMetrics(null)
    const form = new FormData()
    form.append('file', file)
    try{
      const url = mode === 'predict' ? 'http://localhost:8000/batch_predict' : 'http://localhost:8000/batch_eval'
      const res = await fetch(url, { method: 'POST', body: form })
      if (!res.ok) throw new Error('API error')
      const data = await res.json()
      setRows(data.rows || [])
      if (mode === 'eval') setMetrics({ micro_f1: data.micro_f1, macro_f1: data.macro_f1, per_label: data.per_label })
    }catch(e:any){
      setError(String(e))
    }finally{
      setLoading(false)
    }
  }

  const headers = rows.length ? Object.keys(rows[0]) : []

  return (
    <div className="card">
      <h3>Batch CSV</h3>
      <p>CSV must include a <code>text</code> column. Optional label columns: <code>violence</code>, <code>sexual</code>, <code>hate</code> (0/1).</p>
      <div style={{display:'flex', gap:12, alignItems:'center'}}>
        <input type="file" accept=".csv" onChange={onChange} />
        <select value={mode} onChange={e=>setMode(e.target.value as any)}>
          <option value="predict">Predict only</option>
          <option value="eval">Evaluate (if labels provided)</option>
        </select>
        <button className="button" disabled={!file || loading} onClick={run}>
          {loading ? 'Running…' : 'Run'}
        </button>
        {error && <span style={{color:'crimson'}}>{error}</span>}
      </div>

      {metrics && (
        <div style={{marginTop:12}}>
          <strong>Metrics:</strong> Micro-F1 {metrics.micro_f1.toFixed(3)} | Macro-F1 {metrics.macro_f1.toFixed(3)}
        </div>
      )}

      {rows.length > 0 && (
        <div style={{overflow:'auto', marginTop:12, maxHeight: 360}}>
          <table style={{borderCollapse:'collapse', width:'100%'}}>
            <thead>
              <tr>
                {headers.map(h => <th key={h} style={{textAlign:'left', borderBottom:'1px solid #eee'}}>{h}</th>)}
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={i}>
                  {headers.map(h => <td key={h} style={{borderBottom:'1px solid #f5f5f5', padding:'4px 6px'}}>{String(r[h])}</td>)}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
