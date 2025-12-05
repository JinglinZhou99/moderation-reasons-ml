import React from 'react'

type Props = { labels: string[]; probs: number[]; preds: number[] }

const toPct = (x: number) => Math.round(x * 100)

export default function LabelChips({ labels, probs, preds }: Props){
  return (
    <div className="chips">
      {labels.map((l, i) => (
        <div key={l} className="chip">
          <strong>{l}</strong> &nbsp; {toPct(probs[i])}%
          <div className="progress" style={{ width: 140, marginTop: 4 }}>
            <div style={{ width: `${toPct(probs[i])}%` }} />
          </div>
          <div style={{ fontSize: 12, color: preds[i] ? '#0b7' : '#777' }}>
            {preds[i] ? 'flagged' : 'not flagged'}
          </div>
        </div>
      ))}
    </div>
  )
}
