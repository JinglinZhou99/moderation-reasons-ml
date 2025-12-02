import React from 'react'

export type Span = { start: number; end: number; text: string }
export type SpansByLabel = { [label: string]: Span[] }

const COLORS: Record<string,string> = {
  violence: 'rgba(255, 0, 0, 0.22)',
  sexual:   'rgba(255, 165, 0, 0.22)',
  hate:     'rgba(0, 255, 0, 0.22)'
}

function buildMask(text: string, spans: SpansByLabel){
  const arr = Array.from(text).map(ch => ({ ch, bg: '' }))
  Object.entries(spans).forEach(([label, list]) => {
    list.forEach(s => {
      for(let i=s.start;i<s.end;i++){
        if (arr[i]) arr[i].bg = COLORS[label]
      }
    })
  })
  return arr
}

export default function HeatmapText({ text, spans }: { text: string, spans: SpansByLabel }){
  const masked = buildMask(text, spans)
  return (
    <p style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
      {masked.map((t,i) => (
        <span key={i} style={{ background: t.bg }}>{t.ch}</span>
      ))}
    </p>
  )
}
