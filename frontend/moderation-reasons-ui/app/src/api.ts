export async function apiPredict(text: string, model: 'lr'|'bert'='lr') {
  const res = await fetch('http://localhost:8000/predict', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model })
  });
  if (!res.ok) throw new Error('API error');
  return res.json();
}

export async function apiExplain(text: string, model: 'lr'|'bert'='lr') {
  const res = await fetch('http://localhost:8000/explain', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, model })
  });
  if (!res.ok) throw new Error('API error');
  return res.json();
}
