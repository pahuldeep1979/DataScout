import { useEffect, useState } from 'react';
import {
  ArrowDownToLine, ArrowRight, BarChart3, CheckCircle2, CircleAlert, FileText, FlaskConical,
  FolderOpen, GitCompareArrows, Globe, Layers, ListFilter, Search, ShieldCheck, Upload,
} from 'lucide-react';
import './App.css';

const outcomes = {
  supported: { label: 'Supported answer', icon: CheckCircle2 },
  conflict: { label: 'Conflicting sources', icon: GitCompareArrows },
  insufficient: { label: 'Insufficient evidence', icon: CircleAlert },
};
const metrics = { bm25_only: 'BM25', vector_only: 'Local vector', hybrid_rrf: 'Hybrid RRF' };

async function api(path, options) {
  let response;
  try {
    response = await fetch(`/api${path}`, options);
  } catch {
    throw new Error('Could not reach DataScout. Check that the server is running.');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === 'string' ? body.detail : 'DataScout could not complete this request.');
  }
  return response.json();
}

function EvidenceCard({ item, index, workspace, conflicting }) {
  const label = item.page ? `${item.source} · p. ${item.page}` : item.source;
  const link = item.url || (workspace === 'demo'
    ? `/api/demo/documents/${encodeURIComponent(item.source)}#page=${item.page}`
    : null);
  return (
    <li className={conflicting ? 'evidence-card conflicting' : 'evidence-card'}>
      <div className="evidence-meta">
        <span className="citation-num">{index + 1}</span>
        <span>{label}</span>
        {link && <a href={link} target="_blank" rel="noreferrer" aria-label={`Open ${label}`}>
          <ArrowRight size={16} />
        </a>}
      </div>
      {item.value && conflicting && <strong className="evidence-value">{item.value}</strong>}
      <blockquote>{item.quote}</blockquote>
    </li>
  );
}

function Result({ result, workspace }) {
  const { label, icon: Icon } = outcomes[result.verdict];
  return (
    <section className="result" aria-labelledby="result-heading">
      <div className="result-info">
        <span className={`status ${result.verdict}`}><Icon size={15} /> {label}</span>
        <span className="muted timing">{result.latency_ms} ms{result.cache_hit ? ' · Cached' : ''}</span>
      </div>
      <h2 id="result-heading">{result.question}</h2>
      <p className="muted result-scope">Searched {result.scope || (workspace === 'demo' ? 'all demo notices' : 'all uploaded PDFs')}</p>
      {result.source_mode === 'web_fallback' && <p className="web-label"><Globe size={15} /> Public web answer for “{result.publicQuestion}”</p>}
      <p className="answer">{result.answer}</p>
      {result.evidence.length > 0 && (
        <>
          <h3>{result.source_mode === 'web_fallback' ? 'Web evidence' : 'Document evidence'}</h3>
          <ul className="evidence-list">
            {result.evidence.map((item, index) => (
              <EvidenceCard key={`${item.chunk_id || item.url}-${index}`} item={item}
                index={index} workspace={workspace} conflicting={result.verdict === 'conflict'} />
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function Inspector({ result }) {
  return (
    <aside className="inspector">
      <h2><Layers size={17} /> Retrieval trace</h2>
      {!result ? (
        <p className="muted">Run a question to see how DataScout searched and assessed the evidence.</p>
      ) : (
        <>
          <ol className="trace">
            {result.trace_steps.map((step, i) => <li key={i}>{step}</li>)}
          </ol>
          {result.chunks.length > 0 && <details>
            <summary>Retrieved passages ({result.chunks.length})</summary>
            <p className="muted">Rank fusion orders passages. Rank scores measure retrieval priority.</p>
            <ul className="passages">
              {result.chunks.map((chunk) => <li key={chunk.chunk_id}>
                <strong>{chunk.source} · p. {chunk.page}</strong>
                <small>BM25 {chunk.bm25_rank ?? '—'} · Vector {chunk.vector_rank ?? '—'} · RRF {chunk.rrf_score}</small>
                <p>{chunk.text}</p>
              </li>)}
            </ul>
          </details>}
        </>
      )}
    </aside>
  );
}

function Evaluation() {
  const [state, setState] = useState({});
  useEffect(() => {
    let active = true;
    api('/analytics').then((data) => { if (active) setState({ data: data.benchmarks }); })
      .catch((error) => { if (active) setState({ error: error.message }); });
    return () => { active = false; };
  }, []);
  const data = state.data;
  return (
    <section className="evaluation">
      <p className="eyebrow"><FlaskConical size={15} /> FICTIONAL DEMO CORPUS</p>
      <h1>Evaluation</h1>
      {state.error && <p role="alert" className="error">{state.error}</p>}
      {!data && !state.error && <p role="status">Running benchmark…</p>}
      {data && <>
        <div className="metric-row">
          <div><strong>{data.correct}/{data.cases.length}</strong><span>Verdicts and citations matched</span></div>
          <div><strong>{data.retrieval_cases}</strong><span>Labelled retrieval cases</span></div>
          <div><strong>{data.p95_latency_ms} ms</strong><span>P95 local query time</span></div>
        </div>
        <h2>Retrieval on the original question</h2>
        <div className="table-scroll"><table>
          <thead><tr><th>Method</th><th>Recall@{data.k}</th><th>Precision@{data.k}</th><th>MRR</th></tr></thead>
          <tbody>{Object.entries(data.metrics).map(([method, row]) => <tr key={method}>
            <th scope="row">{metrics[method]}</th><td>{row.recall}</td><td>{row.precision}</td><td>{row.mrr}</td>
          </tr>)}</tbody>
        </table></div>
        <h2>Expected and observed results</h2>
        <div className="table-scroll"><table>
          <thead><tr><th>Question</th><th>Expected</th><th>Observed</th><th>Check</th></tr></thead>
          <tbody>{data.cases.map((item, i) => <tr key={i}>
            <th scope="row">{item.question}{item.scope.length > 0 && <small>{item.scope.join(', ')}</small>}</th>
            <td>{item.expected}</td><td>{item.actual}</td><td>{item.correct ? 'Pass' : 'Review'}</td>
          </tr>)}</tbody>
        </table></div>
      </>}
    </section>
  );
}

export default function App() {
  const [tab, setTab] = useState('workbench');
  const [workspace, setWorkspace] = useState('demo');
  const [demo, setDemo] = useState({ documents: [], examples: [] });
  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [loadError, setLoadError] = useState('');
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState('');
  const [publicQuery, setPublicQuery] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [busy, setBusy] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const currentDocuments = workspace === 'demo' ? demo.documents : documents;

  useEffect(() => {
    let active = true;
    Promise.allSettled([api('/health'), api('/demo'), api('/documents')]).then(([server, pack, docs]) => {
      if (!active) return;
      if (server.status === 'fulfilled') setHealth(server.value);
      if (pack.status === 'fulfilled') setDemo(pack.value);
      if (docs.status === 'fulfilled') setDocuments(docs.value);
      const failure = [server, pack, docs].find((item) => item.status === 'rejected');
      if (failure) setLoadError(failure.reason.message);
    });
    return () => { active = false; };
  }, []);

  function switchWorkspace(next) {
    if (next === workspace) return;
    setWorkspace(next);
    setScope('');
    setQuery('');
    setResult(null);
    setError('');
    setUploadStatus(null);
    setPublicQuery('');
    setAdvanced(false);
  }

  async function ask(event) {
    event.preventDefault();
    if (!query.trim() || busy) return;
    const question = query.trim();
    const publicQuestion = workspace === 'documents' && advanced ? publicQuery.trim() : '';
    setBusy(true);
    setError('');
    setResult(null);
    try {
      const data = await api('/query', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: question, workspace, sources: scope ? [scope] : [],
          public_web_query: publicQuestion || null,
        }),
      });
      setResult({ ...data, question, publicQuestion, scope });
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  async function upload(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    const body = new FormData();
    body.append('file', file);
    try {
      const uploaded = await api('/documents/upload', { method: 'POST', body });
      setDocuments(await api('/documents'));
      setResult(null);
      setUploadStatus({ text: `Indexed ${uploaded.filename} · ${uploaded.page_count} pages` });
    } catch (failure) {
      setUploadStatus({ text: failure.message, error: true });
    } finally {
      setUploading(false);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand"><span className="brand-icon"><ShieldCheck size={21} /></span><strong>DataScout</strong></div>
        <nav aria-label="Main navigation">
          <button aria-current={tab === 'workbench' ? 'page' : undefined} onClick={() => setTab('workbench')}>
            <Layers size={16} /> Workbench
          </button>
          <button aria-current={tab === 'evaluation' ? 'page' : undefined} onClick={() => setTab('evaluation')}>
            <BarChart3 size={16} /> Evaluation
          </button>
        </nav>
        <span className="header-state"><i className={health ? 'connected' : ''} /> {health ? 'Server connected' : 'Server status'}</span>
      </header>

      {tab === 'evaluation' ? <main><Evaluation /></main> : (
        <main className="workbench">
          <aside className="library">
            <h2><FolderOpen size={18} /> Source library</h2>
            <div className="segmented" aria-label="Source workspace">
              <button aria-pressed={workspace === 'demo'} disabled={busy || uploading}
                onClick={() => switchWorkspace('demo')}>Demo notices</button>
              <button aria-pressed={workspace === 'documents'} disabled={busy || uploading}
                onClick={() => switchWorkspace('documents')}>My PDFs</button>
            </div>
            {workspace === 'documents' && <>
              <label className="upload">
                <Upload size={17} /> {uploading ? 'Indexing…' : 'Upload PDF'}
                <input type="file" accept=".pdf,application/pdf" disabled={uploading || busy} onChange={upload} />
              </label>
              <p className="muted limits">Text PDFs up to 10 MB and 120 pages. Uploading the same filename replaces it.</p>
              {uploadStatus && <p role={uploadStatus.error ? 'alert' : 'status'}
                className={uploadStatus.error ? 'error' : 'upload-success'}>{uploadStatus.text}</p>}
            </>}
            {currentDocuments.length > 0 ? <ul className="document-list">
              {currentDocuments.map((doc) => <li key={doc.filename} className={scope === doc.filename ? 'selected' : ''}>
                <FileText size={17} />
                <div><strong>{doc.filename}</strong><small>{doc.page_count} pages · {doc.chunk_count} passages</small></div>
                {workspace === 'demo' && <a href={`/api/demo/documents/${encodeURIComponent(doc.filename)}`}
                  target="_blank" rel="noreferrer" aria-label={`Open ${doc.filename}`}><ArrowDownToLine size={16} /></a>}
              </li>)}
            </ul> : <p className="muted">{workspace === 'demo' ? 'Loading demo notices…' : 'Upload a PDF to build your source library.'}</p>}
            {workspace === 'demo' && <p className="library-note"><FlaskConical size={15} /> These are fictional campus notices.</p>}
            {workspace === 'documents' && health?.gemini_configured && <p className="library-note">
              <Globe size={15} /> Gemini is enabled. Questions and relevant document text may be sent to Gemini.
            </p>}
          </aside>

          <div className="main-panel">
            <section className="intro">
              <p className="eyebrow">{workspace === 'demo' ? 'CAMPUS INNOVATION CHALLENGE' : 'DOCUMENT WORKSPACE'}</p>
              <h1>Find the answer.<br />Check the notice.</h1>
              <p>{workspace === 'demo'
                ? 'Ask about registration, then inspect the exact text behind the result.'
                : 'Ask a question and trace the answer back to the source page.'}</p>
            </section>
            {loadError && <p role="alert" className="error">{loadError}</p>}
            <form onSubmit={ask} className="question-form">
              <label htmlFor="question">Your question</label>
              <textarea id="question" value={query} onChange={(e) => setQuery(e.target.value)}
                maxLength={1000} rows={2} placeholder="What is the team registration deadline?" required disabled={busy} />
              <div className="form-actions">
                <div className="scope"><ListFilter size={15} /><label htmlFor="scope" className="sr-only">Search scope</label>
                  <select id="scope" value={scope} onChange={(e) => setScope(e.target.value)} disabled={busy}>
                    <option value="">All {workspace === 'demo' ? 'demo notices' : 'uploaded PDFs'}</option>
                    {currentDocuments.map((doc) => <option key={doc.filename}>{doc.filename}</option>)}
                  </select>
                </div>
                <button type="submit" className="primary" disabled={!query.trim() || busy || uploading}>
                  <Search size={16} /> {busy ? 'Checking…' : 'Check evidence'}
                </button>
              </div>
              {workspace === 'documents' && health?.web_configured && (
                <div className="public-search">
                  <label><input type="checkbox" checked={advanced} onChange={(e) => setAdvanced(e.target.checked)}
                    disabled={busy} /> Try a separate public web query if documents fall short</label>
                  {advanced && <input aria-label="Public web query" placeholder="Enter a question safe to send to Tavily"
                    maxLength={300} required value={publicQuery} onChange={(e) => setPublicQuery(e.target.value)} disabled={busy} />}
                </div>
              )}
            </form>
            {workspace === 'demo' && <div className="examples">
              {demo.examples.map((example) => <button key={example.label} disabled={busy}
                onClick={() => { setQuery(example.query); setScope(''); document.getElementById('question')?.focus(); }}>
                {example.label} <ArrowRight size={13} />
              </button>)}
            </div>}
            <div className="response-area" aria-live="polite" aria-busy={busy}>
              {error && <p role="alert" className="error">{error}</p>}
              {busy && <p role="status" className="muted">Checking sources for direct evidence…</p>}
              {result && <Result result={result} workspace={workspace} />}
            </div>
          </div>

          <Inspector result={result} />
        </main>
      )}
    </div>
  );
}
