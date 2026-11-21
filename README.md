# DataScout

DataScout is the VeriRAG project’s evidence workbench for people who need to act on institutional notices. A campus events desk can use it to answer repeat questions and catch disagreements before sharing a deadline. The demo contains three fictional PDFs about the Campus Innovation Challenge.

## Run locally

Use Python 3.12 and Node.js 22.12+ (or Node 24). From the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
npm --prefix frontend ci
npm --prefix frontend run build
.\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

On macOS/Linux, use `python3.12 -m venv .venv` and substitute `.venv/bin/python` for the Windows executable. Open <http://127.0.0.1:8000>. The demo runs with no API keys and makes no live provider calls. Dependency installation requires network access.

For frontend development, keep FastAPI on port 8000 and run `npm --prefix frontend run dev` in a second terminal. Vite serves the frontend on port 5173 and proxies `/api` to FastAPI.

## Live demo for judges

1. Open **Demo notices**, click **Conflicting deadlines**, then **Check evidence**. The registration circular says 24 November at 5:00 PM IST and the coordinator notice says 25 November at noon IST. Open both page citations from the cards. DataScout leaves the authority decision to the event office.
2. Keep the same question and select **01 Registration circular.pdf** in **Search scope**. Run it again. The result reports the deadline in that document and displays the search scope. It does not resolve which notice governs.
3. Click **Paraphrased question**, then **Check evidence**. The trace shows one rewrite from “How many people can be in a group?” to “maximum team size”. It quotes the handbook’s limit of three.
4. Click **Missing answer**, then **Check evidence**. The notices contain no fee-waiver policy, so the app stops without giving a yes or no answer.
5. Open **Evaluation** to show the expected verdicts, measured results, and retrieval scores.

### 45-second pitch

> Students and coordinators often get answers from circulars, handbooks, and reminders that were written at different times. If those sources disagree, picking a date from whichever passage ranked first can make someone miss a deadline. DataScout shows the passages side by side, links each one to its page, and asks you to confirm the authority when the sources disagree. If the documents never answered the question, it says so. Our demo shows those decisions live, and our evaluation checks both the verdict and the cited pages.

### How it works in plain language

| Component | What it does |
| --- | --- |
| React, Vite, FastAPI, Pydantic | React shows the result and evidence. FastAPI receives uploads and questions. Pydantic checks their format. |
| PyMuPDF and SQLite | The backend extracts text in page order, splits it into short passages, and stores each uploaded PDF’s text and page numbers. SQLite replaces a same-named upload in a single transaction. |
| BM25 | Ranks passages by important matching words. Technical identifiers such as `ERR-701` stay intact. |
| Vector search | Local word and character-trigram vectors rank similar wording using cosine similarity. When Gemini is enabled and every selected document has matching embeddings, the workspace uses Gemini’s semantic embeddings. |
| Reciprocal Rank Fusion | Combines the two rankings. A high rank helps a passage get inspected, but the ranking does not establish that its claim is correct. |
| LangGraph and evidence checks | The workflow assesses direct support, retries with another wording at most once, then returns supported, conflicting, or insufficient evidence. It compares matching policy fields across all selected documents, including when retrieval returns fewer passages. |
| Gemini and Tavily | Optionally assess free-form document evidence and search the public web. The server checks quoted text against the source passage returned by the provider and attaches the page or URL. |

## Requirements and implementation decisions

| Prototype finding | Implemented change |
| --- | --- |
| Without a Gemini key, a PDF upload returned HTTP 500 and a query returned a provider configuration error. | Offline demo, text-PDF indexing, BM25, local vector search, and extraction from explicit policy lines run without a provider. |
| A relevant passage could be taken as sufficient, and synthesis could mix sources without showing an unresolved disagreement. | Separate verdicts for supported, conflict, and insufficient. Matching policy labels with different values get side-by-side quotes. Scope can be narrowed to one source so the user can ask what it says. |
| Web fallback could transmit the rewritten internal question, and its snippet presence was sufficient for synthesis. | Private-document search runs first. Web search is opt-in through a separately entered public question, and it runs only after insufficient document evidence. Conflicts stop on the document route. |
| Replacing a shorter PDF could leave stale chunks. Retrieval ignored `top_k` and query cache was unbounded. | Transactional replacement, validated scope and retrieval limit, and a bounded cache for deterministic demo results. The default first launch imports existing Chroma text and page metadata from `backend/data/chroma_db`. The legacy database stays on disk. |
| The frontend clean install failed on its lockfile. Its API URL was fixed to the developer machine. The benchmark used keyword hits as relevance labels. | A verified lockfile, same-origin API, and reproducible evaluation with explicit expected pages and verdicts. |

## Your own PDFs and optional providers

Open **My PDFs** to upload a readable text PDF. Limits are 10 MB, 120 pages and 1,200 extracted passages per file. Without Gemini, the app can answer matching explicit lines such as `Maximum team size: 3 students` and show related passages for other questions. Scanned PDFs need OCR first. A new upload using the same filename replaces the old entries. Private workspace answers are not cached.

To use Gemini or Tavily, copy `.env.example` to `.env` at the repository root, set the desired keys and restart FastAPI. Gemini receives uploaded passages during embedding and selected text plus questions during assessment. Use this mode only with material you can send to Gemini. Files uploaded without embeddings and legacy-imported passages remain searchable, and use local vector ranking in a mixed selection. Re-upload PDFs with Gemini configured if you need semantic vectors for them. A tool failure falls back to local ranking or abstention. The app checks a quote’s presence in the text, which is separate from judging whether the quote proves a model-generated claim.

Tavily uses the separately entered public query. Both Gemini and Tavily keys are needed for a cited web answer. Results remain explicitly labelled as answers to the public query. The app is a single-workspace local prototype. Before a shared pilot, add sign-in, access control, and upload limits at the network edge.

## Evaluation and next work

```powershell
cd backend
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe -m evaluation.run_benchmarks
```

The benchmark contains eight workflow cases and six answerable retrieval cases. It measures page-level Recall@3, Precision@3, MRR and P95 local workflow latency. It counts a workflow case as correct only when both its verdict and the set of cited pages match the labels. The three retrieval modes tied at 0.833 Recall@3 on the demo set. This fixture does not establish a retrieval advantage or estimate accuracy on unseen campus notices.

For a three-person team, assign one person to curate real public notices and hold out a labelled test set, one to verify provider behaviour and study errors in scope/date handling, and one to own the live demo and gather feedback from a campus coordinator. Prioritise explicit supersession and effective-date metadata next, so a verified replacement can be distinguished from a disagreement needing human review. Scanned PDF support and access control are follow-up work.
