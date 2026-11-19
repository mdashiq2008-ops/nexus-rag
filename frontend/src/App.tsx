import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type FormEvent,
} from 'react'
import {
  ArrowUp,
  BookOpenText,
  Check,
  ChevronDown,
  CircleHelp,
  FileText,
  FolderOpen,
  LoaderCircle,
  MessageSquareText,
  Plus,
  ShieldCheck,
  Sparkles,
  Trash2,
  Upload,
  Wifi,
  WifiOff,
} from 'lucide-react'
import './App.css'

type SearchMode = 'hybrid' | 'hybrid_rerank'

type Document = {
  sha256: string
  filename: string
  page_count: number
  chunk_count: number
  processed_at: string
  pdf_url: string | null
  source_available: boolean
  file_size_bytes: number | null
  ocr_used: boolean
}

type Citation = {
  citation_id: string
  passage_id: string
  filename: string
  page_number: number
  char_start: number
  char_end: number
  ocr_used: boolean
  chunk_ids: number[]
  pdf_url: string | null
}

type PassageSource = {
  filename: string
  page_number: number
  char_start: number
  char_end: number
  ocr_used: boolean
  chunk_ids: number[]
  pdf_url: string | null
}

type SourcePassage = {
  rank: number
  text: string
  score: number
  truncated: boolean
  sources: PassageSource[]
}

type AnswerStatement = {
  text: string
  citation_ids: string[]
}

type ChatResponse = {
  answer: string
  abstained: boolean
  answer_mode: 'extractive'
  search_mode: SearchMode
  statements: AnswerStatement[]
  citations: Citation[]
  source_passages: SourcePassage[]
}

type Message = {
  id: string
  role: 'user' | 'assistant'
  content: string
  response?: ChatResponse
}

type HealthResponse = {
  status: string
  answer_mode: string
  database_status: string
  reranker_available: boolean
  llm: {
    configured: boolean
    available: boolean
    status: string
  }
}

type StatusResponse = {
  backend: string
  database_status: string
  database_path: string
  document_storage: string
  document_count: number
  chunk_count: number
  embedding_count: number
  reranker_available: boolean
  llm: HealthResponse['llm'] & { model_repo: string; model_file: string }
}

type WorkspaceView = 'workspace' | 'library' | 'settings'

async function requestJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = `Request failed (${response.status})`
    try {
      const body = (await response.json()) as { detail?: string }
      if (body.detail) message = body.detail
    } catch {
      // Keep the HTTP status as the user-facing error when the response is not JSON.
    }
    throw new Error(message)
  }
  return (await response.json()) as T
}

function loadWorkspace(): Promise<[HealthResponse, Document[], StatusResponse]> {
  return Promise.all([
    requestJson<HealthResponse>('/api/health'),
    requestJson<Document[]>('/api/documents'),
    requestJson<StatusResponse>('/api/status'),
  ])
}

function citationLabel(citation: Citation) {
  return `${citation.filename} · page ${citation.page_number}`
}

function App() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [workspaceView, setWorkspaceView] = useState<WorkspaceView>('workspace')
  const [messages, setMessages] = useState<Message[]>([])
  const [question, setQuestion] = useState('')
  const [searchMode, setSearchMode] = useState<SearchMode>('hybrid')
  const [topK, setTopK] = useState(5)
  const [candidateK, setCandidateK] = useState(20)
  const [dragging, setDragging] = useState(false)
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null)
  const [loadingDocuments, setLoadingDocuments] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [deletingDocument, setDeletingDocument] = useState<string | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const fileInput = useRef<HTMLInputElement>(null)
  const messagesEnd = useRef<HTMLDivElement>(null)

  const refreshDocuments = useCallback(async () => {
    try {
      const [healthResult, indexedDocuments, statusResult] = await loadWorkspace()
      setDocuments(indexedDocuments)
      setStatus(statusResult)
      setHealth(healthResult)
      setBackendOnline(healthResult.status === 'ok')
      setError('')
    } catch (loadError) {
      setBackendOnline(false)
      setError(
        loadError instanceof Error
          ? loadError.message
          : 'Unable to reach the local NEXUS API.',
      )
    } finally {
      setLoadingDocuments(false)
    }
  }, [])

  useEffect(() => {
    let active = true
    void loadWorkspace()
      .then(([healthResult, indexedDocuments, statusResult]) => {
        if (!active) return
        setDocuments(indexedDocuments)
        setStatus(statusResult)
        setHealth(healthResult)
        setBackendOnline(healthResult.status === 'ok')
      })
      .catch((loadError: unknown) => {
        if (!active) return
        setBackendOnline(false)
        setError(
          loadError instanceof Error
            ? loadError.message
            : 'Unable to reach the local NEXUS API.',
        )
      })
      .finally(() => {
        if (active) setLoadingDocuments(false)
      })
    return () => {
      active = false
    }
  }, [])

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({
      behavior: messages.length > 1 ? 'smooth' : 'auto',
    })
  }, [messages, sending])

  async function handleUpload(event: FormEvent<HTMLInputElement>) {
    const files = event.currentTarget.files
    event.currentTarget.value = ''
    if (!files) return
    await handleDroppedFiles(files)
  }

  async function uploadFile(file: File) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      setError('Only PDF files can be added to NEXUS.')
      return
    }
    setError('')
    setNotice('')
    setUploading(true)
    try {
      const formData = new FormData()
      formData.append('file', file)
      const result = await requestJson<{ status: string; document: Document }>(
        '/api/documents',
        { method: 'POST', body: formData },
      )
      await refreshDocuments()
      setNotice(
        result.status === 'duplicate'
          ? `${result.document.filename} is already in your library.`
          : `${result.document.filename} is ready to search.`,
      )
    } catch (uploadError) {
      setError(
        uploadError instanceof Error
          ? uploadError.message
          : 'The PDF could not be uploaded.',
      )
    } finally {
      setUploading(false)
    }
  }

  async function handleDroppedFiles(files: FileList | null) {
    if (!files || files.length === 0) return
    for (const file of Array.from(files)) {
      await uploadFile(file)
    }
  }

  async function handleDelete(document: Document) {
    if (
      !window.confirm(
        `Remove ${document.filename} from the NEXUS index? Its app-managed PDF copy will also be deleted.`,
      )
    ) {
      return
    }
    setError('')
    setNotice('')
    setDeletingDocument(document.sha256)
    try {
      const result = await requestJson<{
        deleted: boolean
        uploaded_file_deleted: boolean
      }>(
        `/api/documents/${document.sha256}?delete_uploaded_file=true`,
        { method: 'DELETE' },
      )
      await refreshDocuments()
      setNotice(
        result.uploaded_file_deleted
          ? `${document.filename} was removed from the index and its managed PDF was deleted.`
          : `${document.filename} was removed from the index. The original PDF was kept.`,
      )
    } catch (deleteError) {
      setError(
        deleteError instanceof Error
          ? deleteError.message
          : 'The document could not be removed.',
      )
    } finally {
      setDeletingDocument(null)
    }
  }

  async function handleAsk(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const submittedQuestion = question.trim()
    if (!submittedQuestion || sending) return

    setQuestion('')
    setError('')
    setNotice('')
    setMessages((current) => [
      ...current,
      {
        id: `user-${Date.now()}`,
        role: 'user',
        content: submittedQuestion,
      },
    ])
    setSending(true)
    try {
      const response = await requestJson<ChatResponse>('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: submittedQuestion,
          search_mode: searchMode,
          top_k: topK,
          candidate_k: candidateK,
        }),
      })
      setMessages((current) => [
        ...current,
        {
          id: `assistant-${Date.now()}`,
          role: 'assistant',
          content: response.answer,
          response,
        },
      ])
    } catch (chatError) {
      setError(
        chatError instanceof Error
          ? chatError.message
          : 'NEXUS could not answer this question.',
      )
    } finally {
      setSending(false)
    }
  }

  const startQuestion = (suggestion: string) => {
    setQuestion(suggestion)
    document.getElementById('question-input')?.focus()
  }

  return (
    <div className="app-shell">
      <aside className="sidebar" aria-label="NEXUS workspace">
        <div className="brand" aria-label="NEXUS">
          <span className="brand-mark" aria-hidden="true">
            <BookOpenText size={21} strokeWidth={2.2} />
          </span>
          <span className="brand-name">nexus<span>.</span></span>
          <span className="brand-version">LOCAL RAG</span>
        </div>

        <button
          className="new-chat-button"
          type="button"
          onClick={() => {
            setMessages([])
            setQuestion('')
            setError('')
            setNotice('')
            setWorkspaceView('workspace')
          }}
        >
          <Plus size={17} />
          <span>New conversation</span>
        </button>

        <nav className="sidebar-nav" aria-label="Primary">
          <button
            type="button"
            className={workspaceView === 'workspace' ? 'active' : ''}
            onClick={() => setWorkspaceView('workspace')}
          >
            <MessageSquareText size={16} />
            Workspace
          </button>
          <button
            type="button"
            className={workspaceView === 'library' ? 'active' : ''}
            onClick={() => setWorkspaceView('library')}
          >
            <FolderOpen size={16} />
            Document library
          </button>
          <button
            type="button"
            className={workspaceView === 'settings' ? 'active' : ''}
            onClick={() => setWorkspaceView('settings')}
          >
            <ShieldCheck size={16} />
            Settings
          </button>
        </nav>

        <div className="sidebar-label">
          <span>Your library</span>
          <span className="document-count">{documents.length}</span>
        </div>
        <button
          className="upload-button"
          type="button"
          onClick={() => fileInput.current?.click()}
          disabled={uploading}
        >
          {uploading ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Upload size={17} />
          )}
          <span>{uploading ? 'Adding PDF…' : 'Add a PDF'}</span>
        </button>
        <input
          ref={fileInput}
          className="visually-hidden"
          type="file"
          accept="application/pdf,.pdf"
          multiple
          aria-label="Choose a PDF to add"
          onChange={handleUpload}
        />

        <div className="document-list" aria-live="polite">
          {loadingDocuments ? (
            <p className="library-empty">Loading your library…</p>
          ) : documents.length === 0 ? (
            <div className="library-empty">
              <FolderOpen size={19} />
              <span>Your PDFs will appear here.</span>
            </div>
          ) : (
            documents.map((item) => (
              <div className="document-row" key={item.sha256}>
                <span className="document-icon" aria-hidden="true">
                  <FileText size={17} />
                </span>
                <span className="document-info">
                  <span className="document-name" title={item.filename}>
                    {item.filename}
                  </span>
                  <span className="document-meta">
                    {item.page_count} pages · {item.chunk_count} passages
                  </span>
                </span>
                <button
                  className="icon-button delete-document"
                  type="button"
                  aria-label={`Remove ${item.filename}`}
                  title={`Remove ${item.filename}`}
                  disabled={deletingDocument === item.sha256}
                  onClick={() => void handleDelete(item)}
                >
                  {deletingDocument === item.sha256 ? (
                    <LoaderCircle className="spin" size={16} />
                  ) : (
                    <Trash2 size={16} />
                  )}
                </button>
              </div>
            ))
          )}
        </div>

        <div className="sidebar-footer">
          <div className={`connection-status ${backendOnline ? 'online' : ''}`}>
            {backendOnline ? <Wifi size={15} /> : <WifiOff size={15} />}
            <span>
              {backendOnline === null
                ? 'Connecting to local API'
                : backendOnline
                  ? 'Local API connected'
                  : 'Local API offline'}
            </span>
            {backendOnline && <span className="status-dot" aria-hidden="true" />}
          </div>
          <p>Documents stay on this computer.</p>
          <div className="help-link">
            <CircleHelp size={15} />
            <span>Local-first by design</span>
          </div>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div className="workspace-title">
            <span className="workspace-icon" aria-hidden="true">
              <MessageSquareText size={16} />
            </span>
            <span>Workspace</span>
            <ChevronDown size={15} className="muted-icon" aria-hidden="true" />
          </div>
          <div className="topbar-right">
            <span className="private-badge">
              <ShieldCheck size={14} />
              Private & local
            </span>
            <span className="avatar" aria-label="Local workspace">N</span>
          </div>
        </header>

        <section className="conversation" aria-label="Question and answer">
          {workspaceView !== 'workspace' ? (
            <div className="management-view">
              <div className="management-heading">
                <p className="eyebrow">LOCAL WORKSPACE</p>
                <h1>{workspaceView === 'library' ? 'Document library' : 'System settings'}</h1>
              </div>
              {workspaceView === 'library' ? (
                <>
                  <div
                    className={`upload-zone ${dragging ? 'dragging' : ''}`}
                    onDragEnter={(event) => {
                      event.preventDefault()
                      setDragging(true)
                    }}
                    onDragOver={(event) => event.preventDefault()}
                    onDragLeave={() => setDragging(false)}
                    onDrop={(event) => {
                      event.preventDefault()
                      setDragging(false)
                      void handleDroppedFiles(event.dataTransfer.files)
                    }}
                  >
                    <Upload size={24} />
                    <strong>Drop PDF files here</strong>
                    <span>or</span>
                    <button type="button" onClick={() => fileInput.current?.click()}>
                      Browse files
                    </button>
                  </div>
                  <div className="library-grid">
                    {documents.map((document) => (
                      <article className="library-card" key={document.sha256}>
                        <div className="library-card-icon"><FileText size={20} /></div>
                        <div className="library-card-main">
                          <strong>{document.filename}</strong>
                          <span>{document.page_count} pages · {document.chunk_count} chunks</span>
                          <span>
                            {document.file_size_bytes
                              ? `${Math.max(1, Math.round(document.file_size_bytes / 1024))} KiB · `
                              : ''}
                            {document.ocr_used ? 'OCR used' : 'Text extraction'}
                          </span>
                        </div>
                        {document.pdf_url && (
                          <a href={document.pdf_url} target="_blank" rel="noreferrer">Open PDF</a>
                        )}
                        <button
                          type="button"
                          onClick={() => void handleDelete(document)}
                          disabled={deletingDocument === document.sha256}
                        >
                          {deletingDocument === document.sha256 ? 'Removing…' : 'Delete'}
                        </button>
                      </article>
                    ))}
                  </div>
                  {!loadingDocuments && documents.length === 0 && (
                    <p className="empty-library">No indexed PDFs yet. Upload a PDF to begin.</p>
                  )}
                </>
              ) : (
                <div className="settings-grid">
                  <article>
                    <strong>Database</strong>
                    <span>{status?.database_status === 'ok' ? 'Connected' : 'Unavailable'}</span>
                  </article>
                  <article>
                    <strong>Retrieval</strong>
                    <span>Hybrid + optional reranking</span>
                  </article>
                  <article>
                    <strong>Final results</strong>
                    <span>{topK}</span>
                  </article>
                  <article>
                    <strong>Candidate pool</strong>
                    <span>{candidateK}</span>
                  </article>
                  <article>
                    <strong>Cross-encoder</strong>
                    <span>{health?.reranker_available ? 'Available' : 'Unavailable'}</span>
                  </article>
                  <article>
                    <strong>Local model</strong>
                    <span>{status?.llm.status === 'available' ? 'Configured, not started' : 'Not installed / unavailable'}</span>
                  </article>
                  <article className="settings-wide">
                    <strong>Model configuration</strong>
                    <span>{status?.llm.model_repo || 'Not available'} / {status?.llm.model_file || ''}</span>
                  </article>
                </div>
              )}
            </div>
          ) : messages.length === 0 ? (
            <div className="welcome">
              <div className="welcome-icon">
                <Sparkles size={22} />
              </div>
              <p className="eyebrow">YOUR DOCUMENTS, IN CONTEXT</p>
              <h1>What would you like<br />to understand?</h1>
              <p className="welcome-copy">
                Ask a question about your PDFs. NEXUS finds relevant passages
                and links every answer back to its source.
              </p>
              <div className="suggestions" aria-label="Example questions">
                <button
                  type="button"
                  onClick={() => startQuestion('Summarize the key ideas in my documents')}
                >
                  <span className="suggestion-symbol">✦</span>
                  Summarize the key ideas
                  <ArrowUp size={15} className="suggestion-arrow" />
                </button>
                <button
                  type="button"
                  onClick={() => startQuestion('What evidence or recommendations do the documents provide?')}
                >
                  <span className="suggestion-symbol">⌕</span>
                  Find evidence and recommendations
                  <ArrowUp size={15} className="suggestion-arrow" />
                </button>
              </div>
              {documents.length === 0 && !loadingDocuments && (
                <p className="welcome-hint">
                  <Upload size={14} /> Add a PDF from the sidebar to get started.
                </p>
              )}
            </div>
          ) : (
            <div className="message-list">
              {messages.map((message) => (
                <article
                  className={`message message-${message.role}`}
                  key={message.id}
                >
                  {message.role === 'user' ? (
                    <>
                      <span className="user-avatar" aria-hidden="true">Y</span>
                      <p className="user-message-text">{message.content}</p>
                    </>
                  ) : (
                    <>
                      <span className="assistant-avatar" aria-hidden="true">
                        <BookOpenText size={17} />
                      </span>
                      <div className="assistant-message">
                        <div className="answer-heading">
                          <span className="answer-heading-label">NEXUS</span>
                          <span className="answer-mode-badge">
                            <Check size={12} />
                            Extractive · evidence-checked
                          </span>
                        </div>
                        {message.response?.abstained ? (
                          <div className="abstention" role="status">
                            <ShieldCheck size={18} />
                            <div>
                              <strong>Not enough evidence</strong>
                              <p>{message.content}</p>
                            </div>
                          </div>
                        ) : message.response?.statements.length ? (
                          <div className="answer-statements">
                            {message.response.statements.map((statement, index) => (
                              <p key={`${message.id}-statement-${index}`}>
                                {statement.text}{' '}
                                {statement.citation_ids.map((citationId) => {
                                  const citation = message.response?.citations.find(
                                    (item) => item.citation_id === citationId,
                                  )
                                  if (!citation) return null
                                  return citation.pdf_url ? (
                                    <a
                                      className="inline-citation"
                                      href={citation.pdf_url}
                                      key={citationId}
                                      target="_blank"
                                      rel="noreferrer"
                                      aria-label={`Open ${citationLabel(citation)}`}
                                    >
                                      {citationId}
                                    </a>
                                  ) : (
                                    <span className="inline-citation disabled" key={citationId}>
                                      {citationId}
                                    </span>
                                  )
                                })}
                              </p>
                            ))}
                          </div>
                        ) : (
                          <p className="answer-fallback">{message.content}</p>
                        )}

                        {!!message.response?.citations.length && (
                          <div className="citation-list" aria-label="Answer citations">
                            <span className="citation-section-label">SOURCES</span>
                            {message.response.citations.map((citation) =>
                              citation.pdf_url ? (
                                <a
                                  className="citation-card"
                                  href={citation.pdf_url}
                                  key={citation.citation_id}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  <FileText size={16} />
                                  <span className="citation-card-text">
                                    <strong>{citation.filename}</strong>
                                    <span>Page {citation.page_number}</span>
                                  </span>
                                  <span className="citation-id">{citation.citation_id}</span>
                                </a>
                              ) : (
                                <div className="citation-card unavailable" key={citation.citation_id}>
                                  <FileText size={16} />
                                  <span className="citation-card-text">
                                    <strong>{citation.filename}</strong>
                                    <span>Page {citation.page_number} · PDF unavailable</span>
                                  </span>
                                  <span className="citation-id">{citation.citation_id}</span>
                                </div>
                              ),
                            )}
                          </div>
                        )}

                        {!!message.response?.source_passages.length && (
                          <details className="passage-details">
                            <summary>
                              <span>View source passages</span>
                              <span className="passage-count">
                                {message.response.source_passages.length}
                              </span>
                            </summary>
                            <div className="passage-list">
                              {message.response.source_passages.map((passage) => (
                                <div className="passage-card" key={`passage-${passage.rank}`}>
                                  <div className="passage-card-heading">
                                    <span>Passage {passage.rank}</span>
                                    {passage.sources.map((source, index) => {
                                      const label = `${source.filename} · page ${source.page_number}`
                                      return source.pdf_url ? (
                                        <a
                                          href={source.pdf_url}
                                          key={`${source.filename}-${source.page_number}-${index}`}
                                          target="_blank"
                                          rel="noreferrer"
                                        >
                                          {label}
                                        </a>
                                      ) : (
                                        <span key={`${source.filename}-${source.page_number}-${index}`}>
                                          {label}
                                        </span>
                                      )
                                    })}
                                  </div>
                                  <p>{passage.text}</p>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    </>
                  )}
                </article>
              ))}
              {sending && (
                <div className="thinking" role="status" aria-live="polite">
                  <span className="assistant-avatar" aria-hidden="true">
                    <BookOpenText size={17} />
                  </span>
                  <span className="thinking-dots"><i /><i /><i /></span>
                  <span>Searching your documents…</span>
                </div>
              )}
              <div ref={messagesEnd} />
            </div>
          )}
        </section>

        <footer className="composer-area">
          {(error || notice) && (
            <div
              className={`feedback-banner ${error ? 'feedback-error' : 'feedback-success'}`}
              role={error ? 'alert' : 'status'}
            >
              <span>{error || notice}</span>
              <button
                type="button"
                className="icon-button"
                aria-label="Dismiss message"
                onClick={() => {
                  setError('')
                  setNotice('')
                }}
              >
                <span aria-hidden="true">×</span>
              </button>
            </div>
          )}
          <form className="composer" onSubmit={handleAsk}>
            <label className="visually-hidden" htmlFor="question-input">
              Ask a question about your documents
            </label>
            <input
              id="question-input"
              type="text"
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={
                documents.length
                  ? 'Ask anything about your documents…'
                  : 'Add a PDF to start asking questions'
              }
              disabled={sending || !backendOnline || documents.length === 0}
              autoComplete="off"
              maxLength={4000}
            />
            <button
              className="send-button"
              type="submit"
              aria-label="Send question"
              disabled={!question.trim() || sending || !backendOnline || documents.length === 0}
            >
              {sending ? (
                <LoaderCircle className="spin" size={18} />
              ) : (
                <ArrowUp size={19} strokeWidth={2.5} />
              )}
            </button>
          </form>

          <div className="composer-options">
            <div className="search-mode" role="group" aria-label="Search mode">
              <button
                type="button"
                aria-pressed={searchMode === 'hybrid'}
                className={searchMode === 'hybrid' ? 'mode-selected' : ''}
                onClick={() => setSearchMode('hybrid')}
              >
                Hybrid
              </button>
              <button
                type="button"
                aria-pressed={searchMode === 'hybrid_rerank'}
                className={searchMode === 'hybrid_rerank' ? 'mode-selected' : ''}
                onClick={() => setSearchMode('hybrid_rerank')}
                title="Uses a local cross-encoder and additional CPU and memory"
              >
                Hybrid + reranking
              </button>
            </div>
            <label className="result-setting">
              Results
              <select
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                aria-label="Final result count"
              >
                {[3, 5, 10, 20].map((value) => (
                  <option value={value} key={value}>{value}</option>
                ))}
              </select>
            </label>
            <label className="result-setting">
              Candidates
              <select
                value={candidateK}
                onChange={(event) => {
                  const next = Number(event.target.value)
                  setCandidateK(next)
                  if (topK > next) setTopK(next)
                }}
                aria-label="Candidate count"
              >
                {[20, 40, 60].map((value) => (
                  <option value={value} key={value}>{value}</option>
                ))}
              </select>
            </label>
            <span className="local-hint">
              {searchMode === 'hybrid_rerank'
                ? 'Local reranker uses extra CPU & memory'
                : 'Answers are checked against retrieved text'}
            </span>
          </div>
          <p className="privacy-note">
            NEXUS uses your local index. Answers may need verification against the source PDF.
          </p>
        </footer>
      </main>
    </div>
  )
}

export default App
