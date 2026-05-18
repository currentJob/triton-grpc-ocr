import type { HistoryEntry } from '../types'

interface ServerStatus {
  live:   boolean
  ready:  boolean
  models: Record<string, boolean>
}

interface Props {
  serverStatus:    ServerStatus | null
  history:         HistoryEntry[]
  onToggleHistory: () => void
}

export default function Header({ serverStatus, history, onToggleHistory }: Props) {
  const dotClass = !serverStatus
    ? 'status-dot'
    : serverStatus.live && serverStatus.ready
    ? 'status-dot live'
    : serverStatus.live
    ? 'status-dot warn'
    : 'status-dot dead'

  const pillClass = !serverStatus
    ? 'server-pill'
    : serverStatus.live && serverStatus.ready
    ? 'server-pill live'
    : serverStatus.live
    ? 'server-pill warn'
    : 'server-pill dead'

  const statusText = !serverStatus
    ? '연결 중...'
    : serverStatus.live && serverStatus.ready
    ? '서버 정상'
    : serverStatus.live
    ? '모델 로딩 중'
    : '연결 실패'

  const models = serverStatus?.models ?? {}
  const modelEntries = Object.entries(models)

  return (
    <header className="header">
      {/* Logo */}
      <div className="header-logo">
        <div className="header-logo-icon">🔍</div>
        <h1>OCR <span>Demo</span></h1>
      </div>

      {/* Center: model badges */}
      {modelEntries.length > 0 && (
        <div className="header-center">
          <div className="model-badges">
            {modelEntries.map(([name, ready]) => (
              <span key={name} className={`model-badge${ready ? ' ready' : ''}`}>
                {name}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Right: status + history */}
      <div className="header-right">
        <div className={pillClass}>
          <span className={dotClass} />
          <span className="status-text">{statusText}</span>
        </div>

        {history.length > 0 && (
          <button className="history-toggle" onClick={onToggleHistory}>
            <svg viewBox="0 0 20 20" fill="currentColor" width="12" height="12">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.75-13a.75.75 0 00-1.5 0v5c0 .414.336.75.75.75h4a.75.75 0 000-1.5h-3.25V5z" clipRule="evenodd"/>
            </svg>
            히스토리 {history.length}
          </button>
        )}
      </div>
    </header>
  )
}
