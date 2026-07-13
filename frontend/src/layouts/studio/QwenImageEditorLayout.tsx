import { ExternalLink, RefreshCcw } from 'lucide-react'
import { useCallback, useEffect, useState } from 'react'
import './studioLayouts.css'

const QWEN_EDITOR_URL = 'http://127.0.0.1:7865'
const QWEN_HEALTH_URL = `${QWEN_EDITOR_URL}/api/health`

type EditorState = 'checking' | 'online' | 'offline'

export function QwenImageEditorLayout() {
  const [editorState, setEditorState] = useState<EditorState>('checking')
  const [frameKey, setFrameKey] = useState(0)

  const checkEditor = useCallback(async () => {
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 2500)
    try {
      await fetch(QWEN_HEALTH_URL, {
        cache: 'no-store',
        mode: 'no-cors',
        signal: controller.signal,
      })
      setEditorState('online')
    } catch {
      setEditorState('offline')
    } finally {
      window.clearTimeout(timeout)
    }
  }, [])

  useEffect(() => {
    const initialCheck = window.setTimeout(() => void checkEditor(), 0)
    const interval = window.setInterval(() => void checkEditor(), 8000)
    return () => {
      window.clearTimeout(initialCheck)
      window.clearInterval(interval)
    }
  }, [checkEditor])

  const reloadEditor = useCallback(() => {
    setEditorState('checking')
    setFrameKey((current) => current + 1)
    void checkEditor()
  }, [checkEditor])

  return (
    <section className="studio-full-surface qwen-editor-surface" aria-label="Qwen Image Editor">
      <header className="qwen-editor-toolbar">
        <div>
          <span className={`qwen-editor-status is-${editorState}`} aria-hidden="true" />
          <strong>Qwen Image Editor</strong>
          <small>
            {editorState === 'online'
              ? 'Desktop editor connected'
              : editorState === 'checking'
                ? 'Checking local editor'
                : 'Desktop editor is offline'}
          </small>
        </div>
        <div className="qwen-editor-actions">
          <button type="button" onClick={reloadEditor} aria-label="Reload Qwen Image Editor">
            <RefreshCcw size={15} aria-hidden="true" />
            Reload
          </button>
          <a href={QWEN_EDITOR_URL} target="_blank" rel="noreferrer">
            <ExternalLink size={15} aria-hidden="true" />
            Open separately
          </a>
        </div>
      </header>
      <div className="qwen-editor-frame-wrap">
        <iframe
          key={frameKey}
          className="qwen-editor-frame"
          src={QWEN_EDITOR_URL}
          title="Qwen Image Editor"
          allow="clipboard-write"
        />
        {editorState === 'offline' ? (
          <div className="qwen-editor-offline" role="status">
            <strong>Qwen Image Editor is not running</strong>
            <span>Start the desktop editor on 127.0.0.1:7865, then retry.</span>
            <button type="button" onClick={reloadEditor}>Retry connection</button>
          </div>
        ) : null}
      </div>
    </section>
  )
}
