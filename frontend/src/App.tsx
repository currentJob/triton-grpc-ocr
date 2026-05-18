import { useEffect, useRef, useState, useCallback, useMemo } from 'react'
import type { OcrItem, Phase, HistoryEntry } from './types'
import { useOcrApi }    from './hooks/useOcrApi'
import { useImage }     from './hooks/useImage'
import { useRoi }       from './hooks/useRoi'
import { useHistory }   from './hooks/useHistory'
import { useCamera }    from './hooks/useCamera'
import { useLlm }       from './hooks/useLlm'
import Header           from './components/Header'
import CameraOverlay    from './components/CameraOverlay'
import HistoryPanel     from './components/HistoryPanel'
import UploadArea       from './components/UploadArea'
import ImagePanel       from './components/ImagePanel'
import ResultPanel      from './components/ResultPanel'
import type { FilteredItem } from './components/ResultPanel'
import SummarySection   from './components/SummarySection'
import './App.css'

interface ServerStatus {
  live:   boolean
  ready:  boolean
  models: Record<string, boolean>
}

export default function App() {
  const [phase,    setPhase]    = useState<Phase>('ready')
  const [error,    setError]    = useState('')
  const [items,    setItems]    = useState<OcrItem[]>([])
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [currentFilename, setCurrentFilename] = useState('')
  const [threshold,   setThreshold]   = useState(0)
  const [searchQuery, setSearchQuery] = useState('')
  const [fileQueue,   setFileQueue]   = useState<File[]>([])
  const [serverStatus, setServerStatus] = useState<ServerStatus | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const imgRef    = useRef<HTMLImageElement>(null)

  const ocrApi = useOcrApi()
  const image  = useImage()
  const roi    = useRoi(image.natSize, canvasRef, phase)
  const hist   = useHistory()
  const llm    = useLlm()
  const camera = useCamera(processFile)

  /* ── Server health polling ──────────────────────────────────── */
  const checkHealth = useCallback(async () => {
    try {
      const res  = await fetch('/api/health')
      const data = await res.json() as ServerStatus
      setServerStatus(data)
    } catch {
      setServerStatus({ live: false, ready: false, models: {} })
    }
  }, [])

  useEffect(() => {
    checkHealth()
    const id = setInterval(checkHealth, 10_000)
    return () => {
      clearInterval(id)
      camera.stopOnUnmount()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ── State helpers ──────────────────────────────────────────── */
  const resetResults = useCallback(() => {
    setItems([])
    setSelected(new Set())
    roi.clearRoi()
  }, [roi])

  const resetRoi = useCallback(() => {
    resetResults()
    setPhase('roi')
    setThreshold(0)
    setSearchQuery('')
    llm.reset()
  }, [resetResults, llm])

  /* ── File processing ────────────────────────────────────────── */
  function processFile(file: File) {
    if (!file.type.startsWith('image/')) return
    const url = URL.createObjectURL(file)
    resetResults()
    setPhase('roi')
    setError('')
    setThreshold(0)
    setSearchQuery('')
    setCurrentFilename(file.name)
    image.resetTransforms()
    llm.reset()
    const img = new Image()
    img.onload = () => image.setFile(url, img)
    img.src = url
  }

  const processFiles = useCallback((files: File[]) => {
    const imgs = files.filter(f => f.type.startsWith('image/'))
    if (!imgs.length) return
    setFileQueue(imgs.slice(1))
    processFile(imgs[0])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  /* ── OCR ─────────────────────────────────────────────────────── */
  async function runOcr() {
    if (!image.loadedImgRef.current || !image.natSize) return
    setPhase('running')
    setItems([])
    setSelected(new Set())

    const srcImg = await image.getPreprocessedImg()
    let ocrImg = srcImg, offsetX = 0, offsetY = 0

    if (roi.roiFixed) {
      const { x, y, w, h } = roi.roiFixed
      const c = document.createElement('canvas')
      c.width = w; c.height = h
      c.getContext('2d')!.drawImage(srcImg, x, y, w, h, 0, 0, w, h)
      ocrImg = await new Promise<HTMLImageElement>(resolve => {
        const o = new Image()
        o.onload = () => resolve(o)
        o.src    = c.toDataURL()
      })
      offsetX = x; offsetY = y
    }

    try {
      const result    = await ocrApi.predict(ocrImg)
      const offsetted = result.map(it => ({
        ...it,
        box: it.box.map(([bx, by]) => [bx + offsetX, by + offsetY]) as [number, number][],
      }))
      setItems(offsetted)
      setSelected(new Set(offsetted.map((_, i) => i)))
      setPhase('done')
      await hist.addToHistory(image.loadedImgRef.current!, offsetted, currentFilename)
    } catch (e) {
      setError(String(e))
      setPhase('error')
    }
  }

  /* ── History restore ─────────────────────────────────────────── */
  async function restoreHistory(entry: HistoryEntry) {
    setItems(entry.items)
    setSelected(new Set(entry.items.map((_, i) => i)))
    setPhase('done')
    setThreshold(0)
    setSearchQuery('')
    setCurrentFilename(entry.filename)
    image.resetTransforms()
    llm.reset()
    hist.setShowHistory(false)
    const url = (await hist.getPreview(entry.id)) ?? entry.thumb
    const img = new Image()
    img.onload = () => image.setFile(
      url, img,
      entry.natW && entry.natH ? { w: entry.natW, h: entry.natH } : undefined,
    )
    img.src = url
  }

  /* ── Derived values ──────────────────────────────────────────── */
  const filteredWithIdx = useMemo<FilteredItem[]>(() =>
    items
      .map((item, origIdx) => ({ item, origIdx }))
      .filter(({ item }) =>
        (item.recScore * 100) >= threshold &&
        (!searchQuery || item.text.toLowerCase().includes(searchQuery.toLowerCase()))
      ),
    [items, threshold, searchQuery],
  )

  const showResult = image.imageUrl !== null
  const hasQueue   = fileQueue.length > 0

  /* ── Render ──────────────────────────────────────────────────── */
  return (
    <div className="app">
      <Header
        serverStatus={serverStatus}
        history={hist.history}
        onToggleHistory={() => hist.setShowHistory(s => !s)}
      />

      {camera.showCamera && (
        <CameraOverlay
          videoRef={camera.videoRef}
          onStop={camera.stopCamera}
          onCapture={camera.captureCamera}
        />
      )}

      {hist.showHistory && hist.history.length > 0 && (
        <HistoryPanel
          history={hist.history}
          onRestore={restoreHistory}
          onRemove={hist.removeHistory}
          onClose={() => hist.setShowHistory(false)}
        />
      )}

      <main className="main">

        {phase === 'ready' && (
          <UploadArea
            onFiles={processFiles}
            onCameraOpen={camera.startCamera}
          />
        )}

        {showResult && (
          <>
            <div className="result-layout">
              <ImagePanel
                imageUrl={image.imageUrl!}
                imgRef={imgRef}
                canvasRef={canvasRef}
                brightness={image.brightness}
                contrast={image.contrast}
                phase={phase}
                items={items}
                selected={selected}
                natSize={image.natSize}
                roiFixed={roi.roiFixed}
                roiDraw={roi.roiDraw}
                threshold={threshold}
                searchQuery={searchQuery}
                rotation={image.rotation}
                onRotationChange={image.handleRotationChange}
                onBrightnessChange={image.setBrightness}
                onContrastChange={image.setContrast}
                onResetPreprocess={() => { image.setBrightness(100); image.setContrast(100) }}
                onCanvasMouseDown={roi.onCanvasMouseDown}
                onCanvasMouseMove={roi.onCanvasMouseMove}
                onCommitRoi={roi.commitRoi}
                onFileInput={e => { processFiles(Array.from(e.target.files ?? [])); e.target.value = '' }}
                onCameraOpen={camera.startCamera}
                onResetRoi={resetRoi}
                onRunOcr={runOcr}
                onNextFile={hasQueue && phase === 'done'
                  ? () => { processFile(fileQueue[0]); setFileQueue(q => q.slice(1)) }
                  : null}
                fileQueueCount={fileQueue.length}
              />

              <ResultPanel
                phase={phase}
                error={error}
                items={items}
                selected={selected}
                filteredWithIdx={filteredWithIdx}
                threshold={threshold}
                searchQuery={searchQuery}
                llmStatus={llm.llmStatus}
                onToggleItem={i => setSelected(prev => {
                  const n = new Set(prev); n.has(i) ? n.delete(i) : n.add(i); return n
                })}
                onSelectAll={() => setSelected(new Set(items.map((_, i) => i)))}
                onDeselectAll={() => setSelected(new Set())}
                onEditItem={(idx, text) =>
                  setItems(prev => prev.map((it, i) => i === idx ? { ...it, text } : it))
                }
                onThresholdChange={setThreshold}
                onSearchChange={setSearchQuery}
                onRunAnalysis={() =>
                  llm.runAnalysis(items.filter((_, i) => selected.has(i)).map(it => it.text))
                }
              />
            </div>

            {llm.llmStatus !== 'idle' && (
              <SummarySection
                status={llm.llmStatus}
                progress={llm.llmProgress}
                error={llm.llmError}
                summary={llm.summary}
              />
            )}
          </>
        )}

      </main>
    </div>
  )
}
