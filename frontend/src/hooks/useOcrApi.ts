import { useRef } from 'react'
import type { OcrItem } from '../types'

interface ApiResult {
  text:       string
  confidence: number
  bbox:       { x1: number; y1: number; x2: number; y2: number }
}

interface ApiResponse {
  results:    ApiResult[]
  image_size: { width: number; height: number }
  count:      number
}

function imgToBlob(img: HTMLImageElement): Promise<Blob> {
  const c = document.createElement('canvas')
  c.width  = img.naturalWidth
  c.height = img.naturalHeight
  c.getContext('2d')!.drawImage(img, 0, 0)
  return new Promise((resolve, reject) =>
    c.toBlob(
      b => b ? resolve(b) : reject(new Error('이미지 변환 실패 (canvas toBlob)')),
      'image/jpeg',
      0.92,
    )
  )
}

function toOcrItems(results: ApiResult[]): OcrItem[] {
  return results.map(r => ({
    text:     r.text,
    recScore: r.confidence,
    detScore: r.confidence,
    box: [
      [r.bbox.x1, r.bbox.y1],
      [r.bbox.x2, r.bbox.y1],
      [r.bbox.x2, r.bbox.y2],
      [r.bbox.x1, r.bbox.y2],
    ] as [number, number][],
  }))
}

export function useOcrApi() {
  const abortRef = useRef<AbortController | null>(null)

  async function predict(img: HTMLImageElement): Promise<OcrItem[]> {
    /* Cancel any in-flight request */
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    const blob = await imgToBlob(img)
    const form = new FormData()
    form.append('file', blob, 'image.jpg')

    let res: Response
    try {
      res = await fetch('/api/ocr', {
        method: 'POST',
        body:   form,
        signal: abortRef.current.signal,
      })
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') {
        throw new Error('요청이 취소되었습니다.')
      }
      throw new Error('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.')
    }

    if (!res.ok) {
      let detail = `OCR 서버 오류 (HTTP ${res.status})`
      try {
        const json = await res.json() as { detail?: string }
        if (json.detail) detail = json.detail
      } catch { /* ignore */ }
      throw new Error(detail)
    }

    const data = await res.json() as ApiResponse
    return toOcrItems(data.results ?? [])
  }

  function cancel() {
    abortRef.current?.abort()
  }

  return { predict, cancel }
}
