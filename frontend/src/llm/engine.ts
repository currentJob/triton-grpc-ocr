import { pipeline, TextStreamer, env } from '@huggingface/transformers'

const MODEL_ID = 'onnx-community/Qwen2.5-1.5B-Instruct'

export interface LLMLoadProgress {
  file:     string
  loaded:   number
  total:    number
  progress: number
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _pipe: any = null

async function setupEnv(): Promise<void> {
  env.allowRemoteModels = true
  // 로컬 모델 파일 확인 — public/models/llm/에 배치된 경우만 로컬 경로 사용
  // no-cors HEAD로 서버 로그 오염 방지: 실제 fetch가 아닌 캐시 확인
  try {
    const localBase = `${location.origin}${import.meta.env.BASE_URL}models/llm/`
    const res = await fetch(`${localBase}${MODEL_ID}/config.json`, {
      method: 'HEAD',
      cache:  'no-store',
    })
    if (res.ok) {
      env.localModelPath = localBase
      console.info('[LLM] 로컬 모델 사용:', localBase)
      return
    }
  } catch { /* 로컬 파일 없음 — HuggingFace CDN 사용 */ }
  console.info('[LLM] HuggingFace CDN 사용 (브라우저 캐시 적용)')
}

export function isLLMLoaded(): boolean { return _pipe !== null }

export async function loadLLM(onProgress?: (p: LLMLoadProgress) => void): Promise<void> {
  if (_pipe) return
  await setupEnv()

  // @huggingface/transformers v4.x: CPU 백엔드 이름은 'wasm' ('cpu' 는 무효)
  const device: 'webgpu' | 'wasm' = 'gpu' in navigator ? 'webgpu' : 'wasm'

  // WebGPU: q4f16 (4-bit 가중치 + fp16 활성화 → 속도·정확도 향상)
  // WASM  : q4 (fp16 미지원)
  const load = (dev: 'webgpu' | 'wasm') =>
    pipeline('text-generation', MODEL_ID, {
      dtype:  dev === 'webgpu' ? 'q4f16' : 'q4',
      device: dev,
      progress_callback: (info: Record<string, unknown>) => {
        if (info.status === 'progress' && onProgress) {
          onProgress({
            file:     String(info.file     ?? ''),
            loaded:   Number(info.loaded   ?? 0),
            total:    Number(info.total    ?? 0),
            progress: Number(info.progress ?? 0),
          })
        }
      },
    })

  try {
    _pipe = await load(device)
  } catch (e) {
    if (device === 'webgpu') {
      console.warn('[LLM] WebGPU 실패, WASM으로 재시도:', e)
      _pipe = await load('wasm')
    } else {
      throw e
    }
  }
}

export function unloadLLM(): void { _pipe = null }

function buildMessages(texts: string[]): Array<{ role: string; content: string }> {
  const list = texts.map((t, i) => `${i + 1}. ${t}`).join('\n')
  return [
    {
      role:    'system',
      content: '당신은 OCR로 추출된 한국어 텍스트를 분석하고 체계적으로 정리하는 AI 어시스턴트입니다. 간결하고 명확하게 답변하세요.',
    },
    {
      role:    'user',
      content: `이미지에서 OCR로 추출된 텍스트 목록입니다:\n\n${list}\n\n위 내용을 다음 형식으로 정리해주세요:\n\n**문서 유형**: (영수증/명함/문서/표지판/기타)\n**요약**: (1~2문장)\n**핵심 정보**:\n- (중요 항목들)\n\n**정리된 텍스트**:\n(읽기 좋은 순서로 재배열)`,
    },
  ]
}

export async function summarize(
  texts:   string[],
  onToken: (token: string) => void,
): Promise<void> {
  if (!_pipe) throw new Error('LLM이 로딩되지 않았습니다. loadLLM()을 먼저 호출하세요.')

  const messages = buildMessages(texts)

  const streamer = new TextStreamer(_pipe.tokenizer, {
    skip_prompt:         true,
    skip_special_tokens: true,
    callback_function:   (token: string) => onToken(token),
  })

  await _pipe(messages, {
    max_new_tokens: 768,
    temperature:    0.3,
    do_sample:      true,
    streamer,
  })
}
