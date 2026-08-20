import type {
  AiHealthResponse,
  ChatMessage,
  ChatResponse,
  ErrorResponse,
  ImageAnalysisResponse,
  ReportContext,
} from './contracts.js';
import { ApiError } from './contracts.js';

const JSON_HEADERS = { Accept: 'application/json' };

async function parseError(response: Response): Promise<ApiError> {
  let body: ErrorResponse = {};
  try {
    body = (await response.json()) as ErrorResponse;
  } catch {
    // Keep the HTTP status useful when a proxy returns non-JSON text.
  }
  const message = body.message || body.error || response.statusText || `Request failed (${response.status})`;
  const disabled = response.status === 404 || response.status === 501 || body.status === 'disabled' || body.code === 'ai_disabled';
  return new ApiError(message, response.status, disabled, body.request_id, body.disclaimer);
}

async function expectJson<T>(response: Response): Promise<T> {
  if (!response.ok) throw await parseError(response);
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError('The server returned an invalid AI response.', response.status);
  }
}

export async function getAiHealth(signal?: AbortSignal): Promise<AiHealthResponse> {
  const response = await fetch('/api/ai/health', {
    method: 'GET',
    headers: JSON_HEADERS,
    credentials: 'same-origin',
    signal,
  });
  return expectJson<AiHealthResponse>(response);
}


/** Same-origin client. No credentials or provider configuration ever crosses into the browser. */
export async function analyzeImage(
  image: File,
  scanType: string,
  context: ReportContext,
  signal?: AbortSignal,
): Promise<ImageAnalysisResponse> {
  const form = new FormData();
  form.append('image', image, image.name || 'scan-image');
  form.append('scan_type', scanType);
  form.append('context', JSON.stringify(context));
  const response = await fetch('/api/ai/analyze-image', {
    method: 'POST',
    body: form,
    headers: JSON_HEADERS,
    signal,
    credentials: 'same-origin',
  });
  return expectJson<ImageAnalysisResponse>(response);
}

export async function chat(
  messages: ChatMessage[],
  context: unknown,
  signal?: AbortSignal,
): Promise<ChatResponse> {
  const response = await fetch('/api/ai/chat', {
    method: 'POST',
    headers: { ...JSON_HEADERS, 'Content-Type': 'application/json' },
    credentials: 'same-origin',
    body: JSON.stringify({ messages, context }),
    signal,
  });
  return expectJson<ChatResponse>(response);
}
