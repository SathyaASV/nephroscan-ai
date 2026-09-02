export type ChatRole = 'user' | 'assistant';

export interface ChatMessage {
  role: ChatRole;
  content: string;
}

export interface AiHealthResponse {
  enabled: boolean;
  vision_model: string | null;
  chat_model: string | null;
  max_image_bytes: number;
  accepted_image_types: string[];
  max_messages: number;
  max_message_chars: number;
}

export interface ImageAnalysisResponse {
  status: 'ok';
  provider: string;
  model: string;
  summary: string;
  findings: string[];
  limitations: string[];
  next_steps: string[];
  disclaimer: string;
  request_id: string;
}

export interface ChatResponse {
  status: 'ok';
  message: string;
  disclaimer: string;
  model: string;
  request_id: string;
}

export interface ErrorResponse {
  status?: string;
  error?: string;
  message?: string;
  code?: string;
  request_id?: string;
  disclaimer?: string;
}

export interface ReportContext {
  scanType: string;
  clinicalIndication: string;
  symptoms: string;
  patientName: string;
  patientId: string;
  reportText: string;
}

export class ApiError extends Error {
  readonly status: number;
  readonly disabled: boolean;
  readonly requestId?: string;
  readonly disclaimer?: string;

  constructor(message: string, status: number, disabled = false, requestId?: string, disclaimer?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.disabled = disabled;
    this.requestId = requestId;
    this.disclaimer = disclaimer;
  }
}

