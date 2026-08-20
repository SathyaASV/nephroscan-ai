import type { ChatMessage, ReportContext } from './contracts.js';

export type AiCapability = 'unknown' | 'enabled' | 'disabled' | 'unavailable';
export type AiStatus = 'idle' | 'loading' | 'ready' | 'disabled' | 'error';

export interface FrontendState {
  selectedImage: File | null;
  imageStatus: AiStatus;
  imageRequestKey: string;
  chatStatus: AiStatus;
  chatMessages: ChatMessage[];
  aiCapability: AiCapability;
}

export function createFrontendState(): FrontendState {
  return {
    selectedImage: null,
    imageStatus: 'idle',
    imageRequestKey: '',
    chatStatus: 'idle',
    chatMessages: [],
    aiCapability: 'unknown',
  };
}

export function imageRequestKey(file: File, context: ReportContext): string {
  return [file.name, file.size, file.lastModified, context.scanType, context.reportText.slice(0, 160)].join(':');
}

export function addChatMessage(state: FrontendState, message: ChatMessage): void {
  state.chatMessages.push({ role: message.role, content: message.content });
  if (state.chatMessages.length > 20) state.chatMessages.splice(0, state.chatMessages.length - 20);
}

export function resetChat(state: FrontendState): void {
  state.chatMessages.length = 0;
  state.chatStatus = 'idle';
}
