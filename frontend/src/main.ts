import { analyzeImage, chat, getAiHealth } from './api.js';
import type { ApiError, ImageAnalysisResponse } from './contracts.js';
import { getChatElements, getImageInput, getReportBody, escapeHtml, readReportContext, setStatus } from './dom.js';
import type { ChatElements } from './dom.js';
import { addChatMessage, createFrontendState, imageRequestKey, resetChat } from './state.js';

const state = createFrontendState();
const EDUCATIONAL_DISCLAIMER = 'Optional AI assistance only. This educational screening prototype is not a diagnosis and does not replace review by a qualified healthcare professional.';
let reportMutationTimer: number | undefined;
let aiInitialization: Promise<void> | undefined;

async function initializeAi(): Promise<void> {
  try {
    const health = await getAiHealth();
    state.aiCapability = health.enabled ? 'enabled' : 'disabled';
  } catch {
    state.aiCapability = 'unavailable';
  }
  const reportBody = getReportBody();
  const reportText = reportBody?.textContent?.trim() || '';
  if (!reportBody || reportBody.querySelector('.placeholder') || !reportText || reportText.includes('Upload scans') || reportText.includes('Analysis in progress') || reportText.includes('Analysis Error')) return;
  if (state.aiCapability === 'enabled') {
    void enrichCurrentReport();
    return;
  }
  const status = state.aiCapability === 'disabled' ? 'disabled' : 'error';
  renderImageUnavailable(reportBody, state.aiCapability === 'disabled' ? 'Optional multimodal review is disabled on this server.' : 'Optional multimodal review is unavailable right now.', status);
}
function ensureAiInitialized(): Promise<void> {
  if (!aiInitialization) aiInitialization = initializeAi();
  return aiInitialization;
}


function isApiError(error: unknown): error is ApiError {
  return error instanceof Error && error.name === 'ApiError';
}

function imageStatusElement(reportBody: HTMLElement): HTMLElement {
  const existing = reportBody.querySelector<HTMLElement>('[data-ai-image-status]');
  if (existing) return existing;
  const status = document.createElement('div');
  status.className = 'disclaimer-box';
  status.dataset.aiImageStatus = 'true';
  status.dataset.aiStatus = 'idle';
  status.setAttribute('role', 'status');
  status.setAttribute('aria-live', 'polite');
  reportBody.appendChild(status);
  return status;
}

function renderImageAnalysis(reportBody: HTMLElement, result: ImageAnalysisResponse): void {
  const prior = reportBody.querySelector('[data-ai-image-enhancement]');
  prior?.remove();
  const section = document.createElement('section');
  section.className = 'report-section ai-image-enhancement';
  section.dataset.aiImageEnhancement = 'true';
  const findings = result.findings.filter((item) => item.trim().length > 0);
  const limitations = result.limitations.filter((item) => item.trim().length > 0);
  const nextSteps = result.next_steps.filter((item) => item.trim().length > 0);
  const list = (items: string[]): string => items.length
    ? `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`
    : '<p>No additional observations were returned.</p>';
  section.innerHTML = `
    <div class="section-title"><i class="fas fa-sparkles"></i> OPTIONAL MULTIMODAL REVIEW</div>
    <div class="content">
      <p><strong>Summary:</strong> ${escapeHtml(result.summary || 'No summary returned.')}</p>
      <p><strong>Candidate observations:</strong></p>${list(findings)}
      ${limitations.length ? `<p><strong>Limitations:</strong></p>${list(limitations)}` : ''}
      ${nextSteps.length ? `<p><strong>Suggested discussion points:</strong></p>${list(nextSteps)}` : ''}
      <div class="disclaimer-box">${escapeHtml(result.disclaimer || EDUCATIONAL_DISCLAIMER)}</div>
      <div class="ai-meta"><span><i class="fas fa-cloud"></i> ${escapeHtml(result.provider)}</span><span><i class="fas fa-microchip"></i> ${escapeHtml(result.model)}</span><span><i class="fas fa-hashtag"></i> ${escapeHtml(result.request_id)}</span></div>
    </div>`;
  const status = imageStatusElement(reportBody);
  setStatus(status, 'ready', 'Optional multimodal review added. The existing local report remains the primary screening output.');
  reportBody.insertBefore(section, status);
}

function renderImageUnavailable(reportBody: HTMLElement, message: string, status: 'disabled' | 'error', disclaimer = EDUCATIONAL_DISCLAIMER, includeLocalReport = true): void {
  reportBody.querySelector('[data-ai-image-enhancement]')?.remove();
  const element = imageStatusElement(reportBody);
  const suffix = includeLocalReport ? ' Existing local analysis is still available.' : '';
  const copy = `${message}${suffix} ${disclaimer}`;
  if (element.textContent !== copy) setStatus(element, status, copy);
}

async function fileFromPreview(): Promise<File | null> {
  const image = document.querySelector<HTMLImageElement>('#scanPreviewWrap img');
  const source = image?.getAttribute('src');
  if (!source || !source.startsWith('blob:')) return null;
  try {
    const response = await fetch(source);
    const blob = await response.blob();
    if (!blob.size || !blob.type.startsWith('image/')) return null;
    return new File([blob], 'scan-preview', { type: blob.type });
  } catch {
    return null;
  }
}

async function enrichCurrentReport(): Promise<void> {
  const reportBody = getReportBody();
  if (!reportBody || reportBody.querySelector('[data-ai-image-enhancement]')) return;
  const reportText = (reportBody.textContent || '').trim();
  if (isExploratoryScan(readReportContext().scanType)) return;
  if (!reportText || reportBody.querySelector('.placeholder') || reportText.includes('Upload scans') || reportText.includes('Analysis in progress') || reportText.includes('Analysis Error')) return;
  if (state.aiCapability === 'unknown') return;
  if (state.aiCapability !== 'enabled') {
    renderImageUnavailable(reportBody, state.aiCapability === 'disabled' ? 'Optional multimodal review is disabled on this server.' : 'Optional multimodal review is unavailable right now.', state.aiCapability === 'disabled' ? 'disabled' : 'error');
    return;
  }
  const image = state.selectedImage || await fileFromPreview();
  if (!image) return;
  state.selectedImage = image;
  const context = readReportContext();
  const key = imageRequestKey(image, context);
  if (state.imageRequestKey === key || state.imageStatus === 'loading') return;
  state.imageRequestKey = key;
  state.imageStatus = 'loading';
  const status = imageStatusElement(reportBody);
  setStatus(status, 'loading', 'Optional multimodal review is loading. The local screening report remains available.');
  try {
    const result = await analyzeImage(image, context.scanType, context);
    state.imageStatus = 'ready';
    renderImageAnalysis(reportBody, result);
  } catch (error) {
    state.imageStatus = isApiError(error) && error.disabled ? 'disabled' : 'error';
    const message = isApiError(error) && error.disabled
      ? 'Optional multimodal review is disabled on this server.'
      : isApiError(error) && error.status === 429
        ? 'The AI provider is rate limiting multimodal requests; please retry shortly.'
        : 'Optional multimodal review is unavailable right now.';
    const disclaimer = isApiError(error) && error.disclaimer ? error.disclaimer : EDUCATIONAL_DISCLAIMER;
    renderImageUnavailable(reportBody, message, state.imageStatus, disclaimer);
  }
}

function observeReports(): void {
  const reportBody = getReportBody();
  if (!reportBody) return;
  const observer = new MutationObserver(() => {
    window.clearTimeout(reportMutationTimer);
    reportMutationTimer = window.setTimeout(() => void enrichCurrentReport(), 120);
  });
  observer.observe(reportBody, { childList: true, subtree: true, characterData: true });
}

function isExploratoryScan(scanType: string): boolean {
  return scanType === 'other' || scanType === 'other_ai';
}

async function runExploratoryAnalysis(button: HTMLButtonElement): Promise<void> {
  const reportBody = getReportBody();
  const scanType = (document.getElementById('scanType') as HTMLSelectElement | null)?.value || '';
  if (!reportBody || !isExploratoryScan(scanType)) return;
  await ensureAiInitialized();
  if (state.aiCapability !== 'enabled') {
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, state.aiCapability === 'disabled' ? 'Optional multimodal review is disabled on this server.' : 'Optional multimodal review is unavailable right now.', state.aiCapability === 'disabled' ? 'disabled' : 'error', EDUCATIONAL_DISCLAIMER, false);
    return;
  }
  const image = state.selectedImage || await fileFromPreview();
  if (!image) {
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, 'Upload a supported image before running exploratory analysis.', 'error', EDUCATIONAL_DISCLAIMER, false);
    return;
  }
  state.selectedImage = image;
  state.imageStatus = 'loading';
  state.imageRequestKey = `exploratory:${Date.now()}`;
  reportBody.innerHTML = '<div class="loading-report"><div class="spinner-ring"></div><p><i class="fas fa-microchip"></i> Optional multimodal review in progress</p><div style="font-size:.68rem;color:#9aa7b2;">No local classifier is used for this scan type.</div></div>';
  const status = imageStatusElement(reportBody);
  setStatus(status, 'loading', 'Optional multimodal review is loading. This is not a diagnosis.');
  const originalLabel = button.innerHTML;
  button.disabled = true;
  button.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Reviewing...';
  try {
    const result = await analyzeImage(image, scanType, readReportContext());
    state.imageStatus = 'ready';
    reportBody.innerHTML = '';
    renderImageAnalysis(reportBody, result);
  } catch (error) {
    state.imageStatus = isApiError(error) && error.disabled ? 'disabled' : 'error';
    const message = isApiError(error) && error.disabled
      ? 'Optional multimodal review is disabled on this server.'
      : 'Optional multimodal review is unavailable right now.';
    const disclaimer = isApiError(error) && error.disclaimer ? error.disclaimer : EDUCATIONAL_DISCLAIMER;
    reportBody.innerHTML = '';
    renderImageUnavailable(reportBody, message, state.imageStatus, disclaimer, false);
  } finally {
    button.disabled = false;
    button.innerHTML = originalLabel;
  }
}

function connectExploratoryAnalysis(): void {
  const button = document.getElementById('generateBtn') as HTMLButtonElement | null;
  const scanType = document.getElementById('scanType') as HTMLSelectElement | null;
  if (!button || !scanType) return;
  button.addEventListener('click', (event) => {
    if (!isExploratoryScan(scanType.value)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    void runExploratoryAnalysis(button);
  }, true);
}

function captureSelectedImage(): void {
  const input = getImageInput();
  if (!input) return;
  input.addEventListener('change', () => {
    const file = Array.from(input.files || []).find((candidate) => candidate.type.startsWith('image/'));
    if (file) {
      state.selectedImage = file;
      state.imageRequestKey = '';
      state.imageStatus = 'idle';
    }
  }, true);
  document.getElementById('clearBtn')?.addEventListener('click', () => {
    state.selectedImage = null;
    state.imageRequestKey = '';
    state.imageStatus = 'idle';
  }, true);
}

function chatContext(): Record<string, string> {
  const context = readReportContext();
  return {
    scan_type: context.scanType,
    patient: context.patientName || 'not specified',
    report: context.reportText,
    disclaimer: EDUCATIONAL_DISCLAIMER,
  };
}

function chatStatusElement(elements: ChatElements): HTMLElement {
  const parent = elements.messages.parentElement || elements.messages;
  const existing = parent.querySelector<HTMLElement>('[data-ai-chat-status]');
  if (existing) return existing;
  const status = document.createElement('div');
  status.dataset.aiChatStatus = 'true';
  status.style.cssText = 'font-size:.68rem;color:#6b7f8d;margin:.4rem 0;';
  parent.insertBefore(status, elements.messages.nextSibling);
  return status;
}

function replaceLatestLocalReply(elements: ChatElements, text: string): void {
  const replies = elements.messages.querySelectorAll<HTMLElement>('.bot-msg.bot');
  const latest = replies.item(replies.length - 1);
  if (latest) latest.textContent = text;
  else {
    const reply = document.createElement('div');
    reply.className = 'bot-msg bot';
    reply.textContent = text;
    elements.messages.appendChild(reply);
  }
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

async function submitChat(userText: string, elements: ChatElements): Promise<void> {
  if (state.chatStatus === 'loading') return;
  addChatMessage(state, { role: 'user', content: userText });
  await ensureAiInitialized();
  if (state.aiCapability !== 'enabled') {
    state.chatStatus = state.aiCapability === 'disabled' ? 'disabled' : 'error';
    const message = state.aiCapability === 'disabled'
      ? 'Optional AI assistant is disabled on this server; continuing with local guidance.'
      : 'Optional AI assistant is unavailable; continuing with local guidance.';
    setStatus(chatStatusElement(elements), state.chatStatus, `${message} ${EDUCATIONAL_DISCLAIMER}`);
    return;
  }
  state.chatStatus = 'loading';
  elements.send.disabled = true;
  elements.input.setAttribute('aria-busy', 'true');
  setStatus(chatStatusElement(elements), 'loading', 'Optional AI assistant is thinking. Local guidance remains available.');
  try {
    const result = await chat(state.chatMessages, chatContext());
    state.chatStatus = 'ready';
    const reply = `${result.message}\n\n${result.disclaimer || EDUCATIONAL_DISCLAIMER}`;
    addChatMessage(state, { role: 'assistant', content: reply });
    replaceLatestLocalReply(elements, reply);
    setStatus(chatStatusElement(elements), 'ready', 'Optional AI assistant response received. Review it with a qualified professional.');
  } catch (error) {
    state.chatStatus = isApiError(error) && error.disabled ? 'disabled' : 'error';
    const disabled = isApiError(error) && error.disabled;
    const message = disabled
      ? 'Optional AI assistant is disabled on this server; continuing with local guidance.'
      : isApiError(error) && error.status === 429
        ? 'The AI provider is rate limiting requests; please retry shortly. Local guidance remains available.'
        : 'Optional AI assistant is unavailable; continuing with local guidance.';
    const disclaimer = isApiError(error) && error.disclaimer ? error.disclaimer : EDUCATIONAL_DISCLAIMER;
    setStatus(chatStatusElement(elements), state.chatStatus, `${message} ${disclaimer}`);
  } finally {
    elements.send.disabled = false;
    elements.input.removeAttribute('aria-busy');
  }
}

function connectChat(): void {
  const elements = getChatElements();
  if (!elements) return;
  elements.send.addEventListener('click', () => {
    const value = elements.input.value.trim();
    if (!value) return;
    // The legacy listener intentionally remains in place as the deterministic fallback.
    window.setTimeout(() => void submitChat(value, elements), 400);
  }, true);
  elements.quick.addEventListener('click', (event) => {
    const button = (event.target as HTMLElement).closest<HTMLButtonElement>('button[data-q]');
    const value = button?.dataset.q || '';
    if (!value) return;
    if (value === 'clear_chat') {
      resetChat(state);
      const status = elements.messages.parentElement?.querySelector<HTMLElement>('[data-ai-chat-status]');
      status?.remove();
      return;
    }
    window.setTimeout(() => void submitChat(value, elements), 400);
  }, true);
}

function start(): void {
  aiInitialization = initializeAi();
  captureSelectedImage();
  connectExploratoryAnalysis();
  observeReports();
  connectChat();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start, { once: true });
else start();
