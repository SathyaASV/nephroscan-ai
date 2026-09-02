import type { ReportContext } from './contracts.js';

function byId<T extends HTMLElement>(id: string): T | null {
  return document.getElementById(id) as T | null;
}

export function getImageInput(): HTMLInputElement | null {
  return byId<HTMLInputElement>('fileInput');
}
export function getReportBody(): HTMLElement | null {
  return byId<HTMLElement>('reportBody');
}

export interface ChatElements {
  messages: HTMLElement;
  input: HTMLInputElement;
  send: HTMLButtonElement;
  quick: HTMLElement;
}

export function getChatElements(): ChatElements | null {
  const messages = byId<HTMLElement>('healthGuideMessages');
  const input = byId<HTMLInputElement>('healthGuideInput');
  const send = byId<HTMLButtonElement>('healthGuideSendBtn');
  const quick = byId<HTMLElement>('healthGuideQuick');
  if (!messages || !input || !send || !quick) return null;
  return { messages, input, send, quick };
}

function valueOf(id: string): string {
  return (byId<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(id)?.value || '').trim();
}

export function readReportContext(): ReportContext {
  const reportBody = getReportBody();
  return {
    scanType: valueOf('scanType'),
    clinicalIndication: valueOf('clinicalIndication'),
    symptoms: valueOf('symptomsField'),
    patientName: valueOf('patientName'),
    patientId: valueOf('patientIdField'),
    reportText: (reportBody?.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 6000),
  };
}

export function escapeHtml(value: string): string {
  const element = document.createElement('span');
  element.textContent = value;
  return element.innerHTML;
}

export function setStatus(element: HTMLElement, status: string, message: string): void {
  element.dataset.aiStatus = status;
  element.setAttribute('role', 'status');
  element.setAttribute('aria-live', 'polite');
  element.textContent = message;
}
