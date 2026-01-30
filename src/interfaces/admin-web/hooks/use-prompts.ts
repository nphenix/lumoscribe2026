import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/lib/api';

export interface PromptMessage {
  role: string;
  content: string;
}

export interface Prompt {
  id: string;
  scope: string;
  format: string;
  content: string | null;
  messages: PromptMessage[] | null;
  version: number;
  active: boolean;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface PromptScopeSummary {
  scope: string;
  format: string | null;
  latest_version: number;
  active_version: number | null;
  versions: number;
  updated_at: string | null;
}

export function usePrompts(params?: { scope?: string; active?: boolean; format?: string }) {
  return useQuery<Prompt[]>({
    queryKey: ['prompts', params ?? null],
    queryFn: async () => {
      const res = await api.get('/prompts', { params: { limit: 200, offset: 0, ...(params || {}) } });
      return res.data.items || [];
    },
  });
}

export function usePromptScopes(params?: { scope?: string }) {
  return useQuery<PromptScopeSummary[]>({
    queryKey: ['prompt-scopes', params ?? null],
    queryFn: async () => {
      const res = await api.get('/prompts/scopes', { params: { limit: 500, offset: 0, ...(params || {}) } });
      return res.data.items || [];
    },
  });
}

export function usePrompt(id: string) {
  return useQuery<Prompt>({
    queryKey: ['prompts', id],
    queryFn: async () => {
      const res = await api.get(`/prompts/${id}`);
      return res.data;
    },
    enabled: !!id,
  });
}

export function useUpdatePrompt() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { id: string; active?: boolean; description?: string | null }) => {
      const res = await api.patch(`/prompts/${data.id}`, {
        active: data.active,
        description: data.description,
      });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] });
      queryClient.invalidateQueries({ queryKey: ['prompt-scopes'] });
    },
  });
}

export function usePromptDiff() {
  return useMutation({
    mutationFn: async (data: { from_id: string; to_id: string }) => {
      const res = await api.get('/prompts/diff', { params: data });
      return res.data as { from_id: string; to_id: string; scope: string | null; diff: string };
    },
  });
}

export function useCreatePrompt() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: Partial<Prompt>) => {
      // 适配 API 字段
      const payload = {
        scope: data.scope,
        format: data.format || 'text',
        content: data.content,
        messages: data.messages,
        active: data.active,
        description: data.description,
      };
      const res = await api.post('/prompts', payload);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] });
      queryClient.invalidateQueries({ queryKey: ['prompt-scopes'] });
    },
  });
}
