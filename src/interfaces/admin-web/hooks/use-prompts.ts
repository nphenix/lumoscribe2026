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

export function usePrompts() {
  return useQuery<Prompt[]>({
    queryKey: ['prompts'],
    queryFn: async () => {
      const res = await api.get('/prompts');
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
    mutationFn: async (data: { id: string, content: string, is_active?: boolean }) => {
      const res = await api.put(`/prompts/${data.id}`, data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['prompts'] });
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
    },
  });
}
