import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import api from '@/lib/api';

// ================= Providers =================

export interface LLMProvider {
  id: string;
  key: string;
  name: string;
  provider_type: string;
  base_url?: string | null;
  api_key_env?: string | null;
  config?: Record<string, any> | null;
  max_concurrency?: number | null;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderCreate {
  key?: string | null;
  name: string;
  provider_type: string;
  base_url?: string | null;
  api_key?: string | null;
  api_key_env?: string | null;
  config?: Record<string, any> | null;
  max_concurrency?: number | null;
  enabled?: boolean;
  description?: string | null;
}

export interface LLMProviderUpdate {
  key?: string | null;
  name?: string | null;
  provider_type?: string | null;
  base_url?: string | null;
  api_key?: string | null;
  api_key_env?: string | null;
  config?: Record<string, any> | null;
  max_concurrency?: number | null;
  enabled?: boolean | null;
  description?: string | null;
}

export function useLLMProviders() {
  return useQuery<LLMProvider[]>({
    queryKey: ['llm-providers'],
    queryFn: async () => {
      const res = await api.get('/llm/providers', { params: { limit: 200, offset: 0 } });
      return res.data.items || [];
    },
  });
}

export function useCreateLLMProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: LLMProviderCreate) => {
      const res = await api.post('/llm/providers', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-providers'] });
    },
  });
}

export function useUpdateLLMProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { id: string; patch: LLMProviderUpdate }) => {
      const res = await api.patch(`/llm/providers/${data.id}`, data.patch);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-providers'] });
    },
  });
}

export function useDeleteLLMProvider() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/llm/providers/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-providers'] });
    },
  });
}

// ================= Capabilities (Mappings) =================

export interface LLMCapability {
  id: string;
  capability: string;
  provider_id: string;
  priority: number;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMCapabilityUpsertItem {
  id?: string | null;
  capability: string;
  provider_id: string;
  priority?: number;
  enabled?: boolean;
  description?: string | null;
}

export function useLLMCapabilities() {
  return useQuery<LLMCapability[]>({
    queryKey: ['llm-capabilities'],
    queryFn: async () => {
      const res = await api.get('/llm/capabilities', { params: { limit: 500, offset: 0 } });
      return res.data.items || [];
    },
  });
}

export function useUpdateLLMCapability() {
  // 兼容旧命名：实际调用 PATCH /llm/capabilities 批量 upsert
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (item: LLMCapabilityUpsertItem) => {
      const res = await api.patch('/llm/capabilities', { items: [item] });
      return (res.data.items && res.data.items[0]) || null;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-capabilities'] });
    },
  });
}

// ================= CallSites =================

export interface LLMCallSite {
  id: string;
  key: string;
  expected_model_kind: string;
  provider_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
  max_concurrency?: number | null;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMCallSiteCreate {
  key: string;
  expected_model_kind: string;
  provider_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
  max_concurrency?: number | null;
  enabled?: boolean;
  description?: string | null;
}

export interface LLMCallSiteUpdate {
  provider_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
  max_concurrency?: number | null;
  enabled?: boolean;
  description?: string | null;
}

export function useLLMCallSites(params?: {
  key?: string;
  expected_model_kind?: string;
  enabled?: boolean;
  bound?: boolean;
}) {
  return useQuery<LLMCallSite[]>({
    queryKey: ['llm-call-sites', params || {}],
    queryFn: async () => {
      const res = await api.get('/llm/call-sites', {
        params: { limit: 500, offset: 0, ...(params || {}) },
      });
      return res.data.items || [];
    },
  });
}

export function useCreateLLMCallSite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: LLMCallSiteCreate) => {
      const res = await api.post('/llm/call-sites', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-call-sites'] });
    },
  });
}

export function useUpdateLLMCallSite() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: { id: string; patch: LLMCallSiteUpdate }) => {
      const res = await api.patch(`/llm/call-sites/${data.id}`, data.patch);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-call-sites'] });
    },
  });
}
