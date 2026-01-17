import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import api from '@/lib/api';

// ================= Providers =================

export interface LLMProvider {
  id: string;
  name: string;
  provider_type: string;
  base_url?: string | null;
  api_key_env?: string | null;
  config?: Record<string, any> | null;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMProviderCreate {
  name: string;
  provider_type: string;
  base_url?: string | null;
  api_key?: string | null;
  api_key_env?: string | null;
  config?: Record<string, any> | null;
  enabled?: boolean;
  description?: string | null;
}

export function useLLMProviders() {
  return useQuery<LLMProvider[]>({
    queryKey: ['llm-providers'],
    queryFn: async () => {
      const res = await api.get('/llm/providers');
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

// ================= Models =================

export interface LLMModel {
  id: string;
  provider_id: string;
  name: string;
  model_kind: string;
  config?: Record<string, any> | null;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMModelCreate {
  provider_id: string;
  name: string;
  model_kind: string;
  config?: Record<string, any> | null;
  enabled?: boolean;
  description?: string | null;
}

export function useLLMModels() {
  return useQuery<LLMModel[]>({
    queryKey: ['llm-models'],
    queryFn: async () => {
      const res = await api.get('/llm/models');
      return res.data.items || [];
    },
  });
}

export function useCreateLLMModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (data: LLMModelCreate) => {
      const res = await api.post('/llm/models', data);
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-models'] });
    },
  });
}

export function useDeleteLLMModel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/llm/models/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['llm-models'] });
    },
  });
}

// ================= Capabilities (Mappings) =================

export interface LLMCapability {
  id: string;
  capability: string;
  model_id: string;
  priority: number;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMCapabilityUpsertItem {
  id?: string | null;
  capability: string;
  model_id: string;
  priority?: number;
  enabled?: boolean;
  description?: string | null;
}

export function useLLMCapabilities() {
  return useQuery<LLMCapability[]>({
    queryKey: ['llm-capabilities'],
    queryFn: async () => {
      const res = await api.get('/llm/capabilities');
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
  model_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
  enabled: boolean;
  description?: string | null;
  created_at: string;
  updated_at: string;
}

export interface LLMCallSiteCreate {
  key: string;
  expected_model_kind: string;
  model_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
  enabled?: boolean;
  description?: string | null;
}

export interface LLMCallSiteUpdate {
  model_id?: string | null;
  config?: Record<string, any> | null;
  prompt_scope?: string | null;
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
      const res = await api.get('/llm/call-sites', { params });
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
