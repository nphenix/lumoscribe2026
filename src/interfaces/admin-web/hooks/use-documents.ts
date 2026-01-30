import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import api from '@/lib/api';

// --- Sources ---
export interface SourceFile {
  id: string;
  original_filename: string;
  storage_path: string;
  file_size: number;
  content_hash: string;
  created_at: string;
  updated_at: string;
  status: string;
}

export function useSources() {
  return useQuery<SourceFile[]>({
    queryKey: ['sources'],
    queryFn: async () => {
      const res = await api.get('/sources');
      return res.data.items || [];
    },
  });
}

export function useUploadSource() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('workspace_id', 'default');
      const res = await api.post('/sources', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}

export function useDeleteSource() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (id: string) => {
      await api.delete(`/sources/${id}`);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sources'] });
    },
  });
}

// --- Templates ---
export interface Template {
  id: string;
  filename: string;
  storage_path: string;
  is_locked: boolean;
  created_at: string;
  updated_at: string;
}

export function useTemplates() {
  return useQuery<Template[]>({
    queryKey: ['templates'],
    queryFn: async () => {
      const res = await api.get('/templates');
      return res.data.items || [];
    },
  });
}

export function useUploadTemplate() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      const res = await api.post('/templates', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['templates'] });
    },
  });
}

// --- Targets ---
export interface TargetFile {
  id: string;
  filename: string;
  storage_path: string;
  created_at: string;
}

export function useTargets() {
  return useQuery<TargetFile[]>({
    queryKey: ['targets'],
    queryFn: async () => {
      const res = await api.get('/targets');
      return res.data.items || [];
    },
  });
}

// --- Intermediates ---
export interface IntermediateArtifact {
  id: string;
  filename: string;
  artifact_type: string;
  storage_path: string;
  created_at: string;
}

export function useIntermediates() {
  return useQuery<IntermediateArtifact[]>({
    queryKey: ['intermediates'],
    queryFn: async () => {
      const res = await api.get('/intermediates');
      return res.data.items || [];
    },
  });
}
