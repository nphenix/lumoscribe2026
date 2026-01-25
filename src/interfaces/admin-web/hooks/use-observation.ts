import { useQuery } from '@tanstack/react-query';
import api from '@/lib/api';

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  result?: unknown;
  error?: string;
  celery_task_id?: string;
}

export function useJobs() {
  return useQuery<Job[]>({
    queryKey: ['jobs'],
    queryFn: async () => {
      const res = await api.get('/jobs');
      return res.data.items || [];
    },
    refetchInterval: 5000, // Poll every 5 seconds
  });
}

export interface IngestJob {
  id: number;
  job_type: string;
  status: string;
  progress: number;
  celery_task_id?: string | null;
  input_summary?: {
    file_count?: number;
    workspace_id?: string | null;
    source_file_ids?: string[];
    stage?: string;
    options?: Record<string, unknown> | null;
  } | null;
  result_summary?: {
    processed_count?: number;
    success_count?: number;
    failed_count?: number;
    errors?: string[];
    details?: Array<{
      original_chars?: number;
      cleaned_chars?: number;
      removed_chars?: number;
      original_paragraphs?: number;
      cleaned_paragraphs?: number;
      removed_paragraphs?: number;
      ads_removed?: number;
      noise_removed?: number;
      duplicates_removed?: number;
    }>;
  } | null;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  started_at?: string | null;
  finished_at?: string | null;
}

export function useCleaningJobs() {
  return useQuery<IngestJob[]>({
    queryKey: ['ingest-cleaning-jobs'],
    queryFn: async () => {
      const res = await api.get('/ingest/cleaning/jobs', { params: { limit: 50, offset: 0 } });
      return res.data.items || [];
    },
    refetchInterval: 5000,
  });
}

export interface HealthStatus {
  status: string;
  version: string;
  components: {
    db: boolean;
    redis: boolean;
    worker: boolean;
  };
  info?: {
    db: {
      type: string;
      path: string;
      description: string;
    };
    worker: {
      active_count: number;
      description: string;
    };
  };
}

export function useHealth() {
  return useQuery<HealthStatus>({
    queryKey: ['health'],
    queryFn: async () => {
      const res = await api.get('/health');
      return res.data;
    },
    refetchInterval: 30000,
  });
}
