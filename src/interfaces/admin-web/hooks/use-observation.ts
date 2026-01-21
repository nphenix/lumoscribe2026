import { useQuery } from '@tanstack/react-query';
import api from '@/lib/api';

export interface Job {
  id: string;
  type: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
  updated_at: string;
  result?: any;
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
