'use client';

import { useState } from 'react';
import { useJobs, Job, useCleaningJobs, IngestJob } from '@/hooks/use-observation';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Activity, CheckCircle2, XCircle, Clock, Loader2 } from 'lucide-react';

export default function JobsPage() {
  const { data: jobs, isLoading } = useJobs();
  const { data: cleaningJobs } = useCleaningJobs();
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<IngestJob | null>(null);

  const getStatusBadge = (status: Job['status']) => {
    switch (status) {
      case 'completed':
        return <Badge className="bg-green-500"><CheckCircle2 className="w-3 h-3 mr-1" /> 已完成</Badge>;
      case 'failed':
        return <Badge variant="destructive"><XCircle className="w-3 h-3 mr-1" /> 失败</Badge>;
      case 'processing':
        return <Badge className="bg-blue-500"><Loader2 className="w-3 h-3 mr-1 animate-spin" /> 处理中</Badge>;
      default:
        return <Badge variant="secondary"><Clock className="w-3 h-3 mr-1" /> 等待中</Badge>;
    }
  };

  const getCleaningStatusBadge = (status: string) => {
    switch (status) {
      case 'succeeded':
        return <Badge className="bg-green-500"><CheckCircle2 className="w-3 h-3 mr-1" /> 成功</Badge>;
      case 'failed':
        return <Badge variant="destructive"><XCircle className="w-3 h-3 mr-1" /> 失败</Badge>;
      case 'partial':
        return <Badge className="bg-yellow-500"><Clock className="w-3 h-3 mr-1" /> 部分成功</Badge>;
      case 'running':
        return <Badge className="bg-blue-500"><Loader2 className="w-3 h-3 mr-1 animate-spin" /> 运行中</Badge>;
      default:
        return <Badge variant="secondary"><Clock className="w-3 h-3 mr-1" /> 等待中</Badge>;
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">任务观测</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>最近任务</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>任务 ID</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>状态</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead>耗时</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs?.map((job) => (
                <TableRow key={job.id}>
                  <TableCell className="font-mono text-xs">{job.id}</TableCell>
                  <TableCell className="font-medium flex items-center">
                    <Activity className="mr-2 h-4 w-4 text-gray-500" />
                    {job.type}
                  </TableCell>
                  <TableCell>{getStatusBadge(job.status)}</TableCell>
                  <TableCell>{new Date(job.created_at).toLocaleString()}</TableCell>
                  <TableCell className="text-gray-500">
                    {job.updated_at && job.created_at 
                      ? `${((new Date(job.updated_at).getTime() - new Date(job.created_at).getTime()) / 1000).toFixed(1)}s` 
                      : '-'}
                  </TableCell>
                </TableRow>
              ))}
              {jobs?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                    暂无任务。
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>清洗任务</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>任务 ID</TableHead>
                <TableHead>状态</TableHead>
                <TableHead>文件总数</TableHead>
                <TableHead>已处理</TableHead>
                <TableHead>成功</TableHead>
                <TableHead>失败</TableHead>
                <TableHead>开始时间</TableHead>
                <TableHead>完成时间</TableHead>
                <TableHead>耗时</TableHead>
                <TableHead>操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {cleaningJobs?.map((job) => {
                const total = job.input_summary?.file_count ?? null;
                const processed = job.result_summary?.processed_count ?? null;
                const success = job.result_summary?.success_count ?? null;
                const failed = job.result_summary?.failed_count ?? null;
                const startedAt = job.started_at ? new Date(job.started_at) : null;
                const finishedAt = job.finished_at ? new Date(job.finished_at) : null;
                const duration =
                  startedAt && finishedAt
                    ? `${((finishedAt.getTime() - startedAt.getTime()) / 1000).toFixed(1)}s`
                    : '-';
                return (
                  <TableRow key={job.id}>
                    <TableCell className="font-mono text-xs">{job.id}</TableCell>
                    <TableCell>{getCleaningStatusBadge(job.status)}</TableCell>
                    <TableCell>{total ?? '-'}</TableCell>
                    <TableCell>{processed ?? '-'}</TableCell>
                    <TableCell className="text-green-600">{success ?? '-'}</TableCell>
                    <TableCell className="text-red-600">{failed ?? '-'}</TableCell>
                    <TableCell>{job.started_at ? new Date(job.started_at).toLocaleString() : '-'}</TableCell>
                    <TableCell>{job.finished_at ? new Date(job.finished_at).toLocaleString() : '-'}</TableCell>
                    <TableCell className="text-gray-500">{duration}</TableCell>
                    <TableCell>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setSelected(job);
                          setOpen(true);
                        }}
                      >
                        查看详情
                      </Button>
                    </TableCell>
                  </TableRow>
                );
              })}
              {(!cleaningJobs || cleaningJobs.length === 0) && (
                <TableRow>
                  <TableCell colSpan={10} className="text-center py-8 text-gray-500">
                    暂无清洗任务。
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle>清洗任务详情</DialogTitle>
          </DialogHeader>
          {selected && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>任务 ID：<span className="font-mono">{selected.id}</span></div>
                <div>状态：{getCleaningStatusBadge(selected.status)}</div>
                <div>文件总数：{selected.input_summary?.file_count ?? '-'}</div>
                <div>已处理：{selected.result_summary?.processed_count ?? '-'}</div>
                <div>成功：<span className="text-green-600">{selected.result_summary?.success_count ?? '-'}</span></div>
                <div>失败：<span className="text-red-600">{selected.result_summary?.failed_count ?? '-'}</span></div>
              </div>
              <div>
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>原始字符</TableHead>
                      <TableHead>清洗后字符</TableHead>
                      <TableHead>移除字符</TableHead>
                      <TableHead>原始段落</TableHead>
                      <TableHead>清洗后段落</TableHead>
                      <TableHead>移除段落</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {(selected.result_summary?.details || []).map((d, idx) => (
                      <TableRow key={idx}>
                        <TableCell>{d?.original_chars ?? '-'}</TableCell>
                        <TableCell>{d?.cleaned_chars ?? '-'}</TableCell>
                        <TableCell className="text-blue-600">{d?.removed_chars ?? '-'}</TableCell>
                        <TableCell>{d?.original_paragraphs ?? '-'}</TableCell>
                        <TableCell>{d?.cleaned_paragraphs ?? '-'}</TableCell>
                        <TableCell>{d?.removed_paragraphs ?? '-'}</TableCell>
                      </TableRow>
                    ))}
                    {(!selected.result_summary?.details || selected.result_summary?.details.length === 0) && (
                      <TableRow>
                        <TableCell colSpan={6} className="text-center py-6 text-gray-500">
                          暂无详情。
                        </TableCell>
                      </TableRow>
                    )}
                  </TableBody>
                </Table>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
