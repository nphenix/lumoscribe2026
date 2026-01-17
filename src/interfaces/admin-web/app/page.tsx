'use client';

import { useHealth } from '@/hooks/use-observation';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Activity, Database, Server, CheckCircle2, XCircle } from 'lucide-react';

export default function DashboardPage() {
  const { data: health } = useHealth();

  const StatusIcon = ({ status }: { status: boolean }) => 
    status 
      ? <CheckCircle2 className="h-5 w-5 text-green-500" />
      : <XCircle className="h-5 w-5 text-red-500" />;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold tracking-tight">系统概览</h1>
      
      <div className="grid gap-4 md:grid-cols-3">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">系统状态</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">
              {health?.status === 'ok' ? '正常' : '异常'}
            </div>
            <p className="text-xs text-muted-foreground">
              版本 {health?.version || '0.0.0'}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">数据库</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <StatusIcon status={health?.components?.db ?? false} />
              <span className="text-2xl font-bold">
                {health?.components?.db ? '已连接' : '未连接'}
              </span>
            </div>
            {health?.info?.db && (
              <div className="mt-2 space-y-1">
                <p className="text-xs text-muted-foreground">
                  {health.info.db.type}: {health.info.db.path}
                </p>
                <p className="text-xs text-muted-foreground">
                  {health.info.db.description}
                </p>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">任务节点</CardTitle>
            <Server className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="flex items-center gap-2">
              <StatusIcon status={health?.components?.worker ?? false} />
              <span className="text-2xl font-bold">
                {health?.components?.worker ? '在线' : '离线'}
              </span>
            </div>
            {health?.components?.worker && health?.info?.worker ? (
              <div className="mt-2 space-y-1">
                <p className="text-xs text-muted-foreground">
                  活跃节点: {health.info.worker.active_count}
                </p>
                <p className="text-xs text-muted-foreground">
                  {health.info.worker.description}
                </p>
              </div>
            ) : (
              <p className="mt-1 text-xs text-muted-foreground">
                请检查 Worker 服务状态
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
        <Card className="col-span-4">
          <CardHeader>
            <CardTitle>最近活动</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex h-[200px] items-center justify-center text-muted-foreground">
              暂无最近活动。
            </div>
          </CardContent>
        </Card>
        
        <Card className="col-span-3">
          <CardHeader>
            <CardTitle>快速操作</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {/* Add quick action buttons later */}
              <p className="text-sm text-muted-foreground">暂无配置快速操作。</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
