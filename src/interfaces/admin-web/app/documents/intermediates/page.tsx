'use client';

import { useIntermediates, IntermediateArtifact } from '@/hooks/use-documents';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Database } from 'lucide-react';

export default function IntermediatesPage() {
  const { data: intermediates, isLoading } = useIntermediates();

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">中间产物</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>处理过程数据</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>文件名</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>创建时间</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {intermediates?.map((artifact: IntermediateArtifact) => (
                <TableRow key={artifact.id}>
                  <TableCell className="font-medium flex items-center">
                    <Database className="mr-2 h-4 w-4 text-orange-500" />
                    {artifact.filename}
                  </TableCell>
                  <TableCell>{artifact.artifact_type}</TableCell>
                  <TableCell>{new Date(artifact.created_at).toLocaleString()}</TableCell>
                </TableRow>
              ))}
              {intermediates?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-8 text-gray-500">
                    暂无中间产物。
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
