'use client';

import { useTargets, TargetFile } from '@/hooks/use-documents';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Download, FileOutput } from 'lucide-react';
import api from '@/lib/api';

export default function TargetsPage() {
  const { data: targets, isLoading } = useTargets();

  const handleDownload = async (id: string, filename: string) => {
    try {
      const response = await api.get(`/targets/${id}/content`, {
        responseType: 'blob',
      });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (error) {
      console.error('Download failed', error);
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">生成结果</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>已生成文档</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>文件名</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {targets?.map((target: TargetFile) => (
                <TableRow key={target.id}>
                  <TableCell className="font-medium flex items-center">
                    <FileOutput className="mr-2 h-4 w-4 text-green-500" />
                    {target.filename}
                  </TableCell>
                  <TableCell>{new Date(target.created_at).toLocaleString()}</TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="outline" 
                      size="sm"
                      onClick={() => handleDownload(target.id, target.filename)}
                    >
                      <Download className="mr-2 h-4 w-4" />
                      下载
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {targets?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={3} className="text-center py-8 text-gray-500">
                    暂无生成结果。
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
