'use client';

import { useState } from 'react';
import { useTemplates, useUploadTemplate, Template } from '@/hooks/use-documents';
import { Button } from '@/components/ui/button';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Upload, FileText, Lock, Unlock } from 'lucide-react';
import { toast } from 'sonner';
import { Badge } from '@/components/ui/badge';

export default function TemplatesPage() {
  const { data: templates, isLoading } = useTemplates();
  const uploadMutation = useUploadTemplate();
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const handleUpload = async () => {
    if (!selectedFile) return;
    try {
      await uploadMutation.mutateAsync(selectedFile);
      toast.success('模板上传成功');
      setIsUploadOpen(false);
      setSelectedFile(null);
    } catch (error) {
      toast.error('模板上传失败');
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">模板管理</h1>
        <Dialog open={isUploadOpen} onOpenChange={setIsUploadOpen}>
          <DialogTrigger asChild>
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              上传模板
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>上传模板</DialogTitle>
              <DialogDescription>
                支持 Word 文档 (.docx) 或 Markdown (.md) 格式。
              </DialogDescription>
            </DialogHeader>
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Label htmlFor="file">文件</Label>
              <Input 
                id="file" 
                type="file" 
                accept=".docx,.md"
                onChange={(e) => setSelectedFile(e.target.files?.[0] || null)} 
              />
            </div>
            <div className="flex justify-end mt-4">
              <Button onClick={handleUpload} disabled={!selectedFile || uploadMutation.isPending}>
                {uploadMutation.isPending ? '上传中...' : '上传'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>模板列表</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>文件名</TableHead>
                <TableHead>状态</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {templates?.map((template: Template) => (
                <TableRow key={template.id}>
                  <TableCell className="font-medium flex items-center">
                    <FileText className="mr-2 h-4 w-4 text-purple-500" />
                    {template.filename}
                  </TableCell>
                  <TableCell>
                    {template.is_locked ? (
                      <Badge variant="secondary" className="flex w-fit items-center gap-1">
                        <Lock className="h-3 w-3" /> 已锁定
                      </Badge>
                    ) : (
                      <Badge variant="outline" className="flex w-fit items-center gap-1">
                        <Unlock className="h-3 w-3" /> 未锁定
                      </Badge>
                    )}
                  </TableCell>
                  <TableCell>{new Date(template.created_at).toLocaleString()}</TableCell>
                  <TableCell className="text-right">
                    {/* Add actions like Lock/Unlock later */}
                  </TableCell>
                </TableRow>
              ))}
              {templates?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-8 text-gray-500">
                    暂无模板。
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
