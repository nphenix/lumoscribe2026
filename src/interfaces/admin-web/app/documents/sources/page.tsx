'use client';

import { useState } from 'react';
import { useSources, useUploadSource, useDeleteSource, SourceFile } from '@/hooks/use-documents';
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
import { Trash2, Upload, File as FileIcon } from 'lucide-react';
import { toast } from 'sonner';

export default function SourcesPage() {
  const { data: sources, isLoading } = useSources();
  const uploadMutation = useUploadSource();
  const deleteMutation = useDeleteSource();
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [uploadProgress, setUploadProgress] = useState<{ current: number; total: number } | null>(null);

  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;
    
    setUploadProgress({ current: 0, total: selectedFiles.length });
    let successCount = 0;
    let failCount = 0;

    for (let i = 0; i < selectedFiles.length; i++) {
      const file = selectedFiles[i];
      try {
        await uploadMutation.mutateAsync(file);
        successCount++;
      } catch (error) {
        failCount++;
        toast.error(`文件 ${file.name} 上传失败`);
      }
      setUploadProgress({ current: i + 1, total: selectedFiles.length });
    }

    if (successCount > 0) {
      toast.success(`${successCount} 个文件上传成功`);
    }
    
    setIsUploadOpen(false);
    setSelectedFiles([]);
    setUploadProgress(null);
  };

  const handleDelete = async (id: string) => {
    if (confirm('确认删除此文件吗？')) {
      try {
        await deleteMutation.mutateAsync(id);
        toast.success('文件删除成功');
      } catch (error) {
        toast.error('文件删除失败');
      }
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">源文件管理</h1>
        <Dialog open={isUploadOpen} onOpenChange={setIsUploadOpen}>
          <DialogTrigger asChild>
            <Button>
              <Upload className="mr-2 h-4 w-4" />
              上传源文件
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>上传源文件</DialogTitle>
              <DialogDescription>
                请上传需要处理的 PDF 文档。
              </DialogDescription>
            </DialogHeader>
            <div className="grid w-full max-w-sm items-center gap-1.5">
              <Label htmlFor="file">文件（支持多选）</Label>
              <Input 
                id="file" 
                type="file" 
                accept=".pdf"
                multiple
                onChange={(e) => {
                  if (e.target.files) {
                    setSelectedFiles(Array.from(e.target.files));
                  }
                }} 
              />
              {selectedFiles.length > 0 && (
                <p className="text-sm text-muted-foreground">
                  已选择 {selectedFiles.length} 个文件
                </p>
              )}
            </div>
            <div className="flex justify-end mt-4">
              <Button 
                onClick={handleUpload} 
                disabled={selectedFiles.length === 0 || !!uploadProgress}
              >
                {uploadProgress 
                  ? `上传中 (${uploadProgress.current}/${uploadProgress.total})...` 
                  : '批量上传'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>文件列表</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>文件名</TableHead>
                <TableHead>大小</TableHead>
                <TableHead>创建时间</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sources?.map((source: SourceFile) => (
                <TableRow key={source.id}>
                  <TableCell className="font-medium flex items-center">
                    <FileIcon className="mr-2 h-4 w-4 text-blue-500" />
                    {source.original_filename}
                  </TableCell>
                  <TableCell>{(source.file_size / 1024).toFixed(2)} KB</TableCell>
                  <TableCell>{new Date(source.created_at).toLocaleString()}</TableCell>
                  <TableCell className="text-right">
                    <Button 
                      variant="ghost" 
                      size="icon" 
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                      onClick={() => handleDelete(source.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {sources?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-8 text-gray-500">
                    暂无源文件，请上传。
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
