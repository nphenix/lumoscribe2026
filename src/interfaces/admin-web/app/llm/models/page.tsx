'use client';

import { useState } from 'react';
import { useLLMModels, useCreateLLMModel, useDeleteLLMModel, useLLMProviders, LLMModelCreate } from '@/hooks/use-llm';
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
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Plus, Trash2, Cpu } from 'lucide-react';
import { toast } from 'sonner';

export default function ModelsPage() {
  const { data: models, isLoading } = useLLMModels();
  const { data: providers } = useLLMProviders();
  const createMutation = useCreateLLMModel();
  const deleteMutation = useDeleteLLMModel();
  const [isOpen, setIsOpen] = useState(false);
  const [configText, setConfigText] = useState<string>('{}');
  const [formData, setFormData] = useState<LLMModelCreate>({
    name: '',
    provider_id: '',
    model_kind: 'chat',
    enabled: true,
  });

  const handleSubmit = async () => {
    try {
      const config =
        configText && configText.trim().length > 0 ? JSON.parse(configText) : undefined;
      await createMutation.mutateAsync({ ...formData, config });
      toast.success('模型创建成功');
      setIsOpen(false);
      setFormData({
        name: '',
        provider_id: '',
        model_kind: 'chat',
        enabled: true,
      });
      setConfigText('{}');
    } catch (error) {
      toast.error('创建模型失败（请检查 JSON 格式或后端校验）');
    }
  };

  const handleDelete = async (id: string) => {
    if (confirm('确认删除？')) {
      try {
        await deleteMutation.mutateAsync(id);
        toast.success('模型已删除');
      } catch (error) {
        toast.error('删除模型失败');
      }
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">模型管理</h1>
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              添加模型
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>添加模型</DialogTitle>
              <DialogDescription>
                注册供应商提供的具体模型。
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="provider">供应商</Label>
                <Select
                  value={formData.provider_id}
                  onValueChange={(value) => setFormData({ ...formData, provider_id: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择供应商" />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" sideOffset={6}>
                    {providers?.map((p) => (
                      <SelectItem key={p.id} value={p.id}>
                        {p.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="name">模型名称 (ID)</Label>
                <Input
                  id="name"
                  placeholder="gpt-3.5-turbo"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="type">类型</Label>
                <Select
                  value={formData.model_kind}
                  onValueChange={(value) => setFormData({ ...formData, model_kind: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择类型" />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" sideOffset={6}>
                    <SelectItem value="chat">对话 (Chat)</SelectItem>
                    <SelectItem value="embedding">向量 (Embedding)</SelectItem>
                    <SelectItem value="rerank">重排序 (Rerank)</SelectItem>
                    <SelectItem value="multimodal">多模态 (Multimodal)</SelectItem>
                    <SelectItem value="ocr">OCR (MinerU)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="config">参数 (JSON)</Label>
                <Textarea
                  id="config"
                  value={configText}
                  onChange={(e) => setConfigText(e.target.value)}
                  className="font-mono text-sm"
                />
              </div>
              <div className="flex items-center space-x-2">
                <Switch
                  id="active"
                  checked={formData.enabled ?? true}
                  onCheckedChange={(checked) => setFormData({ ...formData, enabled: checked })}
                />
                <Label htmlFor="active">启用</Label>
              </div>
            </div>
            <div className="flex justify-end">
              <Button onClick={handleSubmit} disabled={createMutation.isPending}>
                {createMutation.isPending ? '保存中...' : '保存'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>已注册模型</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>名称</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {models?.map((model) => (
                <TableRow key={model.id}>
                  <TableCell className="font-medium flex items-center">
                    <Cpu className="mr-2 h-4 w-4 text-green-500" />
                    {model.name}
                  </TableCell>
                  <TableCell>{model.model_kind}</TableCell>
                  <TableCell>
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                      model.enabled ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                    }`}>
                      {model.enabled ? '启用' : '禁用'}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                      onClick={() => handleDelete(model.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {models?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={4} className="text-center py-8 text-gray-500">
                    暂无模型。
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
