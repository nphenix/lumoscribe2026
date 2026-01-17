'use client';

import { useState } from 'react';
import { useLLMProviders, useCreateLLMProvider, useDeleteLLMProvider, LLMProviderCreate } from '@/hooks/use-llm';
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
import { Plus, Trash2, Server } from 'lucide-react';
import { toast } from 'sonner';

export default function ProvidersPage() {
  const { data: providers, isLoading } = useLLMProviders();
  const createMutation = useCreateLLMProvider();
  const deleteMutation = useDeleteLLMProvider();
  const [isOpen, setIsOpen] = useState(false);
  const [formData, setFormData] = useState<LLMProviderCreate>({
    name: '',
    base_url: '',
    api_key: '',
    api_key_env: '',
    provider_type: 'openai_compatible',
    enabled: true,
  });

  const handleSubmit = async () => {
    try {
      await createMutation.mutateAsync(formData as any);
      toast.success('供应商创建成功');
      setIsOpen(false);
      setFormData({
        name: '',
        base_url: '',
        api_key: '',
        api_key_env: '',
        provider_type: 'openai_compatible',
        enabled: true,
      });
    } catch (error) {
      toast.error('创建供应商失败');
    }
  };

  const handleDelete = async (id: string) => {
    if (confirm('确认删除？这可能会影响依赖此供应商的模型。')) {
      try {
        await deleteMutation.mutateAsync(id);
        toast.success('供应商已删除');
      } catch (error) {
        toast.error('删除供应商失败');
      }
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">模型供应商</h1>
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="mr-2 h-4 w-4" />
              添加供应商
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>添加模型供应商</DialogTitle>
              <DialogDescription>
                配置一个新的 LLM 服务端点。
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="name">名称</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="type">类型</Label>
                <Select
                  value={formData.provider_type}
                  onValueChange={(value) => setFormData({ ...formData, provider_type: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择类型" />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" sideOffset={6}>
                    <SelectItem value="openai_compatible">OpenAI 兼容</SelectItem>
                    <SelectItem value="ollama">Ollama</SelectItem>
                    <SelectItem value="flagembedding">FlagEmbedding</SelectItem>
                    <SelectItem value="cohere">Cohere</SelectItem>
                    <SelectItem value="huggingface">HuggingFace</SelectItem>
                    <SelectItem value="llamacpp">LlamaCpp</SelectItem>
                    <SelectItem value="mineru">MinerU (OCR/解析)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="base_url">Base URL</Label>
                <Input
                  id="base_url"
                  placeholder="例如：https://api.openai.com/v1"
                  value={formData.base_url}
                  onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="api_key">API Key / Token（明文）</Label>
                <Input
                  id="api_key"
                  type="password"
                  placeholder="可选：直接填写明文 Key/Token"
                  value={formData.api_key || ''}
                  onChange={(e) => setFormData({ ...formData, api_key: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="api_key_env">API Key 环境变量名</Label>
                <Input
                  id="api_key_env"
                  placeholder="可选：例如 OPENAI_API_KEY"
                  value={formData.api_key_env || ''}
                  onChange={(e) => setFormData({ ...formData, api_key_env: e.target.value })}
                />
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
          <CardTitle>已配置供应商</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>名称</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>Base URL</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers?.map((provider) => (
                <TableRow key={provider.id}>
                  <TableCell className="font-medium flex items-center">
                    <Server className="mr-2 h-4 w-4 text-blue-500" />
                    {provider.name}
                  </TableCell>
                  <TableCell>{provider.provider_type}</TableCell>
                  <TableCell className="text-gray-500">{provider.base_url}</TableCell>
                  <TableCell>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        provider.enabled
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {provider.enabled ? '启用' : '禁用'}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      className="text-red-500 hover:text-red-700 hover:bg-red-50"
                      onClick={() => handleDelete(provider.id)}
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {providers?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                    暂无配置。
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
