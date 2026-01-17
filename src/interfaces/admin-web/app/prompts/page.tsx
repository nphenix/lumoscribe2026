'use client';

import { useState } from 'react';
import { usePrompts, useCreatePrompt, Prompt, PromptMessage } from '@/hooks/use-prompts';
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
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Plus, MessageSquare, Edit } from 'lucide-react';
import { toast } from 'sonner';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export default function PromptsPage() {
  const { data: prompts, isLoading } = usePrompts();
  const createMutation = useCreatePrompt();
  const [isOpen, setIsOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [formData, setFormData] = useState<Partial<Prompt>>({
    scope: '',
    format: 'text',
    content: '',
    messages: [],
    active: true,
    description: '',
  });

  const handleEdit = (prompt: Prompt) => {
    // 转换消息为简单的字符串内容（如果是 Chat 模式，暂时只取第一条 User 消息用于编辑）
    let content = prompt.content || '';
    if (prompt.format === 'chat' && prompt.messages && prompt.messages.length > 0) {
        // 尝试找到 user message
        const userMsg = prompt.messages.find(m => m.role === 'user');
        if (userMsg) {
            content = userMsg.content;
        } else {
            // fallback to JSON string if needed, or just first message
            content = prompt.messages[0].content;
        }
    }

    setFormData({
      scope: prompt.scope,
      format: prompt.format,
      content: content,
      messages: prompt.messages || [],
      active: true, // 编辑新版本默认激活
      description: prompt.description || '',
    });
    setIsEditing(true);
    setIsOpen(true);
  };

  const handleCreate = () => {
    setFormData({
      scope: '',
      format: 'text',
      content: '',
      messages: [],
      active: true,
      description: '',
    });
    setIsEditing(false);
    setIsOpen(true);
  };

  const handleSubmit = async () => {
    try {
      if (!formData.scope) {
        toast.error('Scope 不能为空');
        return;
      }
      
      // 简单处理 chat 格式
      let messages: PromptMessage[] = [];
      if (formData.format === 'chat') {
        if (typeof formData.content === 'string') {
            // 构造标准 Chat 模板
            messages = [
                { role: 'system', content: 'You are a helpful assistant.' },
                { role: 'user', content: formData.content }
            ];
            
            // 如果原先有 messages，尝试保留 system prompt
            if (formData.messages && formData.messages.length > 0) {
                const sysMsg = formData.messages.find(m => m.role === 'system');
                if (sysMsg) {
                    messages[0] = sysMsg;
                }
            }
        }
      }

      await createMutation.mutateAsync({
        ...formData,
        messages: formData.format === 'chat' ? messages : undefined,
      });
      
      toast.success(isEditing ? '新版本发布成功' : '提示词创建成功');
      setIsOpen(false);
      setFormData({ scope: '', format: 'text', content: '', messages: [], active: true, description: '' });
      setIsEditing(false);
    } catch (error) {
      toast.error('操作失败');
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">提示词管理</h1>
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button onClick={handleCreate}>
              <Plus className="mr-2 h-4 w-4" />
              新建提示词
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle>{isEditing ? '编辑提示词 (发布新版本)' : '新建提示词'}</DialogTitle>
              <DialogDescription>
                {isEditing 
                    ? `正在为 ${formData.scope} 创建 v${(prompts?.find(p => p.scope === formData.scope)?.version || 0) + 1} 版本。`
                    : '定义新的提示词模板。新版本将自动作为最新版本。'}
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="scope">Scope (作用域)</Label>
                <Input
                  id="scope"
                  placeholder="例如：doc_cleaning:clean_text"
                  value={formData.scope}
                  onChange={(e) => setFormData({ ...formData, scope: e.target.value })}
                  disabled={isEditing}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="description">说明</Label>
                <Input
                  id="description"
                  placeholder="提示词用途说明"
                  value={formData.description || ''}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="format">格式</Label>
                <Select
                  value={formData.format}
                  onValueChange={(value) => setFormData({ ...formData, format: value })}
                  disabled={isEditing}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择格式" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="text">Text (Completion)</SelectItem>
                    <SelectItem value="chat">Chat (Messages)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="grid gap-2">
                <Label htmlFor="content">
                  {formData.format === 'chat' ? 'User Message 模板' : '内容模板'}
                </Label>
                <Textarea
                  id="content"
                  placeholder="输入提示词内容，使用 {变量} 占位..."
                  className="min-h-[200px] font-mono"
                  value={formData.content || ''}
                  onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                />
                {formData.format === 'chat' && (
                    <p className="text-xs text-gray-500">
                        * Chat 模式下，此处仅编辑 User Message。System Message 将默认保留或使用默认值。
                    </p>
                )}
              </div>
            </div>
            <div className="flex justify-end">
              <Button onClick={handleSubmit} disabled={createMutation.isPending}>
                {createMutation.isPending ? '保存中...' : '保存并发布'}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>系统提示词</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Scope</TableHead>
                <TableHead>版本</TableHead>
                <TableHead>格式</TableHead>
                <TableHead>说明</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {prompts?.map((prompt) => (
                <TableRow key={prompt.id}>
                  <TableCell className="font-medium flex items-center">
                    <MessageSquare className="mr-2 h-4 w-4 text-purple-500" />
                    {prompt.scope}
                  </TableCell>
                  <TableCell>v{prompt.version}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{prompt.format}</Badge>
                  </TableCell>
                  <TableCell className="text-gray-500">
                    {prompt.description || '-'}
                  </TableCell>
                  <TableCell>
                    {prompt.active ? (
                      <Badge className="bg-green-500 hover:bg-green-600">启用</Badge>
                    ) : (
                      <Badge variant="secondary">停用</Badge>
                    )}
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="sm" onClick={() => handleEdit(prompt)}>
                      <Edit className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
