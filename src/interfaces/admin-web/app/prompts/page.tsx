'use client';

import { useMemo, useState } from 'react';
import {
  useCreatePrompt,
  usePromptDiff,
  usePromptScopes,
  usePrompts,
  useUpdatePrompt,
  Prompt,
  PromptMessage,
  PromptScopeSummary,
} from '@/hooks/use-prompts';
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
import { Plus, MessageSquare, Edit, Diff, CheckCircle2, XCircle } from 'lucide-react';
import { toast } from 'sonner';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';

export default function PromptsPage() {
  const { data: scopes, isLoading: scopesLoading } = usePromptScopes();
  const [selectedScope, setSelectedScope] = useState<PromptScopeSummary | null>(null);
  const { data: scopePrompts, isLoading: scopePromptsLoading } = usePrompts(
    selectedScope ? { scope: selectedScope.scope } : undefined
  );
  const createMutation = useCreatePrompt();
  const updateMutation = useUpdatePrompt();
  const diffMutation = usePromptDiff();
  const [isOpen, setIsOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [scopeDialogOpen, setScopeDialogOpen] = useState(false);
  const [diffFromId, setDiffFromId] = useState<string>('');
  const [diffToId, setDiffToId] = useState<string>('');
  const [diffText, setDiffText] = useState<string>('');
  const [formData, setFormData] = useState<Partial<Prompt>>({
    scope: '',
    format: 'text',
    content: '',
    messages: [],
    active: true,
    description: '',
  });

  const openNewVersionFromPrompt = (prompt: Prompt) => {
    let content = prompt.content || '';
    if (prompt.format === 'chat' && prompt.messages && prompt.messages.length > 0) {
      const userMsg = prompt.messages.find((m) => m.role === 'user');
      content = userMsg ? userMsg.content : prompt.messages[0].content;
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

  const openScope = (scope: PromptScopeSummary) => {
    setSelectedScope(scope);
    setScopeDialogOpen(true);
    setDiffFromId('');
    setDiffToId('');
    setDiffText('');
  };

  const versions = useMemo(() => {
    return (scopePrompts || []).slice().sort((a, b) => (b.version || 0) - (a.version || 0));
  }, [scopePrompts]);

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
          messages = [
            { role: 'system', content: 'You are a helpful assistant.' },
            { role: 'user', content: formData.content },
          ];
          if (formData.messages && formData.messages.length > 0) {
            const sysMsg = formData.messages.find((m) => m.role === 'system');
            if (sysMsg) messages[0] = sysMsg;
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

  const compare = async () => {
    if (!diffFromId || !diffToId) {
      toast.error('请选择两个版本进行对比');
      return;
    }
    try {
      const res = await diffMutation.mutateAsync({ from_id: diffFromId, to_id: diffToId });
      setDiffText(res.diff || '');
    } catch (e) {
      toast.error('对比失败');
    }
  };

  const setActive = async (prompt: Prompt, active: boolean) => {
    try {
      await updateMutation.mutateAsync({ id: prompt.id, active });
      toast.success(active ? '已启用该版本' : '已停用该版本');
    } catch (e) {
      toast.error('操作失败');
    }
  };

  if (scopesLoading) return <div>加载中...</div>;

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
                    ? `正在为 ${formData.scope} 创建新版本。`
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
          <CardTitle>Scope 列表</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Scope</TableHead>
                <TableHead>格式</TableHead>
                <TableHead>启用版本</TableHead>
                <TableHead>最新版本</TableHead>
                <TableHead>版本数</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {(scopes || []).map((s) => (
                <TableRow key={s.scope}>
                  <TableCell className="font-medium flex items-center">
                    <MessageSquare className="mr-2 h-4 w-4 text-purple-500" />
                    {s.scope}
                  </TableCell>
                  <TableCell>
                    <Badge variant="outline">{s.format || '—'}</Badge>
                  </TableCell>
                  <TableCell>{s.active_version ? `v${s.active_version}` : '—'}</TableCell>
                  <TableCell>{`v${s.latest_version}`}</TableCell>
                  <TableCell>{s.versions}</TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="sm" onClick={() => openScope(s)}>
                      查看
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {(scopes || []).length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                    暂无提示词。
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={scopeDialogOpen} onOpenChange={setScopeDialogOpen}>
        <DialogContent className="sm:max-w-[1100px]" onInteractOutside={(e) => e.preventDefault()}>
          <DialogHeader>
            <DialogTitle>Scope：{selectedScope?.scope}</DialogTitle>
            <DialogDescription>版本列表、启停与差异对比</DialogDescription>
          </DialogHeader>

          {scopePromptsLoading ? (
            <div>加载中...</div>
          ) : (
            <div className="grid gap-6">
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600">
                  {selectedScope?.active_version ? `当前启用 v${selectedScope.active_version}` : '当前无启用版本'}
                </div>
                <Button
                  variant="outline"
                  onClick={() => {
                    const latest = versions[0];
                    if (!latest) {
                      toast.error('该 scope 暂无版本');
                      return;
                    }
                    openNewVersionFromPrompt(latest);
                  }}
                >
                  <Edit className="mr-2 h-4 w-4" />
                  发布新版本
                </Button>
              </div>

              <Card>
                <CardHeader>
                  <CardTitle>版本列表</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>版本</TableHead>
                        <TableHead>状态</TableHead>
                        <TableHead>说明</TableHead>
                        <TableHead>更新时间</TableHead>
                        <TableHead className="text-right">操作</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {versions.map((p) => (
                        <TableRow key={p.id}>
                          <TableCell>{`v${p.version}`}</TableCell>
                          <TableCell>
                            {p.active ? (
                              <Badge className="bg-green-500 hover:bg-green-600">启用</Badge>
                            ) : (
                              <Badge variant="secondary">停用</Badge>
                            )}
                          </TableCell>
                          <TableCell className="text-gray-600">{p.description || '—'}</TableCell>
                          <TableCell className="text-gray-600">
                            {p.updated_at ? String(p.updated_at) : '—'}
                          </TableCell>
                          <TableCell className="text-right">
                            <Button variant="ghost" size="sm" onClick={() => openNewVersionFromPrompt(p)}>
                              <Edit className="mr-2 h-4 w-4" />
                              基于此版本发布
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setActive(p, true)}
                              disabled={updateMutation.isPending}
                            >
                              <CheckCircle2 className="mr-2 h-4 w-4" />
                              启用
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => setActive(p, false)}
                              disabled={updateMutation.isPending}
                            >
                              <XCircle className="mr-2 h-4 w-4" />
                              停用
                            </Button>
                          </TableCell>
                        </TableRow>
                      ))}
                      {versions.length === 0 && (
                        <TableRow>
                          <TableCell colSpan={5} className="text-center py-8 text-gray-500">
                            暂无版本。
                          </TableCell>
                        </TableRow>
                      )}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>差异对比</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
                    <div className="grid gap-2">
                      <Label>版本 A</Label>
                      <Select value={diffFromId} onValueChange={setDiffFromId}>
                        <SelectTrigger>
                          <SelectValue placeholder="选择版本" />
                        </SelectTrigger>
                        <SelectContent>
                          {versions.map((p) => (
                            <SelectItem key={p.id} value={p.id}>
                              {`v${p.version}`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="grid gap-2">
                      <Label>版本 B</Label>
                      <Select value={diffToId} onValueChange={setDiffToId}>
                        <SelectTrigger>
                          <SelectValue placeholder="选择版本" />
                        </SelectTrigger>
                        <SelectContent>
                          {versions.map((p) => (
                            <SelectItem key={p.id} value={p.id}>
                              {`v${p.version}`}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="flex items-end">
                      <Button onClick={compare} disabled={diffMutation.isPending}>
                        <Diff className="mr-2 h-4 w-4" />
                        {diffMutation.isPending ? '对比中...' : '对比'}
                      </Button>
                    </div>
                  </div>
                  <Textarea className="min-h-[260px] font-mono" value={diffText} readOnly />
                </CardContent>
              </Card>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
