'use client';

import { useMemo, useState } from 'react';
import {
  LLMCallSite,
  useLLMCallSites,
  useLLMModels,
  useUpdateLLMCallSite,
} from '@/hooks/use-llm';
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
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Textarea } from '@/components/ui/textarea';
import { Settings2 } from 'lucide-react';
import { toast } from 'sonner';

export default function CallSitesPage() {
  const { data: callSites, isLoading } = useLLMCallSites();
  const { data: models } = useLLMModels();
  const updateMutation = useUpdateLLMCallSite();

  const modelNameById = useMemo(() => {
    const map = new Map<string, string>();
    (models || []).forEach((m) => map.set(m.id, m.name));
    return map;
  }, [models]);

  const [selected, setSelected] = useState<LLMCallSite | null>(null);
  const [configText, setConfigText] = useState<string>('{}');
  const [patchModelId, setPatchModelId] = useState<string>('');
  const [patchPromptScope, setPatchPromptScope] = useState<string>('');
  const [patchEnabled, setPatchEnabled] = useState<boolean>(true);
  const [patchDescription, setPatchDescription] = useState<string>('');

  const openEditor = (cs: LLMCallSite) => {
    setSelected(cs);
    setConfigText(JSON.stringify(cs.config ?? {}, null, 2));
    setPatchModelId(cs.model_id || '');
    setPatchPromptScope(cs.prompt_scope || '');
    setPatchEnabled(cs.enabled);
    setPatchDescription(cs.description || '');
  };

  const closeEditor = () => {
    setSelected(null);
  };

  const handleSave = async () => {
    if (!selected) return;
    try {
      const config =
        configText && configText.trim().length > 0 ? JSON.parse(configText) : undefined;

      await updateMutation.mutateAsync({
        id: selected.id,
        patch: {
          model_id: patchModelId || null,
          config,
          prompt_scope: patchPromptScope || null,
          enabled: patchEnabled,
          description: patchDescription || null,
        },
      });
      toast.success('调用点配置已更新');
      closeEditor();
    } catch (e) {
      toast.error('保存失败：请检查 JSON 格式或后端错误');
    }
  };

  const candidateModels = useMemo(() => {
    if (!selected) return models || [];
    return (models || []).filter((m) => m.model_kind === selected.expected_model_kind);
  }, [models, selected]);

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">LLM 调用点</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>调用点配置</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Key</TableHead>
                <TableHead>说明</TableHead>
                <TableHead>期望类型</TableHead>
                <TableHead>绑定模型</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {callSites?.map((cs) => (
                <TableRow key={cs.id}>
                  <TableCell className="font-mono text-sm">{cs.key}</TableCell>
                  <TableCell className="text-gray-600">
                    {cs.description ? cs.description : '—'}
                  </TableCell>
                  <TableCell>{cs.expected_model_kind}</TableCell>
                  <TableCell className="text-gray-600">
                    {cs.model_id ? modelNameById.get(cs.model_id) || cs.model_id : '未绑定'}
                  </TableCell>
                  <TableCell>
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        cs.enabled
                          ? 'bg-green-100 text-green-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}
                    >
                      {cs.enabled ? '启用' : '禁用'}
                    </span>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => openEditor(cs)}
                      className="text-gray-700 hover:bg-gray-100"
                    >
                      <Settings2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}

              {callSites?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={6} className="text-center py-8 text-gray-500">
                    暂无调用点。请先运行 seed 或初始化脚本注册调用点。
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Dialog open={!!selected} onOpenChange={(open) => (!open ? closeEditor() : null)}>
        <DialogContent className="sm:max-w-[700px]">
          <DialogHeader>
            <DialogTitle>编辑调用点</DialogTitle>
            <DialogDescription>
              绑定模型、设置参数覆盖与提示词 scope（为空则默认使用 key）。
            </DialogDescription>
          </DialogHeader>

          {selected && (
            <div className="grid gap-4 py-2">
              <div className="grid gap-2">
                <Label>Key</Label>
                <Input value={selected.key} disabled />
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="grid gap-2">
                  <Label htmlFor="model_id">绑定模型</Label>
                  <Select value={patchModelId} onValueChange={setPatchModelId}>
                    <SelectTrigger id="model_id" className="w-full">
                      <SelectValue placeholder="选择模型" />
                    </SelectTrigger>
                    <SelectContent position="popper" align="start" sideOffset={6}>
                      {candidateModels.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="grid gap-2">
                  <Label htmlFor="prompt_scope">Prompt Scope</Label>
                  <Input
                    id="prompt_scope"
                    placeholder="为空则使用 key"
                    value={patchPromptScope}
                    onChange={(e) => setPatchPromptScope(e.target.value)}
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="config">参数覆盖 (JSON)</Label>
                <Textarea
                  id="config"
                  value={configText}
                  onChange={(e) => setConfigText(e.target.value)}
                  className="font-mono text-sm min-h-[180px]"
                />
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div className="grid gap-2">
                  <Label htmlFor="description">说明</Label>
                  <Input
                    id="description"
                    value={patchDescription}
                    onChange={(e) => setPatchDescription(e.target.value)}
                  />
                </div>
                <div className="flex items-center gap-2 pt-7">
                  <Switch checked={patchEnabled} onCheckedChange={setPatchEnabled} />
                  <Label>启用</Label>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2">
                <Button variant="outline" onClick={closeEditor}>
                  取消
                </Button>
                <Button onClick={handleSave} disabled={updateMutation.isPending}>
                  {updateMutation.isPending ? '保存中...' : '保存'}
                </Button>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}

