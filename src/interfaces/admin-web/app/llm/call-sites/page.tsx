'use client';

import { useMemo, useState } from 'react';
import {
  LLMCallSite,
  useLLMCallSites,
  useLLMProviders,
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
import { Settings2 } from 'lucide-react';
import { toast } from 'sonner';

export default function CallSitesPage() {
  const { data: callSites, isLoading } = useLLMCallSites();
  const { data: providers } = useLLMProviders();
  const updateMutation = useUpdateLLMCallSite();

  const providerNameById = useMemo(() => {
    const map = new Map<string, string>();
    (providers || []).forEach((p) => map.set(p.id, p.name));
    return map;
  }, [providers]);

  const [selected, setSelected] = useState<LLMCallSite | null>(null);
  const [patchProviderId, setPatchProviderId] = useState<string>('');
  const [patchEnabled, setPatchEnabled] = useState<boolean>(true);
  const [patchDescription, setPatchDescription] = useState<string>('');
  const [patchMaxConcurrency, setPatchMaxConcurrency] = useState<string>('');

  const openEditor = (cs: LLMCallSite) => {
    setSelected(cs);
    setPatchProviderId(cs.provider_id || '');
    setPatchEnabled(cs.enabled);
    setPatchDescription(cs.description || '');
    setPatchMaxConcurrency(
      cs.max_concurrency !== undefined && cs.max_concurrency !== null ? String(cs.max_concurrency) : ''
    );
  };

  const closeEditor = () => {
    setSelected(null);
  };

  const handleSave = async () => {
    if (!selected) return;
    try {
      const maxConcurrency =
        patchMaxConcurrency.trim().length > 0
          ? Number.parseInt(patchMaxConcurrency.trim(), 10)
          : undefined;
      await updateMutation.mutateAsync({
        id: selected.id,
        patch: {
          provider_id: patchProviderId || null,
          enabled: patchEnabled,
          description: patchDescription || null,
          max_concurrency: maxConcurrency,
          // 注意：config_json 和 prompt_scope 字段已从界面移除
          // 如需配置可通过 API 直接调用，或后续添加高级选项界面
        },
      });
      toast.success('调用点配置已更新');
      closeEditor();
    } catch (e) {
      toast.error('保存失败：请检查后端错误');
    }
  };

  const candidateProviders = useMemo(() => {
    // 显示所有已启用的 Provider
    return (providers || []).filter((p) => p.enabled);
  }, [providers]);

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
                <TableHead>绑定 Provider</TableHead>
                <TableHead>并发上限</TableHead>
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
                    {cs.provider_id ? providerNameById.get(cs.provider_id) || cs.provider_id : '未绑定'}
                  </TableCell>
                  <TableCell className="text-gray-600">
                    {cs.max_concurrency !== undefined && cs.max_concurrency !== null
                      ? cs.max_concurrency
                      : '—'}
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
                  <TableCell colSpan={7} className="text-center py-8 text-gray-500">
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
              绑定 Provider 并配置调用点基本设置。
            </DialogDescription>
          </DialogHeader>

          {selected && (
            <div className="grid gap-4 py-2">
              <div className="grid gap-2">
                <Label>Key</Label>
                <Input value={selected.key} disabled />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="provider_id">绑定 Provider</Label>
                <Select value={patchProviderId} onValueChange={setPatchProviderId}>
                  <SelectTrigger id="provider_id" className="w-full">
                    <SelectValue placeholder="选择 Provider" />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" sideOffset={6}>
                    {candidateProviders.length > 0 ? (
                      candidateProviders.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {p.name} ({p.provider_type})
                        </SelectItem>
                      ))
                    ) : (
                      <div className="px-2 py-1.5 text-sm text-gray-500">
                        暂无可用 Provider。请先在"模型供应商"页面创建 Provider。
                      </div>
                    )}
                  </SelectContent>
                </Select>
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
                <div className="grid gap-2">
                  <Label htmlFor="max_concurrency">并发上限</Label>
                  <Input
                    id="max_concurrency"
                    type="number"
                    min="1"
                    step="1"
                    placeholder="可选：留空表示继承 Provider"
                    value={patchMaxConcurrency}
                    onChange={(e) => setPatchMaxConcurrency(e.target.value)}
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
