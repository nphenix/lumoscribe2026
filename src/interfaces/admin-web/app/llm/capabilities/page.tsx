'use client';

import { useLLMCapabilities, useUpdateLLMCapability, useLLMProviders, LLMCapability } from '@/hooks/use-llm';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Wrench } from 'lucide-react';
import { toast } from 'sonner';

export default function CapabilitiesPage() {
  const { data: capabilities, isLoading } = useLLMCapabilities();
  const { data: providers } = useLLMProviders();
  const updateMutation = useUpdateLLMCapability();

  const handleUpsert = async (patch: LLMCapability) => {
    try {
      await updateMutation.mutateAsync({
        id: patch.id,
        capability: patch.capability,
        provider_id: patch.provider_id,
        priority: patch.priority,
        enabled: patch.enabled,
        description: patch.description ?? null,
      });
      toast.success('能力映射已更新');
    } catch (error) {
      toast.error('更新失败');
    }
  };

  if (isLoading) return <div>加载中...</div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold tracking-tight">能力映射</h1>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>模型能力配置</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>能力名称</TableHead>
                <TableHead>描述</TableHead>
                <TableHead>默认 Provider</TableHead>
                <TableHead>优先级</TableHead>
                <TableHead>启用</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {capabilities?.map((cap) => (
                <TableRow key={cap.id}>
                  <TableCell className="font-medium flex items-center">
                    <Wrench className="mr-2 h-4 w-4 text-orange-500" />
                    {cap.capability}
                  </TableCell>
                  <TableCell className="text-gray-500">{cap.description}</TableCell>
                  <TableCell className="w-[300px]">
                    <Select
                      value={cap.provider_id}
                      onValueChange={(value) => handleUpsert({ ...cap, provider_id: value })}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="选择 Provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {providers?.filter(p => p.enabled).map((p) => (
                          <SelectItem key={p.id} value={p.id}>
                            {p.name} ({p.provider_type})
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </TableCell>
                  <TableCell className="w-[120px]">
                    <Input
                      type="number"
                      value={cap.priority}
                      onChange={(e) =>
                        handleUpsert({ ...cap, priority: parseInt(e.target.value || '0', 10) })
                      }
                    />
                  </TableCell>
                  <TableCell className="w-[100px]">
                    <Switch
                      checked={cap.enabled}
                      onCheckedChange={(checked) => handleUpsert({ ...cap, enabled: checked })}
                    />
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
