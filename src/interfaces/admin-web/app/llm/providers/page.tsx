'use client';

import { useState } from 'react';
import {
  useLLMProviders,
  useCreateLLMProvider,
  useDeleteLLMProvider,
  useUpdateLLMProvider,
  LLMProviderCreate,
  LLMProviderUpdate,
  LLMProvider,
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
import { Plus, Trash2, Server, Pencil } from 'lucide-react';
import { toast } from 'sonner';

function makeProviderKey(name: string, providerType: string): string {
  const raw = (name || '').trim().toLowerCase();
  const slug = raw
    .replace(/[^a-z0-9]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_+|_+$/g, '');
  if (slug) return slug.slice(0, 64);
  // fallback：用类型 + 随机短 id（避免中文名称生成空 key）
  const base = (providerType || 'provider').toLowerCase().replace(/[^a-z0-9]+/g, '_') || 'provider';
  const short = (globalThis.crypto?.randomUUID?.() || `${Date.now()}`)
    .replace(/[^a-z0-9]/gi, '')
    .slice(0, 8)
    .toLowerCase();
  return `${base}_${short}`.slice(0, 64);
}

export default function ProvidersPage() {
  const { data: providers, isLoading } = useLLMProviders();
  const createMutation = useCreateLLMProvider();
  const updateMutation = useUpdateLLMProvider();
  const deleteMutation = useDeleteLLMProvider();
  const [isOpen, setIsOpen] = useState(false);
  const [editing, setEditing] = useState<LLMProvider | null>(null);
  const [providerKeyTouched, setProviderKeyTouched] = useState(false);
  const [apiKeyTouched, setApiKeyTouched] = useState(false);
  const [formData, setFormData] = useState<LLMProviderCreate>({
    key: '',
    name: '',
    base_url: '',
    api_key: '',
    api_key_env: '',
    provider_type: 'openai_compatible',
    enabled: true,
  });
  const [providerMaxConcurrency, setProviderMaxConcurrency] = useState<string>('');
  const [openaiConfig, setOpenaiConfig] = useState<{
    model: string;
    temperature: string;
    max_tokens: string;
    timeout_seconds: string;
    stream: boolean;
    thinking_enabled: boolean;
  }>({
    model: '',
    temperature: '0.2',
    max_tokens: '2048',
    timeout_seconds: '60',
    stream: false,
    thinking_enabled: false,
  });
  const [ollamaConfig, setOllamaConfig] = useState<{
    ollama_model: string;
    max_tokens: string;
  }>({
    ollama_model: 'llama3.1',
    max_tokens: '2048',
  });
  const [flagEmbeddingConfig, setFlagEmbeddingConfig] = useState<{
    device: string;
    use_fp16: boolean;
    trust_remote_code: boolean;
    batch_size: string;
    max_tokens: string;
    embedding_model_path: string;
    embedding_dimension: string;
    rerank_model_path: string;
    rerank_top_k: string;
  }>({
    device: 'cpu',
    use_fp16: true,
    trust_remote_code: true,
    batch_size: '32',
    max_tokens: '2048',
    embedding_model_path: '',
    embedding_dimension: '1024',
    rerank_model_path: '',
    rerank_top_k: '10',
  });

  const showBaseUrl =
    formData.provider_type === 'openai_compatible' ||
    formData.provider_type === 'ollama' ||
    formData.provider_type === 'huggingface' ||
    formData.provider_type === 'mineru';

  const handleSubmit = async () => {
    try {
      let config: Record<string, any> | undefined = undefined;
      const maxConcurrency =
        providerMaxConcurrency.trim().length > 0
          ? Number.parseInt(providerMaxConcurrency.trim(), 10)
          : undefined;
      if (formData.provider_type === 'openai_compatible') {
        const next: Record<string, any> = {
          model: openaiConfig.model?.trim() || undefined,
          temperature:
            openaiConfig.temperature?.trim().length > 0
              ? Number.parseFloat(openaiConfig.temperature)
              : undefined,
          max_tokens:
            openaiConfig.max_tokens?.trim().length > 0
              ? Number.parseInt(openaiConfig.max_tokens, 10)
              : undefined,
          timeout_seconds:
            openaiConfig.timeout_seconds?.trim().length > 0
              ? Number.parseInt(openaiConfig.timeout_seconds, 10)
              : undefined,
          stream: openaiConfig.stream,
          // 中台开关：仅对 MiniMax-M2.1 生效（后端会做模型判断）
          thinking_enabled: openaiConfig.thinking_enabled,
        };
        config = Object.fromEntries(Object.entries(next).filter(([, v]) => v !== undefined));
      } else if (formData.provider_type === 'ollama') {
        const next: Record<string, any> = {
          ollama_model: ollamaConfig.ollama_model?.trim() || undefined,
          max_tokens:
            ollamaConfig.max_tokens?.trim().length > 0
              ? Number.parseInt(ollamaConfig.max_tokens, 10)
              : undefined,
        };
        config = Object.fromEntries(Object.entries(next).filter(([, v]) => v !== undefined));
      } else if (formData.provider_type === 'flagembedding') {
        const next: Record<string, any> = {
          // 默认本地模式，不使用远程服务
          remote: false,
          device: flagEmbeddingConfig.device?.trim() || undefined,
          use_fp16: flagEmbeddingConfig.use_fp16,
          trust_remote_code: flagEmbeddingConfig.trust_remote_code,
          batch_size:
            flagEmbeddingConfig.batch_size?.trim().length > 0
              ? Number.parseInt(flagEmbeddingConfig.batch_size, 10)
              : undefined,
          max_tokens:
            flagEmbeddingConfig.max_tokens?.trim().length > 0
              ? Number.parseInt(flagEmbeddingConfig.max_tokens, 10)
              : undefined,
          embedding_model_path: flagEmbeddingConfig.embedding_model_path?.trim() || undefined,
          embedding_dimension:
            flagEmbeddingConfig.embedding_dimension?.trim().length > 0
              ? Number.parseInt(flagEmbeddingConfig.embedding_dimension, 10)
              : undefined,
          rerank_model_path: flagEmbeddingConfig.rerank_model_path?.trim() || undefined,
          rerank_top_k:
            flagEmbeddingConfig.rerank_top_k?.trim().length > 0
              ? Number.parseInt(flagEmbeddingConfig.rerank_top_k, 10)
              : undefined,
        };
        config = Object.fromEntries(Object.entries(next).filter(([, v]) => v !== undefined));
      }

      if (editing) {
        const patch: LLMProviderUpdate = {
          key: formData.key || undefined,
          name: formData.name,
          provider_type: formData.provider_type,
          base_url: formData.base_url || null,
          api_key_env: formData.api_key_env || null,
          enabled: formData.enabled ?? true,
          config: config || null,
          max_concurrency: maxConcurrency,
        };
        if (apiKeyTouched && (formData.api_key || '').trim().length > 0) {
          patch.api_key = formData.api_key;
        }
        await updateMutation.mutateAsync({ id: editing.id, patch });
        toast.success('供应商已更新');
      } else {
        const payload: any = {
          ...formData,
          // key 为空时让后端自动生成
          key: (formData.key || '').trim().length > 0 ? formData.key : undefined,
          config,
          max_concurrency: maxConcurrency,
        };
        await createMutation.mutateAsync(payload);
        toast.success('供应商创建成功');
      }
      setIsOpen(false);
      setEditing(null);
      setProviderKeyTouched(false);
      setApiKeyTouched(false);
      setFormData({
        key: '',
        name: '',
        base_url: '',
        api_key: '',
        api_key_env: '',
        provider_type: 'openai_compatible',
        enabled: true,
      });
      setProviderMaxConcurrency('');
      setOpenaiConfig({
        model: '',
        temperature: '0.2',
        max_tokens: '2048',
        timeout_seconds: '60',
        stream: false,
        thinking_enabled: false,
      });
      setOllamaConfig({
        ollama_model: 'llama3.1',
        max_tokens: '2048',
      });
      setFlagEmbeddingConfig({
        device: 'cpu',
        use_fp16: true,
        trust_remote_code: true,
        batch_size: '32',
        max_tokens: '2048',
        embedding_model_path: '',
        embedding_dimension: '1024',
        rerank_model_path: '',
        rerank_top_k: '10',
      });
    } catch (error) {
      toast.error(editing ? '更新供应商失败' : '创建供应商失败');
    }
  };

  const openCreate = () => {
    setEditing(null);
    setProviderKeyTouched(false);
    setApiKeyTouched(false);
    setFormData({
      key: '',
      name: '',
      base_url: '',
      api_key: '',
      api_key_env: '',
      provider_type: 'openai_compatible',
      enabled: true,
    });
    setProviderMaxConcurrency('');
    setOpenaiConfig({
      model: '',
      temperature: '0.2',
      max_tokens: '2048',
      timeout_seconds: '60',
      stream: false,
      thinking_enabled: false,
    });
    setOllamaConfig({
      ollama_model: 'llama3.1',
      max_tokens: '2048',
    });
    setFlagEmbeddingConfig({
      device: 'cpu',
      use_fp16: true,
      trust_remote_code: true,
      batch_size: '32',
      max_tokens: '2048',
    });
    setIsOpen(true);
  };

  const openEdit = (p: LLMProvider) => {
    setEditing(p);
    setProviderKeyTouched(false);
    setApiKeyTouched(false);
    setFormData({
      key: p.key || '',
      name: p.name || '',
      base_url: p.base_url || '',
      api_key: '',
      api_key_env: p.api_key_env || '',
      provider_type: p.provider_type || 'openai_compatible',
      enabled: p.enabled ?? true,
    });
    setProviderMaxConcurrency(p.max_concurrency !== undefined && p.max_concurrency !== null ? String(p.max_concurrency) : '');
    const cfg = (p.config || {}) as any;
    setOpenaiConfig({
      model: cfg.model || '',
      temperature: typeof cfg.temperature === 'number' ? String(cfg.temperature) : '0.2',
      max_tokens: typeof cfg.max_tokens === 'number' ? String(cfg.max_tokens) : '2048',
      timeout_seconds: typeof cfg.timeout_seconds === 'number' ? String(cfg.timeout_seconds) : '60',
      stream: Boolean(cfg.stream),
      thinking_enabled: cfg.thinking_enabled !== undefined ? Boolean(cfg.thinking_enabled) : false,
    });
    setOllamaConfig({
      ollama_model: cfg.ollama_model || 'llama3.1',
      max_tokens: typeof cfg.max_tokens === 'number' ? String(cfg.max_tokens) : '2048',
    });
    setFlagEmbeddingConfig({
      device: cfg.device || 'cpu',
      use_fp16: cfg.use_fp16 !== undefined ? Boolean(cfg.use_fp16) : true,
      trust_remote_code:
        cfg.trust_remote_code !== undefined ? Boolean(cfg.trust_remote_code) : true,
      batch_size: cfg.batch_size !== undefined ? String(cfg.batch_size) : '32',
      max_tokens: typeof cfg.max_tokens === 'number' ? String(cfg.max_tokens) : '2048',
      embedding_model_path: cfg.embedding_model_path || '',
      embedding_dimension:
        typeof cfg.embedding_dimension === 'number'
          ? String(cfg.embedding_dimension)
          : cfg.embedding_dimension || '1024',
      rerank_model_path: cfg.rerank_model_path || '',
      rerank_top_k:
        typeof cfg.rerank_top_k === 'number'
          ? String(cfg.rerank_top_k)
          : cfg.rerank_top_k || '10',
    });
    setIsOpen(true);
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
            <Button onClick={openCreate}>
              <Plus className="mr-2 h-4 w-4" />
              添加供应商
            </Button>
          </DialogTrigger>
          <DialogContent
            className="sm:max-w-[1000px]"
            // 防止切出窗口/聚焦到地址栏/复制粘贴时触发“focus outside”从而自动关闭弹窗
            onInteractOutside={(e) => e.preventDefault()}
          >
            <DialogHeader>
              <DialogTitle>{editing ? '编辑模型供应商' : '添加模型供应商'}</DialogTitle>
              <DialogDescription>
                配置一个新的 LLM 服务端点。
              </DialogDescription>
            </DialogHeader>
            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="key">唯一标识符（Provider Key）</Label>
                <Input
                  id="key"
                  placeholder="可留空自动生成；建议小写，例如 openai / ollama / mineru"
                  value={formData.key}
                  onChange={(e) => {
                    setProviderKeyTouched(true);
                    setFormData({ ...formData, key: e.target.value });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="name">名称</Label>
                <Input
                  id="name"
                  value={formData.name}
                  onChange={(e) => {
                    const nextName = e.target.value;
                    setFormData((prev) => {
                      const next = { ...prev, name: nextName };
                      if (!providerKeyTouched && (!prev.key || prev.key.trim().length === 0)) {
                        next.key = makeProviderKey(nextName, prev.provider_type);
                      }
                      return next;
                    });
                  }}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="type">类型</Label>
                <Select
                  value={formData.provider_type}
                  onValueChange={(value) => {
                    const next: any = { ...formData, provider_type: value };
                    // 切换类型时，清空不适用字段，减少误配置
                    if (value === 'flagembedding' || value === 'llamacpp') {
                      next.base_url = '';
                    }
                    if (value === 'flagembedding') {
                      next.api_key = '';
                      next.api_key_env = '';
                    }
                    setFormData(next);
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="选择类型" />
                  </SelectTrigger>
                  <SelectContent position="popper" align="start" sideOffset={6}>
                    <SelectItem value="openai_compatible">OpenAI 兼容</SelectItem>
                    <SelectItem value="ollama">Ollama</SelectItem>
                    <SelectItem value="flagembedding">FlagEmbedding</SelectItem>
                    <SelectItem value="huggingface">HuggingFace</SelectItem>
                    <SelectItem value="llamacpp">LlamaCpp</SelectItem>
                    <SelectItem value="mineru">MinerU (OCR/解析)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {showBaseUrl && (
                <div className="grid gap-2">
                  <Label htmlFor="base_url">Base URL</Label>
                  <Input
                    id="base_url"
                    placeholder={
                      formData.provider_type === 'ollama'
                        ? '例如：http://localhost:11434'
                        : '例如：https://api.openai.com/v1'
                    }
                    value={formData.base_url}
                    onChange={(e) => setFormData({ ...formData, base_url: e.target.value })}
                  />
                </div>
              )}
              <div className="grid gap-2">
                <Label htmlFor="api_key">API Key / Token（明文）</Label>
                <Input
                  id="api_key"
                  type="password"
                  placeholder={editing ? '留空表示不修改' : '可选：直接填写明文 Key/Token'}
                  value={formData.api_key || ''}
                  onChange={(e) => {
                    setApiKeyTouched(true);
                    setFormData({ ...formData, api_key: e.target.value });
                  }}
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
              <div className="grid gap-2">
                <Label htmlFor="max_concurrency">并发上限</Label>
                <Input
                  id="max_concurrency"
                  type="number"
                  min="1"
                  step="1"
                  placeholder="可选：留空表示使用默认/不限制"
                  value={providerMaxConcurrency}
                  onChange={(e) => setProviderMaxConcurrency(e.target.value)}
                />
              </div>
              <div className="flex items-center gap-2">
                <Switch
                  id="enabled"
                  checked={formData.enabled ?? true}
                  onCheckedChange={(checked) => setFormData({ ...formData, enabled: checked })}
                />
                <Label htmlFor="enabled">启用</Label>
              </div>

              {formData.provider_type === 'openai_compatible' && (
                <>
                  <div className="grid gap-2">
                    <Label htmlFor="llm_model">LLM_MODEL</Label>
                    <Input
                      id="llm_model"
                      placeholder="可选：例如 gpt-4o-mini"
                      value={openaiConfig.model}
                      onChange={(e) =>
                        setOpenaiConfig((prev) => ({ ...prev, model: e.target.value }))
                      }
                    />
                  </div>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="grid gap-2">
                      <Label htmlFor="llm_temperature">LLM_TEMPERATURE</Label>
                      <Input
                        id="llm_temperature"
                        type="number"
                        step="0.1"
                        value={openaiConfig.temperature}
                        onChange={(e) =>
                          setOpenaiConfig((prev) => ({ ...prev, temperature: e.target.value }))
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="llm_max_tokens">LLM_MAX_TOKENS</Label>
                      <Input
                        id="llm_max_tokens"
                        type="number"
                        value={openaiConfig.max_tokens}
                        onChange={(e) =>
                          setOpenaiConfig((prev) => ({ ...prev, max_tokens: e.target.value }))
                        }
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="grid gap-2">
                      <Label htmlFor="llm_timeout_seconds">LLM_TIMEOUT_SECONDS</Label>
                      <Input
                        id="llm_timeout_seconds"
                        type="number"
                        value={openaiConfig.timeout_seconds}
                        onChange={(e) =>
                          setOpenaiConfig((prev) => ({
                            ...prev,
                            timeout_seconds: e.target.value,
                          }))
                        }
                      />
                    </div>
                    <div className="flex items-center gap-2 pt-7">
                      <Switch
                        id="llm_stream"
                        checked={openaiConfig.stream}
                        onCheckedChange={(checked) =>
                          setOpenaiConfig((prev) => ({ ...prev, stream: checked }))
                        }
                      />
                      <Label htmlFor="llm_stream">LLM_STREAM</Label>
                    </div>
                  </div>
                  <div className="grid gap-1">
                    <div className="flex items-center gap-2">
                      <Switch
                        id="thinking_enabled"
                        checked={openaiConfig.thinking_enabled}
                        onCheckedChange={(checked) =>
                          setOpenaiConfig((prev) => ({ ...prev, thinking_enabled: checked }))
                        }
                      />
                      <Label htmlFor="thinking_enabled">思考模式（仅 MiniMax-M2.1）</Label>
                    </div>
                    <div className="text-xs text-gray-500">
                      关闭时后端会对 MiniMax-M2.1 自动设置 reasoning_split=true，避免 &lt;think&gt; 输出混入正文。
                    </div>
                  </div>
                </>
              )}

              {formData.provider_type === 'ollama' && (
                <>
                  <div className="grid gap-2">
                    <Label htmlFor="ollama_model">OLLAMA_MODEL</Label>
                    <Input
                      id="ollama_model"
                      placeholder="可选：例如 llama3.1"
                      value={ollamaConfig.ollama_model}
                      onChange={(e) =>
                        setOllamaConfig((prev) => ({ ...prev, ollama_model: e.target.value }))
                      }
                    />
                  </div>
                  <div className="grid gap-2">
                    <Label htmlFor="ollama_max_tokens">LLM_MAX_TOKENS</Label>
                    <Input
                      id="ollama_max_tokens"
                      type="number"
                      placeholder="例如 2048"
                      value={ollamaConfig.max_tokens}
                      onChange={(e) =>
                        setOllamaConfig((prev) => ({ ...prev, max_tokens: e.target.value }))
                      }
                    />
                  </div>
                </>
              )}

              {formData.provider_type === 'flagembedding' && (
                <>
                  <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <div className="grid gap-2">
                      <Label htmlFor="fe_device">FLAGEMBEDDING_DEVICE</Label>
                      <Input
                        id="fe_device"
                        placeholder="例如 cpu / cuda / cuda:0"
                        value={flagEmbeddingConfig.device}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({ ...prev, device: e.target.value }))
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="fe_batch_size">FLAGEMBEDDING_BATCH_SIZE</Label>
                      <Input
                        id="fe_batch_size"
                        type="number"
                        value={flagEmbeddingConfig.batch_size}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({ ...prev, batch_size: e.target.value }))
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="fe_max_tokens">LLM_MAX_TOKENS</Label>
                      <Input
                        id="fe_max_tokens"
                        type="number"
                        placeholder="例如 2048"
                        value={flagEmbeddingConfig.max_tokens}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({ ...prev, max_tokens: e.target.value }))
                        }
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <div className="flex items-center gap-2 pt-7">
                      <Switch
                        id="fe_use_fp16"
                        checked={flagEmbeddingConfig.use_fp16}
                        onCheckedChange={(checked) =>
                          setFlagEmbeddingConfig((prev) => ({ ...prev, use_fp16: checked }))
                        }
                      />
                      <Label htmlFor="fe_use_fp16" className="whitespace-nowrap">FLAGEMBEDDING_USE_FP16</Label>
                    </div>
                    <div className="flex items-center gap-2 pt-7">
                      <Switch
                        id="fe_trust_remote_code"
                        checked={flagEmbeddingConfig.trust_remote_code}
                        onCheckedChange={(checked) =>
                          setFlagEmbeddingConfig((prev) => ({
                            ...prev,
                            trust_remote_code: checked,
                          }))
                        }
                      />
                      <Label htmlFor="fe_trust_remote_code" className="whitespace-nowrap">FLAGEMBEDDING_TRUST_REMOTE_CODE</Label>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <div className="grid gap-2">
                      <Label htmlFor="fe_embedding_model_path">EMBEDDING_MODEL_PATH</Label>
                      <Input
                        id="fe_embedding_model_path"
                        placeholder="例如 ./models/bge-m3"
                        value={flagEmbeddingConfig.embedding_model_path}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({
                            ...prev,
                            embedding_model_path: e.target.value,
                          }))
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="fe_embedding_dimension">EMBEDDING_DIMENSION</Label>
                      <Input
                        id="fe_embedding_dimension"
                        type="number"
                        placeholder="例如 1024"
                        value={flagEmbeddingConfig.embedding_dimension}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({
                            ...prev,
                            embedding_dimension: e.target.value,
                          }))
                        }
                      />
                    </div>
                    <div className="grid gap-2">
                      <Label htmlFor="fe_rerank_model_path">RERANK_MODEL_PATH</Label>
                      <Input
                        id="fe_rerank_model_path"
                        placeholder="例如 ./models/bge-reranker-v2-m3"
                        value={flagEmbeddingConfig.rerank_model_path}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({
                            ...prev,
                            rerank_model_path: e.target.value,
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
                    <div className="grid gap-2">
                      <Label htmlFor="fe_rerank_top_k">RERANK_TOP_K</Label>
                      <Input
                        id="fe_rerank_top_k"
                        type="number"
                        placeholder="例如 10"
                        value={flagEmbeddingConfig.rerank_top_k}
                        onChange={(e) =>
                          setFlagEmbeddingConfig((prev) => ({
                            ...prev,
                            rerank_top_k: e.target.value,
                          }))
                        }
                      />
                    </div>
                  </div>
                </>
              )}
            </div>
            <div className="flex justify-end">
              <Button onClick={handleSubmit} disabled={createMutation.isPending || updateMutation.isPending}>
                {(createMutation.isPending || updateMutation.isPending) ? '保存中...' : '保存'}
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
                <TableHead>标识符</TableHead>
                <TableHead>名称</TableHead>
                <TableHead>类型</TableHead>
                <TableHead>Base URL</TableHead>
                <TableHead>并发上限</TableHead>
                <TableHead>状态</TableHead>
                <TableHead className="text-right">操作</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {providers?.map((provider) => (
                <TableRow key={provider.id}>
                  <TableCell className="font-mono text-sm">{provider.key}</TableCell>
                  <TableCell className="font-medium flex items-center">
                    <Server className="mr-2 h-4 w-4 text-blue-500" />
                    {provider.name}
                  </TableCell>
                  <TableCell>{provider.provider_type}</TableCell>
                  <TableCell className="text-gray-500">{provider.base_url}</TableCell>
                  <TableCell className="text-gray-500">
                    {provider.max_concurrency !== undefined && provider.max_concurrency !== null
                      ? provider.max_concurrency
                      : '—'}
                  </TableCell>
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
                      className="text-gray-700 hover:bg-gray-100"
                      onClick={() => openEdit(provider)}
                      title="编辑"
                    >
                      <Pencil className="h-4 w-4" />
                    </Button>
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
                  <TableCell colSpan={7} className="text-center py-8 text-gray-500">
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
