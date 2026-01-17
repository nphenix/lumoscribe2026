'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  LayoutDashboard, 
  Files, 
  FileText, 
  Settings, 
  Cpu, 
  Activity,
  Database,
  Settings2
} from 'lucide-react';
import { cn } from '@/lib/utils';

const navigation = [
  { name: '仪表盘', href: '/', icon: LayoutDashboard },
  { name: '源文件', href: '/documents/sources', icon: Files },
  { name: '模板管理', href: '/documents/templates', icon: FileText },
  { name: '生成结果', href: '/documents/targets', icon: FileText },
  { name: '中间产物', href: '/documents/intermediates', icon: Database },
  { name: '模型配置', href: '/llm/providers', icon: Cpu },
  { name: 'LLM 调用点', href: '/llm/call-sites', icon: Settings2 },
  { name: '提示词', href: '/prompts', icon: Settings },
  { name: '任务观测', href: '/observation/jobs', icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="flex h-full w-64 flex-col bg-gray-900 text-white">
      <div className="flex h-16 items-center px-6 font-bold text-xl border-b border-gray-800">
        Lumoscribe
      </div>
      <nav className="flex-1 space-y-1 px-2 py-4">
        {navigation.map((item) => {
          const isActive = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'group flex items-center px-4 py-2 text-sm font-medium rounded-md transition-colors',
                isActive
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-300 hover:bg-gray-800 hover:text-white'
              )}
            >
              <item.icon className="mr-3 h-5 w-5 flex-shrink-0" />
              {item.name}
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
