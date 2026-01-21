import { Sidebar } from '@/components/layout/sidebar';
import Providers from '@/components/providers';
import '@/app/globals.css';

export const metadata = {
  title: 'Lumoscribe 管理后台',
  description: 'AI 文档生成平台管理后台',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="zh-CN">
      <body className="h-screen flex overflow-hidden bg-gray-100">
        <Providers>
          <Sidebar />
          <main className="flex-1 overflow-auto">
            <div className="container mx-auto p-6">
              {children}
            </div>
          </main>
        </Providers>
      </body>
    </html>
  );
}
