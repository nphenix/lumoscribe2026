import fs from 'fs';
import path from 'path';
import VMindPkg from '@visactor/vmind';
const VMind = VMindPkg.default || VMindPkg;
const Model = VMindPkg.Model || (VMindPkg.default ? VMindPkg.default.Model : undefined) || {};

import dotenv from 'dotenv';

// 加载环境变量 (需要 .env 中包含 OPENAI_API_KEY 或 VITE_GPT_KEY)
dotenv.config();

const API_KEY = process.env.OPENAI_API_KEY || process.env.VITE_GPT_KEY || process.env.LUMO_LLM_OPENAI_API_KEY;
const API_URL = process.env.OPENAI_BASE_URL || process.env.VITE_GPT_URL || process.env.LUMO_LLM_OPENAI_BASE_URL || 'https://api.openai.com/v1/chat/completions';
const MODEL_NAME = process.env.LLM_MODEL || Model.GPT4o;

if (!API_KEY) {
    console.error("❌ 错误: 未找到 API Key。请在 .env 文件中配置 OPENAI_API_KEY, VITE_GPT_KEY 或 LUMO_LLM_OPENAI_API_KEY。");
    process.exit(1);
}

// 初始化 VMind
const vmind = new VMind({
    url: API_URL,
    model: MODEL_NAME,
    headers: { 'Authorization': `Bearer ${API_KEY}` }
});

// 数据文件路径
const BASE_DIR = 'F:/lumoscribe2026/data/intermediates/36c3ef64-70d4-4e26-b37f-952641049e3e/pic_to_json/chart_json';
const FILES = [
    '1a7c4451c1043b2ee6a50d2ddb4b0672dde603b1b5bb46c32b9da467909d8d73.json', // Sankey
    '4c72cbb9b4a0b0acb7d3e0475ede2cc89b4eba9383f4e3728e42b3ecd337dbda.json', // Stacked Area
    'ee68f2293a1739e7a49247ba7f52d366224ad64189a8fa126641a388e22d82f1.json'  // Table
];

async function processFile(filename) {
    const filePath = path.join(BASE_DIR, filename);
    console.log(`正在处理: ${filename}...`);
    
    const content = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    const userPrompt = content.description || "请根据数据生成合适的图表";
    let dataset = [];
    let fieldInfo = []; // VMind 可以自动推断，也可以手动传入

    // --- 数据适配层 ---
    if (content.chart_type === 'stacked_area') {
        // 转换堆叠图数据: {x, series} -> [{Time, Type, Value}, ...]
        content.chart_data.series.forEach(s => {
            s.values.forEach((v, i) => {
                dataset.push({
                    "时间": content.chart_data.x[i],
                    "类型": s.name,
                    "数值": v
                });
            });
        });
    } else if (content.chart_type === 'sankey') {
        // 转换桑基图数据: links -> [{source, target, value}, ...]
        dataset = content.chart_data.links.map(l => ({
            "来源": l.source,
            "去向": l.target,
            "流量": l.value
        }));
    } else if (content.chart_type === 'table') {
        // 表格数据通常已经是扁平的 rows 数组
        dataset = content.chart_data.rows;
    } else {
        console.warn(`未知图表类型: ${content.chart_type}，尝试直接使用数据。`);
        dataset = content.chart_data;
    }

    try {
        // 调用 VMind 生成 Spec
        // 注意：VMind 的 generateChart 会返回 { spec, time, ... }
        const { spec } = await vmind.generateChart(userPrompt, null, dataset);
        return { filename, spec, prompt: userPrompt, success: true };
    } catch (e) {
        console.error(`生成失败 (${filename}):`, e.message);
        return { filename, error: e.message, success: false };
    }
}

async function run() {
    const results = [];
    for (const file of FILES) {
        results.push(await processFile(file));
    }

    // 生成 HTML 预览文件
    const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>VMind 生成结果预览</title>
    <!-- 使用指定版本的 VChart CDN -->
    <script src="https://unpkg.com/@visactor/vchart@1.12.6/build/index.min.js"></script>
    <style>
        body { font-family: sans-serif; padding: 20px; background: #f0f2f5; }
        .chart-card { background: white; padding: 20px; margin-bottom: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .chart-meta { margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 10px; }
        .chart-container { width: 100%; height: 400px; }
        h3 { margin: 0 0 5px 0; color: #333; }
        p { margin: 0; color: #666; font-size: 0.9em; }
        .error-log { color: red; background: #fff0f0; padding: 10px; margin-bottom: 10px; border: 1px solid #ffcccc; display: none; }
    </style>
</head>
<body>
    <h1>VMind 智能生成结果预览</h1>
    <div id="global-error" class="error-log"></div>
    <script>
        window.onerror = function(msg, url, line, col, error) {
            const el = document.getElementById('global-error');
            el.style.display = 'block';
            el.innerHTML += \`<div>❌ [Global Error] \${msg} (Line \${line})</div>\`;
        };
        // 检查 VChart 是否加载
        window.onload = function() {
            if (typeof VChart === 'undefined') {
                const el = document.getElementById('global-error');
                el.style.display = 'block';
                el.innerHTML += "<div>❌ Critical Error: VChart library failed to load! Check your network or CDN URL.</div>";
            }
        }
    </script>
    ${results.map((r, i) => r.success ? `
    <div class="chart-card">
        <div class="chart-meta">
            <h3>文件: ${r.filename}</h3>
            <p><strong>Prompt:</strong> ${r.prompt}</p>
        </div>
        <div id="chart-${i}" class="chart-container"></div>
        <script>
            try {
                const spec${i} = ${JSON.stringify(r.spec)};
                // 确保容器存在
                if (document.getElementById('chart-${i}')) {
                    // 兼容 VChart 默认导出
                    const VChartClass = window.VChart ? (window.VChart.default || window.VChart) : null;
                    if (!VChartClass) throw new Error('VChart class not found');
                    
                    const vchart${i} = new VChartClass(spec${i}, { dom: 'chart-${i}' });
                    vchart${i}.renderAsync();
                }
            } catch (err) {
                document.getElementById('chart-${i}').innerHTML = '<div style="color:red">Render Error: ' + err.message + '</div>';
                console.error('Render Error for chart-${i}:', err);
            }
        </script>
    </div>
    ` : `
    <div class="chart-card" style="border-left: 4px solid red;">
        <h3>❌ ${r.filename} 生成失败</h3>
        <p>${r.error}</p>
    </div>
    `).join('')}
</body>
</html>`;

    const outputPath = path.join(process.cwd(), 'vmind_preview.html');
    fs.writeFileSync(outputPath, htmlContent);
    console.log(`\n✅ 完成! 请在浏览器中打开以下文件查看结果:\n${outputPath}`);
}

run();