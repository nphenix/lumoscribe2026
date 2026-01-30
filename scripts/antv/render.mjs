import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = '';
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', (chunk) => {
      data += chunk;
    });
    process.stdin.on('end', () => resolve(data));
    process.stdin.on('error', reject);
  });
}

function fail(code, message, details) {
  const payload = { ok: false, code, message, details };
  process.stderr.write(JSON.stringify(payload, null, 2));
  process.stderr.write('\n');
  process.exit(1);
}

function parsePayload(raw) {
  try {
    return JSON.parse(raw);
  } catch (e) {
    fail('invalid_json', 'Invalid JSON payload', { error: String(e) });
  }
}

async function renderG2({ width, height, format, spec }) {
  if (!width || !height) fail('invalid_size', 'width/height required');
  const fmt = String(format || '').toLowerCase();
  if (!['png', 'svg', 'jpg', 'jpeg'].includes(fmt)) fail('invalid_format', 'format must be png|svg|jpg', { format });
  const playwright = await tryImportPlaywright();
  if (!playwright) fail('missing_dependency', 'playwright not installed');
  const g2Umd = resolveG2UmdPath();
  if (!g2Umd) fail('missing_dependency', '@antv/g2 UMD bundle not found in node_modules');

  // 使用 8x viewport 实现超高分辨率渲染 (DPI 192)
  const renderScale = 8;
  const browser = await launchChromium(playwright);
  const page = await browser.newPage({ 
    viewport: { 
      width: width * renderScale, 
      height: height * renderScale,
      deviceScaleFactor: renderScale 
    } 
  });
  try {
    await page.setContent(
      `<!doctype html><html><head><meta charset="utf-8"/></head><body style="margin:0"><div id="container"></div></body></html>`,
      { waitUntil: 'load' },
    );
    await page.addScriptTag({ path: g2Umd });
    await page.evaluate(
      async ({ spec, width, height, renderScale }) => {
        const G2 = window.G2;
        if (!G2 || !G2.Chart) throw new Error('G2.Chart not found on window.G2');
        const container = document.getElementById('container');
        // 使用缩放后的尺寸渲染
        const chartWidth = width * renderScale;
        const chartHeight = height * renderScale;
        container.style.width = `${chartWidth}px`;
        container.style.height = `${chartHeight}px`;
        const chart = new G2.Chart({ container, width: chartWidth, height: chartHeight });
        chart.options({ ...(spec || {}), width: chartWidth, height: chartHeight });
        await chart.render();
      },
      { spec: spec || {}, width, height, renderScale },
    );

    // 根据格式选择截图类型，jpg 格式质量更高
    const screenshotType = (fmt === 'jpg' || fmt === 'jpeg') ? 'jpeg' : 'png';
    const quality = (fmt === 'jpg' || fmt === 'jpeg') ? 95 : undefined;
    const image = await page.locator('#container').screenshot({ 
      type: screenshotType,
      quality: quality
    });
    
    if (fmt === 'svg') {
      const b64 = Buffer.from(image).toString('base64');
      const mimeType = (fmt === 'jpg' || fmt === 'jpeg') ? 'image/jpeg' : 'image/png';
      const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><image href="data:${mimeType};base64,${b64}" width="${width}" height="${height}"/></svg>`;
      process.stdout.write(svg);
    } else {
      process.stdout.write(image);
    }
  } finally {
    await page.close();
    await browser.close();
  }
}

async function tryImportPlaywright() {
  try {
    return await import('playwright');
  } catch (e) {
    return null;
  }
}

async function launchChromium(playwright) {
  try {
    return await playwright.chromium.launch();
  } catch (e) {
    try {
      return await playwright.chromium.launch({ channel: 'msedge' });
    } catch (e2) {
      try {
        return await playwright.chromium.launch({ channel: 'chrome' });
      } catch (e3) {
        fail('missing_dependency', 'chromium executable not available for playwright', {
          error: String(e),
          msedge: String(e2),
          chrome: String(e3),
        });
      }
    }
  }
}

function resolveRepoRoot() {
  const __filename = fileURLToPath(import.meta.url);
  const __dirname = path.dirname(__filename);
  return path.resolve(__dirname, '..', '..');
}

function resolveG2UmdPath() {
  const repoRoot = resolveRepoRoot();
  const cand = path.join(repoRoot, 'node_modules', '@antv', 'g2', 'dist', 'g2.min.js');
  return fs.existsSync(cand) ? cand : null;
}

function resolveS2UmdPath() {
  const repoRoot = resolveRepoRoot();
  const cand = path.join(repoRoot, 'node_modules', '@antv', 's2', 'dist', 's2.min.js');
  return fs.existsSync(cand) ? cand : null;
}

function resolveInfographicUmdPath() {
  const repoRoot = resolveRepoRoot();
  const cand = path.join(repoRoot, 'node_modules', '@antv', 'infographic', 'dist', 'index.umd.js');
  return fs.existsSync(cand) ? cand : null;
}

async function renderS2Table({ width, height, format, spec, theme }) {
  const fmt = String(format || '').toLowerCase();
  const playwright = await tryImportPlaywright();
  if (!playwright) fail('missing_dependency', 'playwright not installed');
  const s2Umd = resolveS2UmdPath();
  if (!s2Umd) fail('missing_dependency', '@antv/s2 UMD bundle not found in node_modules');

  // 使用 8x viewport 实现超高分辨率渲染 (DPI 192)
  const renderScale = 8;
  const browser = await launchChromium(playwright);
  const page = await browser.newPage({ 
    viewport: { 
      width: width * renderScale, 
      height: height * renderScale,
      deviceScaleFactor: renderScale 
    } 
  });
  try {
    await page.setContent(
      `<!doctype html><html><head><meta charset="utf-8"/></head><body style="margin:0"><div id="container"></div></body></html>`,
      { waitUntil: 'load' },
    );
    await page.addScriptTag({ path: s2Umd });
    await page.evaluate(
      async ({ spec, theme, width, height, renderScale }) => {
        const { TableSheet } = window.S2 || {};
        if (!TableSheet) throw new Error('S2 TableSheet not found on window.S2');
        const container = document.getElementById('container');
        // 使用缩放后的尺寸渲染
        const chartWidth = width * renderScale;
        const chartHeight = height * renderScale;
        container.style.width = `${chartWidth}px`;
        container.style.height = `${chartHeight}px`;
        container.style.background = '#ffffff';
        const dataCfg = {
          fields: { columns: spec.columns || [] },
          data: spec.rows || [],
          meta: [],
        };
        const options = {
          width: chartWidth,
          height: chartHeight,
          showDefaultHeaderActionIcon: false,
        };
        const s2 = new TableSheet(container, dataCfg, options);
        if (String(theme || '').toLowerCase() === 'whitepaper-default' && typeof s2.setThemeCfg === 'function') {
          s2.setThemeCfg({
            palette: {
              basicColors: [
                '#111827',
                '#1f2937',
                '#f9fafb',
                '#ffffff',
                '#f3f4f6',
                '#e5e7eb',
                '#d1d5db',
                '#9ca3af',
                '#ffffff',
                '#e5e7eb',
                '#d1d5db',
                '#9ca3af',
                '#d1d5db',
                '#d1d5db',
                '#111827',
              ],
              semanticColors: {
                red: '#ef4444',
                green: '#22c55e',
                yellow: '#f59e0b',
              },
              others: {
                results: '#111827',
                highlight: '#2563eb',
              },
            },
            theme: {
              background: { color: '#ffffff', opacity: 1 },
            },
          });
        }
        await s2.render();
      },
      { spec: spec || {}, theme, width, height, renderScale },
    );

    // 根据格式选择截图类型，jpg 格式质量更高
    const screenshotType = (fmt === 'jpg' || fmt === 'jpeg') ? 'jpeg' : 'png';
    const quality = (fmt === 'jpg' || fmt === 'jpeg') ? 95 : undefined;
    const image = await page.locator('#container').screenshot({ 
      type: screenshotType,
      quality: quality
    });
    
    if (fmt === 'svg') {
      const b64 = Buffer.from(image).toString('base64');
      const mimeType = (fmt === 'jpg' || fmt === 'jpeg') ? 'image/jpeg' : 'image/png';
      const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><image href="data:${mimeType};base64,${b64}" width="${width}" height="${height}"/></svg>`;
      process.stdout.write(svg);
    } else {
      process.stdout.write(image);
    }
  } finally {
    await page.close();
    await browser.close();
  }
}

async function renderInfographic({ width, height, format, spec }) {
  const fmt = String(format || '').toLowerCase();
  const playwright = await tryImportPlaywright();
  if (!playwright) fail('missing_dependency', 'playwright not installed');
  const umd = resolveInfographicUmdPath();
  if (!umd) fail('missing_dependency', '@antv/infographic UMD bundle not found in node_modules');

  // 使用 8x viewport 实现超高分辨率渲染 (DPI 192)
  const renderScale = 8;
  const browser = await launchChromium(playwright);
  const page = await browser.newPage({ 
    viewport: { 
      width: width * renderScale, 
      height: height * renderScale,
      deviceScaleFactor: renderScale 
    } 
  });
  try {
    await page.setContent(
      `<!doctype html><html><head><meta charset="utf-8"/></head><body style="margin:0;background:#ffffff;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Helvetica Neue',Arial,sans-serif"><div id="container" style="width:${width * renderScale}px;height:${height * renderScale}px"></div></body></html>`,
      { waitUntil: 'load' },
    );
    await page.addScriptTag({ path: umd });
    await page.evaluate(
      async ({ spec, width, height, renderScale }) => {
        const { Infographic } = window.AntVInfographic || window.Infographic || {};
        if (!Infographic) throw new Error('Infographic not found on window');
        const ig = new Infographic({ container: '#container', width: width * renderScale, height: height * renderScale, editable: false });
        ig.render(String(spec.syntax || ''));
      },
      { spec: spec || {}, width, height, renderScale },
    );

    // 根据格式选择截图类型，jpg 格式质量更高
    const screenshotType = (fmt === 'jpg' || fmt === 'jpeg') ? 'jpeg' : 'png';
    const quality = (fmt === 'jpg' || fmt === 'jpeg') ? 95 : undefined;
    const image = await page.locator('#container').screenshot({ 
      type: screenshotType,
      quality: quality
    });
    
    if (fmt === 'svg') {
      const b64 = Buffer.from(image).toString('base64');
      const mimeType = (fmt === 'jpg' || fmt === 'jpeg') ? 'image/jpeg' : 'image/png';
      const svg = `<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}"><image href="data:${mimeType};base64,${b64}" width="${width}" height="${height}"/></svg>`;
      process.stdout.write(svg);
    } else {
      process.stdout.write(image);
    }
  } finally {
    await page.close();
    await browser.close();
  }
}

async function main() {
  const raw = await readStdin();
  const payload = parsePayload(raw);

  const engine = String(payload.engine || '').toLowerCase();
  const width = Number(payload.width || 800);
  const height = Number(payload.height || 460);
  const format = String(payload.format || 'svg').toLowerCase();
  const spec = payload.spec || {};
  const theme = payload.theme;

  if (engine === 'g2') {
    await renderG2({ width, height, format, spec });
    return;
  }
  if (engine === 's2') {
    await renderS2Table({ width, height, format, spec, theme });
    return;
  }
  if (engine === 'infographic') {
    await renderInfographic({ width, height, format, spec });
    return;
  }
  fail('invalid_engine', 'engine must be g2|s2|infographic', { engine });
}

main().catch((e) => fail('runtime_error', 'render failed', { error: String(e), stack: e?.stack }));
