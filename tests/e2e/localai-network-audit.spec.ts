/**
 * E2E test: Audit ALL network requests when Load Models is clicked
 * Captures requests to HuggingFace, xethub, model URLs
 * Run with: npx playwright test tests/e2e/localai-network-audit.spec.ts --config=playwright.network.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-network-audit';

interface RequestRecord {
  url: string;
  method: string;
  status?: number;
  statusText?: string;
  size?: number;
  headers?: Record<string, string>;
  error?: string;
}

test.describe('LocalAI - Network Audit', () => {
  test('capture all network requests when Load Models clicked', async ({ page }) => {
    const allRequests: RequestRecord[] = [];
    const requestMap = new Map<string, RequestRecord>();

    // Capture all requests
    page.on('request', (request) => {
      const url = request.url();
      if (!requestMap.has(url)) {
        requestMap.set(url, {
          url,
          method: request.method(),
        });
      }
    });

    // Capture all responses
    page.on('response', async (response) => {
      const url = response.url();
      const req = requestMap.get(url) || { url, method: 'GET' };
      req.status = response.status();
      req.statusText = response.statusText();
      const cl = response.headers()['content-length'];
      if (cl) req.size = parseInt(cl, 10);
      else {
        try {
          const body = await response.body();
          req.size = body?.length ?? 0;
        } catch {
          req.size = 0;
        }
      }
      requestMap.set(url, req);
    });

    // Capture request failures
    page.on('requestfailed', (request) => {
      const url = request.url();
      const req = requestMap.get(url) || { url, method: request.method() };
      req.error = request.failure()?.errorText ?? 'unknown';
      requestMap.set(url, req);
    });

    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // Open DevTools and Network tab BEFORE clicking
    await page.keyboard.press('F12');
    await page.waitForTimeout(500);
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+E' : 'Control+Shift+E');
    await page.waitForTimeout(500);

    // Clear request map for fresh capture
    requestMap.clear();

    // Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // Wait 15 seconds
    await page.waitForTimeout(15_000);

    // Collect all requests
    const requests = Array.from(requestMap.values());

    // Filter for HuggingFace, xethub, model-related
    const hfRequests = requests.filter(
      (r) =>
        r.url.includes('huggingface.co') ||
        r.url.includes('xethub.hf.co') ||
        r.url.includes('hf.co') ||
        r.url.includes('models') ||
        r.url.includes('.onnx') ||
        r.url.includes('.bin') ||
        r.url.includes('config.json') ||
        r.url.includes('tokenizer')
    );

    // Screenshot Network tab
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/network-all-requests.png`,
      fullPage: true,
    });

    // Report
    console.log('\n=== ALL REQUESTS (HuggingFace/xethub/model-related) ===');
    if (hfRequests.length === 0) {
      console.log('NONE - No requests to HuggingFace, xethub.hf.co, or model URLs were made.');
    } else {
      hfRequests.forEach((r, i) => {
        console.log(`\n${i + 1}. ${r.url}`);
        console.log(`   Method: ${r.method}`);
        console.log(`   Status: ${r.status ?? 'N/A'} ${r.statusText ?? ''}`);
        console.log(`   Size: ${r.size ?? 0} bytes`);
        if (r.error) console.log(`   Error: ${r.error}`);
      });
    }

    // Also list ALL requests for context
    console.log('\n=== ALL REQUEST URLS (first 50) ===');
    requests.slice(0, 50).forEach((r, i) => {
      const err = r.error ? ` [${r.error}]` : '';
      const status = r.status ? ` [${r.status}]` : '';
      const size = r.size !== undefined ? ` ${r.size}B` : '';
      console.log(`${i + 1}. ${r.url}${status}${size}${err}`);
    });

    // Failed requests
    const failed = requests.filter((r) => r.status && r.status >= 400);
    console.log('\n=== FAILED REQUESTS (4xx/5xx) ===');
    failed.forEach((r, i) => {
      console.log(`${i + 1}. [${r.status}] ${r.url} - ${r.error ?? ''}`);
    });
  });
});
