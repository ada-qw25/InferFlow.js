/**
 * E2E test: Capture ALL network requests when Load Models is clicked
 * Run with: npx playwright test tests/e2e/localai-network-full.spec.ts --config=playwright.network.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-network-full-screenshots';

interface NetworkEntry {
  url: string;
  status?: number;
  statusText?: string;
  size?: string;
  contentLength?: number;
  error?: string;
  failed?: boolean;
}

test.describe('LocalAI - Full Network Capture', () => {
  test('capture all network requests when Load Models clicked', async ({ page }) => {
    const allRequests: NetworkEntry[] = [];
    const failedRequests: NetworkEntry[] = [];

    // Capture all responses
    page.on('response', async (response) => {
      const url = response.url();
      const status = response.status();
      const headers = response.headers();
      const contentLength = headers['content-length'];
      let size: string | undefined;
      if (contentLength) {
        const bytes = parseInt(contentLength, 10);
        size = bytes >= 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} B`;
      } else {
        size = '(no content-length)';
      }
      allRequests.push({
        url,
        status,
        statusText: response.statusText(),
        size,
        contentLength: contentLength ? parseInt(contentLength, 10) : undefined,
      });
    });

    // Capture failed requests
    page.on('requestfailed', (request) => {
      const failure = request.failure();
      failedRequests.push({
        url: request.url(),
        error: failure?.errorText || 'Unknown',
        failed: true,
      });
    });

    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // Open DevTools, Network tab
    await page.keyboard.press('F12');
    await page.waitForTimeout(500);
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+E' : 'Control+Shift+E');
    await page.waitForTimeout(500);

    // Clear previous captures - we want only Load Models requests
    allRequests.length = 0;
    failedRequests.length = 0;

    // Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // Wait 15 seconds
    await page.waitForTimeout(15_000);

    // Screenshot Network tab
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/network-tab-all-requests.png`,
      fullPage: true,
    });

    // Filter for HuggingFace / model URLs
    const hfUrls = ['huggingface.co', 'hf.co', 'xethub.hf.co', 'cdn-lfs', 'huggingface'];
    const modelRequests = allRequests.filter((r) =>
      hfUrls.some((h) => r.url.toLowerCase().includes(h))
    );
    const otherRequests = allRequests.filter(
      (r) => !hfUrls.some((h) => r.url.toLowerCase().includes(h))
    );

    // Combine with failed
    const allFailed = failedRequests;

    console.log('\n=== REQUESTS TO HUGGINGFACE / MODEL URLs ===');
    if (modelRequests.length === 0) {
      console.log('NONE - No requests were made to HuggingFace or model download URLs');
    } else {
      modelRequests.forEach((r, i) => {
        console.log(`${i + 1}. URL: ${r.url}`);
        console.log(`   Status: ${r.status || 'N/A'} ${r.statusText || ''}`);
        console.log(`   Size: ${r.size || 'N/A'}`);
      });
    }

    console.log('\n=== FAILED REQUESTS (blocked/CORS/net::ERR_*) ===');
    if (allFailed.length === 0) {
      console.log('None');
    } else {
      allFailed.forEach((r, i) => {
        console.log(`${i + 1}. URL: ${r.url}`);
        console.log(`   Error: ${r.error}`);
      });
    }

    console.log('\n=== ALL REQUEST URLS (first 50) ===');
    [...allRequests, ...allFailed.map((f) => ({ ...f, status: 0 }))].slice(0, 50).forEach((r, i) => {
      const status = r.failed ? `FAILED: ${r.error}` : `${r.status}`;
      const size = r.size || r.error || '';
      console.log(`${i + 1}. [${status}] ${r.size || ''} ${r.url}`);
    });

    console.log(`\nTotal requests captured: ${allRequests.length}`);
    console.log(`Failed requests: ${allFailed.length}`);
    console.log(`HuggingFace/model requests: ${modelRequests.length}`);
  });
});
