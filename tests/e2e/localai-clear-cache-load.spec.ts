/**
 * E2E test: Clear IndexedDB cache, then Load Models and check for LOADING state
 * Run with: npx playwright test tests/e2e/localai-clear-cache-load.spec.ts --config=playwright.network.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-clear-cache-screenshots';

test.describe('LocalAI - Clear Cache Then Load', () => {
  test('clear IndexedDB, load models, check LOADING state and HuggingFace requests', async ({
    page,
  }) => {
    const consoleMessages: Array<{ type: string; text: string }> = [];
    const allRequests: Array<{ url: string; status: number }> = [];

    page.on('console', (msg) => {
      consoleMessages.push({ type: msg.type(), text: msg.text() });
    });

    page.on('response', (response) => {
      allRequests.push({ url: response.url(), status: response.status() });
    });

    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // Clear IndexedDB - delete common cache database names
    await page.evaluate(() => {
      const names = ['edgeflow-cache', 'model-cache', 'huggingface-cache', 'cache', 'opfs', 'ort-cache', 'default'];
      for (const name of names) {
        try {
          indexedDB.deleteDatabase(name);
        } catch (_) {}
      }
    });
    await page.waitForTimeout(1000);

    // Clear request/console capture for fresh Load Models run
    allRequests.length = 0;
    consoleMessages.length = 0;

    // Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // After 5 seconds - screenshot and report dot status
    await page.waitForTimeout(5_000);
    const statusAt5s = await page.evaluate(() => {
      const results: Record<string, string> = {};
      const modelNames = ['all-MiniLM-L6-v2', 'distilbart-mnli', 'distilbert-squad'];
      const allText = document.body.innerText;
      for (const name of modelNames) {
        const nameIdx = allText.indexOf(name);
        if (nameIdx >= 0) {
          const snippet = allText.slice(nameIdx, nameIdx + 150);
          if (snippet.includes('READY')) results[name] = 'READY';
          else if (snippet.includes('LOADING')) results[name] = 'LOADING';
          else if (snippet.includes('ERROR')) results[name] = 'ERROR';
          else if (snippet.includes('IDLE')) results[name] = 'IDLE';
          else results[name] = 'UNKNOWN';
        } else results[name] = 'NOT_FOUND';
      }
      return results;
    });
    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-at-5s.png`, fullPage: true });

    console.log('\n=== DOT STATUS AT 5 SECONDS ===');
    console.log('all-MiniLM-L6-v2:', statusAt5s['all-MiniLM-L6-v2'], '(grey=IDLE, orange=LOADING, green=READY, red=ERROR)');
    console.log('distilbart-mnli:', statusAt5s['distilbart-mnli']);
    console.log('distilbert-squad:', statusAt5s['distilbert-squad']);

    // After 30 seconds total - another screenshot
    await page.waitForTimeout(25_000);
    const statusAt30s = await page.evaluate(() => {
      const results: Record<string, string> = {};
      const modelNames = ['all-MiniLM-L6-v2', 'distilbart-mnli', 'distilbert-squad'];
      const allText = document.body.innerText;
      for (const name of modelNames) {
        const nameIdx = allText.indexOf(name);
        if (nameIdx >= 0) {
          const snippet = allText.slice(nameIdx, nameIdx + 150);
          if (snippet.includes('READY')) results[name] = 'READY';
          else if (snippet.includes('LOADING')) results[name] = 'LOADING';
          else if (snippet.includes('ERROR')) results[name] = 'ERROR';
          else if (snippet.includes('IDLE')) results[name] = 'IDLE';
          else results[name] = 'UNKNOWN';
        } else results[name] = 'NOT_FOUND';
      }
      return results;
    });
    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-at-30s.png`, fullPage: true });

    console.log('\n=== DOT STATUS AT 30 SECONDS ===');
    console.log('all-MiniLM-L6-v2:', statusAt30s['all-MiniLM-L6-v2']);
    console.log('distilbart-mnli:', statusAt30s['distilbart-mnli']);
    console.log('distilbert-squad:', statusAt30s['distilbert-squad']);

    // Check for huggingface.co requests
    const hfRequests = allRequests.filter((r) =>
      r.url.toLowerCase().includes('huggingface.co')
    );

    console.log('\n=== REQUESTS TO HUGGINGFACE.CO ===');
    if (hfRequests.length === 0) {
      console.log('NONE - No requests were made to huggingface.co');
    } else {
      hfRequests.forEach((r, i) => console.log(`${i + 1}. [${r.status}] ${r.url}`));
    }

    // Check console for download/loading messages
    const downloadMessages = consoleMessages.filter(
      (m) =>
        m.text.toLowerCase().includes('download') ||
        m.text.toLowerCase().includes('loading') ||
        m.text.toLowerCase().includes('fetch') ||
        m.text.toLowerCase().includes('huggingface')
    );

    console.log('\n=== CONSOLE: download/loading/fetch/huggingface messages ===');
    if (downloadMessages.length === 0) {
      console.log('None');
    } else {
      downloadMessages.forEach((m, i) => console.log(`${i + 1}. [${m.type}] ${m.text}`));
    }

    // Console errors
    const errors = consoleMessages.filter((m) => m.type === 'error');
    console.log('\n=== CONSOLE ERRORS ===');
    errors.forEach((e, i) => console.log(`${i + 1}. ${e.text}`));

    console.log('\n=== KEY QUESTION ===');
    const anyLoading = Object.values(statusAt5s).includes('LOADING') || Object.values(statusAt30s).includes('LOADING');
    console.log('Do models enter LOADING state (orange/amber pulsing)?', anyLoading ? 'YES' : 'NO');
    console.log('Are there requests to huggingface.co?', hfRequests.length > 0 ? 'YES' : 'NO');
  });
});
