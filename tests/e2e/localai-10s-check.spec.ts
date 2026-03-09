/**
 * E2E test: Check model dot status at 10s and 60s after Load Models
 * Run with: npx playwright test tests/e2e/localai-10s-check.spec.ts --config=playwright.localai.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-10s-screenshots';

test.describe('LocalAI - 10s Dot Status Check', () => {
  test('check model dots at 10s and 60s', async ({ page }) => {
    const consoleErrors: string[] = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
        console.log(`[CONSOLE ERROR] ${msg.text()}`);
      }
    });

    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // 1. Initial screenshot
    await page.screenshot({ path: `${SCREENSHOT_DIR}/00-initial.png`, fullPage: true });

    // 2. Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // 3. Wait exactly 10 seconds, screenshot, report dot status
    await page.waitForTimeout(10_000);
    const statusAt10s = await page.evaluate(() => {
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

    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-at-10s.png`, fullPage: true });

    console.log('\n=== DOT STATUS AT 10 SECONDS ===');
    console.log('all-MiniLM-L6-v2:', statusAt10s['all-MiniLM-L6-v2'], '(grey=IDLE, orange-amber pulsing=LOADING, green=READY, red=ERROR)');
    console.log('distilbart-mnli:', statusAt10s['distilbart-mnli']);
    console.log('distilbert-squad:', statusAt10s['distilbert-squad']);

    // 4. Wait until 60 seconds total (50 more seconds)
    await page.waitForTimeout(50_000);
    await page.screenshot({ path: `${SCREENSHOT_DIR}/02-at-60s.png`, fullPage: true });

    // 5. Console errors
    console.log('\n=== CONSOLE ERRORS ===');
    consoleErrors.forEach((e, i) => console.log(`${i + 1}. ${e}`));
  });
});
