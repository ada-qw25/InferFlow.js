/**
 * E2E test for LocalAI Knowledge Base - model loading flow
 * Run with: npx playwright test tests/e2e/localai-load-models.spec.ts --config=playwright.localai.config.ts
 * Uses headed mode for WebGPU/WebNN support. Ensure app is running on port 5173.
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-load-screenshots';
const WAIT_SECONDS = 150;
const SCREENSHOT_INTERVAL_SEC = 30;

test.describe('LocalAI Knowledge Base - Model Loading', () => {
  test('load models and capture loading progress', async ({ page }) => {
    test.setTimeout(180_000); // 150s wait + buffer
    const consoleErrors: string[] = [];

    page.on('console', (msg) => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error') {
        consoleErrors.push(text);
        console.log(`[CONSOLE ERROR] ${text}`);
      } else if (type === 'warning' && (text.includes('WASM') || text.includes('404'))) {
        consoleErrors.push(`[WARNING] ${text}`);
        console.log(`[CONSOLE WARNING] ${text}`);
      }
    });

    // Navigate to the app
    const response = await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });
    expect(response?.status()).toBe(200);

    // 1. Initial screenshot
    await page.screenshot({ path: `${SCREENSHOT_DIR}/00-initial.png`, fullPage: true });

    // 2. Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // 3 & 4. Wait up to 150 seconds, screenshot every 30 seconds, report state at each checkpoint
    const startTime = Date.now();
    let screenshotCount = 1;
    while (Date.now() - startTime < WAIT_SECONDS * 1000) {
      await page.waitForTimeout(SCREENSHOT_INTERVAL_SEC * 1000);
      const elapsed = screenshotCount * SCREENSHOT_INTERVAL_SEC;
      const checkpointStatuses = await page.evaluate(() => {
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
      console.log(`\n--- Checkpoint at ${elapsed}s ---`);
      console.log(`all-MiniLM-L6-v2: ${checkpointStatuses['all-MiniLM-L6-v2']}`);
      console.log(`distilbart-mnli: ${checkpointStatuses['distilbart-mnli']}`);
      console.log(`distilbert-squad: ${checkpointStatuses['distilbert-squad']}`);
      await page.screenshot({
        path: `${SCREENSHOT_DIR}/${String(screenshotCount).padStart(2, '0')}-at-${elapsed}s.png`,
        fullPage: true,
      });
      screenshotCount++;
      if (screenshotCount * SCREENSHOT_INTERVAL_SEC >= WAIT_SECONDS) break;
    }

    // 5. Extract final model statuses from page text
    const finalStatuses = await page.evaluate(() => {
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
        } else {
          results[name] = 'NOT_FOUND';
        }
      }
      return results;
    });

    // Final screenshot
    await page.screenshot({ path: `${SCREENSHOT_DIR}/99-final.png`, fullPage: true });

    // Log results for report
    console.log('\n=== MODEL STATUS REPORT ===');
    console.log('all-MiniLM-L6-v2:', finalStatuses['all-MiniLM-L6-v2'] || 'UNKNOWN');
    console.log('distilbart-mnli:', finalStatuses['distilbart-mnli'] || 'UNKNOWN');
    console.log('distilbert-squad:', finalStatuses['distilbert-squad'] || 'UNKNOWN');
    console.log('\n=== CONSOLE ERRORS ===');
    consoleErrors.forEach((e, i) => console.log(`${i + 1}. ${e}`));
  });
});
