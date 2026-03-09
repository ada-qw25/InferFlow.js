/**
 * E2E test: Check for LOADING state and capture all console messages
 * Run with: npx playwright test tests/e2e/localai-loading-check.spec.ts --config=playwright.network.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-loading-screenshots';

test.describe('LocalAI - Loading State Check', () => {
  test('check for LOADING state and console messages', async ({ page }) => {
    const consoleMessages: Array<{ type: string; text: string }> = [];

    page.on('console', (msg) => {
      const type = msg.type();
      const text = msg.text();
      consoleMessages.push({ type, text });
      if (type === 'error') console.log(`[ERROR] ${text}`);
      else if (text.includes('cache') || text.includes('Evicting') || text.includes('download') || text.includes('progress') || text.includes('loaded'))
        console.log(`[${type}] ${text}`);
    });

    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // 1. Initial screenshot
    await page.screenshot({ path: `${SCREENSHOT_DIR}/00-initial.png`, fullPage: true });

    // 2. Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // 3. Within 5 seconds - screenshot and report
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
    console.log('all-MiniLM-L6-v2:', statusAt5s['all-MiniLM-L6-v2']);
    console.log('distilbart-mnli:', statusAt5s['distilbart-mnli']);
    console.log('distilbert-squad:', statusAt5s['distilbert-squad']);

    // 4. Screenshots at 30s, 60s, 90s, 120s (wait from previous checkpoint)
    const checkpoints = [30, 60, 90, 120];
    let prevSec = 5;
    for (const sec of checkpoints) {
      await page.waitForTimeout((sec - prevSec) * 1000);
      prevSec = sec;
      const status = await page.evaluate(() => {
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
      console.log(`\n=== DOT STATUS AT ${sec}s ===`);
      console.log('all-MiniLM-L6-v2:', status['all-MiniLM-L6-v2']);
      console.log('distilbart-mnli:', status['distilbart-mnli']);
      console.log('distilbert-squad:', status['distilbert-squad']);
      await page.screenshot({ path: `${SCREENSHOT_DIR}/02-at-${sec}s.png`, fullPage: true });
    }

    // Report console messages of interest
    const errors = consoleMessages.filter((m) => m.type === 'error');
    const cacheEvictProgress = consoleMessages.filter(
      (m) =>
        m.text.toLowerCase().includes('cache') ||
        m.text.toLowerCase().includes('evicting') ||
        m.text.toLowerCase().includes('download') ||
        m.text.toLowerCase().includes('progress') ||
        m.text.toLowerCase().includes('loaded from')
    );

    console.log('\n=== CONSOLE ERRORS ===');
    errors.forEach((e, i) => console.log(`${i + 1}. ${e.text}`));
    console.log('\n=== CONSOLE: cache/Evicting/download/progress/loaded ===');
    cacheEvictProgress.forEach((m, i) => console.log(`${i + 1}. [${m.type}] ${m.text}`));
  });
});
