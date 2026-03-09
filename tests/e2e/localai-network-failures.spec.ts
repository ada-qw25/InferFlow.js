/**
 * E2E test: Capture failed network requests when Load Models is clicked
 * Run with: npx playwright test tests/e2e/localai-network-failures.spec.ts --config=playwright.localai.config.ts
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5173/';
const SCREENSHOT_DIR = 'test-results/localai-network-screenshots';

test.describe('LocalAI - Failed Network Requests', () => {
  test('capture failed requests when Load Models is clicked', async ({ page }) => {
    const failedRequests: Array<{ url: string; status: number }> = [];

    // Capture failed responses (4xx, 5xx)
    page.on('response', (response) => {
      const status = response.status();
      if (status >= 400) {
        const url = response.url();
        failedRequests.push({ url, status });
        console.log(`[FAILED] ${status} ${url}`);
      }
    });

    // Navigate to the app
    await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });

    // Open DevTools (F12) - Network tab
    await page.keyboard.press('F12');
    await page.waitForTimeout(500);

    // Switch to Network tab: Cmd+Shift+E (Mac) or Ctrl+Shift+E (Win/Linux)
    await page.keyboard.press(process.platform === 'darwin' ? 'Meta+Shift+E' : 'Control+Shift+E');
    await page.waitForTimeout(500);

    // Clear captured failures from initial page load - we want only Load Models failures
    failedRequests.length = 0;

    // Click Load Models
    const loadModelsBtn = page.locator('button:has-text("Load Models")').first();
    await expect(loadModelsBtn).toBeVisible({ timeout: 5000 });
    await loadModelsBtn.click();

    // Wait 15 seconds
    await page.waitForTimeout(15_000);

    // Screenshot - may capture DevTools if visible
    await page.screenshot({
      path: `${SCREENSHOT_DIR}/network-failed-requests.png`,
      fullPage: true,
    });

    // Log all failed request URLs
    console.log('\n=== FAILED REQUEST URLS (exact as captured) ===');
    const uniqueUrls = [...new Map(failedRequests.map((f) => [f.url, f])).values()];
    uniqueUrls.forEach(({ url, status }, i) => {
      console.log(`${i + 1}. [${status}] ${url}`);
    });
  });
});
