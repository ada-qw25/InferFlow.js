/**
 * E2E test for LocalAI Knowledge Base app at http://localhost:5174/
 * Run with: npx playwright test tests/e2e/localai-knowledge-base.spec.ts
 * Ensure the app is running on port 5174 before running.
 */
import { test, expect } from '@playwright/test';

const BASE_URL = 'http://localhost:5174/';
const SCREENSHOT_DIR = 'test-results/localai-screenshots';

test.describe('LocalAI Knowledge Base App', () => {
  test.beforeEach(async ({ page }) => {
    // Capture console messages for error reporting
    page.on('console', (msg) => {
      const type = msg.type();
      const text = msg.text();
      if (type === 'error') {
        console.log(`[CONSOLE ERROR] ${text}`);
      }
    });
  });

  test('initial page load and UI elements', async ({ page }) => {
    // Navigate to the app
    const response = await page.goto(BASE_URL, { waitUntil: 'networkidle', timeout: 15_000 });
    expect(response?.status()).toBe(200);

    // Take initial screenshot
    await page.screenshot({ path: `${SCREENSHOT_DIR}/01-initial-page.png`, fullPage: true });

    // Check for sidebar
    const sidebar = page.locator('[class*="sidebar"], aside, nav, [data-testid*="sidebar"]').first();
    const sidebarVisible = await sidebar.isVisible().catch(() => false);

    // Check for model status panel
    const modelStatus = page.locator('text=/model|Model/i').first();
    const modelStatusVisible = await modelStatus.isVisible().catch(() => false);

    // Check for upload zone
    const uploadZone = page.locator('input[type="file"], [class*="upload"], [class*="dropzone"], [role="button"]:has-text("upload"), [role="button"]:has-text("Upload")').first();
    const uploadZoneVisible = await uploadZone.isVisible().catch(() => false);

    // Check for search bar
    const searchBar = page.locator('input[type="search"], input[placeholder*="search" i], input[placeholder*="Search" i], [class*="search"] input').first();
    const searchBarVisible = await searchBar.isVisible().catch(() => false);

    // Check for Load Models button
    const loadModelsBtn = page.locator('button:has-text("Load Models"), [role="button"]:has-text("Load Models")').first();
    const loadModelsVisible = await loadModelsBtn.isVisible().catch(() => false);

    // Log findings
    console.log('UI Elements found:');
    console.log('- Sidebar visible:', sidebarVisible);
    console.log('- Model status panel visible:', modelStatusVisible);
    console.log('- Upload zone visible:', uploadZoneVisible);
    console.log('- Search bar visible:', searchBarVisible);
    console.log('- Load Models button visible:', loadModelsVisible);

    // Click Load Models if present
    if (loadModelsVisible) {
      await loadModelsBtn.click();
      await page.waitForTimeout(2000); // Wait for models to load
      await page.screenshot({ path: `${SCREENSHOT_DIR}/02-after-load-models.png`, fullPage: true });
    } else {
      await page.screenshot({ path: `${SCREENSHOT_DIR}/02-no-load-models-btn.png`, fullPage: true });
    }
  });
});
