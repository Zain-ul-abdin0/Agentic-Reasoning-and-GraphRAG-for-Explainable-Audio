const { chromium } = require('playwright');
const path = require('path');
const os = require('os');

(async () => {
  const browser = await chromium.launch({headless: true, channel: 'chrome'});
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  for (const [name, width, height] of [['desktop',1440,1000],['mobile',390,844]]) {
    await page.setViewportSize({width,height});
    await page.goto(process.env.TEST_BASE_URL || 'http://127.0.0.1:8001/');
    await page.locator('[data-tab="evaluation"]').click();
    await page.waitForFunction(() => document.querySelector('#sharedRetrievalMetrics tbody')?.rows.length === 2);
    await page.locator('#runRetrievalBenchmark').click();
    await page.waitForFunction(() => !document.querySelector('#runRetrievalBenchmark').disabled);
    const rows = await page.locator('#sharedRetrievalQueries tbody tr').count();
    if (rows !== 30) throw new Error(`Expected all 30 paired rows, got ${rows}`);
    const dimensions = await page.evaluate(() => ({width: innerWidth, body: document.documentElement.scrollWidth}));
    if (dimensions.body > dimensions.width + 1) throw new Error(`Page overflow: ${JSON.stringify(dimensions)}`);
    await page.locator('#runRetrievalBenchmark').scrollIntoViewIfNeeded();
    await page.screenshot({path:path.join(os.tmpdir(), `thesis-retrieval-${name}.png`)});
    console.log(name, 'all 30 rows rendered, no page overflow');
  }
  await page.locator('#evalTopK').fill('2');
  await page.locator('#runRetrievalBenchmark').click();
  await page.waitForFunction(() => !document.querySelector('#runRetrievalBenchmark').disabled);
  if (!(await page.locator('#retrievalBenchmarkStatus').textContent()).includes('k=2')) throw new Error('Top k was not applied');
  await page.locator('#evalTopK').fill('3');
  await page.locator('#runRetrievalBenchmark').click();
  await page.waitForFunction(() => !document.querySelector('#runRetrievalBenchmark').disabled);
  if (errors.length) throw new Error(errors.join('\n'));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });
