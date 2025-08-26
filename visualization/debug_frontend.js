#!/usr/bin/env node

/**
 * Script de débogage pour identifier les problèmes du frontend visualization
 */

const axios = require('axios').default;

const API_BASE = 'http://localhost:5009';

async function testEndpoint(endpoint, description) {
  try {
    console.log(`🔍 Testing ${description}...`);
    const response = await axios.get(`${API_BASE}${endpoint}`, {
      timeout: 5000,
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
      }
    });
    
    console.log(`✅ ${description} - Status: ${response.status}`);
    
    if (response.data) {
      const data = response.data;
      if (Array.isArray(data)) {
        console.log(`   📊 Array with ${data.length} items`);
      } else if (typeof data === 'object') {
        const keys = Object.keys(data);
        console.log(`   📋 Object with keys: ${keys.slice(0, 5).join(', ')}${keys.length > 5 ? '...' : ''}`);
        
        // Tests spécifiques pour chaque type de données
        if (data.data && data.data.timestamps) {
          console.log(`   📈 Market data: ${data.data.timestamps.length} timestamps`);
        }
        if (data.signals) {
          console.log(`   🚨 Signals: ${data.signals.buy?.length || 0} buy, ${data.signals.sell?.length || 0} sell`);
        }
        if (data.indicators) {
          const indicatorKeys = Object.keys(data.indicators);
          console.log(`   📊 Indicators: ${indicatorKeys.join(', ')}`);
        }
      }
    }
    
    return response.data;
  } catch (error) {
    console.log(`❌ ${description} - Error: ${error.message}`);
    if (error.response) {
      console.log(`   Status: ${error.response.status}`);
      console.log(`   Data: ${JSON.stringify(error.response.data).substring(0, 200)}...`);
    }
    return null;
  }
}

async function main() {
  console.log('🚀 Debugging ROOT Visualization Frontend API Connections\n');
  
  const endpoints = [
    ['/api/available-symbols', 'Available Symbols'],
    ['/api/configured-symbols', 'Configured Symbols'],
    ['/api/charts/market/BTCUSDC?interval=5m&limit=10', 'Market Data (BTCUSDC)'],
    ['/api/charts/signals/BTCUSDC', 'Trading Signals (BTCUSDC)'],
    ['/api/charts/indicators/BTCUSDC?indicators=rsi,macd,ema,vwap&interval=5m&limit=10', 'Technical Indicators'],
    ['/api/portfolio/summary', 'Portfolio Summary'],
    ['/api/portfolio/balances', 'Portfolio Balances'],
    ['/api/statistics/global', 'Global Statistics'],
    ['/api/statistics/strategies', 'Strategy Statistics'],
    ['/api/trade-cycles?limit=5', 'Trade Cycles']
  ];
  
  const results = [];
  
  for (const [endpoint, description] of endpoints) {
    const result = await testEndpoint(endpoint, description);
    results.push({ endpoint, description, success: !!result, data: result });
    console.log(''); // Empty line for readability
  }
  
  console.log('📋 Summary:');
  console.log('='.repeat(50));
  
  const successful = results.filter(r => r.success);
  const failed = results.filter(r => !r.success);
  
  console.log(`✅ Successful endpoints: ${successful.length}/${results.length}`);
  console.log(`❌ Failed endpoints: ${failed.length}/${results.length}`);
  
  if (failed.length > 0) {
    console.log('\n❌ Failed endpoints:');
    failed.forEach(f => console.log(`   - ${f.description}: ${f.endpoint}`));
  }
  
  if (successful.length > 0) {
    console.log('\n✅ Working endpoints:');
    successful.forEach(s => console.log(`   - ${s.description}: ${s.endpoint}`));
  }
  
  // Test spécifique pour l'endpoint getAllChartData
  console.log('\n🔍 Testing combined chart data (like frontend does)...');
  
  try {
    const [marketData, indicators, signals] = await Promise.all([
      testEndpoint('/api/charts/market/SOLUSDC?interval=1m&limit=100', 'Market Data (SOLUSDC)'),
      testEndpoint('/api/charts/indicators/SOLUSDC?indicators=rsi,macd,ema,vwap&interval=1m&limit=100', 'Indicators (SOLUSDC)'),
      testEndpoint('/api/charts/signals/SOLUSDC', 'Signals (SOLUSDC)')
    ]);
    
    if (marketData && indicators && signals) {
      console.log('✅ Combined data test successful - all required data available');
    } else {
      console.log('❌ Combined data test failed - missing required data');
    }
  } catch (error) {
    console.log(`❌ Combined data test error: ${error.message}`);
  }
  
  console.log('\n🎯 Recommendations:');
  if (failed.length === 0) {
    console.log('✅ All API endpoints are working correctly!');
    console.log('💡 If frontend still shows empty data, check:');
    console.log('   - Browser console errors');
    console.log('   - Network tab in developer tools');
    console.log('   - Frontend proxy configuration (vite.config.ts)');
    console.log('   - State management (Zustand store)');
  } else {
    console.log('🔧 Fix the following issues:');
    console.log('   1. Ensure all ROOT services are running');
    console.log('   2. Check database connectivity');
    console.log('   3. Verify service configurations');
    console.log('   4. Check logs: docker logs roottrading-visualization-1');
  }
}

main().catch(console.error);