import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Layout from '@/components/Layout';
import Dashboard from '@/components/Dashboard';
import SignalsPage from '@/components/Signals/SignalsPage';
import CyclesPage from '@/components/Cycles/CyclesPage';
import StatisticsPage from '@/components/Statistics/StatisticsPage';
import ManualTradingPage from '@/components/ManualTrading/ManualTradingPage';
import PositionTrackerPage from '@/components/ManualTrading/PositionTrackerPage';

function App() {
  return (
    <div className="min-h-screen bg-dark-500">
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/signals" element={<SignalsPage />} />
            <Route path="/cycles" element={<CyclesPage />} />
            <Route path="/statistics" element={<StatisticsPage />} />
            <Route path="/manual-trading" element={<ManualTradingPage />} />
            <Route path="/position-tracker" element={<PositionTrackerPage />} />
          </Routes>
        </Layout>
      </Router>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#2d3748',
            color: '#ffffff',
            border: '1px solid #4a5568',
          },
          success: {
            iconTheme: {
              primary: '#26a69a',
              secondary: '#ffffff',
            },
          },
          error: {
            iconTheme: {
              primary: '#ef5350',
              secondary: '#ffffff',
            },
          },
        }}
      />
    </div>
  );
}

export default App;