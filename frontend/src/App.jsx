import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import TradeView from './components/TradeView';
import CycleList from './components/CycleList';
import StatsPanel from './components/StatsPanel';

// Import icons
import { ChartPie, ArrowsLeftRight, List, BarChart2 } from 'lucide-react';

const App = () => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <Router>
      <div className="flex h-screen bg-gray-100">
        {/* Sidebar */}
        <div className={`${sidebarOpen ? 'w-64' : 'w-20'} bg-gray-900 text-white transition-all duration-300 flex flex-col`}>
          <div className="p-4 flex items-center justify-between">
            <h1 className={`text-xl font-bold ${sidebarOpen ? 'block' : 'hidden'}`}>RootTrading</h1>
            <button 
              onClick={() => setSidebarOpen(!sidebarOpen)} 
              className="p-1 rounded-full hover:bg-gray-700"
            >
              <ArrowsLeftRight size={20} />
            </button>
          </div>
          
          <nav className="flex-1 pt-4">
            <ul>
              <li>
                <Link to="/" className="flex items-center py-3 px-4 hover:bg-gray-800">
                  <ChartPie size={20} />
                  {sidebarOpen && <span className="ml-4">Dashboard</span>}
                </Link>
              </li>
              <li>
                <Link to="/trades" className="flex items-center py-3 px-4 hover:bg-gray-800">
                  <List size={20} />
                  {sidebarOpen && <span className="ml-4">Trades</span>}
                </Link>
              </li>
              <li>
                <Link to="/stats" className="flex items-center py-3 px-4 hover:bg-gray-800">
                  <BarChart2 size={20} />
                  {sidebarOpen && <span className="ml-4">Statistics</span>}
                </Link>
              </li>
            </ul>
          </nav>
          
          <div className="p-4">
            <div className={`flex items-center ${sidebarOpen ? 'justify-between' : 'justify-center'}`}>
              <div className={`${sidebarOpen ? 'block' : 'hidden'}`}>
                <p className="text-sm text-gray-400">Mode:</p>
                <p className="font-medium">Demo</p>
              </div>
              <div className="h-3 w-3 bg-green-500 rounded-full"></div>
            </div>
          </div>
        </div>

        {/* Main content */}
        <div className="flex-1 overflow-auto">
          <div className="p-4 sm:p-6 lg:p-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/trades" element={<TradeView />} />
              <Route path="/stats" element={<StatsPanel />} />
            </Routes>
          </div>
        </div>
      </div>
    </Router>
  );
};

export default App;