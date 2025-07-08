import React from 'react';
import { Toaster } from 'react-hot-toast';
import Layout from '@/components/Layout';
import Dashboard from '@/components/Dashboard';

function App() {
  return (
    <div className="min-h-screen bg-dark-500">
      <Layout>
        <Dashboard />
      </Layout>
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