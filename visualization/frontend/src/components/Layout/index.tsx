import React from 'react';
import Header from './Header';
import Footer from './Footer';
import WebSocketIndicator from './WebSocketIndicator';

interface LayoutProps {
  children: React.ReactNode;
}

function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-dark-500 text-white">
      <WebSocketIndicator />
      <Header />
      <main className="container mx-auto px-4 py-6 max-w-7xl">
        {children}
      </main>
      <Footer />
    </div>
  );
}

export default Layout;