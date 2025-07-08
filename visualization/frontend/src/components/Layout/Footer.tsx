import React from 'react';

function Footer() {
  return (
    <footer className="bg-dark-200 border-t border-gray-700 py-4 mt-12">
      <div className="container mx-auto px-4 max-w-7xl">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-2">
            <span>© 2024 RootTrading</span>
            <span>•</span>
            <span>Plateforme de trading avancée</span>
          </div>
          
          <div className="flex items-center space-x-4">
            <span className="text-xs opacity-75">
              Powered by React + TypeScript + Tailwind CSS
            </span>
          </div>
        </div>
      </div>
    </footer>
  );
}

export default Footer;