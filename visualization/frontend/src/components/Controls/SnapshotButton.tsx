import React, { useState } from 'react';
import { Camera } from 'lucide-react';
import html2canvas from 'html2canvas';
import { useChartStore } from '@/stores/useChartStore';

function SnapshotButton() {
  const [isCapturing, setIsCapturing] = useState(false);
  const { config } = useChartStore();
  const { symbol: selectedSymbol, interval: selectedInterval } = config;

  const captureCharts = async () => {
    setIsCapturing(true);
    
    try {
      // Capturer le graphique principal
      const mainChartElement = document.querySelector('.market-chart-container');
      const rsiChartElement = document.querySelector('.rsi-chart-container');
      const macdChartElement = document.querySelector('.macd-chart-container');

      if (!mainChartElement || !rsiChartElement || !macdChartElement) {
        throw new Error('Charts not found');
      }

      // Capturer chaque graphique
      const mainCanvas = await html2canvas(mainChartElement as HTMLElement, {
        backgroundColor: '#1a1a1a',
        scale: 2,
      });
      
      const rsiCanvas = await html2canvas(rsiChartElement as HTMLElement, {
        backgroundColor: '#1a1a1a',
        scale: 2,
      });
      
      const macdCanvas = await html2canvas(macdChartElement as HTMLElement, {
        backgroundColor: '#1a1a1a',
        scale: 2,
      });

      // Créer un canvas combiné
      const combinedCanvas = document.createElement('canvas');
      const ctx = combinedCanvas.getContext('2d');
      
      if (!ctx) throw new Error('Could not get canvas context');

      // Calculer les dimensions
      const padding = 20;
      const spacing = 10;
      combinedCanvas.width = mainCanvas.width;
      combinedCanvas.height = mainCanvas.height + rsiCanvas.height + macdCanvas.height + (spacing * 2) + (padding * 2);

      // Fond noir
      ctx.fillStyle = '#1a1a1a';
      ctx.fillRect(0, 0, combinedCanvas.width, combinedCanvas.height);

      // Dessiner les graphiques
      let yOffset = padding;
      ctx.drawImage(mainCanvas, 0, yOffset);
      yOffset += mainCanvas.height + spacing;
      
      ctx.drawImage(rsiCanvas, 0, yOffset);
      yOffset += rsiCanvas.height + spacing;
      
      ctx.drawImage(macdCanvas, 0, yOffset);

      // Ajouter timestamp et infos
      ctx.fillStyle = '#ffffff';
      ctx.font = '16px monospace';
      const displayTimestamp = new Date().toLocaleString('fr-FR');
      const info = `${selectedSymbol} - ${selectedInterval} - ${displayTimestamp}`;
      ctx.fillText(info, padding, combinedCanvas.height - 5);

      // Convertir en base64
      const combinedImage = combinedCanvas.toDataURL('image/png');

      // Créer un lien de téléchargement
      const link = document.createElement('a');
      const fileTimestamp = new Date().toLocaleString('fr-FR').replace(/[/:]/g, '-');
      const filename = `${selectedSymbol}_${selectedInterval}_${fileTimestamp}.png`;
      
      link.download = filename;
      link.href = combinedImage;
      
      // Déclencher le téléchargement
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      // Afficher une notification de succès
      const notification = document.createElement('div');
      notification.className = 'fixed bottom-4 right-4 bg-green-600 text-white px-4 py-2 rounded-lg shadow-lg z-50';
      notification.textContent = `Snapshot téléchargé: ${filename}`;
      document.body.appendChild(notification);
      
      setTimeout(() => {
        notification.remove();
      }, 3000);

    } catch (error) {
      console.error('Error capturing snapshot:', error);
      
      // Afficher une notification d'erreur
      const notification = document.createElement('div');
      notification.className = 'fixed bottom-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg z-50';
      notification.textContent = `Erreur: ${error instanceof Error ? error.message : 'Erreur inconnue'}`;
      document.body.appendChild(notification);
      
      setTimeout(() => {
        notification.remove();
      }, 3000);
    } finally {
      setIsCapturing(false);
    }
  };

  return (
    <button
      onClick={captureCharts}
      disabled={isCapturing}
      className="px-3 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
      title="Capturer les graphiques"
    >
      <Camera className="w-4 h-4" />
      <span className="text-sm font-medium">
        {isCapturing ? 'Capture...' : 'Snapshot'}
      </span>
    </button>
  );
}

export default SnapshotButton;