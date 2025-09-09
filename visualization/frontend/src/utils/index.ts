import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatNumber(num: number, decimals: number = 2): string {
  if (num === undefined || num === null || isNaN(num)) {
    return '0';
  }
  return num.toLocaleString('fr-FR', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export function formatCurrency(amount: number, currency: string = 'USDC'): string {
  if (amount === undefined || amount === null || isNaN(amount)) {
    return `0.00 ${currency}`;
  }
  return `${formatNumber(amount, 2)} ${currency}`;
}

export function formatPercent(value: number): string {
  if (value === undefined || value === null || isNaN(value)) {
    return '0.00%';
  }
  const sign = value >= 0 ? '+' : '';
  return `${sign}${formatNumber(value, 2)}%`;
}

export function formatVolume(volume: number): string {
  if (volume >= 1e9) {
    return `${formatNumber(volume / 1e9, 1)}B`;
  } else if (volume >= 1e6) {
    return `${formatNumber(volume / 1e6, 1)}M`;
  } else if (volume >= 1e3) {
    return `${formatNumber(volume / 1e3, 1)}K`;
  }
  return formatNumber(volume, 0);
}

export function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString('fr-FR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

export function formatDate(timestamp: string): string {
  return new Date(timestamp).toLocaleDateString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}

export function formatDateTime(timestamp: string): string {
  return new Date(timestamp).toLocaleString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function debounce<T extends (...args: any[]) => void>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

export function throttle<T extends (...args: any[]) => void>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  return (...args: Parameters<T>) => {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    }
  };
}

export function getVisiblePrices(
  marketData: any,
  startTime: Date,
  endTime: Date
): number[] {
  const visiblePrices: number[] = [];
  const startMs = startTime.getTime();
  const endMs = endTime.getTime();
  
  for (let i = 0; i < marketData.timestamps.length; i++) {
    const timestampMs = new Date(marketData.timestamps[i]).getTime();
    if (timestampMs >= startMs && timestampMs <= endMs) {
      visiblePrices.push(marketData.high[i]);
      visiblePrices.push(marketData.low[i]);
    }
  }
  
  return visiblePrices;
}

export function getDefaultTimeRange(interval: string): number {
  const ranges = {
    '1m': 3 * 60 * 60 * 1000,     // 3 heures
    '5m': 24 * 60 * 60 * 1000,    // 24 heures
    '15m': 48 * 60 * 60 * 1000,   // 48 heures
    '3m': 6 * 60 * 60 * 1000,  // 6 heures
    '1h': 7 * 24 * 60 * 60 * 1000, // 7 jours
    '1d': 90 * 24 * 60 * 60 * 1000, // 90 jours
  };
  
  return ranges[interval as keyof typeof ranges] || ranges['1m'];
}

export function generateId(): string {
  return Math.random().toString(36).substr(2, 9);
}

export function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}