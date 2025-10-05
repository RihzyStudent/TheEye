import { useEffect, useRef } from 'react';

interface StarFieldProps {
  starCount?: number;
}

export function StarField({ starCount = 100 }: StarFieldProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const createStar = () => {
      const star = document.createElement('div');
      star.className = 'star';
      star.style.left = Math.random() * 100 + '%';
      star.style.top = Math.random() * 100 + '%';
      star.style.width = Math.random() * 3 + 1 + 'px';
      star.style.height = star.style.width;
      star.style.animationDelay = Math.random() * 2 + 's';
      star.style.animationDuration = (Math.random() * 3 + 2) + 's';
      return star;
    };

    // Clear existing stars
    container.innerHTML = '';

    // Create stars
    for (let i = 0; i < starCount; i++) {
      container.appendChild(createStar());
    }
  }, [starCount]);

  return <div ref={containerRef} className="star-field" />;
}