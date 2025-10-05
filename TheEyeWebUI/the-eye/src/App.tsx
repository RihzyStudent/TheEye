import { useEffect } from 'react';
import { StarField } from './components/StarField';
import { ExoplanetDetectionScreen } from './components/ExoplanetDetectionScreen';
import { Toaster } from './components/ui/sonner';

export default function App() {
  useEffect(() => {
    // Apply dark theme by default for the cosmic experience
    document.documentElement.classList.add('dark');
  }, []);

  return (
    <div className="min-h-screen relative">
      {/* Star Field Background */}
      <StarField starCount={150} />
      
      {/* Exoplanet Detection Screen */}
      <div className="relative z-10">
        <ExoplanetDetectionScreen onBack={() => {}} />
      </div>

      {/* Toast Notifications */}
      <Toaster 
        position="top-center"
        theme="dark"
        toastOptions={{
          style: {
            background: '#0B1426',
            border: '1px solid #FFD700',
            color: '#FFFFFF'
          }
        }}
      />
    </div>
  );
}