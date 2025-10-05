import { motion } from 'motion/react';
import { Telescope } from 'lucide-react';

interface SplashScreenProps {
  onComplete: () => void;
}

export function SplashScreen({ onComplete }: SplashScreenProps) {
  return (
    <motion.div 
      className="fixed inset-0 cosmic-bg flex items-center justify-center z-50"
      initial={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="text-center relative">
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 1.5, ease: "easeOut" }}
          className="mb-6"
        >
          <Telescope className="w-24 h-24 text-stellar-gold mx-auto rotating-animation" />
        </motion.div>
        
        <motion.h1 
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="text-4xl font-bold text-pure-white mb-2"
          style={{ fontFamily: 'Montserrat, sans-serif' }}
        >
          AstroExplorer
        </motion.h1>
        
        <motion.p 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8, duration: 0.8 }}
          className="text-stellar-gold text-lg"
        >
          Discover the Universe
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2, duration: 0.5 }}
          className="mt-8"
        >
          <motion.button
            onClick={onComplete}
            className="px-8 py-3 bg-stellar-gold text-cosmic-deep-blue font-semibold rounded-full hover:bg-opacity-90 transition-all"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            Start Exploration
          </motion.button>
        </motion.div>
      </div>
    </motion.div>
  );
}