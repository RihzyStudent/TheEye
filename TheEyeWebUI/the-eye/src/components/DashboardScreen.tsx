import { motion } from 'motion/react';
import { 
  Globe, 
  Star, 
  Rocket, 
  Zap, 
  Brain, 
  User, 
  Menu,
  ArrowLeft,
  Cpu 
} from 'lucide-react';
import { useState } from 'react';
import { Button } from './ui/button';

interface DashboardScreenProps {
  onNavigate: (screen: string) => void;
  onBack: () => void;
}

const menuItems = [
  { 
    id: 'solar-system',
    icon: <Globe className="w-8 h-8" />,
    title: 'Solar System',
    description: 'Explore planets and moons',
    color: 'bg-blue-500'
  },
  { 
    id: 'constellations',
    icon: <Star className="w-8 h-8" />,
    title: 'Constellations',
    description: 'Interactive star map',
    color: 'bg-purple-500'
  },
  { 
    id: 'space-exploration',
    icon: <Rocket className="w-8 h-8" />,
    title: 'Space Exploration',
    description: 'Missions and spacecraft',
    color: 'bg-green-500'
  },
  { 
    id: 'cosmic-phenomena',
    icon: <Zap className="w-8 h-8" />,
    title: 'Cosmic Phenomena',
    description: 'Extraordinary events',
    color: 'bg-red-500'
  },
  { 
    id: 'quiz',
    icon: <Brain className="w-8 h-8" />,
    title: 'Cosmic Quiz',
    description: 'Test your knowledge',
    color: 'bg-yellow-500'
  },
  { 
    id: 'ai-ml',
    icon: <Cpu className="w-8 h-8" />,
    title: 'Exoplanet Detection',
    description: 'AI-Powered Discovery',
    color: 'bg-cyan-500'
  },
  { 
    id: 'profile',
    icon: <User className="w-8 h-8" />,
    title: 'Profile',
    description: 'Achievements and progress',
    color: 'bg-indigo-500'
  }
];

export function DashboardScreen({ onNavigate, onBack }: DashboardScreenProps) {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <div className="min-h-screen cosmic-bg relative overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-6 relative z-10">
        <Button 
          variant="ghost" 
          size="icon"
          onClick={onBack}
          className="text-pure-white hover:bg-white/10"
        >
          <ArrowLeft className="w-6 h-6" />
        </Button>
        
        <h1 className="text-2xl font-bold text-pure-white">
          AstroExplorer
        </h1>
        
        <Button 
          variant="ghost" 
          size="icon"
          onClick={toggleMenu}
          className="text-pure-white hover:bg-white/10"
        >
          <Menu className="w-6 h-6" />
        </Button>
      </div>

      {/* Central Navigation Hub */}
      <div className="flex items-center justify-center min-h-[calc(100vh-100px)] p-6">
        <div className="relative">
          {/* Central Hub Button */}
          <motion.div
            className="relative z-10"
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.95 }}
          >
            <button
              onClick={toggleMenu}
              className="w-24 h-24 bg-stellar-gold rounded-full flex items-center justify-center shadow-lg shadow-stellar-gold/50"
            >
              <Star className="w-12 h-12 text-cosmic-deep-blue" />
            </button>
          </motion.div>

          {/* Radial Menu Items */}
          {menuItems.map((item, index) => {
            const angle = (index * (360 / menuItems.length)) - 90; // Distribute evenly, starting from top
            const radius = 120;
            const x = Math.cos((angle * Math.PI) / 180) * radius;
            const y = Math.sin((angle * Math.PI) / 180) * radius;

            return (
              <motion.div
                key={item.id}
                className="absolute top-12 left-12"
                initial={{ scale: 0, x: 0, y: 0 }}
                animate={isMenuOpen ? {
                  scale: 1,
                  x: x,
                  y: y,
                  transition: { delay: index * 0.1, type: "spring", stiffness: 300 }
                } : {
                  scale: 0,
                  x: 0,
                  y: 0,
                  transition: { delay: (menuItems.length - 1 - index) * 0.05 }
                }}
              >
                <motion.button
                  onClick={() => onNavigate(item.id)}
                  className={`w-16 h-16 ${item.color} rounded-full flex items-center justify-center text-white shadow-lg relative group`}
                  whileHover={{ scale: 1.2 }}
                  whileTap={{ scale: 0.9 }}
                >
                  {item.icon}
                  
                  {/* Tooltip */}
                  <div className="absolute -bottom-16 left-1/2 transform -translate-x-1/2 opacity-0 group-hover:opacity-100 transition-opacity bg-black/80 text-white text-xs p-2 rounded whitespace-nowrap">
                    {item.title}
                  </div>
                </motion.button>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="absolute bottom-6 left-6 right-6">
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-stellar-gold font-bold text-xl">85%</div>
            <div className="text-gray-300 text-sm">Progress</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-stellar-gold font-bold text-xl">12</div>
            <div className="text-gray-300 text-sm">Achievements</div>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-lg p-3">
            <div className="text-stellar-gold font-bold text-xl">7</div>
            <div className="text-gray-300 text-sm">Day Streak</div>
          </div>
        </div>
      </div>
    </div>
  );
}