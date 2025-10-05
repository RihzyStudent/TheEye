import { motion, useDragControls } from 'motion/react';
import { useState, useRef } from 'react';
import { ArrowLeft, Star, Award } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { toast } from 'sonner@2.0.3';

interface ConstellationsScreenProps {
  onBack: () => void;
}

interface Star {
  id: string;
  x: number;
  y: number;
  brightness: number;
  constellation?: string;
}

interface Constellation {
  name: string;
  stars: string[];
  description: string;
  mythology: string;
  completed: boolean;
}

const constellations: Constellation[] = [
  {
    name: 'Ursa Major',
    stars: ['star-1', 'star-2', 'star-3', 'star-4', 'star-5', 'star-6', 'star-7'],
    description: 'One of the most recognizable constellations in the northern hemisphere.',
    mythology: 'In Greek mythology, it represents Callisto transformed into a bear.',
    completed: false
  },
  {
    name: 'Orion',
    stars: ['star-8', 'star-9', 'star-10', 'star-11', 'star-12'],
    description: 'The celestial hunter, visible in winter.',
    mythology: 'Orion was a great hunter in Greek mythology.',
    completed: false
  }
];

export function ConstellationsScreen({ onBack }: ConstellationsScreenProps) {
  const [stars] = useState<Star[]>([
    // Osa Mayor
    { id: 'star-1', x: 150, y: 120, brightness: 0.9 },
    { id: 'star-2', x: 180, y: 100, brightness: 0.8 },
    { id: 'star-3', x: 220, y: 110, brightness: 0.9 },
    { id: 'star-4', x: 260, y: 120, brightness: 0.7 },
    { id: 'star-5', x: 240, y: 160, brightness: 0.8 },
    { id: 'star-6', x: 200, y: 170, brightness: 0.9 },
    { id: 'star-7', x: 170, y: 150, brightness: 0.8 },
    
    // Orión
    { id: 'star-8', x: 100, y: 300, brightness: 0.9 },
    { id: 'star-9', x: 130, y: 280, brightness: 0.8 },
    { id: 'star-10', x: 160, y: 290, brightness: 0.9 },
    { id: 'star-11', x: 140, y: 330, brightness: 0.7 },
    { id: 'star-12', x: 120, y: 350, brightness: 0.8 },
    
    // Estrellas adicionales
    ...Array.from({ length: 20 }, (_, i) => ({
      id: `extra-${i}`,
      x: Math.random() * 350 + 50,
      y: Math.random() * 400 + 100,
      brightness: Math.random() * 0.5 + 0.3
    }))
  ]);

  const [connectedStars, setConnectedStars] = useState<string[]>([]);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentPath, setCurrentPath] = useState<string[]>([]);
  const [completedConstellations, setCompletedConstellations] = useState<string[]>([]);
  const [selectedConstellation, setSelectedConstellation] = useState<string | null>(null);

  const handleStarPress = (starId: string) => {
    if (!isDrawing) {
      setIsDrawing(true);
      setCurrentPath([starId]);
    } else {
      const newPath = [...currentPath, starId];
      setCurrentPath(newPath);
      
      // Check if this completes a constellation
      const completedConstellation = constellations.find(constellation => 
        constellation.stars.length === newPath.length &&
        constellation.stars.every(star => newPath.includes(star))
      );
      
      if (completedConstellation && !completedConstellations.includes(completedConstellation.name)) {
        setCompletedConstellations([...completedConstellations, completedConstellation.name]);
        setConnectedStars([...connectedStars, ...newPath]);
        toast.success(`Constellation ${completedConstellation.name} completed!`);
        setIsDrawing(false);
        setCurrentPath([]);
      }
    }
  };

  const resetDrawing = () => {
    setIsDrawing(false);
    setCurrentPath([]);
  };

  const getStarPosition = (starId: string) => {
    const star = stars.find(s => s.id === starId);
    return star ? { x: star.x, y: star.y } : { x: 0, y: 0 };
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
          Constellations
        </h1>
        
        <Button 
          variant="ghost" 
          size="sm"
          onClick={resetDrawing}
          className="text-pure-white hover:bg-white/10"
        >
          Reset
        </Button>
      </div>

      {/* Star Map */}
      <div className="relative flex-1 p-6">
        <div className="relative w-full h-96 bg-black/30 rounded-lg border border-stellar-gold/30 overflow-hidden">
          <svg className="absolute inset-0 w-full h-full">
            {/* Draw constellation lines */}
            {constellations.map(constellation => {
              if (!completedConstellations.includes(constellation.name)) return null;
              
              return (
                <g key={constellation.name}>
                  {constellation.stars.slice(0, -1).map((starId, index) => {
                    const start = getStarPosition(starId);
                    const end = getStarPosition(constellation.stars[index + 1]);
                    return (
                      <line
                        key={`${starId}-${constellation.stars[index + 1]}`}
                        x1={start.x}
                        y1={start.y}
                        x2={end.x}
                        y2={end.y}
                        stroke="#FFD700"
                        strokeWidth="2"
                        opacity="0.8"
                      />
                    );
                  })}
                </g>
              );
            })}
            
            {/* Draw current path */}
            {currentPath.slice(0, -1).map((starId, index) => {
              const start = getStarPosition(starId);
              const end = getStarPosition(currentPath[index + 1]);
              return (
                <line
                  key={`current-${starId}-${currentPath[index + 1]}`}
                  x1={start.x}
                  y1={start.y}
                  x2={end.x}
                  y2={end.y}
                  stroke="#FFD700"
                  strokeWidth="2"
                  opacity="0.5"
                  strokeDasharray="4 4"
                />
              );
            })}
          </svg>

          {/* Stars */}
          {stars.map(star => (
            <motion.button
              key={star.id}
              className={`absolute w-3 h-3 rounded-full cursor-pointer ${
                currentPath.includes(star.id) ? 'bg-stellar-gold' :
                connectedStars.includes(star.id) ? 'bg-yellow-300' :
                'bg-white'
              }`}
              style={{
                left: star.x,
                top: star.y,
                opacity: star.brightness,
                transform: 'translate(-50%, -50%)'
              }}
              whileHover={{ scale: 1.5 }}
              whileTap={{ scale: 0.8 }}
              onClick={() => handleStarPress(star.id)}
              animate={{
                boxShadow: currentPath.includes(star.id) 
                  ? '0 0 10px #FFD700' 
                  : '0 0 4px rgba(255,255,255,0.5)'
              }}
            />
          ))}
        </div>

        {/* Instructions */}
        <div className="mt-4 text-center">
          <p className="text-gray-300 text-sm">
            {isDrawing 
              ? `Connecting ${currentPath.length} stars... Tap the next star`
              : 'Tap a star to start forming a constellation'
            }
          </p>
        </div>

        {/* Constellation Info */}
        <div className="grid grid-cols-2 gap-4 mt-6">
          {constellations.map(constellation => (
            <Card 
              key={constellation.name}
              className={`cursor-pointer transition-all ${
                completedConstellations.includes(constellation.name)
                  ? 'bg-stellar-gold/20 border-stellar-gold'
                  : 'bg-white/10 border-gray-600'
              }`}
              onClick={() => setSelectedConstellation(constellation.name)}
            >
              <CardContent className="p-4">
                <div className="flex items-center justify-between mb-2">
                  <h3 className="text-pure-white font-semibold">
                    {constellation.name}
                  </h3>
                  {completedConstellations.includes(constellation.name) && (
                    <Award className="w-5 h-5 text-stellar-gold" />
                  )}
                </div>
                <p className="text-gray-300 text-sm">
                  {constellation.stars.length} stars
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>

      {/* Constellation Detail Modal */}
      {selectedConstellation && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/70 flex items-center justify-center p-6 z-50"
          onClick={() => setSelectedConstellation(null)}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="max-w-md w-full"
          >
            {(() => {
              const constellation = constellations.find(c => c.name === selectedConstellation);
              if (!constellation) return null;
              
              return (
                <Card className="bg-cosmic-deep-blue border-stellar-gold">
                  <CardContent className="p-6">
                    <div className="flex items-center mb-4">
                      <Star className="w-6 h-6 text-stellar-gold mr-3" />
                      <h3 className="text-xl font-bold text-pure-white">
                        {constellation.name}
                      </h3>
                    </div>
                    
                    <p className="text-gray-300 mb-4">
                      {constellation.description}
                    </p>
                    
                    <div className="mb-4">
                      <h4 className="text-stellar-gold font-semibold mb-2">Mythology:</h4>
                      <p className="text-gray-300 text-sm">
                        {constellation.mythology}
                      </p>
                    </div>
                    
                    <Button 
                      onClick={() => setSelectedConstellation(null)}
                      className="w-full bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400"
                    >
                      Close
                    </Button>
                  </CardContent>
                </Card>
              );
            })()}
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}