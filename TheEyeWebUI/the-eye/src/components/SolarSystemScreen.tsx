import { motion, useMotionValue, useTransform } from 'motion/react';
import { useState, useRef } from 'react';
import { ArrowLeft, Info } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';

interface SolarSystemScreenProps {
  onBack: () => void;
}

const planets = [
  {
    name: 'Mercury',
    size: 'w-8 h-8',
    color: 'bg-gray-400',
    distance: '57.9 million km',
    description: 'The closest planet to the Sun with extreme temperatures.',
    facts: ['Day: 88 Earth days', 'No atmosphere', 'Rocky surface']
  },
  {
    name: 'Venus',
    size: 'w-10 h-10',
    color: 'bg-yellow-600',
    distance: '108.2 million km',
    description: 'The hottest planet in the solar system.',
    facts: ['Toxic atmosphere', 'Extreme greenhouse effect', 'Rotates backwards']
  },
  {
    name: 'Earth',
    size: 'w-12 h-12',
    color: 'bg-blue-500',
    distance: '149.6 million km',
    description: 'Our home, the only known planet with life.',
    facts: ['70% water', 'Breathable atmosphere', 'One natural moon']
  },
  {
    name: 'Mars',
    size: 'w-10 h-10',
    color: 'bg-red-500',
    distance: '227.9 million km',
    description: 'The red planet, target of future human missions.',
    facts: ['Two small moons', 'Seasons like Earth', 'Evidence of ancient water']
  },
  {
    name: 'Jupiter',
    size: 'w-20 h-20',
    color: 'bg-orange-400',
    distance: '778.5 million km',
    description: 'The largest gas giant in the solar system.',
    facts: ['Great Red Spot', 'Over 80 moons', 'Protects Earth from asteroids']
  },
  {
    name: 'Saturn',
    size: 'w-18 h-18',
    color: 'bg-yellow-400',
    distance: '1.432 billion km',
    description: 'Famous for its impressive rings.',
    facts: ['Rings of ice and rock', 'Over 80 moons', 'Lower density than water']
  },
  {
    name: 'Uranus',
    size: 'w-14 h-14',
    color: 'bg-cyan-400',
    distance: '2.867 billion km',
    description: 'An ice giant that rotates on its side.',
    facts: ['Rotates sideways', 'Vertical rings', 'Very cold: -224°C']
  },
  {
    name: 'Neptune',
    size: 'w-14 h-14',
    color: 'bg-blue-700',
    distance: '4.515 billion km',
    description: 'The farthest planet with supersonic winds.',
    facts: ['Winds at 2,100 km/h', 'Diamond rain', 'Orbits the Sun in 165 years']
  }
];

export function SolarSystemScreen({ onBack }: SolarSystemScreenProps) {
  const [selectedPlanet, setSelectedPlanet] = useState<number | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const x = useMotionValue(0);
  const scale = useTransform(x, [-1000, 0, 1000], [0.8, 1, 0.8]);

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
          Solar System
        </h1>
        
        <div className="w-10 h-10" />
      </div>

      {/* Solar System Slider */}
      <div className="flex-1 flex items-center">
        <motion.div
          ref={containerRef}
          className="w-full h-80 relative overflow-hidden"
          drag="x"
          dragConstraints={{ left: -1200, right: 0 }}
          style={{ x, scale }}
        >
          {/* Sun */}
          <motion.div
            className="absolute left-20 top-1/2 transform -translate-y-1/2"
            whileHover={{ scale: 1.1 }}
          >
            <div className="w-16 h-16 bg-stellar-gold rounded-full shadow-lg shadow-stellar-gold/50 flex items-center justify-center">
              <div className="w-12 h-12 bg-yellow-300 rounded-full animate-pulse" />
            </div>
            <p className="text-pure-white text-center mt-2 font-semibold">Sun</p>
          </motion.div>

          {/* Planets */}
          {planets.map((planet, index) => (
            <motion.div
              key={planet.name}
              className="absolute top-1/2 transform -translate-y-1/2 cursor-pointer"
              style={{ left: `${200 + index * 200}px` }}
              whileHover={{ scale: 1.2 }}
              whileTap={{ scale: 0.9 }}
              onClick={() => setSelectedPlanet(index)}
            >
              <div className={`${planet.size} ${planet.color} rounded-full shadow-lg mx-auto floating-animation`} />
              <p className="text-pure-white text-center mt-2 font-semibold text-sm">
                {planet.name}
              </p>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Instructions */}
      <div className="absolute bottom-24 left-6 right-6 text-center">
        <p className="text-gray-300 text-sm">
          👆 Swipe horizontally to explore • Tap a planet for more information
        </p>
      </div>

      {/* Planet Info Modal */}
      {selectedPlanet !== null && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 bg-black/70 flex items-center justify-center p-6 z-50"
          onClick={() => setSelectedPlanet(null)}
        >
          <motion.div
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.8, opacity: 0 }}
            onClick={(e) => e.stopPropagation()}
            className="max-w-md w-full"
          >
            <Card className="bg-cosmic-deep-blue border-stellar-gold">
              <CardContent className="p-6">
                <div className="flex items-center mb-4">
                  <div className={`${planets[selectedPlanet].size} ${planets[selectedPlanet].color} rounded-full mr-4`} />
                  <div>
                    <h3 className="text-xl font-bold text-pure-white">
                      {planets[selectedPlanet].name}
                    </h3>
                    <p className="text-stellar-gold text-sm">
                      {planets[selectedPlanet].distance} from the Sun
                    </p>
                  </div>
                </div>
                
                <p className="text-gray-300 mb-4">
                  {planets[selectedPlanet].description}
                </p>
                
                <div className="space-y-2">
                  <h4 className="text-stellar-gold font-semibold flex items-center">
                    <Info className="w-4 h-4 mr-2" />
                    Fun Facts:
                  </h4>
                  {planets[selectedPlanet].facts.map((fact, index) => (
                    <p key={index} className="text-gray-300 text-sm">
                      • {fact}
                    </p>
                  ))}
                </div>
                
                <Button 
                  onClick={() => setSelectedPlanet(null)}
                  className="w-full mt-6 bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400"
                >
                  Close
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </motion.div>
      )}
    </div>
  );
}