import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft, Zap, Eye, CircleDot, Orbit, Sparkles, Moon } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from './ui/dialog';
import { Badge } from './ui/badge';

interface Phenomenon {
  id: string;
  name: string;
  icon: any;
  category: string;
  description: string;
  rarity: 'common' | 'rare' | 'very-rare';
  color: string;
  details: string;
  facts: string[];
}

const phenomena: Phenomenon[] = [
  {
    id: '1',
    name: 'Solar Eclipse',
    icon: Moon,
    category: 'Eclipse',
    description: 'The Moon passes between Earth and Sun, blocking sunlight',
    rarity: 'rare',
    color: '#FFD700',
    details: 'A solar eclipse occurs when the Moon passes between the Sun and Earth, casting a shadow on Earth. Total solar eclipses are visible from a specific location on Earth only once every 375 years on average.',
    facts: [
      'Total solar eclipses last up to 7.5 minutes',
      'The Moon perfectly covers the Sun due to cosmic coincidence',
      'Ancient civilizations feared solar eclipses'
    ]
  },
  {
    id: '2',
    name: 'Supernova',
    icon: Sparkles,
    category: 'Stellar Death',
    description: 'Massive explosion marking the death of a star',
    rarity: 'very-rare',
    color: '#FF6B6B',
    details: 'A supernova is the explosive death of a massive star. In just seconds, a supernova can release more energy than our Sun will produce in its entire lifetime. The explosion creates most of the heavy elements in the universe.',
    facts: [
      'Brightest events in the universe',
      'Can outshine entire galaxies temporarily',
      'Create neutron stars and black holes'
    ]
  },
  {
    id: '3',
    name: 'Black Hole',
    icon: CircleDot,
    category: 'Extreme Gravity',
    description: 'Region of spacetime where gravity is so strong nothing can escape',
    rarity: 'rare',
    color: '#1a1a2e',
    details: 'Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their lives. The boundary around a black hole is called the event horizon.',
    facts: [
      'Time slows down near black holes',
      'Supermassive black holes exist at galaxy centers',
      'First image captured in 2019'
    ]
  },
  {
    id: '4',
    name: 'Aurora Borealis',
    icon: Zap,
    category: 'Atmospheric',
    description: 'Natural light display caused by solar particles',
    rarity: 'common',
    color: '#4ECDC4',
    details: 'The Northern Lights occur when charged particles from the Sun interact with gases in Earth\'s atmosphere. The different colors depend on which gas is being excited and at what altitude.',
    facts: [
      'Best viewed near magnetic poles',
      'Colors indicate different atmospheric gases',
      'Can be predicted using solar activity'
    ]
  },
  {
    id: '5',
    name: 'Meteor Shower',
    icon: Orbit,
    category: 'Cosmic Debris',
    description: 'Multiple meteors appearing to radiate from one point',
    rarity: 'common',
    color: '#95E1D3',
    details: 'Meteor showers occur when Earth passes through trails of comet debris. As these particles enter our atmosphere at high speeds, they burn up, creating streaks of light across the sky.',
    facts: [
      'Peak viewing is after midnight',
      'Perseids shower produces 60 meteors per hour',
      'Named after the constellation they appear from'
    ]
  },
  {
    id: '6',
    name: 'Gravitational Waves',
    icon: Eye,
    category: 'Spacetime Ripples',
    description: 'Ripples in the fabric of spacetime',
    rarity: 'very-rare',
    color: '#A78BFA',
    details: 'Gravitational waves are ripples in spacetime caused by massive accelerating objects, like colliding black holes. First directly detected in 2015, they confirmed a major prediction of Einstein\'s general relativity.',
    facts: [
      'Travel at the speed of light',
      'Detected by LIGO observatory',
      'Created by colliding black holes and neutron stars'
    ]
  }
];

const getRarityColor = (rarity: Phenomenon['rarity']) => {
  switch (rarity) {
    case 'common':
      return 'bg-green-500/20 text-green-300';
    case 'rare':
      return 'bg-blue-500/20 text-blue-300';
    case 'very-rare':
      return 'bg-purple-500/20 text-purple-300';
  }
};

interface CosmicPhenomenaScreenProps {
  onBack: () => void;
}

export function CosmicPhenomenaScreen({ onBack }: CosmicPhenomenaScreenProps) {
  const [selectedPhenomenon, setSelectedPhenomenon] = useState<Phenomenon | null>(null);

  return (
    <div className="min-h-screen cosmic-bg p-6 overflow-y-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-7xl mx-auto mb-8"
      >
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-stellar-gold hover:text-yellow-400 transition-colors mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Dashboard
        </button>

        <div className="text-center mb-8">
          <motion.h1
            initial={{ scale: 0.9 }}
            animate={{ scale: 1 }}
            className="text-pure-white mb-3"
          >
            Cosmic Phenomena
          </motion.h1>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Explore the most spectacular and mysterious events in the universe
          </p>
        </div>
      </motion.div>

      {/* Phenomena Grid */}
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {phenomena.map((phenomenon, index) => {
            const Icon = phenomenon.icon;
            return (
              <motion.div
                key={phenomenon.id}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05, rotateY: 5 }}
                whileTap={{ scale: 0.95 }}
              >
                <Card
                  className="bg-cosmic-purple/20 border-stellar-gold/30 hover:border-stellar-gold cursor-pointer h-full backdrop-blur-sm relative overflow-hidden group"
                  onClick={() => setSelectedPhenomenon(phenomenon)}
                >
                  {/* Animated background effect */}
                  <motion.div
                    className="absolute inset-0 opacity-0 group-hover:opacity-20 transition-opacity"
                    style={{ 
                      background: `radial-gradient(circle at 50% 50%, ${phenomenon.color}, transparent 70%)`
                    }}
                    animate={{
                      scale: [1, 1.2, 1],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />

                  <CardHeader>
                    <div className="flex items-start justify-between mb-3">
                      <motion.div
                        whileHover={{ rotate: [0, -10, 10, -10, 0] }}
                        transition={{ duration: 0.5 }}
                        className="p-3 rounded-lg relative"
                        style={{ backgroundColor: `${phenomenon.color}30` }}
                      >
                        <Icon className="w-8 h-8" style={{ color: phenomenon.color }} />
                        
                        {/* Pulsing glow effect */}
                        <motion.div
                          className="absolute inset-0 rounded-lg"
                          style={{ backgroundColor: phenomenon.color }}
                          animate={{
                            opacity: [0, 0.3, 0],
                            scale: [1, 1.2, 1]
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                      </motion.div>
                      <Badge className={getRarityColor(phenomenon.rarity)}>
                        {phenomenon.rarity}
                      </Badge>
                    </div>
                    <CardTitle className="text-pure-white">{phenomenon.name}</CardTitle>
                    <CardDescription className="text-stellar-gold">
                      {phenomenon.category}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-300">{phenomenon.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Detail Dialog */}
      <Dialog open={!!selectedPhenomenon} onOpenChange={() => setSelectedPhenomenon(null)}>
        <DialogContent className="bg-cosmic-deep-blue border-stellar-gold max-w-2xl">
          <AnimatePresence mode="wait">
            {selectedPhenomenon && (
              <motion.div
                key={selectedPhenomenon.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
              >
                <DialogHeader>
                  <div className="flex items-center gap-4 mb-4">
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                      className="p-4 rounded-full"
                      style={{ backgroundColor: `${selectedPhenomenon.color}30` }}
                    >
                      <selectedPhenomenon.icon 
                        className="w-8 h-8" 
                        style={{ color: selectedPhenomenon.color }} 
                      />
                    </motion.div>
                    <div>
                      <DialogTitle className="text-pure-white">
                        {selectedPhenomenon.name}
                      </DialogTitle>
                      <DialogDescription className="text-stellar-gold">
                        {selectedPhenomenon.category}
                      </DialogDescription>
                    </div>
                  </div>
                </DialogHeader>

                <div className="space-y-4 mt-4">
                  <div>
                    <h3 className="text-pure-white mb-2">About</h3>
                    <p className="text-gray-300">{selectedPhenomenon.details}</p>
                  </div>

                  <div>
                    <h3 className="text-pure-white mb-3">Interesting Facts</h3>
                    <ul className="space-y-2">
                      {selectedPhenomenon.facts.map((fact, index) => (
                        <motion.li
                          key={index}
                          initial={{ opacity: 0, x: -20 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className="flex items-start gap-2 text-gray-300"
                        >
                          <span className="text-stellar-gold mt-1">✦</span>
                          {fact}
                        </motion.li>
                      ))}
                    </ul>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </DialogContent>
      </Dialog>

      {/* Animated particles */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[...Array(30)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full"
            style={{
              width: Math.random() * 4 + 1,
              height: Math.random() * 4 + 1,
              backgroundColor: phenomena[Math.floor(Math.random() * phenomena.length)].color
            }}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: 0.2
            }}
            animate={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: [0.2, 0.8, 0.2]
            }}
            transition={{
              duration: 5 + Math.random() * 10,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>
    </div>
  );
}