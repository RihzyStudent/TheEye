import { useState } from 'react';
import { motion } from 'motion/react';
import { ArrowLeft, Rocket, Satellite, Radio, Globe } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { toast } from 'sonner@2.0.3';

interface Mission {
  id: string;
  name: string;
  year: number;
  description: string;
  achievement: string;
  type: 'satellite' | 'rocket' | 'probe' | 'telescope';
  status: 'completed' | 'active' | 'planned';
}

const missions: Mission[] = [
  {
    id: '1',
    name: 'Sputnik 1',
    year: 1957,
    description: 'First artificial satellite in Earth orbit',
    achievement: 'Launched the Space Age',
    type: 'satellite',
    status: 'completed'
  },
  {
    id: '2',
    name: 'Apollo 11',
    year: 1969,
    description: 'First crewed mission to land on the Moon',
    achievement: 'First humans on the Moon',
    type: 'rocket',
    status: 'completed'
  },
  {
    id: '3',
    name: 'Voyager 1',
    year: 1977,
    description: 'Exploring interstellar space',
    achievement: 'Farthest human-made object',
    type: 'probe',
    status: 'active'
  },
  {
    id: '4',
    name: 'Hubble Space Telescope',
    year: 1990,
    description: 'Revolutionary space observatory',
    achievement: 'Deep field images of the universe',
    type: 'telescope',
    status: 'active'
  },
  {
    id: '5',
    name: 'International Space Station',
    year: 1998,
    description: 'Continuous human presence in space',
    achievement: 'International collaboration in space',
    type: 'satellite',
    status: 'active'
  },
  {
    id: '6',
    name: 'James Webb Space Telescope',
    year: 2021,
    description: 'Next-generation infrared observatory',
    achievement: 'Deepest views of the early universe',
    type: 'telescope',
    status: 'active'
  },
  {
    id: '7',
    name: 'Artemis Program',
    year: 2025,
    description: 'Return humans to the Moon',
    achievement: 'Sustainable lunar presence',
    type: 'rocket',
    status: 'planned'
  },
  {
    id: '8',
    name: 'Mars Sample Return',
    year: 2028,
    description: 'Retrieve samples from Mars',
    achievement: 'First samples returned from Mars',
    type: 'probe',
    status: 'planned'
  }
];

const getIcon = (type: Mission['type']) => {
  switch (type) {
    case 'satellite':
      return Satellite;
    case 'rocket':
      return Rocket;
    case 'probe':
      return Radio;
    case 'telescope':
      return Globe;
  }
};

interface SpaceExplorationScreenProps {
  onBack: () => void;
}

export function SpaceExplorationScreen({ onBack }: SpaceExplorationScreenProps) {
  const [selectedMission, setSelectedMission] = useState<Mission | null>(null);
  const [activeTab, setActiveTab] = useState('all');

  const handleMissionClick = (mission: Mission) => {
    setSelectedMission(mission);
    toast.success(`Exploring ${mission.name}`, {
      description: mission.achievement
    });
  };

  const filteredMissions = activeTab === 'all' 
    ? missions 
    : missions.filter(m => m.status === activeTab);

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
            Space Exploration
          </motion.h1>
          <p className="text-gray-300 max-w-2xl mx-auto">
            Discover humanity's greatest achievements in space exploration, from the first satellite to future missions
          </p>
        </div>

        {/* Tabs Filter */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-4 bg-cosmic-purple/30">
            <TabsTrigger value="all">All</TabsTrigger>
            <TabsTrigger value="completed">Completed</TabsTrigger>
            <TabsTrigger value="active">Active</TabsTrigger>
            <TabsTrigger value="planned">Planned</TabsTrigger>
          </TabsList>
        </Tabs>
      </motion.div>

      {/* Mission Timeline */}
      <div className="max-w-7xl mx-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
          {filteredMissions.map((mission, index) => {
            const Icon = getIcon(mission.type);
            return (
              <motion.div
                key={mission.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.98 }}
              >
                <Card
                  className="bg-cosmic-purple/20 border-stellar-gold/30 hover:border-stellar-gold/60 transition-all cursor-pointer h-full backdrop-blur-sm"
                  onClick={() => handleMissionClick(mission)}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between mb-3">
                      <motion.div
                        whileHover={{ rotate: 360 }}
                        transition={{ duration: 0.6 }}
                        className="p-3 bg-stellar-gold/20 rounded-lg"
                      >
                        <Icon className="w-6 h-6 text-stellar-gold" />
                      </motion.div>
                      <Badge 
                        variant={mission.status === 'active' ? 'default' : 'secondary'}
                        className={
                          mission.status === 'completed' 
                            ? 'bg-green-500/20 text-green-300' 
                            : mission.status === 'active'
                            ? 'bg-stellar-gold/20 text-stellar-gold'
                            : 'bg-blue-500/20 text-blue-300'
                        }
                      >
                        {mission.status}
                      </Badge>
                    </div>
                    <CardTitle className="text-pure-white">{mission.name}</CardTitle>
                    <CardDescription className="text-stellar-gold">
                      {mission.year}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-300 mb-3">{mission.description}</p>
                    <div className="pt-3 border-t border-stellar-gold/20">
                      <p className="text-sm text-stellar-gold/80">
                        🏆 {mission.achievement}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Floating Particles */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-stellar-gold rounded-full"
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: 0.3
            }}
            animate={{
              y: [null, Math.random() * window.innerHeight],
              opacity: [0.3, 0.8, 0.3]
            }}
            transition={{
              duration: 3 + Math.random() * 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        ))}
      </div>
    </div>
  );
}