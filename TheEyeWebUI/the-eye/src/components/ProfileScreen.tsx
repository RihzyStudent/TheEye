import { useState } from 'react';
import { motion } from 'motion/react';
import { ArrowLeft, Trophy, Star, Target, Award, BookOpen, Rocket, Zap } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Progress } from './ui/progress';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Avatar, AvatarFallback } from './ui/avatar';

interface Achievement {
  id: string;
  name: string;
  description: string;
  icon: any;
  unlocked: boolean;
  progress: number;
  maxProgress: number;
  rarity: 'common' | 'rare' | 'legendary';
}

interface LearningStats {
  screensVisited: number;
  quizzesTaken: number;
  correctAnswers: number;
  totalQuestions: number;
  constellationsCompleted: number;
  planetsExplored: number;
  phenomenaDiscovered: number;
  missionsLearned: number;
}

const achievements: Achievement[] = [
  {
    id: '1',
    name: 'First Steps',
    description: 'Complete the onboarding tutorial',
    icon: Rocket,
    unlocked: true,
    progress: 1,
    maxProgress: 1,
    rarity: 'common'
  },
  {
    id: '2',
    name: 'Solar Explorer',
    description: 'Visit all planets in the solar system',
    icon: Target,
    unlocked: true,
    progress: 8,
    maxProgress: 8,
    rarity: 'rare'
  },
  {
    id: '3',
    name: 'Constellation Master',
    description: 'Complete 5 constellations',
    icon: Star,
    unlocked: false,
    progress: 3,
    maxProgress: 5,
    rarity: 'rare'
  },
  {
    id: '4',
    name: 'Quiz Champion',
    description: 'Get 100% on a cosmic quiz',
    icon: Trophy,
    unlocked: false,
    progress: 0,
    maxProgress: 1,
    rarity: 'legendary'
  },
  {
    id: '5',
    name: 'Knowledge Seeker',
    description: 'Discover all cosmic phenomena',
    icon: BookOpen,
    unlocked: false,
    progress: 4,
    maxProgress: 6,
    rarity: 'rare'
  },
  {
    id: '6',
    name: 'Space Pioneer',
    description: 'Learn about 5 space missions',
    icon: Rocket,
    unlocked: true,
    progress: 5,
    maxProgress: 5,
    rarity: 'common'
  },
  {
    id: '7',
    name: 'Perfect Streak',
    description: 'Answer 5 quiz questions correctly in a row',
    icon: Zap,
    unlocked: false,
    progress: 2,
    maxProgress: 5,
    rarity: 'legendary'
  },
  {
    id: '8',
    name: 'Cosmic Scholar',
    description: 'Visit all sections of the app',
    icon: Award,
    unlocked: true,
    progress: 5,
    maxProgress: 5,
    rarity: 'rare'
  }
];

const learningStats: LearningStats = {
  screensVisited: 5,
  quizzesTaken: 3,
  correctAnswers: 18,
  totalQuestions: 24,
  constellationsCompleted: 3,
  planetsExplored: 8,
  phenomenaDiscovered: 4,
  missionsLearned: 5
};

const getRarityColor = (rarity: Achievement['rarity']) => {
  switch (rarity) {
    case 'common':
      return 'bg-gray-500/20 text-gray-300';
    case 'rare':
      return 'bg-blue-500/20 text-blue-300';
    case 'legendary':
      return 'bg-purple-500/20 text-purple-300';
  }
};

const getRarityGlow = (rarity: Achievement['rarity']) => {
  switch (rarity) {
    case 'common':
      return '#9CA3AF';
    case 'rare':
      return '#3B82F6';
    case 'legendary':
      return '#A855F7';
  }
};

interface ProfileScreenProps {
  onBack: () => void;
}

export function ProfileScreen({ onBack }: ProfileScreenProps) {
  const [activeTab, setActiveTab] = useState('overview');

  const totalAchievements = achievements.length;
  const unlockedAchievements = achievements.filter(a => a.unlocked).length;
  const overallProgress = (unlockedAchievements / totalAchievements) * 100;
  const quizAccuracy = Math.round((learningStats.correctAnswers / learningStats.totalQuestions) * 100);

  return (
    <div className="min-h-screen cosmic-bg p-6 overflow-y-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-6xl mx-auto mb-8"
      >
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-stellar-gold hover:text-yellow-400 transition-colors mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Dashboard
        </button>

        {/* Profile Header */}
        <Card className="bg-cosmic-purple/30 border-stellar-gold/50 backdrop-blur-sm mb-6">
          <CardContent className="p-6">
            <div className="flex items-center gap-6">
              <motion.div
                whileHover={{ scale: 1.05 }}
                className="relative"
              >
                <Avatar className="w-24 h-24 border-4 border-stellar-gold">
                  <AvatarFallback className="bg-stellar-gold text-cosmic-deep-blue text-2xl">
                    CS
                  </AvatarFallback>
                </Avatar>
                <motion.div
                  className="absolute -top-1 -right-1 bg-stellar-gold rounded-full p-2"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                >
                  <Star className="w-4 h-4 text-cosmic-deep-blue" />
                </motion.div>
              </motion.div>

              <div className="flex-1">
                <h2 className="text-pure-white mb-1">Cosmic Scholar</h2>
                <p className="text-gray-300 mb-3">Space Explorer • Level 5</p>
                <div className="flex gap-4">
                  <div className="text-center">
                    <div className="text-2xl text-stellar-gold">{unlockedAchievements}</div>
                    <div className="text-sm text-gray-400">Achievements</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl text-stellar-gold">{quizAccuracy}%</div>
                    <div className="text-sm text-gray-400">Quiz Accuracy</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl text-stellar-gold">{learningStats.screensVisited}</div>
                    <div className="text-sm text-gray-400">Sections Visited</div>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6">
              <div className="flex justify-between mb-2">
                <span className="text-sm text-gray-300">Overall Progress</span>
                <span className="text-sm text-stellar-gold">{Math.round(overallProgress)}%</span>
              </div>
              <Progress value={overallProgress} className="h-2" />
            </div>
          </CardContent>
        </Card>

        {/* Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full max-w-md mx-auto grid-cols-2 bg-cosmic-purple/30">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="achievements">Achievements</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
              >
                <Card className="bg-cosmic-purple/20 border-stellar-gold/30 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-pure-white flex items-center gap-2">
                      <Rocket className="w-5 h-5 text-stellar-gold" />
                      Missions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl text-stellar-gold mb-1">
                      {learningStats.missionsLearned}
                    </div>
                    <p className="text-sm text-gray-400">Missions Learned</p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
              >
                <Card className="bg-cosmic-purple/20 border-stellar-gold/30 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-pure-white flex items-center gap-2">
                      <Target className="w-5 h-5 text-stellar-gold" />
                      Planets
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl text-stellar-gold mb-1">
                      {learningStats.planetsExplored}
                    </div>
                    <p className="text-sm text-gray-400">Planets Explored</p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
              >
                <Card className="bg-cosmic-purple/20 border-stellar-gold/30 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-pure-white flex items-center gap-2">
                      <Star className="w-5 h-5 text-stellar-gold" />
                      Constellations
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl text-stellar-gold mb-1">
                      {learningStats.constellationsCompleted}
                    </div>
                    <p className="text-sm text-gray-400">Constellations Formed</p>
                  </CardContent>
                </Card>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <Card className="bg-cosmic-purple/20 border-stellar-gold/30 backdrop-blur-sm">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-lg text-pure-white flex items-center gap-2">
                      <Zap className="w-5 h-5 text-stellar-gold" />
                      Phenomena
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl text-stellar-gold mb-1">
                      {learningStats.phenomenaDiscovered}
                    </div>
                    <p className="text-sm text-gray-400">Phenomena Discovered</p>
                  </CardContent>
                </Card>
              </motion.div>
            </div>

            {/* Quiz Statistics */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-6"
            >
              <Card className="bg-cosmic-purple/30 border-stellar-gold/50 backdrop-blur-sm">
                <CardHeader>
                  <CardTitle className="text-pure-white flex items-center gap-2">
                    <Trophy className="w-6 h-6 text-stellar-gold" />
                    Quiz Performance
                  </CardTitle>
                  <CardDescription className="text-gray-300">
                    Your knowledge testing statistics
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div>
                      <div className="text-sm text-gray-400 mb-2">Quizzes Taken</div>
                      <div className="text-3xl text-pure-white">{learningStats.quizzesTaken}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400 mb-2">Correct Answers</div>
                      <div className="text-3xl text-green-400">
                        {learningStats.correctAnswers}/{learningStats.totalQuestions}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400 mb-2">Accuracy Rate</div>
                      <div className="text-3xl text-stellar-gold">{quizAccuracy}%</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </TabsContent>

          {/* Achievements Tab */}
          <TabsContent value="achievements" className="mt-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {achievements.map((achievement, index) => {
                const Icon = achievement.icon;
                const glowColor = getRarityGlow(achievement.rarity);
                
                return (
                  <motion.div
                    key={achievement.id}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    whileHover={{ scale: achievement.unlocked ? 1.05 : 1 }}
                  >
                    <Card 
                      className={`${
                        achievement.unlocked
                          ? 'bg-cosmic-purple/30 border-stellar-gold/50'
                          : 'bg-cosmic-purple/10 border-stellar-gold/20 opacity-60'
                      } backdrop-blur-sm transition-all h-full relative overflow-hidden`}
                    >
                      {achievement.unlocked && (
                        <motion.div
                          className="absolute inset-0 opacity-10"
                          style={{ background: `radial-gradient(circle at 50% 50%, ${glowColor}, transparent 70%)` }}
                          animate={{ scale: [1, 1.2, 1] }}
                          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
                        />
                      )}

                      <CardHeader>
                        <div className="flex items-start justify-between mb-3">
                          <motion.div
                            animate={achievement.unlocked ? { rotate: [0, 10, -10, 0] } : {}}
                            transition={{ duration: 2, repeat: Infinity }}
                            className={`p-3 rounded-lg ${
                              achievement.unlocked ? 'bg-stellar-gold/20' : 'bg-gray-500/10'
                            }`}
                          >
                            <Icon 
                              className={`w-6 h-6 ${
                                achievement.unlocked ? 'text-stellar-gold' : 'text-gray-500'
                              }`} 
                            />
                          </motion.div>
                          <Badge className={getRarityColor(achievement.rarity)}>
                            {achievement.rarity}
                          </Badge>
                        </div>
                        <CardTitle className={`${
                          achievement.unlocked ? 'text-pure-white' : 'text-gray-500'
                        }`}>
                          {achievement.name}
                        </CardTitle>
                        <CardDescription className="text-gray-400">
                          {achievement.description}
                        </CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="space-y-2">
                          <div className="flex justify-between text-sm">
                            <span className="text-gray-400">Progress</span>
                            <span className={achievement.unlocked ? 'text-stellar-gold' : 'text-gray-500'}>
                              {achievement.progress}/{achievement.maxProgress}
                            </span>
                          </div>
                          <Progress 
                            value={(achievement.progress / achievement.maxProgress) * 100} 
                            className="h-1.5"
                          />
                        </div>
                      </CardContent>
                    </Card>
                  </motion.div>
                );
              })}
            </div>
          </TabsContent>
        </Tabs>
      </motion.div>

      {/* Floating achievement particles */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            initial={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
              opacity: 0.2
            }}
            animate={{
              y: [null, Math.random() * window.innerHeight],
              opacity: [0.2, 0.6, 0.2]
            }}
            transition={{
              duration: 4 + Math.random() * 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <Trophy className="w-3 h-3 text-stellar-gold" />
          </motion.div>
        ))}
      </div>
    </div>
  );
}