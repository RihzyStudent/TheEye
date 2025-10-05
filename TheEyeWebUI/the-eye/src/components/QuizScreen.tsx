import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { ArrowLeft, CheckCircle2, XCircle, Trophy, Star } from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent } from './ui/card';
import { Progress } from './ui/progress';
import { toast } from 'sonner@2.0.3';

interface Question {
  id: number;
  question: string;
  options: string[];
  correctAnswer: number;
  explanation: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

const questions: Question[] = [
  {
    id: 1,
    question: 'What is the closest planet to the Sun?',
    options: ['Venus', 'Mercury', 'Mars', 'Earth'],
    correctAnswer: 1,
    explanation: 'Mercury is the closest planet to the Sun, orbiting at an average distance of about 58 million kilometers.',
    difficulty: 'easy'
  },
  {
    id: 2,
    question: 'How many moons does Mars have?',
    options: ['1', '2', '3', '0'],
    correctAnswer: 1,
    explanation: 'Mars has two small moons: Phobos and Deimos, both discovered in 1877 by Asaph Hall.',
    difficulty: 'medium'
  },
  {
    id: 3,
    question: 'What is a supernova?',
    options: [
      'A new star being born',
      'A planet collision',
      'The explosive death of a massive star',
      'A type of galaxy'
    ],
    correctAnswer: 2,
    explanation: 'A supernova is the explosive death of a massive star, releasing enormous amounts of energy and creating heavy elements.',
    difficulty: 'medium'
  },
  {
    id: 4,
    question: 'What is the Great Red Spot on Jupiter?',
    options: [
      'A volcanic eruption',
      'A giant storm',
      'An impact crater',
      'A mountain range'
    ],
    correctAnswer: 1,
    explanation: 'The Great Red Spot is a massive storm on Jupiter that has been raging for at least 300 years, large enough to fit two or three Earths.',
    difficulty: 'easy'
  },
  {
    id: 5,
    question: 'What causes the Northern Lights (Aurora Borealis)?',
    options: [
      'Reflection of ice crystals',
      'Solar particles interacting with Earth\'s atmosphere',
      'Lightning in the upper atmosphere',
      'Volcanic ash'
    ],
    correctAnswer: 1,
    explanation: 'The Northern Lights are caused by charged particles from the Sun colliding with gases in Earth\'s atmosphere, primarily oxygen and nitrogen.',
    difficulty: 'medium'
  },
  {
    id: 6,
    question: 'What is the nearest galaxy to the Milky Way?',
    options: ['Andromeda', 'Triangulum', 'Canis Major Dwarf', 'Whirlpool'],
    correctAnswer: 2,
    explanation: 'The Canis Major Dwarf Galaxy is the closest known galaxy to the Milky Way, though the Andromeda Galaxy is the nearest large spiral galaxy.',
    difficulty: 'hard'
  },
  {
    id: 7,
    question: 'How long does it take light from the Sun to reach Earth?',
    options: ['8 seconds', '8 minutes', '8 hours', '8 days'],
    correctAnswer: 1,
    explanation: 'Light from the Sun takes approximately 8 minutes and 20 seconds to reach Earth, traveling at about 300,000 km/s.',
    difficulty: 'easy'
  },
  {
    id: 8,
    question: 'What is the hottest planet in our solar system?',
    options: ['Mercury', 'Venus', 'Mars', 'Jupiter'],
    correctAnswer: 1,
    explanation: 'Venus is the hottest planet with surface temperatures around 465°C due to its thick atmosphere creating a runaway greenhouse effect.',
    difficulty: 'medium'
  }
];

interface QuizScreenProps {
  onBack: () => void;
}

export function QuizScreen({ onBack }: QuizScreenProps) {
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [showResult, setShowResult] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [streak, setStreak] = useState(0);

  const progress = ((currentQuestion + 1) / questions.length) * 100;
  const question = questions[currentQuestion];

  const handleAnswerSelect = (index: number) => {
    if (answered) return;
    
    setSelectedAnswer(index);
    setAnswered(true);

    const isCorrect = index === question.correctAnswer;
    
    if (isCorrect) {
      setScore(score + 1);
      setStreak(streak + 1);
      toast.success('Correct! 🎉', {
        description: question.explanation
      });
    } else {
      setStreak(0);
      toast.error('Incorrect', {
        description: question.explanation
      });
    }
  };

  const handleNext = () => {
    if (currentQuestion < questions.length - 1) {
      setCurrentQuestion(currentQuestion + 1);
      setSelectedAnswer(null);
      setAnswered(false);
    } else {
      setShowResult(true);
    }
  };

  const handleRestart = () => {
    setCurrentQuestion(0);
    setSelectedAnswer(null);
    setScore(0);
    setShowResult(false);
    setAnswered(false);
    setStreak(0);
  };

  const getScoreMessage = () => {
    const percentage = (score / questions.length) * 100;
    if (percentage === 100) return 'Perfect! You are a cosmic master! 🌟';
    if (percentage >= 80) return 'Excellent! You know your space! 🚀';
    if (percentage >= 60) return 'Good job! Keep learning! 🌙';
    if (percentage >= 40) return 'Not bad! Keep exploring! ⭐';
    return 'Keep studying the cosmos! 🌍';
  };

  if (showResult) {
    return (
      <div className="min-h-screen cosmic-bg p-6 flex items-center justify-center">
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="max-w-2xl w-full"
        >
          <Card className="bg-cosmic-purple/30 border-stellar-gold backdrop-blur-sm">
            <CardContent className="p-8 text-center">
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: "spring", delay: 0.2 }}
                className="mb-6"
              >
                <Trophy className="w-24 h-24 text-stellar-gold mx-auto" />
              </motion.div>

              <h2 className="text-pure-white mb-2">Quiz Complete!</h2>
              <p className="text-gray-300 mb-6">{getScoreMessage()}</p>

              <div className="mb-8">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.3 }}
                  className="text-6xl font-bold text-stellar-gold mb-2"
                >
                  {score}/{questions.length}
                </motion.div>
                <p className="text-gray-300">
                  {Math.round((score / questions.length) * 100)}% Correct
                </p>
              </div>

              <div className="flex gap-4 justify-center">
                <Button
                  onClick={handleRestart}
                  className="bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400"
                >
                  Try Again
                </Button>
                <Button
                  onClick={onBack}
                  variant="outline"
                  className="border-stellar-gold text-stellar-gold hover:bg-stellar-gold/10"
                >
                  Back to Dashboard
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Celebration particles */}
          <div className="fixed inset-0 pointer-events-none">
            {[...Array(50)].map((_, i) => (
              <motion.div
                key={i}
                className="absolute"
                initial={{
                  x: window.innerWidth / 2,
                  y: window.innerHeight / 2,
                  opacity: 1
                }}
                animate={{
                  x: Math.random() * window.innerWidth,
                  y: Math.random() * window.innerHeight,
                  opacity: 0
                }}
                transition={{
                  duration: 2,
                  delay: i * 0.02,
                  ease: "easeOut"
                }}
              >
                <Star className="w-4 h-4 text-stellar-gold" />
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen cosmic-bg p-6 overflow-y-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-3xl mx-auto mb-8"
      >
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-stellar-gold hover:text-yellow-400 transition-colors mb-6"
        >
          <ArrowLeft className="w-5 h-5" />
          Back to Dashboard
        </button>

        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-gray-300">
              Question {currentQuestion + 1} of {questions.length}
            </span>
            <div className="flex items-center gap-4">
              {streak > 0 && (
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  className="flex items-center gap-1 text-stellar-gold"
                >
                  <span className="text-sm">🔥 Streak: {streak}</span>
                </motion.div>
              )}
              <span className="text-stellar-gold">
                Score: {score}
              </span>
            </div>
          </div>
          <Progress value={progress} className="h-2" />
        </div>
      </motion.div>

      {/* Question Card */}
      <div className="max-w-3xl mx-auto">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentQuestion}
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ type: "spring", stiffness: 300, damping: 30 }}
          >
            <Card className="bg-cosmic-purple/30 border-stellar-gold/50 backdrop-blur-sm mb-6">
              <CardContent className="p-8">
                <div className="mb-6">
                  <div className="flex items-center gap-2 mb-4">
                    <span className={`px-3 py-1 rounded-full text-sm ${
                      question.difficulty === 'easy' 
                        ? 'bg-green-500/20 text-green-300'
                        : question.difficulty === 'medium'
                        ? 'bg-yellow-500/20 text-yellow-300'
                        : 'bg-red-500/20 text-red-300'
                    }`}>
                      {question.difficulty}
                    </span>
                  </div>
                  <h2 className="text-pure-white">
                    {question.question}
                  </h2>
                </div>

                <div className="space-y-3">
                  {question.options.map((option, index) => {
                    const isSelected = selectedAnswer === index;
                    const isCorrect = index === question.correctAnswer;
                    const showCorrect = answered && isCorrect;
                    const showIncorrect = answered && isSelected && !isCorrect;

                    return (
                      <motion.button
                        key={index}
                        whileHover={!answered ? { scale: 1.02 } : {}}
                        whileTap={!answered ? { scale: 0.98 } : {}}
                        onClick={() => handleAnswerSelect(index)}
                        disabled={answered}
                        className={`w-full p-4 rounded-lg text-left transition-all border-2 ${
                          showCorrect
                            ? 'bg-green-500/20 border-green-500 text-green-300'
                            : showIncorrect
                            ? 'bg-red-500/20 border-red-500 text-red-300'
                            : isSelected
                            ? 'bg-stellar-gold/20 border-stellar-gold text-pure-white'
                            : 'bg-cosmic-purple/20 border-stellar-gold/30 text-gray-300 hover:border-stellar-gold/60'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span>{option}</span>
                          {showCorrect && <CheckCircle2 className="w-5 h-5" />}
                          {showIncorrect && <XCircle className="w-5 h-5" />}
                        </div>
                      </motion.button>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {answered && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="flex justify-center"
              >
                <Button
                  onClick={handleNext}
                  className="bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400"
                  size="lg"
                >
                  {currentQuestion < questions.length - 1 ? 'Next Question' : 'See Results'}
                </Button>
              </motion.div>
            )}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Floating particles */}
      <div className="fixed inset-0 pointer-events-none z-0">
        {[...Array(15)].map((_, i) => (
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
              opacity: [0.3, 1, 0.3]
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