import { motion } from 'motion/react';
import { ChevronRight, Rocket, Star, Globe } from 'lucide-react';
import { Button } from './ui/button';
import { useState } from 'react';

interface OnboardingScreenProps {
  onComplete: () => void;
}

const onboardingSteps = [
  {
    icon: <Globe className="w-16 h-16 text-stellar-gold" />,
    title: "Explore the Solar System",
    description: "Discover the planets, their moons, and unique characteristics in an interactive experience."
  },
  {
    icon: <Star className="w-16 h-16 text-stellar-gold" />,
    title: "Navigate the Constellations",
    description: "Learn about constellations and the stories behind the brightest stars."
  },
  {
    icon: <Rocket className="w-16 h-16 text-stellar-gold" />,
    title: "Space Missions",
    description: "Discover the missions that have taken humanity beyond our planet."
  }
];

export function OnboardingScreen({ onComplete }: OnboardingScreenProps) {
  const [currentStep, setCurrentStep] = useState(0);

  const nextStep = () => {
    if (currentStep < onboardingSteps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete();
    }
  };

  const currentStepData = onboardingSteps[currentStep];

  return (
    <div className="min-h-screen cosmic-bg flex items-center justify-center p-6">
      <div className="max-w-md w-full text-center">
        <motion.div
          key={currentStep}
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -50 }}
          transition={{ duration: 0.5 }}
          className="mb-8"
        >
          <div className="mb-6 floating-animation">
            {currentStepData.icon}
          </div>
          
          <h2 className="text-2xl font-bold text-pure-white mb-4">
            {currentStepData.title}
          </h2>
          
          <p className="text-gray-300 mb-8 leading-relaxed">
            {currentStepData.description}
          </p>
        </motion.div>

        <div className="flex justify-center mb-8 space-x-2">
          {onboardingSteps.map((_, index) => (
            <div
              key={index}
              className={`w-3 h-3 rounded-full transition-all duration-300 ${
                index === currentStep ? 'bg-stellar-gold' : 'bg-gray-600'
              }`}
            />
          ))}
        </div>

        <Button
          onClick={nextStep}
          className="w-full bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400 transition-all group"
        >
          {currentStep === onboardingSteps.length - 1 ? 'Get Started' : 'Next'}
          <ChevronRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
        </Button>

        {currentStep > 0 && (
          <button
            onClick={() => setCurrentStep(currentStep - 1)}
            className="mt-4 text-gray-400 hover:text-pure-white transition-colors"
          >
            Previous
          </button>
        )}
      </div>
    </div>
  );
}