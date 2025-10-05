import { motion } from 'motion/react';
import { useState, useRef } from 'react';
import { 
  Upload, 
  Send, 
  Brain, 
  Sparkles, 
  FileText, 
  Download,
  Database,
  Settings,
  TrendingUp,
  Globe2,
  Activity,
  BarChart3,
  RefreshCw,
  Info,
  Sliders,
  Plus
} from 'lucide-react';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from './ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { toast } from 'sonner';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Slider } from './ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Switch } from './ui/switch';
import { Badge } from './ui/badge';

interface ExoplanetDetectionScreenProps {
  onBack: () => void;
}

interface ModelStats {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  totalSamples: number;
  confirmedExoplanets: number;
  falsePositives: number;
  lastTrained: string;
  modelVersion: string;
}

interface ExoplanetData {
  orbitalPeriod: string;
  transitDuration: string;
  planetaryRadius: string;
  transitDepth: string;
  stellarRadius: string;
  stellarMagnitude: string;
  temperature: string;
}

export function ExoplanetDetectionScreen({ onBack }: ExoplanetDetectionScreenProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // Model parameters
  const [learningRate, setLearningRate] = useState([0.001]);
  const [epochs, setEpochs] = useState([100]);
  const [batchSize, setBatchSize] = useState([32]);
  const [selectedDataset, setSelectedDataset] = useState('kepler');
  const [enablePreprocessing, setEnablePreprocessing] = useState(true);
  const [removeFalsePositives, setRemoveFalsePositives] = useState(true);
  
  // Manual entry form
  const [manualData, setManualData] = useState<ExoplanetData>({
    orbitalPeriod: '',
    transitDuration: '',
    planetaryRadius: '',
    transitDepth: '',
    stellarRadius: '',
    stellarMagnitude: '',
    temperature: ''
  });

  // Model statistics (mock data - will be updated from your backend)
  const [modelStats, setModelStats] = useState<ModelStats>({
    accuracy: 0.952,
    precision: 0.943,
    recall: 0.968,
    f1Score: 0.955,
    totalSamples: 15420,
    confirmedExoplanets: 3847,
    falsePositives: 234,
    lastTrained: 'Not trained yet',
    modelVersion: 'v1.0.0'
  });

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setSelectedFile(file);
      toast.success(`Dataset "${file.name}" loaded successfully`);
    }
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      setSelectedFile(file);
      toast.success(`Dataset "${file.name}" loaded successfully`);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const classifyData = async () => {
    setIsProcessing(true);
    setProgress(0);
    
    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 10, 90));
      }, 200);

      // TODO: Replace with actual API call to your exoplanet detection model
      /*
      const formData = new FormData();
      if (selectedFile) {
        formData.append('dataset', selectedFile);
      } else {
        formData.append('data', JSON.stringify(manualData));
      }
      formData.append('preprocessing', enablePreprocessing.toString());
      formData.append('removeFalsePositives', removeFalsePositives.toString());
      
      const response = await fetch('YOUR_API_ENDPOINT/classify', {
        method: 'POST',
        body: formData,
        headers: {
          'Authorization': `Bearer ${YOUR_API_KEY}`,
        }
      });
      
      const data = await response.json();
      setResult(data);
      */

      // Mock response for demonstration
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const mockResult = {
        classification: 'CONFIRMED EXOPLANET',
        confidence: 0.947,
        planetType: 'Hot Jupiter',
        details: {
          orbitalPeriod: manualData.orbitalPeriod || '3.52 days',
          transitDuration: manualData.transitDuration || '2.8 hours',
          planetaryRadius: manualData.planetaryRadius || '1.2 Jupiter radii',
          estimatedMass: '0.89 Jupiter masses',
          distanceFromStar: '0.048 AU',
          equilibriumTemp: manualData.temperature || '1450 K',
          stellarType: 'G-type main-sequence',
          hostStarTemp: '5778 K'
        },
        features: [
          'Transit signature detected',
          'Periodic dimming pattern',
          'Doppler shift confirmed',
          'Low false positive probability'
        ],
        similarExoplanets: [
          'HD 209458 b',
          '51 Pegasi b',
          'WASP-12b'
        ],
        dataQuality: 'High'
      };
      
      setResult(mockResult);
      setProgress(100);
      clearInterval(progressInterval);
      toast.success('Classification complete!');
      
    } catch (error) {
      console.error('Classification error:', error);
      toast.error('Classification failed. Please check your backend connection.');
    } finally {
      setIsProcessing(false);
    }
  };

  const trainModel = async () => {
    setIsTraining(true);
    setProgress(0);
    
    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 5, 95));
      }, 500);

      // TODO: Replace with actual training API call
      /*
      const response = await fetch('YOUR_API_ENDPOINT/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${YOUR_API_KEY}`,
        },
        body: JSON.stringify({
          dataset: selectedDataset,
          learningRate: learningRate[0],
          epochs: epochs[0],
          batchSize: batchSize[0],
          preprocessing: enablePreprocessing,
          removeFalsePositives: removeFalsePositives
        })
      });
      
      const data = await response.json();
      setModelStats(data.stats);
      */

      // Mock training simulation
      await new Promise(resolve => setTimeout(resolve, 5000));
      
      setModelStats({
        accuracy: 0.952 + Math.random() * 0.03,
        precision: 0.943 + Math.random() * 0.03,
        recall: 0.968 + Math.random() * 0.02,
        f1Score: 0.955 + Math.random() * 0.03,
        totalSamples: 15420,
        confirmedExoplanets: 3847,
        falsePositives: Math.floor(200 + Math.random() * 50),
        lastTrained: new Date().toLocaleString(),
        modelVersion: 'v1.0.0'
      });
      
      setProgress(100);
      clearInterval(progressInterval);
      toast.success('Model training completed successfully!');
      
    } catch (error) {
      console.error('Training error:', error);
      toast.error('Training failed. Please check your backend connection.');
    } finally {
      setIsTraining(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setResult(null);
    setProgress(0);
    setManualData({
      orbitalPeriod: '',
      transitDuration: '',
      planetaryRadius: '',
      transitDepth: '',
      stellarRadius: '',
      stellarMagnitude: '',
      temperature: ''
    });
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const updateManualData = (field: keyof ExoplanetData, value: string) => {
    setManualData(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="min-h-screen cosmic-bg relative overflow-hidden">
      {/* Animated Background Particles */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-1 h-1 bg-stellar-gold/30 rounded-full"
            initial={{ 
              x: Math.random() * window.innerWidth, 
              y: Math.random() * window.innerHeight 
            }}
            animate={{
              x: Math.random() * window.innerWidth,
              y: Math.random() * window.innerHeight,
            }}
            transition={{
              duration: 20 + Math.random() * 10,
              repeat: Infinity,
              repeatType: 'reverse'
            }}
          />
        ))}
      </div>

      {/* Header */}
      <div className="flex items-center justify-between p-6 relative z-10">
        <div className="flex items-center gap-3">
          <Globe2 className="w-8 h-8 text-stellar-gold" />
          <div>
            <h1 className="text-2xl font-bold text-pure-white">The Eye</h1>
            <p className="text-sm text-gray-300">AI-Powered Exoplanet Detection</p>
          </div>
        </div>
        
        <Button 
          variant="ghost" 
          size="sm"
          onClick={handleReset}
          className="text-pure-white hover:bg-white/10"
        >
          Reset
        </Button>
      </div>

      <div className="max-w-7xl mx-auto p-6 space-y-6 relative z-10">
        {/* Model Statistics Dashboard */}
        <Card className="bg-cosmic-deep-blue/80 backdrop-blur-sm border-stellar-gold/30">
          <CardHeader>
            <CardTitle className="text-pure-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-stellar-gold" />
              Model Performance Statistics
            </CardTitle>
            <CardDescription className="text-gray-300">
              Current model: {modelStats.modelVersion} | Last trained: {modelStats.lastTrained}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid md:grid-cols-4 gap-4">
              <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">Accuracy</span>
                  <TrendingUp className="w-4 h-4 text-green-400" />
                </div>
                <p className="text-3xl font-bold text-stellar-gold">
                  {(modelStats.accuracy * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">Precision</span>
                  <Activity className="w-4 h-4 text-blue-400" />
                </div>
                <p className="text-3xl font-bold text-pure-white">
                  {(modelStats.precision * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">Recall</span>
                  <Activity className="w-4 h-4 text-purple-400" />
                </div>
                <p className="text-3xl font-bold text-pure-white">
                  {(modelStats.recall * 100).toFixed(1)}%
                </p>
              </div>
              
              <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-gray-400 text-sm">F1 Score</span>
                  <BarChart3 className="w-4 h-4 text-cyan-400" />
                </div>
                <p className="text-3xl font-bold text-pure-white">
                  {(modelStats.f1Score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
            
            <div className="mt-4 grid md:grid-cols-3 gap-4">
              <div className="bg-black/20 rounded p-3 text-center">
                <p className="text-gray-400 text-sm">Total Samples</p>
                <p className="text-xl font-bold text-pure-white">{modelStats.totalSamples.toLocaleString()}</p>
              </div>
              <div className="bg-black/20 rounded p-3 text-center">
                <p className="text-gray-400 text-sm">Confirmed Exoplanets</p>
                <p className="text-xl font-bold text-green-400">{modelStats.confirmedExoplanets.toLocaleString()}</p>
              </div>
              <div className="bg-black/20 rounded p-3 text-center">
                <p className="text-gray-400 text-sm">False Positives</p>
                <p className="text-xl font-bold text-red-400">{modelStats.falsePositives}</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="grid lg:grid-cols-3 gap-6">
          {/* Main Content - Left Column */}
          <div className="lg:col-span-2 space-y-6">
            <Tabs defaultValue="upload" className="w-full">
              <TabsList className="grid w-full grid-cols-2 bg-white/10">
                <TabsTrigger value="upload" className="data-[state=active]:bg-stellar-gold data-[state=active]:text-cosmic-deep-blue">
                  <Database className="w-4 h-4 mr-2" />
                  Dataset Upload
                </TabsTrigger>
                <TabsTrigger value="manual" className="data-[state=active]:bg-stellar-gold data-[state=active]:text-cosmic-deep-blue">
                  <Plus className="w-4 h-4 mr-2" />
                  Manual Entry
                </TabsTrigger>
              </TabsList>

              {/* Dataset Upload Tab */}
              <TabsContent value="upload" className="space-y-4">
                <Card className="bg-cosmic-deep-blue/50 border-stellar-gold/30">
                  <CardHeader>
                    <CardTitle className="text-pure-white flex items-center gap-2">
                      <Upload className="w-5 h-5 text-stellar-gold" />
                      Upload Exoplanet Dataset
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                      Upload CSV/JSON files from Kepler, K2, TESS, or custom datasets
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label className="text-gray-300 mb-2 block">Select Dataset Source</Label>
                      <Select value={selectedDataset} onValueChange={setSelectedDataset}>
                        <SelectTrigger className="bg-black/30 border-stellar-gold/30 text-pure-white">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="kepler">NASA Kepler Mission</SelectItem>
                          <SelectItem value="k2">NASA K2 Mission</SelectItem>
                          <SelectItem value="tess">NASA TESS Mission</SelectItem>
                          <SelectItem value="custom">Custom Dataset</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {/* Drag & Drop Zone */}
                    <div
                      onDrop={handleDrop}
                      onDragOver={handleDragOver}
                      className="border-2 border-dashed border-stellar-gold/50 rounded-lg p-8 text-center cursor-pointer hover:border-stellar-gold transition-colors bg-black/20"
                      onClick={() => fileInputRef.current?.click()}
                    >
                      {selectedFile ? (
                        <div className="space-y-2">
                          <FileText className="w-16 h-16 text-stellar-gold mx-auto" />
                          <p className="text-stellar-gold">{selectedFile.name}</p>
                          <p className="text-gray-400 text-sm">
                            {(selectedFile.size / 1024).toFixed(2)} KB
                          </p>
                          <Badge className="bg-green-500/20 text-green-400 border-green-500">
                            Ready for processing
                          </Badge>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          <Upload className="w-16 h-16 text-stellar-gold mx-auto" />
                          <p className="text-pure-white">Drag & drop dataset file</p>
                          <p className="text-gray-400 text-sm">Supports CSV, JSON formats</p>
                        </div>
                      )}
                    </div>
                    
                    <input
                      ref={fileInputRef}
                      type="file"
                      onChange={handleFileSelect}
                      className="hidden"
                      accept=".csv,.json,.txt"
                    />
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Manual Entry Tab */}
              <TabsContent value="manual" className="space-y-4">
                <Card className="bg-cosmic-deep-blue/50 border-stellar-gold/30">
                  <CardHeader>
                    <CardTitle className="text-pure-white flex items-center gap-2">
                      <Sparkles className="w-5 h-5 text-stellar-gold" />
                      Manual Data Entry
                    </CardTitle>
                    <CardDescription className="text-gray-300">
                      Enter exoplanet parameters for individual classification
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid md:grid-cols-2 gap-4">
                      <div>
                        <Label className="text-gray-300">Orbital Period (days)</Label>
                        <Input
                          type="number"
                          step="0.001"
                          placeholder="e.g., 3.52"
                          value={manualData.orbitalPeriod}
                          onChange={(e) => updateManualData('orbitalPeriod', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Transit Duration (hours)</Label>
                        <Input
                          type="number"
                          step="0.1"
                          placeholder="e.g., 2.8"
                          value={manualData.transitDuration}
                          onChange={(e) => updateManualData('transitDuration', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Planetary Radius (R⊕)</Label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="e.g., 1.2"
                          value={manualData.planetaryRadius}
                          onChange={(e) => updateManualData('planetaryRadius', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Transit Depth (%)</Label>
                        <Input
                          type="number"
                          step="0.001"
                          placeholder="e.g., 0.015"
                          value={manualData.transitDepth}
                          onChange={(e) => updateManualData('transitDepth', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Stellar Radius (R☉)</Label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="e.g., 1.0"
                          value={manualData.stellarRadius}
                          onChange={(e) => updateManualData('stellarRadius', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Stellar Magnitude</Label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="e.g., 5.5"
                          value={manualData.stellarMagnitude}
                          onChange={(e) => updateManualData('stellarMagnitude', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div className="md:col-span-2">
                        <Label className="text-gray-300">Equilibrium Temperature (K)</Label>
                        <Input
                          type="number"
                          step="1"
                          placeholder="e.g., 1450"
                          value={manualData.temperature}
                          onChange={(e) => updateManualData('temperature', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* Classification Button */}
            <Card className="bg-gradient-to-r from-cosmic-purple to-cosmic-deep-blue border-stellar-gold">
              <CardContent className="p-6">
                <Button
                  onClick={classifyData}
                  disabled={isProcessing || isTraining}
                  className="w-full bg-stellar-gold text-cosmic-deep-blue hover:bg-yellow-400 transition-all disabled:opacity-50 h-14 text-lg"
                >
                  {isProcessing ? (
                    <>
                      <Brain className="w-5 h-5 mr-2 animate-pulse" />
                      Classifying...
                    </>
                  ) : (
                    <>
                      <Send className="w-5 h-5 mr-2" />
                      Classify Exoplanet
                    </>
                  )}
                </Button>

                {isProcessing && (
                  <div className="mt-4 space-y-2">
                    <Progress value={progress} className="h-2" />
                    <p className="text-center text-sm text-gray-300">
                      Analyzing data... {progress}%
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Right Column - Model Settings */}
          <div className="space-y-6">
            <Card className="bg-cosmic-deep-blue/80 backdrop-blur-sm border-stellar-gold/30">
              <CardHeader>
                <CardTitle className="text-pure-white flex items-center gap-2">
                  <Sliders className="w-5 h-5 text-stellar-gold" />
                  Hyperparameters
                </CardTitle>
                <CardDescription className="text-gray-300">
                  Adjust model training parameters
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300">Learning Rate</Label>
                    <span className="text-stellar-gold text-sm">{learningRate[0]}</span>
                  </div>
                  <Slider
                    value={learningRate}
                    onValueChange={setLearningRate}
                    min={0.0001}
                    max={0.01}
                    step={0.0001}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300">Epochs</Label>
                    <span className="text-stellar-gold text-sm">{epochs[0]}</span>
                  </div>
                  <Slider
                    value={epochs}
                    onValueChange={setEpochs}
                    min={10}
                    max={500}
                    step={10}
                    className="w-full"
                  />
                </div>

                <div>
                  <div className="flex justify-between mb-2">
                    <Label className="text-gray-300">Batch Size</Label>
                    <span className="text-stellar-gold text-sm">{batchSize[0]}</span>
                  </div>
                  <Slider
                    value={batchSize}
                    onValueChange={setBatchSize}
                    min={8}
                    max={128}
                    step={8}
                    className="w-full"
                  />
                </div>

                <div className="space-y-3 pt-4 border-t border-stellar-gold/20">
                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Data Preprocessing</Label>
                    <Switch
                      checked={enablePreprocessing}
                      onCheckedChange={setEnablePreprocessing}
                    />
                  </div>

                  <div className="flex items-center justify-between">
                    <Label className="text-gray-300">Remove False Positives</Label>
                    <Switch
                      checked={removeFalsePositives}
                      onCheckedChange={setRemoveFalsePositives}
                    />
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Train Model Button */}
            <Card className="bg-gradient-to-br from-stellar-gold/20 to-cosmic-purple/20 border-stellar-gold">
              <CardContent className="p-6">
                <Button
                  onClick={trainModel}
                  disabled={isTraining || isProcessing}
                  className="w-full bg-cosmic-purple text-pure-white hover:bg-cosmic-purple/80 h-12"
                >
                  {isTraining ? (
                    <>
                      <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                      Training Model...
                    </>
                  ) : (
                    <>
                      <Settings className="w-5 h-5 mr-2" />
                      Train/Retrain Model
                    </>
                  )}
                </Button>

                {isTraining && (
                  <div className="mt-4 space-y-2">
                    <Progress value={progress} className="h-2" />
                    <p className="text-center text-sm text-gray-300">
                      Training in progress... {progress}%
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>

            {/* Info Card */}
            <Card className="bg-white/5 border-stellar-gold/20">
              <CardHeader>
                <CardTitle className="text-pure-white text-sm flex items-center gap-2">
                  <Info className="w-4 h-4 text-stellar-gold" />
                  About This Tool
                </CardTitle>
              </CardHeader>
              <CardContent className="text-gray-300 text-sm space-y-2">
                <p>
                  This AI/ML tool helps classify exoplanet candidates from transit data.
                </p>
                <p>
                  Trained on NASA's Kepler, K2, and TESS mission datasets to identify confirmed exoplanets.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Results Section */}
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="bg-cosmic-deep-blue border-stellar-gold shadow-2xl shadow-stellar-gold/20">
              <CardHeader>
                <CardTitle className="text-pure-white flex items-center justify-between">
                  <span className="flex items-center gap-2">
                    <Sparkles className="w-6 h-6 text-stellar-gold" />
                    Classification Results
                  </span>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-stellar-gold hover:bg-white/10"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Main Classification */}
                <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-lg p-6 border-2 border-green-500/50">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-2xl font-bold text-green-400">
                      {result.classification}
                    </h3>
                    <Badge className="bg-green-500 text-white px-3 py-1">
                      {result.planetType}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="flex-1">
                      <p className="text-gray-300 text-sm mb-1">Confidence Level</p>
                      <Progress value={result.confidence * 100} className="h-3" />
                    </div>
                    <span className="text-stellar-gold text-2xl font-bold">
                      {(result.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                {/* Planetary Details */}
                <div className="grid md:grid-cols-2 gap-4">
                  <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                    <h4 className="text-stellar-gold mb-3">Orbital Characteristics</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Orbital Period:</span>
                        <span className="text-pure-white">{result.details.orbitalPeriod}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Transit Duration:</span>
                        <span className="text-pure-white">{result.details.transitDuration}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Distance from Star:</span>
                        <span className="text-pure-white">{result.details.distanceFromStar}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                    <h4 className="text-stellar-gold mb-3">Physical Properties</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Radius:</span>
                        <span className="text-pure-white">{result.details.planetaryRadius}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Estimated Mass:</span>
                        <span className="text-pure-white">{result.details.estimatedMass}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Temperature:</span>
                        <span className="text-pure-white">{result.details.equilibriumTemp}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                    <h4 className="text-stellar-gold mb-3">Host Star</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-400">Stellar Type:</span>
                        <span className="text-pure-white">{result.details.stellarType}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">Temperature:</span>
                        <span className="text-pure-white">{result.details.hostStarTemp}</span>
                      </div>
                    </div>
                  </div>

                  <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                    <h4 className="text-stellar-gold mb-3">Data Quality</h4>
                    <div className="space-y-2">
                      <Badge className="bg-green-500/20 text-green-400 border-green-500">
                        {result.dataQuality} Quality Data
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Detection Features */}
                <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                  <h4 className="text-stellar-gold mb-3">Detection Features</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.features.map((feature: string, index: number) => (
                      <Badge
                        key={index}
                        className="px-3 py-1 bg-stellar-gold/20 border border-stellar-gold/50 text-stellar-gold"
                      >
                        {feature}
                      </Badge>
                    ))}
                  </div>
                </div>

                {/* Similar Exoplanets */}
                <div className="bg-black/30 rounded-lg p-4 border border-stellar-gold/20">
                  <h4 className="text-stellar-gold mb-3">Similar Known Exoplanets</h4>
                  <div className="flex flex-wrap gap-2">
                    {result.similarExoplanets.map((planet: string, index: number) => (
                      <Badge
                        key={index}
                        variant="outline"
                        className="px-3 py-1 border-blue-400 text-blue-400"
                      >
                        {planet}
                      </Badge>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        )}

        {/* Integration Instructions */}
        <Card className="bg-white/5 border-stellar-gold/20">
          <CardHeader>
            <CardTitle className="text-pure-white text-sm">Backend Integration Instructions</CardTitle>
          </CardHeader>
          <CardContent className="text-gray-300 text-sm space-y-2">
            <p>To connect your exoplanet detection ML model:</p>
            <ol className="list-decimal list-inside space-y-1 ml-2">
              <li>Update the <code className="bg-black/50 px-2 py-1 rounded text-stellar-gold">classifyData</code> function with your classification API endpoint</li>
              <li>Update the <code className="bg-black/50 px-2 py-1 rounded text-stellar-gold">trainModel</code> function with your training API endpoint</li>
              <li>Configure your API authentication in environment variables</li>
              <li>Map the response data to match the result display format</li>
            </ol>
            <p className="pt-2">See <code className="bg-black/50 px-2 py-1 rounded text-stellar-gold">/AI_ML_INTEGRATION_GUIDE.md</code> for detailed integration steps.</p>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
