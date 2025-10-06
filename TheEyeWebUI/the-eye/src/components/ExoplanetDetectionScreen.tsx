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

// Flask Backend API Configuration
const API_BASE_URL = 'http://localhost:5001';

// Development mode - set to true to use mock data when Flask backend is not running
const DEV_MODE = false; // change to true for development without backend

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
  transitDepth: string;
  planetaryRadius: string;
  planetEquilibriumTemp: string;
  stellarEffectiveTemp: string;
  stellarLogG: string;
  stellarRadius: string;
  ra: string;
  dec: string;
}

export function ExoplanetDetectionScreen({ onBack }: ExoplanetDetectionScreenProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  // FITS file processing
  const [fitsFile, setFitsFile] = useState<File | null>(null);
  const [tpfFile, setTpfFile] = useState<File | null>(null);
  const [targetId, setTargetId] = useState('');
  const [searchType, setSearchType] = useState<'search' | 'data'>('data');
  const [mission, setMission] = useState<'Kepler' | 'TESS'>('Kepler');
  const [manualStellarParams, setManualStellarParams] = useState({
    stellarMass: '1.0',
    stellarRadius: '1.0',
    stellarTeff: '5800',
    stellarLogg: '4.5',
    ra: '0.0',
    dec: '0.0'
  });
  const fitsInputRef = useRef<HTMLInputElement>(null);
  const tpfInputRef = useRef<HTMLInputElement>(null);
  
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
    transitDepth: '',
    planetaryRadius: '',
    planetEquilibriumTemp: '',
    stellarEffectiveTemp: '',
    stellarLogG: '',
    stellarRadius: '',
    ra: '',
    dec: ''
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

  // Mock training response for development
  const mockTrainResponse = async () => {
    return new Promise<any>((resolve) => {
      setTimeout(() => {
        resolve({
          success: true,
          stats: {
            accuracy: 0.947 + Math.random() * 0.02,
            precision: 0.940 + Math.random() * 0.02,
            recall: 0.965 + Math.random() * 0.02,
            f1Score: 0.952 + Math.random() * 0.02,
            totalSamples: 15420,
            confirmedExoplanets: 3847,
            falsePositives: 234,
            lastTrained: new Date().toLocaleString(),
            modelVersion: 'v1.0.0'
          }
        });
      }, 2000);
    });
  };

  // Mock classification response for development
  const mockClassifyResponse = async () => {
    return new Promise<any>((resolve) => {
      setTimeout(() => {
        const isExoplanet = Math.random() > 0.3;
        resolve({
          success: true,
          classification: isExoplanet ? 'CONFIRMED EXOPLANET' : 'FALSE POSITIVE',
          confidence: 0.85 + Math.random() * 0.15,
          planetType: isExoplanet ? ['Hot Jupiter', 'Super-Earth', 'Neptune-like', 'Rocky Planet'][Math.floor(Math.random() * 4)] : null,
          details: {
            orbitalPeriod: `${manualData.orbitalPeriod || '3.52'} days`,
            transitDuration: `${manualData.transitDuration || '2.8'} hours`,
            planetaryRadius: `${manualData.planetaryRadius || '1.2'} Earth radii`,
            estimatedMass: '1.07 Earth masses',
            distanceFromStar: '0.169 AU',
            equilibriumTemp: `${manualData.planetEquilibriumTemp || '1450'} K`,
            stellarType: 'G-type main-sequence',
            hostStarTemp: `${manualData.stellarEffectiveTemp || '5778'} K`
          },
          features: [
            'Transit signature detected',
            'Periodic dimming pattern confirmed',
            'Doppler shift measured',
            'Low false positive probability'
          ],
          similarExoplanets: [
            'HD 209458 b',
            '51 Pegasi b',
            'WASP-12b'
          ],
          dataQuality: 'High'
        });
      }, 1500);
    });
  };

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

      let data;

      if (DEV_MODE) {
        // Use mock data in development mode
        data = await mockClassifyResponse();
      } else {
        // Use real Flask backend
        let response;
        
        if (selectedFile) {
          // File upload mode - show info toast for large files
          toast.info('🔬 Processing CSV... Large files may take several minutes. Check the backend console for progress!', { duration: 60000 });
          
          const formData = new FormData();
          formData.append('dataset', selectedFile);
          formData.append('preprocessing', enablePreprocessing.toString());
          formData.append('removeFalsePositives', removeFalsePositives.toString());
          
          // Create AbortController for timeout (10 minutes for large CSVs)
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 600000); // 10 minutes
          
          response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            body: formData,
            signal: controller.signal
          });
          
          clearTimeout(timeoutId);
        } else {
          // Manual entry mode
          response = await fetch(`${API_BASE_URL}/classify`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              data: manualData,
              preprocessing: enablePreprocessing,
              removeFalsePositives: removeFalsePositives
            })
          });
        }
        
        data = await response.json();
      }
      
      if (data.success) {
        setResult(data);
        setProgress(100);
        clearInterval(progressInterval);
        
        // Show different toast based on whether it's CSV or manual entry
        if (data.csv_summary) {
          toast.success(
            `✅ CSV processed: ${data.csv_summary.confirmed_exoplanets}/${data.csv_summary.total_rows} confirmed exoplanets!`,
            { duration: 5000 }
          );
        } else {
          toast.success(`✅ ${data.classification} (${(data.confidence * 100).toFixed(1)}% confidence)`);
        }
      } else {
        throw new Error(data.error || 'Classification failed');
      }
      
    } catch (error: any) {
      console.error('Classification error:', error);
      if (error.name === 'AbortError') {
        toast.error('⏱️ Classification timed out (>10 min). Your CSV might be too large.');
      } else if (DEV_MODE) {
        toast.error('Classification failed in development mode.');
      } else {
        toast.error('❌ Classification failed. Is the Flask backend running on port 5001?');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const processFitsFile = async () => {
    // Validation based on search type
    if (searchType === 'search' && !targetId) {
      toast.error('Target ID is required for Archive Download mode');
      return;
    }
    
    if (searchType === 'data' && !fitsFile && !targetId) {
      toast.error('Please upload a FITS file or provide a Target ID');
      return;
    }

    setIsProcessing(true);
    setProgress(10);

    try {
      toast.info('🔬 Processing FITS data - this may take 1-2 minutes. Watch the backend console for detailed progress!', { duration: 120000 });

      const formData = new FormData();
      
      // Add target ID if provided
      if (targetId) {
        formData.append('target', targetId);
        formData.append('search_type', searchType);
      }
      
      // Add FITS file if provided
      if (fitsFile) {
        formData.append('fits_file', fitsFile);
        formData.append('mission', mission);
        formData.append('search_type', 'data');
      }
      
      // Add TPF file if provided
      if (tpfFile) {
        formData.append('tpf_file', tpfFile);
      }
      
      // Add manual stellar parameters if no target ID
      if (!targetId || searchType === 'data') {
        formData.append('stellar_mass', manualStellarParams.stellarMass);
        formData.append('stellar_radius', manualStellarParams.stellarRadius);
        formData.append('stellar_teff', manualStellarParams.stellarTeff);
        formData.append('stellar_logg', manualStellarParams.stellarLogg);
        formData.append('ra', manualStellarParams.ra);
        formData.append('dec', manualStellarParams.dec);
      }

      // Create AbortController for timeout (3 minutes for FITS processing)
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes

      setProgress(20);
      const response = await fetch(`${API_BASE_URL}/process_fits`, {
        method: 'POST',
        body: formData,
        signal: controller.signal
      });

      clearTimeout(timeoutId);
      setProgress(80);

      const data = await response.json();
      setProgress(100);

      if (data.success) {
        setResult(data);
        toast.success(`✅ Analysis complete! ${data.classification} (${(data.confidence * 100).toFixed(1)}% confidence)`);
        
        // Show key results
        if (data.tls_results) {
          toast.info(`🔭 Period: ${data.tls_results.period.toFixed(2)} days, SDE: ${data.tls_results.sde.toFixed(2)}`, { duration: 8000 });
        }
      } else {
        throw new Error(data.error || 'FITS processing failed');
      }
    } catch (error: any) {
      console.error('FITS processing error:', error);
      if (error.name === 'AbortError') {
        toast.error('⏱️ FITS processing timed out (>3 min). The target may be too large or complex.');
      } else {
        toast.error(`❌ FITS processing failed: ${error.message || 'Unknown error'}`);
      }
    } finally {
      setIsProcessing(false);
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const trainModel = async () => {
    setIsTraining(true);
    setProgress(0);
    
    try {
      const progressInterval = setInterval(() => {
        setProgress(prev => Math.min(prev + 5, 95));
      }, 500);

      let data;

      if (DEV_MODE) {
        // Use mock data in development mode
        data = await mockTrainResponse();
      } else {
        // Use real Flask backend
        const response = await fetch(`${API_BASE_URL}/train`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
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
        
        data = await response.json();
      }
      
      if (data.success) {
        setModelStats(data.stats);
        setProgress(100);
        clearInterval(progressInterval);
        toast.success('Model training completed successfully!');
      } else {
        throw new Error(data.error || 'Training failed');
      }
      
    } catch (error) {
      console.error('Training error:', error);
      if (DEV_MODE) {
        toast.error('Training failed in development mode.');
      } else {
        toast.error('Training failed. Is the Flask backend running on port 5001?');
      }
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
      transitDepth: '',
      planetaryRadius: '',
      planetEquilibriumTemp: '',
      stellarEffectiveTemp: '',
      stellarLogG: '',
      stellarRadius: '',
      ra: '',
      dec: ''
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
        
        <div className="flex items-center gap-3">
          {DEV_MODE && (
            <Badge variant="outline" className="border-yellow-500 text-yellow-500 bg-yellow-500/10">
              🧪 Dev Mode (Mock Data)
            </Badge>
          )}
          <Button 
            variant="ghost" 
            size="sm"
            onClick={handleReset}
            className="text-pure-white hover:bg-white/10"
          >
            Reset
          </Button>
        </div>
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
                <p className="text-gray-400 text-sm">Candidate Exoplanets</p>
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
              <TabsList className="grid w-full grid-cols-3 bg-white/10">
                <TabsTrigger value="upload" className="data-[state=active]:bg-stellar-gold data-[state=active]:text-cosmic-deep-blue">
                  <Database className="w-4 h-4 mr-2" />
                  Dataset Upload
                </TabsTrigger>
                <TabsTrigger value="fits" className="data-[state=active]:bg-stellar-gold data-[state=active]:text-cosmic-deep-blue">
                  <Activity className="w-4 h-4 mr-2" />
                  FITS Analysis
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

              {/* FITS File Analysis Tab */}
              <TabsContent value="fits" className="space-y-4">
                <Card className="bg-cosmic-deep-blue/50 border-stellar-gold/30">
                  <CardHeader>
                    <CardTitle className="text-stellar-gold flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      FITS File Processing
                    </CardTitle>
                    <CardDescription className="text-white/70">
                      Process light curve FITS files using LightKurve analysis
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Target ID or FITS Upload Options */}
                    <div className="space-y-4">
                      <div className="flex items-center justify-between">
                        <Label className="text-white font-semibold">Processing Mode</Label>
                        <Select value={searchType} onValueChange={(value: 'search' | 'data') => setSearchType(value)}>
                          <SelectTrigger className="w-[200px] bg-white/10 border-stellar-gold/30 text-white">
                            <SelectValue />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="search">Download from Archive</SelectItem>
                            <SelectItem value="data">Upload FITS File</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      {/* Target ID Input (for search mode or with FITS) */}
                      <div className="space-y-2">
                        <Label htmlFor="targetId" className="text-white flex items-center gap-2">
                          Target ID 
                          {searchType === 'search' ? (
                            <Badge variant="outline" className="bg-stellar-gold/20 text-stellar-gold border-stellar-gold">Required</Badge>
                          ) : (
                            <Badge variant="outline" className="bg-white/10 text-white/70 border-white/30">Optional</Badge>
                          )}
                        </Label>
                        <Input
                          id="targetId"
                          type="text"
                          placeholder="e.g., KIC 12345678, TIC 789012, EPIC 345678"
                          value={targetId}
                          onChange={(e) => setTargetId(e.target.value)}
                          className="bg-white/10 border-stellar-gold/30 text-white placeholder:text-white/50"
                        />
                        <p className="text-xs text-white/60">
                          {searchType === 'search' 
                            ? '⚡ Downloads light curve and auto-fetches stellar parameters from NASA archive'
                            : '💡 Optional: Provide Target ID to auto-fetch stellar parameters, or specify them manually below'}
                        </p>
                      </div>

                      {/* Mission Selection */}
                      {searchType === 'data' && (
                        <div className="space-y-2">
                          <Label className="text-white">Mission</Label>
                          <Select value={mission} onValueChange={(value: 'Kepler' | 'TESS') => setMission(value)}>
                            <SelectTrigger className="bg-white/10 border-stellar-gold/30 text-white">
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="Kepler">Kepler</SelectItem>
                              <SelectItem value="TESS">TESS</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      )}

                      {/* FITS File Upload */}
                      {searchType === 'data' && (
                        <div className="space-y-4">
                          <div className="space-y-2">
                            <Label htmlFor="fitsFile" className="text-white">
                              Light Curve FITS File *
                            </Label>
                            <div className="flex gap-2">
                              <Input
                                id="fitsFile"
                                ref={fitsInputRef}
                                type="file"
                                accept=".fits,.fit"
                                onChange={(e) => setFitsFile(e.target.files?.[0] || null)}
                                className="bg-white/10 border-stellar-gold/30 text-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-stellar-gold file:text-cosmic-deep-blue hover:file:bg-stellar-gold/80"
                              />
                            </div>
                            {fitsFile && (
                              <p className="text-sm text-stellar-gold">
                                Selected: {fitsFile.name}
                              </p>
                            )}
                          </div>

                          <div className="space-y-2">
                            <Label htmlFor="tpfFile" className="text-white">
                              Target Pixel File (optional)
                            </Label>
                            <div className="flex gap-2">
                              <Input
                                id="tpfFile"
                                ref={tpfInputRef}
                                type="file"
                                accept=".fits,.fit"
                                onChange={(e) => setTpfFile(e.target.files?.[0] || null)}
                                className="bg-white/10 border-stellar-gold/30 text-white file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-stellar-gold file:text-cosmic-deep-blue hover:file:bg-stellar-gold/80"
                              />
                            </div>
                            {tpfFile && (
                              <p className="text-sm text-stellar-gold">
                                Selected: {tpfFile.name}
                              </p>
                            )}
                            <p className="text-xs text-white/60">
                              Optional: Improves SAP correction accuracy
                            </p>
                          </div>
                        </div>
                      )}

                      {/* Manual Stellar Parameters (shown only if no target ID and uploading FITS) */}
                      {!targetId && searchType === 'data' && (
                        <div className="space-y-4 p-4 bg-white/5 rounded-lg border border-stellar-gold/20">
                          <h4 className="text-white font-semibold flex items-center gap-2">
                            <Globe2 className="w-4 h-4" />
                            Manual Stellar Parameters (Optional)
                          </h4>
                          <p className="text-xs text-white/60">
                            Leave blank to use defaults: M=1.0☉, R=1.0☉, Teff=5800K, log(g)=4.5
                          </p>
                          
                          <div className="grid grid-cols-2 gap-4">
                            <div className="space-y-2">
                              <Label htmlFor="stellarMass" className="text-white text-sm">
                                Stellar Mass (M☉)
                              </Label>
                              <Input
                                id="stellarMass"
                                type="number"
                                step="0.01"
                                value={manualStellarParams.stellarMass}
                                onChange={(e) => setManualStellarParams(prev => ({ ...prev, stellarMass: e.target.value }))}
                                className="bg-white/10 border-stellar-gold/30 text-white"
                                placeholder="1.0"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="stellarRadius" className="text-white text-sm">
                                Stellar Radius (R☉)
                              </Label>
                              <Input
                                id="stellarRadius"
                                type="number"
                                step="0.01"
                                value={manualStellarParams.stellarRadius}
                                onChange={(e) => setManualStellarParams(prev => ({ ...prev, stellarRadius: e.target.value }))}
                                className="bg-white/10 border-stellar-gold/30 text-white"
                                placeholder="1.0"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="stellarTeff" className="text-white text-sm">
                                Effective Temp (K)
                              </Label>
                              <Input
                                id="stellarTeff"
                                type="number"
                                step="1"
                                value={manualStellarParams.stellarTeff}
                                onChange={(e) => setManualStellarParams(prev => ({ ...prev, stellarTeff: e.target.value }))}
                                className="bg-white/10 border-stellar-gold/30 text-white"
                                placeholder="5800"
                              />
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="stellarLogg" className="text-white text-sm">
                                log(g)
                              </Label>
                              <Input
                                id="stellarLogg"
                                type="number"
                                step="0.01"
                                value={manualStellarParams.stellarLogg}
                                onChange={(e) => setManualStellarParams(prev => ({ ...prev, stellarLogg: e.target.value }))}
                                className="bg-white/10 border-stellar-gold/30 text-white"
                                placeholder="4.5"
                              />
                            </div>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Process Button */}
                    <Button 
                      onClick={processFitsFile}
                      disabled={
                        isProcessing || 
                        (searchType === 'search' && !targetId) ||
                        (searchType === 'data' && !fitsFile && !targetId)
                      }
                      className="w-full bg-gradient-to-r from-stellar-gold via-cosmic-purple to-stellar-gold bg-size-200 bg-pos-0 hover:bg-pos-100 transition-all duration-500 text-cosmic-deep-blue font-bold py-6"
                    >
                      {isProcessing ? (
                        <>
                          <RefreshCw className="w-5 h-5 mr-2 animate-spin" />
                          Processing FITS Data...
                        </>
                      ) : (
                        <>
                          <Brain className="w-5 h-5 mr-2" />
                          Analyze with LightKurve + ML
                        </>
                      )}
                    </Button>

                    {isProcessing && (
                      <div className="space-y-2">
                        <Progress value={progress} className="h-2" />
                        <p className="text-sm text-center text-white/70">
                          {progress < 20 && 'Loading light curve...'}
                          {progress >= 20 && progress < 40 && 'Correcting and detrending...'}
                          {progress >= 40 && progress < 60 && 'Running Transit Least Squares...'}
                          {progress >= 60 && progress < 80 && 'Applying SAP corrections...'}
                          {progress >= 80 && 'Classifying with ML model...'}
                        </p>
                      </div>
                    )}
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
                        <Label className="text-gray-300">Planet Equilibrium Temp (K)</Label>
                        <Input
                          type="number"
                          step="1"
                          placeholder="e.g., 1450"
                          value={manualData.planetEquilibriumTemp}
                          onChange={(e) => updateManualData('planetEquilibriumTemp', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Stellar Effective Temp (K)</Label>
                        <Input
                          type="number"
                          step="1"
                          placeholder="e.g., 5778"
                          value={manualData.stellarEffectiveTemp}
                          onChange={(e) => updateManualData('stellarEffectiveTemp', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">Stellar Log G (cgs)</Label>
                        <Input
                          type="number"
                          step="0.01"
                          placeholder="e.g., 4.44"
                          value={manualData.stellarLogG}
                          onChange={(e) => updateManualData('stellarLogG', e.target.value)}
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
                        <Label className="text-gray-300">RA - Right Ascension (deg)</Label>
                        <Input
                          type="number"
                          step="0.0001"
                          placeholder="e.g., 285.6789"
                          value={manualData.ra}
                          onChange={(e) => updateManualData('ra', e.target.value)}
                          className="bg-black/30 border-stellar-gold/30 text-pure-white mt-1"
                        />
                      </div>
                      
                      <div>
                        <Label className="text-gray-300">DEC - Declination (deg)</Label>
                        <Input
                          type="number"
                          step="0.0001"
                          placeholder="e.g., 38.7833"
                          value={manualData.dec}
                          onChange={(e) => updateManualData('dec', e.target.value)}
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

                {/* CSV Upload Summary - shown when processing CSV files */}
                {result.csv_summary && (
                  <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-lg p-6 border-2 border-blue-500/50">
                    <h3 className="text-xl font-bold text-blue-400 mb-4 flex items-center gap-2">
                      <FileText className="w-5 h-5" />
                      CSV Analysis Summary
                    </h3>
                    <div className="grid grid-cols-3 gap-4 mb-4">
                      <div className="bg-black/30 rounded-lg p-3 text-center">
                        <p className="text-gray-400 text-sm">Total Rows</p>
                        <p className="text-2xl font-bold text-white">{result.csv_summary.total_rows}</p>
                      </div>
                      <div className="bg-black/30 rounded-lg p-3 text-center">
                        <p className="text-gray-400 text-sm">Confirmed Exoplanets</p>
                        <p className="text-2xl font-bold text-green-400">{result.csv_summary.confirmed_exoplanets}</p>
                      </div>
                      <div className="bg-black/30 rounded-lg p-3 text-center">
                        <p className="text-gray-400 text-sm">False Positives</p>
                        <p className="text-2xl font-bold text-red-400">{result.csv_summary.false_positives}</p>
                      </div>
                    </div>
                    <p className="text-sm text-gray-300 italic">
                      Showing detailed results for Row 0 below. All {result.csv_summary.total_rows} rows were classified - check backend console for full list.
                    </p>
                  </div>
                )}

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


      </div>
    </div>
  );
}
