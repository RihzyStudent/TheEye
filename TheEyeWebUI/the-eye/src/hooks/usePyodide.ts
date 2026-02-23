import { useEffect, useState, useRef, useCallback } from 'react';
import { loadPyodide, PyodideInterface } from 'pyodide';

interface PyodideState {
  pyodide: PyodideInterface | null;
  loading: boolean;
  error: string | null;
  ready: boolean;
}

/**
 * Hook to initialize and manage Pyodide instance
 * Loads Python runtime and required packages (numpy, pandas, scikit-learn)
 */
export function usePyodide() {
  const [state, setState] = useState<PyodideState>({
    pyodide: null,
    loading: true,
    error: null,
    ready: false,
  });

  const pyodideRef = useRef<PyodideInterface | null>(null);
  const initializingRef = useRef(false);

  useEffect(() => {
    // Prevent multiple initializations
    if (initializingRef.current || pyodideRef.current) {
      return;
    }

    initializingRef.current = true;

    const initPyodide = async () => {
      try {
        console.log('🚀 Loading Pyodide...');
        
        // Load Pyodide
        const pyodide = await loadPyodide({
          indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
        });

        console.log('📦 Loading Python packages...');
        
        // Load required packages
        await pyodide.loadPackage(['numpy', 'pandas', 'scikit-learn']);
        
        console.log('✅ Pyodide and packages loaded successfully');

        pyodideRef.current = pyodide;
        setState({
          pyodide,
          loading: false,
          error: null,
          ready: true,
        });
      } catch (error) {
        console.error('❌ Failed to load Pyodide:', error);
        setState({
          pyodide: null,
          loading: false,
          error: error instanceof Error ? error.message : 'Failed to load Pyodide',
          ready: false,
        });
      }
    };

    initPyodide();
  }, []);

  /**
   * Run Python code in Pyodide
   */
  const runPython = useCallback(async (code: string) => {
    if (!pyodideRef.current) {
      throw new Error('Pyodide not initialized');
    }

    try {
      return await pyodideRef.current.runPythonAsync(code);
    } catch (error) {
      console.error('Python execution error:', error);
      throw error;
    }
  }, []);

  /**
   * Run Python code and return result as JavaScript object
   */
  const runPythonAndReturn = useCallback(async <T = any>(code: string): Promise<T> => {
    if (!pyodideRef.current) {
      throw new Error('Pyodide not initialized');
    }

    try {
      const result = await pyodideRef.current.runPythonAsync(code);
      // Convert Python objects to JavaScript
      return result?.toJs ? result.toJs({ dict_converter: Object.fromEntries }) : result;
    } catch (error) {
      console.error('Python execution error:', error);
      throw error;
    }
  }, []);

  /**
   * Load Python file content into Pyodide's filesystem
   */
  const loadPythonModule = useCallback(async (moduleName: string, content: string) => {
    if (!pyodideRef.current) {
      throw new Error('Pyodide not initialized');
    }

    try {
      // Write the module to Pyodide's filesystem
      pyodideRef.current.FS.writeFile(`${moduleName}.py`, content);
      console.log(`✅ Loaded module: ${moduleName}.py`);
    } catch (error) {
      console.error(`Failed to load module ${moduleName}:`, error);
      throw error;
    }
  }, []);

  /**
   * Set a global variable in Python
   */
  const setPythonVariable = useCallback((name: string, value: any) => {
    if (!pyodideRef.current) {
      throw new Error('Pyodide not initialized');
    }

    try {
      pyodideRef.current.globals.set(name, value);
    } catch (error) {
      console.error(`Failed to set Python variable ${name}:`, error);
      throw error;
    }
  }, []);

  /**
   * Get a global variable from Python
   */
  const getPythonVariable = useCallback(<T = any>(name: string): T => {
    if (!pyodideRef.current) {
      throw new Error('Pyodide not initialized');
    }

    try {
      const value = pyodideRef.current.globals.get(name);
      return value?.toJs ? value.toJs({ dict_converter: Object.fromEntries }) : value;
    } catch (error) {
      console.error(`Failed to get Python variable ${name}:`, error);
      throw error;
    }
  }, []);

  return {
    ...state,
    runPython,
    runPythonAndReturn,
    loadPythonModule,
    setPythonVariable,
    getPythonVariable,
  };
}
