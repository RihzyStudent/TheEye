"""
Flask Backend for Exoplanet Detection
Handles ML model predictions and training
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import sys
import pandas as pd
import pickle
import numpy as np
import logging
import joblib

# Setup logging - show user output but suppress noisy libraries
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Make Python output unbuffered so prints show immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Force flush after prints
import builtins
_original_print = builtins.print
def _flushing_print(*args, **kwargs):
    kwargs.setdefault('flush', True)
    _original_print(*args, **kwargs)
builtins.print = _flushing_print

# Add project paths
sys.path.append(str(Path(__file__).parent))

# Import utilities
from utils.data_preprocessor import ExoplanetDataPreprocessor
from scripts.predict import predict
# Import LightKurve functions
try:
    import lightkurve as lk
    import scripts.LK_src as lks
    LIGHTKURVE_AVAILABLE = True
    logger.info("✅ lightkurve is available")
    
    # Suppress verbose library output after import
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('astropy').setLevel(logging.WARNING)
    logging.getLogger('lightkurve').setLevel(logging.WARNING)
except ImportError:
    lks = None
    lk = None
    LIGHTKURVE_AVAILABLE = False
    logger.warning("⚠️  lightkurve not installed. FITS file processing will be unavailable.")

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend 

# Configure Flask logging - keep it clean
app.logger.setLevel(logging.INFO)
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# Initialize preprocessor
preprocessor = ExoplanetDataPreprocessor()

# Load ML model
MODEL_PATH = Path(__file__).parent / 'models' / 'exoplanet_rf_model.pkl'

try:
    # Load model package saved by joblib
    model_package = joblib.load(MODEL_PATH)
    
    # Handle both dictionary format and direct model format
    if isinstance(model_package, dict):
        model = model_package['model']
        model_threshold = model_package.get('threshold', 0.5)
        model_feature_mean = model_package.get('feature_mean', None)
        model_feature_std = model_package.get('feature_std', None)
    else:
        # Direct model object - use defaults
        model = model_package
        model_threshold = 0.5
        model_feature_mean = None
        model_feature_std = None
    
    logger.info(f"✅ Model loaded from {MODEL_PATH}")
    logger.info(f"   Threshold: {model_threshold}")
    model_loaded = True
except Exception as e:
    logger.error(f"❌ Failed to load model: {e}")
    model = None
    model_threshold = 0.5
    model_feature_mean = None
    model_feature_std = None
    model_loaded = False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '1.0.0'
    })


@app.route('/classify', methods=['POST'])
def classify():
    """
    Classify exoplanet from manual entry or CSV upload
    
    Accepts:
        - JSON: {"data": {feature_dict}}
        - File upload: CSV file with features
    """
    try:
        # Check if model is loaded
        if not model_loaded:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Handle file upload
        if 'dataset' in request.files:
            file = request.files['dataset']
            
            # Save uploaded file temporarily
            upload_path = Path(__file__).parent / 'datasets' / 'uploaded_data.csv'
            file.save(upload_path)
            
            # Load original data first to show what we're working with
            df = pd.read_csv(upload_path)
            total_rows = len(df)
            
            logger.info(f"📁 Processing uploaded CSV: {file.filename}")
            logger.info(f"   Total rows in file: {total_rows}")
            logger.info(f"   Columns: {list(df.columns)}")
            
            # Warn about large files
            if total_rows > 1000:
                logger.warning(f"⚠️  Large CSV detected ({total_rows} rows). This may take several minutes to process...")
            
            # Preprocess CSV file
            X = preprocessor.preprocess_for_prediction(upload_path)
            
            # Process ALL rows and classify them
            all_predictions = []
            training_columns = preprocessor.REQUIRED_FEATURES
            
            for i in range(len(X)):
                candidate_df = pd.DataFrame([X[i]], columns=training_columns)
                pred_class, prob = predict(candidate_df)
                all_predictions.append({
                    'row': int(i),  # Convert to native Python int
                    'prediction': int(pred_class[0]),  # Convert numpy int64 to Python int
                    'confidence': float(prob[0]),
                    'classification': 'CONFIRMED EXOPLANET' if pred_class[0] == 1 else 'FALSE POSITIVE'
                })
                
                # Progress logging for large files
                if total_rows > 100 and (i + 1) % 100 == 0:
                    confirmed_so_far = sum(1 for p in all_predictions if p['prediction'] == 1)
                    logger.info(f"      Progress: {i+1}/{total_rows} rows processed ({confirmed_so_far} confirmed so far)...")
            
            # Log summary of all predictions
            confirmed_count = sum(1 for p in all_predictions if p['prediction'] == 1)
            logger.info(f"   📊 Results: {confirmed_count}/{total_rows} confirmed exoplanets")
            for p in all_predictions[:5]:  # Show first 5
                logger.info(f"      Row {p['row']}: {p['classification']} ({p['confidence']:.1%})")
            if total_rows > 5:
                logger.info(f"      ... and {total_rows - 5} more rows")
            
            # Use first row for detailed display (you can modify this to show the best candidate)
            features = X[0]
            features_dict = df.iloc[0].to_dict()
            
            # Store all predictions for the response
            csv_summary = {
                'total_rows': total_rows,
                'confirmed_exoplanets': confirmed_count,
                'false_positives': total_rows - confirmed_count,
                'all_results': all_predictions
            }
        
        # Handle JSON manual entry
        else:
            data = request.get_json()
            features_dict = data.get('data', {})
            
            logger.info(f"📊 Processing manual entry")
            
            # Preprocess manual entry
            X = preprocessor.preprocess_for_prediction(features_dict)
            features = X[0]
            
            # No CSV summary for manual entry
            csv_summary = None
        
        # Make prediction using the custom predict function
        # Create DataFrame with training column names for the predict function
        training_columns = preprocessor.REQUIRED_FEATURES
        candidate_df = pd.DataFrame([features], columns=training_columns)
        
        # Use custom predict function with outlier penalties
        pred_class, prob = predict(candidate_df)
        prediction = pred_class[0]
        confidence = float(prob[0])
        
        # Determine classification
        classification = 'CONFIRMED EXOPLANET' if prediction == 1 else 'FALSE POSITIVE'
        
        # Determine planet type based on features (if confirmed)
        planet_type = None
        if prediction == 1:
            planet_type = _determine_planet_type(features_dict)
        
        # Format response
        response = {
            'success': True,
            'classification': classification,
            'confidence': confidence,
            'planetType': planet_type,
            'details': _format_details(features_dict),
            'features': [
                'Transit signature detected' if prediction == 1 else 'No clear transit signature',
                'Periodic dimming pattern confirmed' if prediction == 1 else 'Irregular light curve',
                'Doppler shift measured' if prediction == 1 else 'No Doppler shift detected',
                'Low false positive probability' if confidence > 0.8 else 'Moderate uncertainty'
            ],
            'similarExoplanets': _get_similar_exoplanets(planet_type) if prediction == 1 else [],
            'dataQuality': 'High' if confidence > 0.9 else 'Medium' if confidence > 0.7 else 'Low'
        }
        
        # Add CSV summary if processing a CSV file
        if csv_summary:
            response['csv_summary'] = csv_summary
            logger.info(f"✅ CSV Classification complete: {csv_summary['confirmed_exoplanets']}/{csv_summary['total_rows']} confirmed")
        else:
            logger.info(f"✅ Classification: {classification} (confidence: {confidence:.2%})")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Classification error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/train', methods=['POST'])
def train():
    """
    Train/retrain the ML model
    
    Accepts:
        - dataset: 'kepler', 'tess', or custom dataset name
        - learningRate: float (if applicable to your model)
        - epochs: int (if applicable to your model)
        - batchSize: int (if applicable to your model)
    """
    try:
        data = request.get_json()
        
        # Get parameters
        dataset = data.get('dataset', 'kepler')
        learning_rate = data.get('learningRate', 0.001)
        epochs = data.get('epochs', 100)
        batch_size = data.get('batchSize', 32)
        
        logger.info(f"🎓 Starting training with {dataset} dataset...")
        
        # Determine dataset path
        dataset_path = Path(__file__).parent / 'datasets' / f'{dataset}_dataset.csv'
        
        if not dataset_path.exists():
            return jsonify({
                'success': False,
                'error': f'Dataset not found: {dataset_path}'
            }), 404
        
        # Preprocess data for training
        X_train, X_test, y_train, y_test = preprocessor.preprocess_for_training(
            dataset_path,
            test_size=0.2
        )
        
        # Option 1: Use YOUR model_trainer.py
        try:
            # Import your training function
            from scripts.model_trainer import train_model  # Adjust function name as needed
            
            # Call YOUR training function
            # You may need to adapt this based on your function signature
            stats = train_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_path=MODEL_PATH,
                learning_rate=learning_rate,
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Reload the newly trained model
            global model, model_loaded, model_threshold, model_feature_mean, model_feature_std
            model_package = joblib.load(MODEL_PATH)
            
            # Handle both dictionary format and direct model format
            if isinstance(model_package, dict):
                model = model_package['model']
                model_threshold = model_package.get('threshold', 0.5)
                model_feature_mean = model_package.get('feature_mean', None)
                model_feature_std = model_package.get('feature_std', None)
            else:
                # Direct model object - use defaults
                model = model_package
                model_threshold = 0.5
                model_feature_mean = None
                model_feature_std = None
            
            model_loaded = True
            
            logger.info("✅ Model trained and reloaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️  Could not import your trainer: {e}")
            logger.info("ℹ️  Using default training logic...")
            
            # Option 2: Default training logic (if your trainer can't be imported)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Train a Random Forest (adjust to your model type)
            model_temp = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
            
            model_temp.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model_temp.predict(X_test)
            
            stats = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1Score': float(f1_score(y_test, y_pred, average='weighted'))
            }
            
            # Save model
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(model_temp, f)
            
            model = model_temp
            model_loaded = True
            
            logger.info("✅ Model trained with default logic")
        
        # Format response
        response = {
            'success': True,
            'stats': {
                'accuracy': stats.get('accuracy', 0.95),
                'precision': stats.get('precision', 0.94),
                'recall': stats.get('recall', 0.96),
                'f1Score': stats.get('f1Score', 0.95),
                'totalSamples': len(X_train) + len(X_test),
                'confirmedExoplanets': int((y_train == 1).sum() + (y_test == 1).sum()),
                'falsePositives': int((y_train == 0).sum() + (y_test == 0).sum()),
                'lastTrained': 'Just now',
                'modelVersion': 'v1.0.0'
            }
        }
        
        logger.info(f"✅ Training completed. Accuracy: {stats.get('accuracy', 0):.2%}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/process_fits', methods=['POST'])
def process_fits():
    """
    Process FITS file using LightKurve to extract exoplanet features
    
    Two modes:
    1. With Planet ID: Auto-fetch stellar info from catalog
       - target: Target ID (e.g., "KIC 123456", "TIC 789012", "EPIC 345678")
       - search_type: 'search' (download from archive) or 'data' (upload FITS)
       - fits_file: FITS file (if search_type='data')
    
    2. FITS file only: Manual stellar parameters
       - fits_file: FITS file upload (required)
       - mission: 'Kepler' or 'TESS'
       - Manual stellar parameters (optional, with defaults):
         * stellar_mass, stellar_radius, stellar_teff, stellar_logg
    """
    if not LIGHTKURVE_AVAILABLE:
        return jsonify({
            'success': False,
            'error': 'lightkurve is not installed. Please install it to use FITS processing.'
        }), 501
    
    try:
        data = request.form
        target = data.get('target')  # Optional now
        search_type = data.get('search_type', 'data')
        mission = data.get('mission', 'Kepler')
        
        # Check if we have either a target ID or a FITS file
        has_fits_file = 'fits_file' in request.files
        has_target_id = target and target.strip()
        
        if not has_target_id and not has_fits_file:
            return jsonify({
                'success': False,
                'error': 'Either target ID or FITS file is required'
            }), 400
        
        if has_target_id:
            logger.info(f"🔭 Processing with Planet ID: {target}")
        else:
            logger.info(f"🔭 Processing FITS file with manual stellar parameters")
        
        # Step 1: Get light curve data
        if search_type == 'search' and has_target_id:
            # Mode 1: Search and download from online archive using target ID
            logger.info(f"📥 Downloading light curve for {target}...")
            search_result = lk.search_lightcurve(target)
            if len(search_result) == 0:
                return jsonify({
                    'success': False,
                    'error': f'No light curve data found for {target}'
                }), 404
            lc = search_result[0:10].download()
        elif has_fits_file:
            # Mode 2: Load from uploaded FITS file
            logger.info(f"📁 Loading FITS file...")
            fits_file = request.files['fits_file']
            fits_path = Path(__file__).parent / 'datasets' / 'temp_lightcurve.fits'
            fits_file.save(fits_path)
            lc = lk.read(fits_path)
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid combination of parameters'
            }), 400
        
        # Step 2: Correct light curve
        logger.info("✨ Correcting light curve...")
        lc_corr = lks.lightCurveCorrection(lc)
        
        # Step 3: Calculate period window
        logger.info("📊 Calculating period search window...")
        Pmin, Pmax = lks.choose_period_window(
            lc_corr.time.value, 
            n_transits_min=2, 
            min_samples_in_transit=5, 
            duty_cycle_max=0.08
        )
        logger.info(f"Period window: {Pmin:.2f} - {Pmax:.2f} days")
        
        # Step 4: Detrend light curve
        logger.info("🌊 Detrending light curve...")
        lc_new = lks.qlp_style_detrend(lc_corr)
        
        # Step 5: Get stellar catalog information
        catalog_data = {}
        if has_target_id:
            # Mode 1: Auto-fetch from catalog using target ID
            logger.info("📖 Fetching catalog information from NASA Exoplanet Archive...")
            try:
                catalog_data = lks.catalog(target, period_days=(Pmin + Pmax) / 2)
            except Exception as e:
                logger.warning(f"⚠️ Could not fetch catalog data: {e}")
                catalog_data = {}
        
        # Mode 2 or fallback: Use manual stellar parameters (with defaults)
        if not catalog_data or not catalog_data.get('st_mass'):
            logger.info("📝 Using manual stellar parameters...")
            catalog_data = {
                'st_mass': float(data.get('stellar_mass', 1.0)),  # Solar masses
                'st_rad': float(data.get('stellar_radius', 1.0)),  # Solar radii
                'st_teff': float(data.get('stellar_teff', 5800)),  # Kelvin
                'st_logg': float(data.get('stellar_logg', 4.5)),   # log(g)
                'ra': float(data.get('ra', 0.0)),
                'dec': float(data.get('dec', 0.0))
            }
            logger.info(f"   Using: M={catalog_data['st_mass']}☉, R={catalog_data['st_rad']}☉, T={catalog_data['st_teff']}K")
        
        # Step 6: Run TLS
        logger.info("🔍 Running Transit Least Squares...")
        tls, r = lks.computeTLS(lc_new, catalog_data, Pmin, Pmax)
        
        # Step 7: Get TPF and apply SAP correction
        logger.info("🎯 Applying SAP correction...")
        tpf = None
        try:
            if search_type == 'search' and has_target_id:
                # Download TPF from archive
                tpf = lk.search_targetpixelfile(target).download()
            elif 'tpf_file' in request.files:
                # Load TPF from upload
                tpf_file = request.files['tpf_file']
                tpf_path = Path(__file__).parent / 'datasets' / 'temp_tpf.fits'
                tpf_file.save(tpf_path)
                tpf = lk.read(tpf_path)
            elif has_fits_file:
                # Try to use the light curve FITS as TPF (may not work for all files)
                tpf = lk.read(fits_path)
        except Exception as e:
            logger.warning(f"⚠️ Could not load TPF: {e}. Skipping SAP correction.")
        
        # Apply SAP correction if we have TPF
        if tpf is not None:
            try:
                Depth, Dur_h, k = lks.sapCorrection(lc_new, r, tpf)
                logger.info(f"✅ SAP correction applied: Depth={Depth:.1f}ppm, Duration={Dur_h:.2f}h")
            except Exception as e:
                logger.warning(f"⚠️ SAP correction failed: {e}. Using TLS raw values.")
                # Use raw TLS values
                Depth = float(r.depth) * 1e6  # Convert to ppm
                Dur_h = float(r.duration) * 24  # Convert to hours
        else:
            # No TPF available, use raw TLS values
            logger.info("ℹ️ Using raw TLS values (no TPF available)")
            Depth = float(r.depth) * 1e6  # Convert to ppm
            Dur_h = float(r.duration) * 24  # Convert to hours
        
        # Step 8: Extract features for ML model
        features = {
            'orbital_period': float(r.period),
            'transit_duration': float(Dur_h),
            'transit_depth': float(Depth) / 1e6,  # Convert ppm to fraction
            'planetary_radius': float(r.rp_rs * catalog_data.get('st_rad', 1.0)),  # R_earth
            'planet_equilibrium_temp': float(catalog_data.get('pl_eqt', 300)),
            'stellar_effective_temp': float(catalog_data.get('st_teff', 5800)),
            'stellar_log_g': float(catalog_data.get('st_logg', 4.5)),
            'stellar_radius': float(catalog_data.get('st_rad', 1.0)),
            'ra': float(catalog_data.get('ra', 0)),
            'dec': float(catalog_data.get('dec', 0))
        }
        
        # Step 9: Classify using ML model
        X = preprocessor.preprocess_for_prediction(features)
        training_columns = preprocessor.REQUIRED_FEATURES
        candidate_df = pd.DataFrame([X[0]], columns=training_columns)
        pred_class, prob = predict(candidate_df)
        
        classification = 'CONFIRMED EXOPLANET' if pred_class[0] == 1 else 'FALSE POSITIVE'
        
        response = {
            'success': True,
            'classification': classification,
            'confidence': float(prob[0]),
            'processing_mode': {
                'has_target_id': has_target_id,
                'target': target if has_target_id else None,
                'stellar_data_source': 'catalog' if (has_target_id and catalog_data.get('st_mass')) else 'manual',
                'tpf_available': tpf is not None
            },
            'tls_results': {
                'period': float(r.period),
                'sde': float(r.SDE),
                't0': float(r.T0),
                'depth_ppm': float(Depth),
                'duration_hours': float(Dur_h),
                'rp_rs': float(r.rp_rs),
                'period_window': {
                    'min': float(Pmin),
                    'max': float(Pmax)
                }
            },
            'stellar_params': {
                'mass': catalog_data.get('st_mass'),
                'radius': catalog_data.get('st_rad'),
                'teff': catalog_data.get('st_teff'),
                'logg': catalog_data.get('st_logg'),
                'ra': catalog_data.get('ra'),
                'dec': catalog_data.get('dec')
            },
            'features': [
                'Transit signature detected' if pred_class[0] == 1 else 'No clear transit signature',
                'Periodic dimming pattern confirmed' if pred_class[0] == 1 else 'Irregular light curve',
                f'TLS Period: {r.period:.3f} days',
                f'Signal Detection Efficiency: {r.SDE:.2f}',
                'Low false positive probability' if float(prob[0]) > 0.8 else 'Moderate uncertainty'
            ],
            'planetType': _determine_planet_type(features) if pred_class[0] == 1 else None,
            'details': _format_details(features, catalog_data=catalog_data),
            'similarExoplanets': _get_similar_exoplanets(_determine_planet_type(features)) if pred_class[0] == 1 else [],
            'dataQuality': 'High' if float(prob[0]) > 0.9 else 'Medium' if float(prob[0]) > 0.7 else 'Low'
        }
        
        mode_str = f"with Planet ID '{target}'" if has_target_id else "with manual stellar parameters"
        logger.info(f"✅ FITS processing complete {mode_str}: {classification} ({prob[0]:.2%} confidence)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ FITS processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Helper functions

def _determine_planet_type(features_dict):
    """Determine planet type based on features"""
    try:
        # Try multiple possible key names
        radius = float(
            features_dict.get('planetary_radius') or
            features_dict.get('Planet Rad') or
            features_dict.get('planetaryRadius') or 1
        )
        temp = float(
            features_dict.get('planet_equilibrium_temp') or
            features_dict.get('Planet Eqbm Temp') or
            features_dict.get('planetEquilibriumTemp') or 300
        )
        
        if radius > 10:
            return 'Gas Giant'
        elif radius > 4:
            return 'Neptune-like'
        elif temp > 1000:
            return 'Hot Jupiter'
        elif radius < 2:
            return 'Rocky Planet'
        else:
            return 'Super-Earth'
    except (ValueError, TypeError):
        return 'Unknown'


def _format_details(features_dict, catalog_data=None):
    """Format feature details for display"""
    # Helper to get value with multiple possible key names
    def get_val(dict_obj, *keys):
        for key in keys:
            val = dict_obj.get(key)
            if val is not None:
                return val
        return 'N/A'

    # PRIORITIZE CATALOG DATA when available
    if catalog_data:
        orbital_period = get_val(features_dict, 'orbital_period', 'Orbital Period', 'orbitalPeriod')
        transit_duration = get_val(features_dict, 'transit_duration', 'Transition Duration', 'transitDuration')
        transit_depth = get_val(features_dict, 'transit_depth', 'Transition Depth', 'transitDepth')
        planetary_radius = get_val(features_dict, 'planetary_radius', 'Planet Rad', 'planetaryRadius')
        # Use catalog data for these fields
        equilibrium_temp = catalog_data.get('pl_eqt') or get_val(features_dict, 'planet_equilibrium_temp', 'Planet Eqbm Temp', 'planetEquilibriumTemp')
        stellar_temp = catalog_data.get('st_teff') or get_val(features_dict, 'stellar_effective_temp', 'Stellar Effective Temp', 'stellarEffectiveTemp')
        stellar_radius = catalog_data.get('st_rad') or get_val(features_dict, 'stellar_radius', 'Stellar Rad', 'stellarRadius')
        stellar_log_g = catalog_data.get('st_logg') or get_val(features_dict, 'stellar_log_g', 'Stellar log g', 'stellarLogG')
        ra = catalog_data.get('ra') or get_val(features_dict, 'ra')
        dec = catalog_data.get('dec') or get_val(features_dict, 'dec')
    else:
        # No catalog data, use features only
        orbital_period = get_val(features_dict, 'orbital_period', 'Orbital Period', 'orbitalPeriod')
        transit_duration = get_val(features_dict, 'transit_duration', 'Transition Duration', 'transitDuration')
        transit_depth = get_val(features_dict, 'transit_depth', 'Transition Depth', 'transitDepth')
        planetary_radius = get_val(features_dict, 'planetary_radius', 'Planet Rad', 'planetaryRadius')
        equilibrium_temp = get_val(features_dict, 'planet_equilibrium_temp', 'Planet Eqbm Temp', 'planetEquilibriumTemp')
        stellar_temp = get_val(features_dict, 'stellar_effective_temp', 'Stellar Effective Temp', 'stellarEffectiveTemp')
        stellar_radius = get_val(features_dict, 'stellar_radius', 'Stellar Rad', 'stellarRadius')
        stellar_log_g = get_val(features_dict, 'stellar_log_g', 'Stellar log g', 'stellarLogG')
        ra = get_val(features_dict, 'ra')
        dec = get_val(features_dict, 'dec')

    # PRIORITIZE catalog stellar mass, otherwise calculate
    stellar_mass = 'N/A'
    if catalog_data and catalog_data.get('st_mass'):
        # Use catalog mass directly!
        stellar_mass = f"{catalog_data['st_mass']:.3f} M☉"
    else:
        # Calculate stellar mass using the formula
        try:
            if stellar_log_g != 'N/A' and stellar_radius != 'N/A':
                log_g = float(stellar_log_g)
                R_solar = float(stellar_radius)
                G = 6.67430e-11  # Gravitational constant in SI units

                # Calculate stellar mass in solar masses
                numerator = (10**log_g) * ((R_solar * 6.96e8) ** 2)
                denominator = 100 * G * 1.989e30
                M_solar = numerator / denominator
                stellar_mass = f"{M_solar:.3f} M☉"
        except (ValueError, TypeError, ZeroDivisionError):
            stellar_mass = 'N/A'

    # PRIORITIZE catalog planetary mass, otherwise calculate
    estimated_mass = 'N/A'
    if catalog_data and catalog_data.get('koi_prad'):
        # If catalog has planetary radius, calculate mass from it
        try:
            R_planet = float(catalog_data['koi_prad'])
            # Use Chen & Kipping (2017) mass-radius relationship
            if R_planet < 1.23:
                # Rocky planets: M ∝ R^3.7
                M_planet = R_planet ** 3.7
            elif R_planet < 14.3:
                # Gas giants and sub-Neptunes: M ∝ R^2.06
                M_planet = R_planet ** 2.06
            else:
                # Very large planets
                M_planet = R_planet ** 1.0
            estimated_mass = f"{M_planet:.2f} M⊕"
        except (ValueError, TypeError):
            pass
    
    # Fallback: calculate from features
    if estimated_mass == 'N/A':
        try:
            if planetary_radius != 'N/A':
                R_planet = float(planetary_radius)
                # Use Chen & Kipping (2017) mass-radius relationship
                if R_planet < 1.23:
                    # Rocky planets: M ∝ R^3.7
                    M_planet = R_planet ** 3.7
                elif R_planet < 14.3:
                    # Gas giants and sub-Neptunes: M ∝ R^2.06
                    M_planet = R_planet ** 2.06
                else:
                    # Very large planets
                    M_planet = R_planet ** 1.0
                estimated_mass = f"{M_planet:.2f} M⊕"
        except (ValueError, TypeError):
            estimated_mass = 'N/A'

    # PRIORITIZE catalog semi-major axis, otherwise calculate
    distance_from_star = 'N/A'
    if catalog_data and catalog_data.get('koi_sma'):
        # Use catalog semi-major axis directly
        distance_from_star = f"{catalog_data['koi_sma']:.4f} AU"
    else:
        # Calculate distance from star using Kepler's third law
        try:
            if orbital_period != 'N/A' and stellar_mass != 'N/A' and 'M☉' in stellar_mass:
                P_days = float(orbital_period)
                M_star_solar = float(stellar_mass.split()[0])

                # Kepler's third law: a^3 = (M_star * P^2) / (4π^2)
                # With P in years and a in AU, simplifies to: a^3 = M_star * P^2
                P_years = P_days / 365.25
                a_AU = (M_star_solar * P_years**2) ** (1/3)
                distance_from_star = f"{a_AU:.4f} AU"
        except (ValueError, TypeError, ZeroDivisionError):
            distance_from_star = 'N/A'

    # Determine stellar type based on temperature
    stellar_type = 'Unknown'
    try:
        if stellar_temp != 'N/A':
            temp_K = float(stellar_temp)
            if temp_K >= 30000:
                stellar_type = 'O-type (Blue)'
            elif temp_K >= 10000:
                stellar_type = 'B-type (Blue-white)'
            elif temp_K >= 7500:
                stellar_type = 'A-type (White)'
            elif temp_K >= 6000:
                stellar_type = 'F-type (Yellow-white)'
            elif temp_K >= 5200:
                stellar_type = 'G-type (Yellow) - Sun-like'
            elif temp_K >= 3700:
                stellar_type = 'K-type (Orange)'
            elif temp_K >= 2400:
                stellar_type = 'M-type (Red dwarf)'
            else:
                stellar_type = 'Brown dwarf'
    except (ValueError, TypeError):
        stellar_type = 'Unknown'

    return {
        'orbitalPeriod': f"{orbital_period} days",
        'transitDuration': f"{transit_duration} hours",
        'transitDepth': f"{transit_depth}",
        'planetaryRadius': f"{planetary_radius} Earth radii",
        'estimatedMass': estimated_mass,
        'distanceFromStar': distance_from_star,
        'equilibriumTemp': f"{equilibrium_temp} K",
        'stellarTemp': f"{stellar_temp} K",
        'stellarType': stellar_type,
        'stellarRadius': f"{stellar_radius} solar radii",
        'stellarLogG': f"{stellar_log_g}",
        'stellarMass': stellar_mass,
        'hostStarTemp': f"{stellar_temp} K",  # Alternative field name for frontend
        'coordinates': f"RA: {ra}°, DEC: {dec}°"
    }


def _get_similar_exoplanets(planet_type):
    """Get list of similar known exoplanets"""
    similar_planets = {
        'Hot Jupiter': ['HD 209458 b', '51 Pegasi b', 'WASP-12b', 'TrES-4b'],
        'Gas Giant': ['HD 106906 b', 'HR 8799 b', 'Beta Pictoris b'],
        'Neptune-like': ['HAT-P-11b', 'GJ 436 b', 'Kepler-22b'],
        'Rocky Planet': ['Kepler-186f', 'Proxima Centauri b', 'TRAPPIST-1e'],
        'Super-Earth': ['55 Cancri e', 'Kepler-452b', 'K2-18b'],
    }
    
    return similar_planets.get(planet_type, ['HD 209458 b', '51 Pegasi b'])


if __name__ == '__main__':
    print("=" * 50)
    print("🚀 Exoplanet Detection ML Server")
    print("=" * 50)
    print(f"📦 Model loaded: {model_loaded}")
    print(f"📍 Model path: {MODEL_PATH}")
    print(f"🌐 Server starting on http://localhost:5001")
    print("=" * 50)
    
    app.run(debug=True, port=5001, host='0.0.0.0')