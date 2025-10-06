import lightkurve as lk
import numpy as np
import astropy.units as u
from astropy.constants import G
from transitleastsquares import transitleastsquares
from lightkurve import SFFCorrector
from ldtk import LDPSetCreator, BoxcarFilter
from lightkurve.correctors import DesignMatrix, RegressionCorrector
import requests
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
from astropy.constants import G, M_sun, R_sun, R_earth



# global Planet
#
# search_result = lk.search_lightcurve(Planet)
# lc = search_result[0:10].download()
#
def lightCurveCorrection(lc):
    print("🔧 Correcting light curve...")
    lc = lc.remove_nans().remove_outliers().normalize()
    print("   Removed NaNs and outliers, normalized flux")
    
    X = np.vstack([lc.centroid_col.value, lc.centroid_row.value]).T
    dm = DesignMatrix(X, name="centroids").append_constant()
    rc = RegressionCorrector(lc)
    lc_corr = rc.correct(dm, sigma=5)
    print("✅ Light curve correction complete")
    return lc_corr

def correct_tls_depth_duration(results, time_days, f_target=None):
    cadence_hr = float(np.median(np.diff(np.asarray(time_days))) * 24.0)

    # TLS raw outputs
    T_fit_hr = float(results.duration) * 24
    depth_frac = float(results.depth)

    # Default: no dilution if not supplied
    if f_target is None:
        f_target = 1.0

    # Smearing correction (box convolved with box of width = cadence)
    T_true_hr = T_fit_hr - cadence_hr
    if T_true_hr <= 0:
        T_true_hr = T_fit_hr * 0.8  # Fallback to prevent errors

    # Physical corrections
    k_smear = 1.0 + (cadence_hr / T_true_hr)   # boosts depth
    k_dilute = 1.0 / f_target                   # undilute depth
    k_total = k_smear * k_dilute

    # Apply physical corrections
    depth_ppm_corr = (1 - depth_frac) * 1e6 * k_total
    duration_hr_corr = T_true_hr

    return depth_ppm_corr, duration_hr_corr, k_total


def _contiguous_spans(time, gap_days=0.75):
    t = np.asarray(time)
    t = t[np.isfinite(t)]
    if t.size == 0: return []
    cuts = np.where(np.diff(t) > gap_days)[0] + 1
    segs = np.split(t, cuts)
    return [(s[0], s[-1], s[-1]-s[0]) for s in segs if s.size > 1]

def choose_period_window(time,
                         n_transits_min=2,
                         min_samples_in_transit=4,
                         duty_cycle_max=0.08,
                         period_floor_days=0.3,
                         period_cap_days=None,
                         gap_days=0.75):
    spans = _contiguous_spans(time, gap_days=gap_days)
    if not spans:
        raise ValueError("No valid time samples.")
    # use the largest contiguous segment
    Tseg = max(spans, key=lambda x: x[2])[2]
    # cadence (median spacing)
    t = np.asarray(time)
    dt = np.median(np.diff(np.sort(t[np.isfinite(t)])))
    # P_max from seeing at least N transits in the segment
    if n_transits_min < 2:
        Pmax = Tseg
    else:
        Pmax = Tseg / (n_transits_min - 1)
    if period_cap_days is not None:
        Pmax = min(Pmax, period_cap_days)
    # P_min from needing m samples in-transit and limiting duty cycle
    Tmin = min_samples_in_transit * dt
    Pmin = max(Tmin / duty_cycle_max, period_floor_days)
    # safety ordering
    if Pmin >= Pmax:
        # fallback: relax constraints just enough to be valid
        Pmin = 0.9 * Pmax
    return float(Pmin), float(Pmax)


def qlp_style_detrend(lc, knot_spacing_days=2.0, k=3, n_iter=3, sigma=3.0, gap_days=0.75):
    # 1) keep only good cadences
    if hasattr(lc, "quality"):
        lc = lc[lc.quality == 0]
    t = lc.time.value
    f = lc.flux.value / np.nanmedian(lc.flux.value)
    e = lc.flux_err.value if lc.flux_err is not None else None

    good = np.isfinite(t) & np.isfinite(f)
    if e is not None:
        good &= np.isfinite(e)
    t, f = t[good], f[good]
    if e is not None:
        e = e[good]

    # 2) split into segments (like per orbit/quarter); here by time gaps
    cut = np.where(np.diff(t) > gap_days)[0] + 1
    idx_segments = np.split(np.arange(len(t)), cut)

    tout, fout, eout = [], [], []
    for idx in idx_segments:
        ts = t[idx]; fs = f[idx]; es = e[idx] if e is not None else None
        if len(ts) < 10:
            continue

        # internal knots every ~knot_spacing_days (avoid endpoints)
        ks = knot_spacing_days
        if ts.max() - ts.min() > 2*ks:
            knots = np.arange(ts.min() + ks, ts.max() - ks, ks)
        else:
            knots = None

        mask = np.ones_like(fs, bool)
        for _ in range(n_iter):
            # fit spline to current inliers
            if knots is not None and len(knots) > 0 and len(ts[mask]) > (k+1):
                try:
                    spl = LSQUnivariateSpline(ts[mask], fs[mask], knots, k=k)
                except Exception:
                    spl = UnivariateSpline(ts[mask], fs[mask], k=k, s=len(ts[mask])*0.5)
            else:
                spl = UnivariateSpline(ts[mask], fs[mask], k=k, s=len(ts[mask])*0.5)

            trend = spl(ts)
            resid = fs - trend
            mad = 1.4826 * np.nanmedian(np.abs(resid - np.nanmedian(resid))) + 1e-12
            mask = np.abs(resid) < sigma * mad  # 3σ clipping

        trend = spl(ts)
        fd = fs / trend
        tout.append(ts)
        fout.append(fd)
        if es is not None:
            eout.append(es / trend)

    # 3) stitch segments and renormalize
    t_final = np.concatenate(tout) if len(tout) else t
    f_final = np.concatenate(fout) if len(fout) else f
    f_final = f_final / np.nanmedian(f_final)
    e_final = (np.concatenate(eout) if e is not None and len(eout) else None)

    return lk.LightCurve(time=t_final, flux=f_final, flux_err=e_final)

def catalog_info(s, period_days=None):
    """
    Query NASA Exoplanet Archive for stellar/planetary parameters
    Uses cumulative table for Kepler targets
    """
    BASE_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query="
    
    # Build query based on target type
    if s.startswith("KIC"):
        KIC = int(s.split()[1])
        if not isinstance(KIC, int):
            raise TypeError('KIC ID must be of type "int"')
        # Use cumulative table (Kepler mission data)
        query = f"SELECT+*+FROM+cumulative+WHERE+kepid={KIC}"
    elif s.startswith("EPIC"):
        EPIC = int(s.split()[1])
        if not isinstance(EPIC, int):
            raise TypeError('EPIC ID must be of type "int"')
        if (EPIC < 201000001) or (EPIC > 251813738):
            raise TypeError("EPIC ID must be in range 201000001 to 251813738")
        query = f"SELECT+*+FROM+k2pandc+WHERE+epic_hostname={EPIC}"
    elif s.startswith("TIC"):
        TIC = int(s.split()[1])
        if not isinstance(TIC, int):
            raise TypeError('TIC ID must be of type "int"')
        query = f"SELECT+*+FROM+toi+WHERE+tid={TIC}"
    else:
        raise ValueError(f"Unknown target format: {s}. Must start with KIC, EPIC, or TIC")
    
    URL = BASE_URL + query + "&format=json"
    print(f"Querying: {URL}")
    
    try:
        response = requests.get(URL, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error querying database: {e}")
        return {}
    
    data = {}
    if response.status_code == 200:
        json_data = response.json()
        if len(json_data) == 0:
            print(f"No data found for {s}")
            return {}
        
        data = json_data[0]
        print("Data fetched successfully!")
        print(f"Available columns: {list(data.keys())[:20]}...")  # Show first 20 columns
        print(f"Has st_mass: {data.get('st_mass')}, Has koi_srad: {data.get('koi_srad')}, Has koi_slogg: {data.get('koi_slogg')}")
        
        # Map cumulative table column names to standard names
        # Cumulative table uses koi_* prefixes
        if 'koi_srad' in data and ('st_rad' not in data or data.get('st_rad') is None):
            data['st_rad'] = data['koi_srad']
        if 'koi_slogg' in data and ('st_logg' not in data or data.get('st_logg') is None):
            data['st_logg'] = data['koi_slogg']
        if 'koi_steff' in data and ('st_teff' not in data or data.get('st_teff') is None):
            data['st_teff'] = data['koi_steff']
        
        # Calculate stellar mass from log(g) and radius if not provided
        if "st_mass" not in data or data["st_mass"] is None:
            if data.get("st_logg") and data.get("st_rad"):
                # M = g * R^2 / G, where g = 10^(log g) cm/s^2
                g_cgs = 10**data["st_logg"]  # cm/s^2
                R_cgs = data["st_rad"] * R_sun.cgs.value  # cm
                M_cgs = (g_cgs * R_cgs**2) / G.cgs.value  # g
                solar_mass = M_cgs / M_sun.cgs.value  # solar masses
                
                # Estimate error propagation
                try:
                    logg_err = max(abs(data.get("st_loggerr1", 0)), abs(data.get("st_loggerr2", 0)))
                    rad_err = max(abs(data.get("st_raderr1", 0)), abs(data.get("st_raderr2", 0)))
                    
                    # Simplified error propagation
                    mass_err_frac = np.sqrt((np.log(10) * logg_err)**2 + (2 * rad_err / data["st_rad"])**2)
                    solar_err = solar_mass * mass_err_frac
                except:
                    solar_err = 0.1 * solar_mass  # 10% default error
                
                data["st_mass"] = solar_mass
                data["st_masserr"] = solar_err
                print(f"Calculated stellar mass: {solar_mass:.3f} ± {solar_err:.3f} M_sun")
        
        # Calculate semi-major axis if we have period and stellar mass
        if period_days and data.get("st_mass"):
            # Kepler's third law: a^3 = G*M*P^2 / (4*pi^2)
            a_AU = ((G.value * data["st_mass"] * M_sun.value * (period_days * 86400)**2) / (4 * np.pi**2))**(1/3) / 1.496e11
            data["a"] = a_AU
            print(f"Calculated semi-major axis: {a_AU:.3f} AU")
    
    return data

# Alias for convenience
catalog = catalog_info
def computeTLS(lc_new, catalog, Pmin=None, Pmax=None):
    """
    Compute Transit Least Squares for exoplanet detection (OPTIMIZED for speed)
    
    Args:
        lc_new: Detrended light curve
        catalog: Stellar catalog information
        Pmin: Minimum period to search (optional)
        Pmax: Maximum period to search (optional)
    """
    print("⚡ Starting TLS computation...")
    
    # OPTIMIZATION: Limit data size for faster processing
    n_points = len(lc_new.time.value)
    if n_points > 10000:
        print(f"   Large dataset detected ({n_points} points). Subsampling to 10000 points for speed...")
        # Subsample but keep the same time range
        indices = np.linspace(0, n_points - 1, 10000, dtype=int)
        time_sub = lc_new.time.value[indices]
        flux_sub = lc_new.flux[indices]
        flux_err_sub = lc_new.flux_err[indices]
    else:
        time_sub = lc_new.time.value
        flux_sub = lc_new.flux
        flux_err_sub = lc_new.flux_err
        print(f"   Processing {n_points} data points...")
    
    # If period bounds not provided, calculate them with tighter constraints
    if Pmin is None or Pmax is None:
        Pmin_calc, Pmax_calc = choose_period_window(
            time_sub, 
            n_transits_min=2, 
            min_samples_in_transit=5, 
            duty_cycle_max=0.08
        )
        Pmin = Pmin if Pmin is not None else max(Pmin_calc, 0.5)  # At least 0.5 days
        Pmax = Pmax if Pmax is not None else min(Pmax_calc, 30.0)  # Cap at 30 days
    
    print(f"   Period search range: {Pmin:.2f} - {Pmax:.2f} days")
    
    tls = transitleastsquares(time_sub, flux_sub, flux_err_sub)

    # OPTIMIZED parameters for speed (2-3x faster)
    tls_params = {
        "period_max": Pmax,
        "period_min": Pmin,
        "limb_dark": "quadratic",  # Fixed typo: was "quadric"
        "transit_template": "default",  # Faster than "box"
        "oversampling_factor": 2,  # Reduced from 5 to 2 (2.5x speedup!)
        "duration_grid_step": 1.15,  # Coarser grid (was default 1.1)
        "n_transits_min": 2,
        "T0_fit_margin": 0.05  # Small margin instead of 0
    }

    # Only add catalog parameters if they exist and are not None
    if catalog.get("st_rad") is not None:
        tls_params["R_star"] = catalog["st_rad"]
        if catalog.get("st_raderr1") is not None and catalog.get("st_raderr2") is not None:
            tls_params["R_star_min"] = catalog["st_rad"] + catalog["st_raderr2"]
            tls_params["R_star_max"] = catalog["st_rad"] + catalog["st_raderr1"]

    if catalog.get("st_mass") is not None:
        tls_params["M_star"] = catalog["st_mass"]
        if catalog.get("st_masserr") is not None:
            tls_params["M_star_min"] = catalog["st_mass"] - catalog["st_masserr"]
            tls_params["M_star_max"] = catalog["st_mass"] + catalog["st_masserr"]

    if catalog.get("a") is not None:
        tls_params["a"] = catalog["a"]

    if catalog.get("pl_rade") is not None:
        tls_params["rp"] = catalog["pl_rade"]

    print("   Running TLS search (this may take 30-60 seconds)...")
    r = tls.power(**tls_params)
    print(f"✅ TLS complete! Found period: {r.period:.3f} days, SDE: {r.SDE:.2f}")
    
    return tls, r


def sapCorrection(lc_new, r, tpf):
    # Access header of the primary extension or the flux extension
    hdr = tpf.hdu[1].header
    crowdsap = hdr.get("CROWDSAP", None)
    flfrcsap = hdr.get("FLFRCSAP", None)
    print("CROWDSAP:", crowdsap)
    print(flfrcsap)
    
    return correct_tls_depth_duration(r, lc_new.time.value,flfrcsap)
#
# SDE = float(r.SDE)
# P = float(r.period)
# T0 = float(r.T0+2457000.0)
#
# print(f"\nTLS Results:")
# print(f"Period: {P:.6f} days")
# print(f"T0: {T0:.6f} BJD")
# print(f"Transit Duration: {Dur_h:.2f} hours")
# print(f"Transit Depth: {Depth:.1f} ppm")
# print(f"SDE: {SDE:.1f}")
#


