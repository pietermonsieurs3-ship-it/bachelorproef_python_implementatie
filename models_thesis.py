import numpy as np
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import lightkurve as lk
from scipy.signal import stft
from scipy.ndimage import gaussian_filter1d
from scipy.signal.windows import dpss
from numpy.fft import rfft
from scipy.signal import find_peaks
from numpy.linalg import svd
from scipy.ndimage import gaussian_filter1d

class Thesis:
    """
    Minimal thesis analysis class:
    - load data
    - plot time series
    - compute Fourier spectrum (FFT)
    """

    def __init__(self):
        self.t = None
        self.y = None
        self.filename = None
        self.y_original = None
        self.y_err = None

    # -------------------------
    # Load data
    # -------------------------
    
    @staticmethod
    def mag_to_flux(mag):
        mag = np.array(mag)
        flux = 10 ** (-0.4 * (mag - np.nanmedian(mag)))
        flux = flux / np.nanmedian(flux)
        return flux
    
    def load_data(self, filename):
        data = np.loadtxt(filename)

        self.t = data[:, 0] # first colomn
        self.y = data[:, 1] # second column
        self.y_err = data[:, 2] # third column
        
        self.t = np.asarray(self.t)
        self.y = np.asarray(self.y)
        self.y_err = np.asarray(self.y_err)

        mask = np.isfinite(self.t) & np.isfinite(self.y)
        self.t = self.t[mask]
        self.y = self.y[mask]
        self.y_err = self.y_err[mask]
        
        print(np.min(self.y_err), np.max(self.y_err))
        self.y_original = self.y.copy()
        self.filename = filename.split("/")[-1].replace(".txt", "")

        print(f"[LOAD] {filename}")
        print(f"[LOAD] Number of data points: {len(self.t)}")

        return self.t, self.y, self.y_err

    # -------------------------
    # Time series plot
    # -------------------------
    def plot_time_series(self, savepath=None):
        plt.figure()
        plt.errorbar(
            self.t,
            self.y,
            yerr=self.y_err,
            fmt='o',
            markersize=3,
            alpha=0.7,
            elinewidth=1,
            capsize=2
        )
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude (unitless)")
        plt.title(f"{self.filename} — Time Series")
            
        plt.savefig("name1.jpg", dpi=150, bbox_inches="tight")
        plt.show()

    # -------------------------
    # LombScargle spectrum
    # -------------------------
    
    def lomb_scargle_spectrum(self, min_freq=0.01, max_freq=20, samples_per_peak=20):

        flux = self.mag_to_flux(self.y)

        # --- KEY: use baseline ---
        baseline = self.t.max() - self.t.min()

        df = 1 / (samples_per_peak * baseline)

        freq = np.arange(min_freq, max_freq, df)

        ls = LombScargle(self.t, flux)
        
        power = ls.power(freq)

        return freq, power
        
    def plot_spectrum_lk(self, savepath=None):

        pg = self.lightkurve_periodogram()

        freq = pg.frequency.value
        power = pg.power.value

        plt.figure(figsize=(7,4))

        plt.plot(freq, power / np.max(power), color="black")

        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized Power")
        plt.title(f"{self.filename} — Lightkurve Periodogram")

        plt.xlim(0, 20)

        plt.savefig("name2.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
    # -------------------------
    # Find dominant frequency
    # -------------------------
    def find_dominant_frequency(self, freq, power):
        """
        Returns the frequency with highest power
        """
        idx = np.argmax(power)
        peaks, _ = find_peaks(power, height=0.2*np.max(power))
        f_peak = freq[peaks[np.argmax(power[peaks])]]
        p_peak = power[idx]

        return f_peak, p_peak
    
    # -------------------------
    # Plot spectrum
    # -------------------------
    def plot_spectrum(self, freq, power, f_peak=None, savepath=None):

        fig, ax = plt.subplots()

        ax.plot(freq, power, color="black", lw=1)

        ax.set_xlabel("Frequency (cycles/day)")
        ax.set_ylabel("Power")
        ax.set_title(f"{self.filename} — Periodogram (Enhanced)")

        # highlight peak
        if f_peak is not None:
            idx = np.argmin(np.abs(freq - f_peak))
            ax.scatter(freq[idx], power[idx], color="red", s=80, label="f₀")

            # harmonics
            for n in range(2, 6):
                fh = n * f_peak
                if fh > freq.max():
                    continue
                i = np.argmin(np.abs(freq - fh))
                ax.scatter(freq[i], power[i], color="orange", s=40)

            ax.legend()

        # add zoom inset (this gives “Mohammed richness feel”)
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes

        axins = inset_axes(ax, width="45%", height="45%", loc="upper right")
        axins.plot(freq, power, color="gray")

        if f_peak is not None:
            zoom_mask = (freq > f_peak*0.8) & (freq < f_peak*1.2)
            axins.plot(freq[zoom_mask], power[zoom_mask], color="blue")

        plt.savefig("name11.jpg", dpi=150, bbox_inches="tight")

        plt.show()
    
    # -------------------------
    # Sine model
    # -------------------------
    def sine_model(self, t, A, phi, f, C):
        return A * np.sin(2 * np.pi * f * t + phi) + C
        
    def fft_spectrum(self, min_freq=0.01):
        """
        FFT-based spectrum (for comparison only).
        Works by interpolating onto a uniform grid.
        """

        y = self.mag_to_flux(self.y)
        y = y - np.mean(y)

        # interpolate to uniform grid (critical fix)
        t_uniform = np.linspace(self.t.min(), self.t.max(), len(self.t))
        y_uniform = np.interp(t_uniform, self.t, y)

        # apply window (important!)
        window = np.hanning(len(y_uniform))
        y_uniform *= window

        dt = t_uniform[1] - t_uniform[0]
        n = len(y_uniform)

        fft_vals = np.fft.rfft(y_uniform)
        power = np.abs(fft_vals) ** 2
        freq = np.fft.rfftfreq(n, d=dt)

        # remove low-frequency drift
        mask = freq >= min_freq

        return freq[mask], power[mask]
    
    def plot_fft_spectrum(self, savepath=None):
        """
        FFT spectrum plot (for comparison with LS and Lightkurve).
        """

        freq, power = self.fft_spectrum()

        # normalize for fair comparison
        power = power / np.max(power)

        plt.figure()

        plt.plot(freq, power, color="purple", lw=1)

        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized Power")
        plt.title(f"{self.filename} — FFT Spectrum")

        plt.xlim(0, 20)

        if savepath:
            plt.savefig(savepath, dpi=150, bbox_inches="tight")
        else:
            plt.savefig("fft_spectrum.jpg", dpi=150, bbox_inches="tight")

        plt.show()

    # -------------------------
    # Residuals for fitting
    # -------------------------
    def _residuals_sine(self, params, t, y, f):
        A, phi, C = params
        model = self.sine_model(t, A, phi, f, C)
        return y - model
        
    # -------------------------
    def fit_sine(self, f_init, y_data=None):

        if y_data is None:
            y_data = self.y

        def model(params, t):
            A, f, phi, C = params
            return A * np.sin(2 * np.pi * f * t + phi) + C

        def residuals(params, t, y):
            return y - model(params, t)

        A0 = np.std(y_data)
        phi0 = 0.0
        C0 = np.mean(y_data)
        f0 = f_init

        params0 = [A0, f0, phi0, C0]

        result = least_squares(
            residuals,
            params0,
            args=(self.t, y_data)
        )

        A, f, phi, C = result.x

        phi = (phi + np.pi) % (2*np.pi) - np.pi
        
        phi = np.mod(phi, 2*np.pi)
        if phi > np.pi:
            phi -= 2*np.pi

        # covariance → uncertainties
        J = result.jac
        residual_var = np.sum(result.fun**2) / (len(result.fun) - len(result.x))

        U, s, Vt = svd(J, full_matrices=False)
        cov = Vt.T @ np.diag(1 / (s**2 + 1e-12)) @ Vt * residual_var
        errors = np.sqrt(np.diag(cov))

        σA = errors[0]
        σf = errors[1]
        σφ = errors[2]

        return A, f, phi, C, σA, σf, σφ
        
    # -------------------------
    # Plot fitted sine model
    # -------------------------
    def plot_fit(self, f, A, phi, C, savepath=None):
        model = self.sine_model(self.t, A, phi, f, C)

        plt.figure()
        plt.scatter(self.t, self.y, s=10)
        plt.plot(self.t, model)
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude (unitless)")
        plt.title(f"{self.filename} — Sine Fit")
        
        plt.savefig("name3.jpg", dpi=150, bbox_inches="tight")
        
        plt.show()
        
    def plot_phase_folded(self, f, savepath=None):
        phase = (self.t * f) % 1

        plt.figure()
        plt.scatter(phase, self.y, c=phase, cmap="viridis", s=8)
        plt.colorbar(label="Phase")
        
        plt.xlabel("Phase (0–1 cycle)")
        plt.ylabel("Magnitude (unitless)")
        plt.title(f"{self.filename} — Phase-folded light curve (f = {f:.6f})")
        
        plt.savefig("name4.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
        # plt.scatter(self.t, self.y, c=self.t, cmap="plasma", s=8)
        # plt.colorbar(label="Time")
        
    def plot_spectral_window(self, max_freq=20, nfreq=10000, savepath=None, zoom_xlim=(0, 0.4)):

        t = self.t
        freq = np.linspace(0, max_freq, nfreq)

        window_power = np.abs(
            np.sum(np.exp(2j * np.pi * np.outer(freq, t)), axis=1)
        )**2

        window_power /= np.max(window_power)

        # --------------------
        # FULL PLOT
        # --------------------
        plt.figure()
        plt.plot(freq, window_power)
        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized spectral window power")
        plt.title(f"{self.filename} — Spectral Window (Full)")

        plt.savefig("namebaby.jpg", dpi=150, bbox_inches="tight")

        plt.show()

        # --------------------
        # ZOOMED PLOT
        # --------------------
        plt.figure()
        plt.plot(freq, window_power)
        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized spectral window power")
        plt.title(f"{self.filename} — Spectral Window (Zoom)")

        plt.xlim(*zoom_xlim)

        plt.savefig("namebabyzoomin.jpg", dpi=150, bbox_inches="tight")

        plt.show()
    
    def plot_petersen_diagram(self, freqs, amplitudes, savepath=None):

        periods = 1 / np.array(freqs)

        # assume fundamental = strongest amplitude mode
        idx0 = np.argmax(amplitudes)
        P0 = periods[idx0]

        idx1 = np.argsort(amplitudes)[-2]
        P1 = periods[idx1]

        ratio = P1 / P0

        plt.figure()

        plt.scatter(P0, ratio, s=80)
        plt.annotate("Your star", (P0, ratio))

        plt.xlabel("Period (days)")
        plt.ylabel("Period Ratio (P / P0)")
        plt.title(f"{self.filename} — Petersen Diagram")

        plt.savefig("name6.jpg", dpi=150, bbox_inches="tight")

        plt.show()
    
    def plot_data_with_model(self, f, A, phi, C, savepath=None, xlim=None, ylim=None):
        """
        Combined plot:
        - raw time series
        - best-fit sine model
        """

        model = self.sine_model(self.t, A, phi, f, C)
  
        plt.figure()

        # data
        plt.errorbar(
            self.t,
            self.y,
            yerr=self.y_err,
            fmt='o',
            markersize=3,
            alpha=0.6,
            label="Data"
        )

        # model (smooth curve → sort time!)
        idx = np.argsort(self.t)
        t_sorted = self.t[idx]

        model_sorted = self.sine_model(t_sorted, A, phi, f, C)
        plt.plot(t_sorted, model_sorted, color="red", label="Model")

        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude (unitless)")
        plt.title(f"{self.filename} — Time Series and Sine Fit (f = {f:.6f})")
        plt.legend()
        
        # -------------------------
        # ADD ZOOM CONTROL HERE
        # -------------------------
        if xlim is not None:
            plt.xlim(xlim)

        if ylim is not None:
            plt.ylim(ylim)

        plt.savefig("name7.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
    # -------------------------
    # Prewhitening (multi-frequency)
    # -------------------------
    def prewhiten(self, n_freqs=40, min_freq=0.01, max_freq=20):

        residual = self.y_original.copy()

        frequencies = []
        amplitudes = []
        phases = []
        offsets = []

        freq_errors = []
        amp_errors = []
        phase_errors = []

        for i in range(n_freqs):

            print(f"\n--- Iteration {i+1} ---")
            snr_list = []
            # LombScargle on residual ONLY (no self.y hacking)
            baseline = self.t.max() - self.t.min()
            df = 1 / (10 * baseline)   # oversampling factor ~5. For dhpeg: 10 works best.
            freq = np.arange(min_freq, max_freq, df)

            ls = LombScargle(self.t, residual)
            power = ls.power(freq)
            
            # if signal/noise < 2 or power.max() < threshold:
              #  break

            f_peak = freq[np.argmax(power)]

            print(f"Found frequency: {f_peak:.6f}")

            # skip duplicates BEFORE fitting
            if self.is_duplicate(f_peak, frequencies, tol=0.02):
                print("Duplicate detected → skipping")
                continue
    
            # fit on ORIGINAL data (standard approach)
            A, f_fit, phi, C, σA, σf, σφ = self.fit_sine(f_peak, y_data=residual)

            # enforce positive amplitude
            if A < 0:
                A = -A
                phi += np.pi

            frequencies.append(f_fit)
            amplitudes.append(A)
            phases.append(phi)
            offsets.append(C)

            freq_errors.append(σf)
            amp_errors.append(σA)
            phase_errors.append(σφ)

            # subtract model
            model = A * np.sin(2*np.pi*f_fit*self.t + phi)
            residual = residual - model

            # keep residual centered (prevents drift)
            residual = residual - np.mean(residual)
            
            noise_level = np.std(residual)
            snr_i = A / (noise_level + 1e-12)
            
            snr_list.append(snr_i)

        return (
            frequencies,
            amplitudes,
            phases,
            offsets,
            
            freq_errors,
            amp_errors,
            phase_errors,
            
            snr_list
        )
         
    def is_duplicate(self, f, freq_list, tol=0.005):
        """
        Checks if frequency is already present within tolerance
        """
        return any(abs(f - f0) < tol for f0 in freq_list)
    
    def multi_sine_model(self, t, params, freqs):
      
        model = np.zeros_like(t)

        n = len(freqs)

        for i in range(n):
            A = params[2*i]
            phi = params[2*i + 1]
            model += A * np.sin(2 * np.pi * freqs[i] * t + phi)

        C = params[-1]
        model += C

        return model
        
    def _residuals_multi(self, params, t, y, freqs):
        model = self.multi_sine_model(t, params, freqs)
        return y - model

    def fit_multi_sine(self, freqs_init):

        n = len(freqs_init)
  
        def model(params, t):
            y = np.zeros_like(t)

            for i in range(n):
                A = params[3*i]
                f = params[3*i + 1]
                phi = params[3*i + 2]
                y += A * np.sin(2 * np.pi * f * t + phi)

            C = params[-1]
            return y + C

        def residuals(params, t, y):
            return y - model(params, t)

        # ---- initial guesses ----
        params0 = []

        for f in freqs_init:
            params0 += [
                np.std(self.y),  # A
                f,               # frequency
                0.0              # phase
            ]

        params0 += [np.mean(self.y)]  # offset

        result = least_squares(
            residuals,
            params0,
            args=(self.t, self.y),
            max_nfev=5000
        )

        params = result.x

        amplitudes = []
        freqs = []
        phases = []

        for i in range(n):
            A = params[3*i]
            f = params[3*i + 1]
            phi = params[3*i + 2]

            # enforce stable amplitude-phase form
            if A < 0:
                A = -A
                phi += np.pi

            phi = (phi + np.pi) % (2*np.pi) - np.pi

            amplitudes.append(A)
            freqs.append(f)
            phases.append(phi)

        C = params[-1]

        model = self.multi_sine_model(self.t, params, freqs)

        rms = self.compute_rms(self.y, model)
        errors = self.parameter_uncertainties(result)

        signal = max(amplitudes)
        snr = signal / (rms + 1e-12)

        return amplitudes, freqs, phases, C, rms, snr, errors, result

    @staticmethod
    def pick_top_frequencies(freq, power, n_peaks=2, min_sep=0.2):
        peaks, _ = find_peaks(power)

        peak_freqs = freq[peaks]
        peak_power = power[peaks]

        # sort by power
        sorted_idx = np.argsort(peak_power)[::-1]

        selected = []

        for i in sorted_idx:
            f = peak_freqs[i]

            if all(abs(f - s) > min_sep for s in selected):
                selected.append(f)

            if len(selected) == n_peaks:
                break

        return sorted(selected)
    
    def compute_rms(self, y_true, y_model):
        residuals = y_true - y_model
        return np.sqrt(np.mean(residuals**2))
    
    def parameter_uncertainties(self, result):
    
        if len(result.x) >= len(self.y):
            raise ValueError("Model too complex for data size")
   
        J = result.jac
        residuals = result.fun

        # variance estimate
        s_sq = np.sum(residuals**2) / (len(residuals) - len(result.x))

        cov = np.linalg.pinv(J.T @ J) * s_sq
        errors = np.sqrt(np.diag(cov))

        return errors
    
    def compute_fourier_parameters(self, amplitudes, phases):
        """
        Computes simple Fourier-style diagnostics
        for multi-frequency fits.
        """
        
        if len(amplitudes) < 2:
            raise ValueError("Need at least 2 frequencies for Fourier parameters")

        A1, A2 = amplitudes[0], amplitudes[1]
        phi1, phi2 = phases[0], phases[1]

        eps = 1e-12
        R21 = A2 / (A1 + eps)
        R31 = A3 / (A1 + eps)

        phi21 = phi2 - phi1
        phi21 = (phi21 + np.pi) % (2*np.pi) - np.pi

        return R21, phi21
    
    def monte_carlo_frequency(self, f0, n_sim=200, window=0.05, fit_width=5):

        best_freqs = []
        residuals = self.y - np.mean(self.y)

        freq = np.linspace(f0 - window, f0 + window, 2000)

        for _ in range(n_sim):

            # noise realisation
            y_noisy = self.y + np.random.normal(
                0, np.std(residuals), size=len(self.y)
            )

            ls = LombScargle(self.t, y_noisy)
            power = ls.power(freq)

            # ---- local refinement instead of raw argmax ----
            idx_peak = np.argmax(power)

            # take local region around peak
            i1 = max(0, idx_peak - fit_width)
            i2 = min(len(freq), idx_peak + fit_width)

            f_local = freq[i1:i2]
            p_local = power[i1:i2]

            # quadratic fit: p = af^2 + bf + c
            coeffs = np.polyfit(f_local, p_local, 2)

            a, b, c = coeffs

            # vertex of parabola = -b / (2a)
            if a != 0:
                f_refined = -b / (2 * a)
            else:
                f_refined = freq[idx_peak]

            best_freqs.append(f_refined)

        best_freqs = np.array(best_freqs)

        return np.mean(best_freqs), np.std(best_freqs), best_freqs
    
    def fit_frequency_ls(self, f0):
        """
        Least-squares refinement INCLUDING frequency as a free parameter.
        Returns f and its formal uncertainty.
        """

        def model(params, t):
            A, f, phi, C = params
            return A * np.sin(2 * np.pi * f * t + phi) + C

        def residuals(params, t, y):
            return y - model(params, t)

        # initial guesses
        A0 = np.std(self.y)
        f_init = f0
        phi0 = 0.0
        C0 = np.mean(self.y)

        params0 = [A0, f_init, phi0, C0]

        result = least_squares(
            residuals,
            params0,
            args=(self.t, self.y)
        )

        A, f, phi, C = result.x

        # --- uncertainty from covariance ---
        J = result.jac
        residual_var = np.sum(result.fun**2) / (len(result.fun) - len(result.x))

        cov = np.linalg.pinv(J.T @ J) * residual_var
        errors = np.sqrt(np.diag(cov))

        f_err = errors[1]  # frequency is index 1

        return f, f_err, result
    
    def plot_phase_residuals(self, f, model, savepath=None):

        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.lines import Line2D

        # -------------------------
        # phase + residuals
        # -------------------------
        phase = (self.t * f) % 1
        residuals = self.y - model

        # sort by phase (for cleaner visual structure)
        idx = np.argsort(phase)
        phase = phase[idx]
        residuals = residuals[idx]

        # -------------------------
        # figure (slightly compact)
        # -------------------------
        plt.figure(figsize=(6.2, 4.3))

        # -------------------------
        # axis limits (robust padding)
        # -------------------------
        ymin, ymax = np.min(residuals), np.max(residuals)
        pad = 0.08 * (ymax - ymin + 1e-12)
        ymin -= pad
        ymax += pad

        # -------------------------
        # background (simple + non-distorting)
        # -------------------------
        plt.axhspan(ymin, ymax, color="#ff5a5a", alpha=0.10, zorder=0)

        # narrow "good-fit" band around zero (physically meaningful)
        sigma = np.std(residuals)
        tol = 0.4 * sigma

        plt.axhspan(-tol, tol, color="#00cc66", alpha=0.22, zorder=1)

        # zero line (keep neutral, not semantic color)
        plt.axhline(0, linestyle="--", color="black", linewidth=1.2, zorder=2)

        # -------------------------
        # data
        # -------------------------
        plt.scatter(
            phase,
            residuals,
            s=9,
            alpha=0.85,
            color="#1f77b4",
            edgecolor="none",
            zorder=3
        )

        plt.xlim(0, 1)
        plt.ylim(ymin, ymax)

        plt.xlabel("Phase")
        plt.ylabel("Residuals")
        plt.title(f"{self.filename} — Phase-Folded Residuals")

        # -------------------------
        # legend (clean + informative)
        # -------------------------
        good_patch = mpatches.Patch(color="#00cc66", alpha=0.22, label="Low-residual zone (±0.4σ)")
        bad_patch = mpatches.Patch(color="#ff5a5a", alpha=0.10, label="Background")
        zero_line = Line2D([0], [0], color="black", linestyle="--", linewidth=1.2, label="Zero residual")
        data_patch = Line2D([0], [0], marker='o', color='w',
                        markerfacecolor="#1f77b4",
                        markersize=6,
                        label="Residuals")

        legend = plt.legend(
            handles=[good_patch, bad_patch, zero_line, data_patch],
            title="Legend",
            loc="lower right",
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=0.95
        )

        legend.get_title().set_fontweight("bold")
        plt.tight_layout()
        plt.savefig("name8.jpg", dpi=150, bbox_inches="tight")
        plt.show()
  
    def lightkurve_periodogram(self, min_period=0.05, max_period=20, oversample_factor=10):

        flux = self.mag_to_flux(self.y)
        lc = lk.LightCurve(time=self.t, flux=flux)

        lc = lc.remove_nans()
        if len(lc) > 50:
            window = min(401, len(lc)//5)
            # lc = lc.flatten(window_length=window)

        pg = lc.to_periodogram(
            method="lombscargle",
            minimum_period=min_period,
            maximum_period=max_period,
            oversample_factor=oversample_factor
        )

        return pg
    
    def plot_rich_spectrum(self):

        pg = self.lightkurve_periodogram()

        freq = pg.frequency.value
        power = pg.power.value
        period = pg.period.value

        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        # -------------------------
        # Frequency view
        # -------------------------
        ax[0].plot(freq, power, color="black")
        ax[0].set_xlabel("Frequency (cycles/day)")
        ax[0].set_ylabel("Power")
        ax[0].set_title("Frequency Spectrum")
        ax[0].set_xlim(0, 20)   # zoom low-frequency structure
        
        # -------------------------
        # Period view (THIS is what gives detail feel)
        # -------------------------
        ax[1].plot(period, power, color="darkblue")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Period (days)")
        ax[1].set_ylabel("Power")
        ax[1].set_title("Period Spectrum (log scale)")
        ax[1].set_xlim(0.1, 10)
   
        plt.tight_layout()
        plt.show()
    
    def fft_spectrum(self):
        """
        FFT-based spectrum (uniform-grid approximation for comparison only).
        """

        y = self.mag_to_flux(self.y)
        y = y - np.mean(y)

        # interpolate to uniform grid (IMPORTANT improvement)
        t_uniform = np.linspace(self.t.min(), self.t.max(), len(self.t))
        y_uniform = np.interp(t_uniform, self.t, y)

        dt = t_uniform[1] - t_uniform[0]
        n = len(y_uniform)

        fft_vals = np.fft.rfft(y_uniform)
        power = np.abs(fft_vals)**2
        freq = np.fft.rfftfreq(n, d=dt)

        return freq, power
    
    def compare_spectra(self):
        """
        LS vs Lightkurve vs FFT comparison
        """

        # 1. Lightkurve
        pg = self.lightkurve_periodogram()
        f_lk = pg.frequency.value
        p_lk = pg.power.value

        # 2. FFT
        f_fft, p_fft = self.fft_spectrum()

        # 3. LS (your old method)
        f_ls, p_ls = self.lomb_scargle_spectrum()

        plt.figure(figsize=(12,6))

        plt.plot(f_ls, p_ls / np.max(p_ls), label="Lomb-Scargle", alpha=0.6)
        plt.plot(f_fft, p_fft / np.max(p_fft), label="FFT", alpha=0.6)
        plt.plot(f_lk, p_lk / np.max(p_lk), label="Lightkurve LS", alpha=0.8)

        plt.xlabel("Frequency")
        plt.ylabel("Normalized Power")
        plt.title(f"{self.filename} — Spectrum Comparison")
        plt.legend()
        plt.savefig("name9.jpg", dpi=150, bbox_inches="tight")
        plt.show()
        
    def compute_fourier_params_harmonics(self, amplitudes, phases):
        """
        Proper RR Lyrae Fourier parameters
        assumes amplitudes are [A1, A2, A3, ...]
        """

        A1 = amplitudes[0]
        A2 = amplitudes[1]
        A3 = amplitudes[2] if len(amplitudes) > 2 else 0

        phi1 = phases[0]
        phi2 = phases[1]
        phi3 = phases[2] if len(phases) > 2 else 0

        R21 = A2 / A1
        R31 = A3 / A1

        phi21 = phi2 - 2 * phi1
        phi31 = phi3 - 3 * phi1

        # wrap phases
        phi21 = (phi21 + np.pi) % (2*np.pi) - np.pi
        phi31 = (phi31 + np.pi) % (2*np.pi) - np.pi

        return R21, R31, phi21, phi31
        
    def compute_fourier_params_with_errors(self, amplitudes, phases, errors):
        """
        Computes Fourier parameters + propagated uncertainties
        Assumes:
            errors = [σA1, σφ1, σA2, σφ2, σA3, σφ3, ...]
        """

        # --- values ---
        A1, A2, A3 = amplitudes[0], amplitudes[1], amplitudes[2]
        phi1, phi2, phi3 = phases[0], phases[1], phases[2]

        # --- ratios ---
        R21 = A2 / A1
        R31 = A3 / A1

        # --- phase params ---
        phi21 = phi2 - 2 * phi1
        phi31 = phi3 - 3 * phi1

        # wrap phases
        phi21 = (phi21 + np.pi) % (2*np.pi) - np.pi
        phi31 = (phi31 + np.pi) % (2*np.pi) - np.pi

        # =========================
        # ERROR PROPAGATION
        # =========================

        σA1, σφ1 = errors[0], errors[1]
        σA2, σφ2 = errors[2], errors[3]
        σA3, σφ3 = errors[4], errors[5]

        # ---- R21 uncertainty ----
        σR21 = R21 * np.sqrt((σA2/A2)**2 + (σA1/A1)**2)

        # ---- R31 uncertainty ----
        σR31 = R31 * np.sqrt((σA3/A3)**2 + (σA1/A1)**2)

        # ---- phase uncertainties (linear propagation) ----
        σphi21 = np.sqrt(σφ2**2 + (2*σφ1)**2)
        σphi31 = np.sqrt(σφ3**2 + (3*σφ1)**2)

        return (
            R21, σR21,
            R31, σR31,
            phi21, σphi21,
            phi31, σphi31
        )
    
    def plot_phase_diagram(self, f0, A=None, phi=None, C=None, savepath=None):

        f0 = float(f0)
        phase = ((self.t - self.t.min()) * f0) % 1

        idx = np.argsort(phase)
        phase_sorted = phase[idx]
        y_sorted = self.y[idx]

        plt.figure()

        plt.scatter(phase_sorted, y_sorted, s=8, alpha=0.6, label="Data")

        if A is not None and phi is not None and C is not None:
            model = self.sine_model(self.t, A, phi, f0, C)
            model_sorted = model[idx]
            plt.plot(phase_sorted, model_sorted, color="red", label="Model")

        plt.xlabel("Phase (0–1)")
        plt.ylabel("Magnitude")
        plt.title(f"{self.filename} — Phase Diagram (f0 = {f0:.6f})")
        plt.legend()
        plt.savefig("name10.jpg", dpi=150, bbox_inches="tight")
        plt.show()
        


    def compute_stft_spectrum(self, nperseg=512, min_freq=0, max_freq=20):

        y = self.mag_to_flux(self.y)
        y = y - np.mean(y)

        dt = np.median(np.diff(self.t))

        f, t_spec, Zxx = stft(
            y,
            fs=1/dt,
            nperseg=nperseg,
            noverlap=int(0.75*nperseg)
        )

        power = np.abs(Zxx)**2

        # -------------------------
        # RIDGE extraction
        # -------------------------
        ridge_freqs = []

        for i in range(power.shape[1]):

            col = power[:, i]

            # normalize column
            col = col / (np.sum(col) + 1e-12)

            # compute weighted average frequency (center of mass)
            f_center = np.sum(f * col)

            ridge_freqs.append(f_center)

        ridge_freqs = np.array(ridge_freqs)

        # estimate global offset vs LS peak  
        f_ls, p_ls = self.lomb_scargle_spectrum()
        f0_ls = f_ls[np.argmax(p_ls)]

        f0_stft = np.median(ridge_freqs)

        # compute tiny correction
        shift = f0_ls - f0_stft

        # apply correction
        ridge_freqs = ridge_freqs + shift

        # histogram → pseudo-spectrum
        hist, bins = np.histogram(
            ridge_freqs,
            bins=200,
            range=(min_freq, max_freq),
            density=True
        )

        f_hist = 0.5 * (bins[:-1] + bins[1:])

        return f_hist, hist
    
  
    def multitaper_spectrum(self, nw=2, k=3, min_freq=0.0, max_freq=20.0):
        """
        Multitaper spectral estimate (Thomson method).
        Returns frequency and averaged power spectrum.
        """

        y = self.mag_to_flux(self.y)
        y = y - np.mean(y)

        dt = np.median(np.diff(self.t))
        fs = 1 / dt
        n = len(y)

        # DPSS tapers
        tapers = dpss(n, nw, Kmax=k)

        spectra = []

        for taper in tapers:
            y_tapered = y * taper
            fft_vals = rfft(y_tapered)
            power = np.abs(fft_vals) ** 2
            spectra.append(power)

        spectra = np.array(spectra)
        power_mean = np.mean(spectra / np.sum(spectra, axis=1)[:, None], axis=0)

        freq = np.fft.rfftfreq(n, d=dt)

        mask = (freq >= min_freq) & (freq <= max_freq)

        return freq[mask], power_mean[mask]
    
        
    # def plot_combined_spectrum_dual(self, savepath_global=None, savepath_zoom=None, zoom_width=1.0):
    def plot_combined_spectrum_4way(
        self,
        savepath_global=None,
        savepath_zoom=None,
        zoom_width=0.75,
        freq_max=20
    ):
        """
        4-way spectral comparison:
        - Lomb-Scargle
        - Lightkurve LS
        - FFT
        - STFT (collapsed)
        """

        # -------------------------
        # Compute spectra ONCE
        # -------------------------
        f_ls, p_ls = self.lomb_scargle_spectrum()
        pg = self.lightkurve_periodogram()
        f_lk, p_lk = pg.frequency.value, pg.power.value
        f_fft, p_fft = self.fft_spectrum()
        f_stft, p_stft = self.compute_stft_spectrum()
        f_mt, p_mt = self.multitaper_spectrum()
        
        # -------------------------
        # PEAK DETECTION (ALL METHODS)
        # -------------------------
        f_ls_peak = f_ls[np.argmax(p_ls)]
        f_lk_peak = f_lk[np.argmax(p_lk)]
        f_fft_peak = f_fft[np.argmax(p_fft)]
        f_stft_peak = f_stft[np.argmax(p_stft)]
        f_mt_peak = f_mt[np.argmax(p_mt)]
        
        print("\n=== PEAK FREQUENCIES ===")
        print(f"Lomb-Scargle : {f_ls_peak:.6f} cycles/day") 
        print(f"Lightkurve   : {f_lk_peak:.6f} cycles/day")
        print(f"FFT          : {f_fft_peak:.6f} cycles/day")
        # print(f"STFT         : {f_stft_peak:.6f} cycles/day")
        print(f"Multitaper   : {f_mt_peak:.6f} cycles/day")
        
        # -------------------------
        # Safety normalization
        # -------------------------
        p_ls = p_ls / (np.max(p_ls) + 1e-12)
        p_lk = p_lk / (np.max(p_lk) + 1e-12)
        p_fft = p_fft / (np.max(p_fft) + 1e-12)
        p_stft = p_stft / (np.max(p_stft) + 1e-12)
        p_mt = p_mt / (np.max(p_mt) + 1e-12)
        # p_mt = np.power(p_mt, 1.5)
        p_mt = p_mt / (np.max(p_mt) + 1e-12)
        
        # -------------------------
        # reference frequency (LS peak)
        # -------------------------
        f0 = f_ls[np.argmax(p_ls)]

        # =========================================================
        # 1. GLOBAL PLOT
        # =========================================================
        plt.figure(figsize=(12, 5))

        plt.plot(f_ls, p_ls, label="Lomb–Scargle", color="black", alpha=0.9)
        plt.plot(f_lk, p_lk, label="Lightkurve", color="blue", alpha=0.7)
        plt.plot(f_fft, p_fft, label="FFT", color="red", alpha=0.6)
        # plt.plot(f_stft, p_stft, label="STFT", color="green", alpha=0.7)
        plt.plot(f_mt, p_mt, label="Multitaper", color="green", alpha=0.8)
        
        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized Power")
        plt.title(f"{self.filename} — Global Spectrum Comparison")

        plt.xlim(0, freq_max)
        plt.legend()
        plt.tight_layout()

        plt.savefig("name11.jpg", dpi=150, bbox_inches="tight")

        plt.show()

        # =========================================================
        # 2. ZOOMED PLOT
        # =========================================================

        mask_ls = (f_ls > f0 - zoom_width) & (f_ls < f0 + zoom_width)
        mask_lk = (f_lk > f0 - zoom_width) & (f_lk < f0 + zoom_width)
        mask_fft = (f_fft > f0 - zoom_width) & (f_fft < f0 + zoom_width)
        mask_stft = (f_stft > f0 - zoom_width) & (f_stft < f0 + zoom_width)
        mask_mt = (f_mt > f0 - zoom_width) & (f_mt < f0 + zoom_width)
        
        # combine values for safe ylim scaling
        all_vals = np.concatenate([
            p_ls[mask_ls],
            p_lk[mask_lk],
            p_fft[mask_fft],
            p_stft[mask_stft],
            p_mt[mask_mt]
        ])

        ymin = 0
        ymax = np.percentile(all_vals, 99.5)

        plt.figure(figsize=(12, 5))

        plt.plot(f_ls[mask_ls], p_ls[mask_ls], label="Lomb–Scargle", color="black")
        plt.plot(f_lk[mask_lk], p_lk[mask_lk], label="Lightkurve", color="blue")
        plt.plot(f_fft[mask_fft], p_fft[mask_fft], label="FFT", color="red")
        # plt.plot(f_stft[mask_stft], p_stft[mask_stft], label="STFT", color="green")
        plt.plot(f_mt[mask_mt], p_mt[mask_mt], label="Multitaper", color="green")
        
        # plt.axvline(f0, color="red", linestyle="--", alpha=0.6, label="Main peak")

        plt.xlabel("Frequency (cycles/day)")
        plt.ylabel("Normalized Power")
        plt.title(f"{self.filename} — Global Spectrum Comparison (zoom-in)")

        plt.xlim(f0 - zoom_width, f0 + zoom_width)
        plt.ylim(ymin, ymax * 1.05)

        plt.legend()
        plt.tight_layout()

        plt.savefig("name12.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
    def get_peak_info(self, freq, power):
        idx = np.argmax(power)
        f_peak = freq[idx]
        p_peak = power[idx]

        # simple SNR definition (robust + comparable)
        noise = np.median(power)
        snr = p_peak / (noise + 1e-12)

        return f_peak, p_peak, snr
    
    def build_spectrum_table(self):
        import pandas as pd

        results = []

        # ---------------- Lomb-Scargle ----------------
        f_ls, p_ls = self.lomb_scargle_spectrum()
        f, p, snr = self.get_peak_info(f_ls, p_ls)

        results.append(["Lomb-Scargle", f, snr, "uneven sampling optimized"])

        # ---------------- Lightkurve ----------------
        pg = self.lightkurve_periodogram()
        f_lk, p_lk = pg.frequency.value, pg.power.value
        f, p, snr = self.get_peak_info(f_lk, p_lk)

        results.append(["Lightkurve LS", f, snr, "pipeline implementation"])

        # ---------------- FFT ----------------
        f_fft, p_fft = self.fft_spectrum()
        f, p, snr = self.get_peak_info(f_fft, p_fft)

        results.append(["FFT", f, snr, "requires interpolation"])

        # ---------------- Multitaper ----------------
        f_mt, p_mt = self.multitaper_spectrum()
        f, p, snr = self.get_peak_info(f_mt, p_mt)

        results.append(["Multitaper", f, snr, "robust to noise"])

        # build dataframe
        df = pd.DataFrame(results, columns=[
            "Method", "Peak frequency (c/d)", "Peak strength (SNR)", "Notes"
        ])

        return df
        
    def print_spectrum_table(self):
        df = self.build_spectrum_table()
        print("\n=== SPECTRUM COMPARISON TABLE ===\n")
        print(df.to_string(index=False))
        return df
    
    def plot_multisine_reconstruction(self, freqs, amplitudes, phases, C=0.0, savepath=None):

        # -------------------------
        # model construction
        # -------------------------
        t_sorted = np.sort(self.t)

        model = np.zeros_like(t_sorted)

        for A, f, phi in zip(amplitudes, freqs, phases):
            model += A * np.sin(2 * np.pi * f * t_sorted + phi)

        model += C

        # -------------------------
        # raw data
        # -------------------------
        plt.figure(figsize=(10, 5))

        plt.errorbar(
            self.t,
            self.y,
            yerr=self.y_err,
            fmt='o',
            markersize=3,
            alpha=0.5,
            label="Raw data"
        )

        # -------------------------
        # model
        # -------------------------
        plt.plot(
            t_sorted,
            model,
            color="red",
            linewidth=2,
            label="26-mode sine reconstruction"
        )

        # -------------------------
        # styling
        # -------------------------
        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude (unitless)")
        plt.title(f"{self.filename} — Multi-Sine Fit")
        plt.legend()

        plt.tight_layout()

        plt.savefig("thefit.jpg", dpi=150, bbox_inches="tight")

        plt.show()
    
    def plot_multisine_reconstruction(self, freqs, amplitudes, phases, C=0.0, savepath=None):

        import numpy as np
        import matplotlib.pyplot as plt

        # sort by frequency (optional but cleaner)
        idx = np.argsort(freqs)
        freqs = np.array(freqs)[idx]
        amplitudes = np.array(amplitudes)[idx]
        phases = np.array(phases)[idx]

        t = self.t

        model = np.zeros_like(t)

        # build full reconstruction
        for A, f, phi in zip(amplitudes, freqs, phases):
            model += A * np.sin(2*np.pi*f*t + phi)

        model += C

        # ---------------- plot ----------------
        plt.figure(figsize=(10,5))
        
        # ERROR BARS (key change)
        plt.errorbar(
            t,
            self.y,
            yerr=self.y_err,
            fmt='o',
            markersize=3,
            alpha=0.5,
            elinewidth=1,
            capsize=2,
            label="Data"
        )

        # plt.scatter(t, self.y, s=8, alpha=0.6, label="Data")
        plt.plot(t, model, color="red", lw=2, label="11-mode reconstruction")

        plt.xlabel("Time (days)")
        plt.ylabel("Magnitude")
        plt.title(f"{self.filename} — Multi-Sine Fit")
        plt.legend()
        plt.xlim(3540, 3541)
        # plt.xlim(2700, 2700.1)
        # plt.ylim(-0.15, -0.10)

        plt.savefig("babyyes.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
    def plot_phase_residual_diagram(self, f0, model):
    
        # phase
        phase = ((self.t - self.t.min()) * f0) % 1

        # residuals
        residuals = self.y - model

        # sort for clean plotting
        idx = np.argsort(phase)
        phase = phase[idx]
        residuals = residuals[idx]

        plt.figure()

        plt.errorbar(
            phase,
            residuals,
            yerr=self.y_err,
            fmt='o',
            ms=3,
            alpha=0.6
        )

        plt.axhline(0, color='black', linestyle='--', linewidth=1)

        plt.xlim(0, 1)
        plt.xlabel("Phase")
        plt.ylabel("Residuals")
        plt.title(f"{self.filename} — Phase Residual Diagram (f0={f0:.6f})")
        print(np.std(self.y))
        print(np.std(self.y - model))
        plt.tight_layout()
        plt.show()
        
    def plot_phase_diagram(self, f0):

        phase = ((self.t - self.t.min()) * f0) % 1

        idx = np.argsort(phase)

        plt.figure()

        plt.errorbar(
            phase[idx],
            self.y_original[idx],   # IMPORTANT CHANGE
            yerr=self.y_err[idx],
            fmt='o',
            ms=2,
            alpha=0.6,
            elinewidth=0.8,
            capsize=0
        )

        plt.xlim(0, 1)
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.title(f"{self.filename} — Phase diagram (f0={f0:.6f})")

        plt.gca().invert_yaxis()  # optional but matches astronomy convention

        plt.show()
    
    def plot_phase_diagram_period04(
        self,
        f0,
        bins=50,
        show_binned=True,
        show_model=False,
        A=None,
        phi=None,
        C=None,
        savepath=None
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        # -------------------------
        # USE RAW DATA (CRITICAL)
        # -------------------------
        t = self.t
        y = self.y_original if self.y_original is not None else self.y
        yerr = self.y_err

        # -------------------------
        # PHASE FOLDING
        # -------------------------
        phase = ((self.t - self.t.min()) * f0) % 1

        idx = np.argsort(phase)
        phase = phase[idx]
        y = y[idx]
        yerr = yerr[idx]

        # -------------------------
        # FIGURE
        # -------------------------
        plt.figure(figsize=(7, 4.5))

        # raw scatter (Period04 style = dense cloud)
        plt.errorbar(
            phase,
            y,
            yerr=yerr,
            fmt='o',
            ms=2.5,
            alpha=0.45,
            elinewidth=0.6,
            capsize=0,
            color='black'
        )

        # -------------------------
        # BINNING (THIS IS THE KEY PERIOD04 FEATURE)
        # -------------------------
        if show_binned:
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            bin_means = np.zeros(bins)
            bin_stds = np.zeros(bins)

            for i in range(bins):
                mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])

                if np.sum(mask) > 3:
                    bin_means[i] = np.mean(y[mask])
                    bin_stds[i] = np.std(y[mask])
                else:
                    bin_means[i] = np.nan
                    bin_stds[i] = np.nan

            plt.plot(
                bin_centers,
                bin_means,
                color='red',
                linewidth=2,
                label="Binned light curve (Period04-style)"
            )

            plt.fill_between(
                bin_centers,
                bin_means - bin_stds,
                bin_means + bin_stds,
                color='red',
                alpha=0.2
            )

        # -------------------------
        # OPTIONAL MODEL OVERLAY
        # -------------------------
        if show_model and A is not None and phi is not None and C is not None:
            phase_dense = np.linspace(0, 1, 1000)
            model = A * np.sin(2 * np.pi * phase_dense + phi) + C

            plt.plot(
                phase_dense,
                model,
                color='blue',
                linewidth=2,
                label="Best-fit sine"
            )

        # -------------------------
        # FINAL STYLE
        # -------------------------
        plt.xlim(0, 1)
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")
        plt.title(f"{self.filename} — Period04-style phase diagram (f₀ = {f0:.6f})")

        plt.gca().invert_yaxis()  # astronomy convention (optional but nice)
        plt.legend()
        plt.tight_layout()
        
        plt.show()
        
    def plot_phase_folded_lightcurve(
        self,
        f0,
        bins=100,
        show_binned=True,
        show_errorbars=True,
        invert_y=True,
        savepath=None
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        # -------------------------
        # ALWAYS USE RAW DATA
        # -------------------------
        t = self.t
        y = self.y_original if self.y_original is not None else self.y
        yerr = self.y_err

        # -------------------------
        # PHASE FOLDING
        # -------------------------
        phase = (t * f0) % 1

        idx = np.argsort(phase)
        phase = phase[idx]
        y = y[idx]
        yerr = yerr[idx]

        # -------------------------
        # FIGURE
        # -------------------------
        plt.figure(figsize=(7, 4.5))

        # RAW FOLDED LIGHT CURVE
        if show_errorbars:
            plt.errorbar(
                phase,
                y,
                yerr=yerr,
                fmt='o',
                ms=2.2,
                alpha=0.35,
                elinewidth=0.7,
                capsize=0,
                color='black'
            )
        else:
            plt.scatter(
                phase,
                y,
                s=6,
                alpha=0.4,
                color='black'
            )

        # -------------------------
        # BINNING (THIS CREATES THE "PERIOD04 LOOK")
        # -------------------------
        if show_binned:
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            bin_means = np.full(bins, np.nan)
            bin_stds = np.full(bins, np.nan)

            for i in range(bins):
                mask = (phase >= bin_edges[i]) & (phase < bin_edges[i + 1])

                if np.sum(mask) > 5:
                    bin_means[i] = np.mean(y[mask])
                    bin_stds[i] = np.std(y[mask])

            plt.plot(
                bin_centers,
                bin_means,
                color='red',
                linewidth=2,
                label="Binned curve"
            )

            plt.fill_between(
                bin_centers,
                bin_means - bin_stds,
                bin_means + bin_stds,
                color='red',
                alpha=0.2
            )

        # -------------------------
        # STYLE (ASTRONOMY STANDARD)
        # -------------------------
        plt.xlim(0, 1)
        plt.xlabel("Phase")
        plt.ylabel("Magnitude")

        plt.title(f"{self.filename} — Phase-folded light curve (f0 = {f0:.6f})")

        if invert_y:
            plt.gca().invert_yaxis()
        
        plt.legend()
        plt.tight_layout()

        plt.savefig("yesman.jpg", dpi=150, bbox_inches="tight")

        plt.show()
        
    def plot_combined_phase_fold_4way(self, savepath=None):

        import numpy as np
        import matplotlib.pyplot as plt
        
        def binned_phase_curve(phase, y, bins=50):
            bin_edges = np.linspace(0, 1, bins + 1)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

            digitized = np.digitize(phase, bin_edges)

            means = np.array([
                np.nanmean(y[digitized == i]) if np.any(digitized == i) else np.nan
                for i in range(1, len(bin_edges))
            ])

            return bin_centers, means 
        
        def add_trend(ax, phase, y, trend_color, label_prefix):
            bx, by = binned_phase_curve(phase, y)

            order = np.argsort(phase)
            phase_sorted = phase[order]
            y_sorted = y[order]
            y_smooth = gaussian_filter1d(y_sorted, sigma=10)

            # uncertainty (neutral grey for ALL methods)
            local_std = np.nanstd(y_sorted) * 0.25

           # ax.plot(
             #   phase_sorted,
             #   y_smooth,
             #   color=trend_color,
             #   linewidth=2.5,
             #   label=f"{label_prefix} smoothed"
            #)

            #ax.fill_between(
             #   phase_sorted,
              #  y_smooth - local_std,
              #  y_smooth + local_std,
              #  color="grey",
              #  alpha=0.15,
               # label=f"{label_prefix} uncertainty"
            #)

        # -------------------------
        # 1. GET PEAK FREQUENCIES
        # -------------------------
        f_ls, p_ls = self.lomb_scargle_spectrum()
        lk_pg = self.lightkurve_periodogram()
        f_lk, p_lk = lk_pg.frequency.value, lk_pg.power.value
        f_fft, p_fft = self.fft_spectrum()
        f_mt, p_mt = self.multitaper_spectrum()

        f_ls_peak = f_ls[np.argmax(p_ls)]
        f_lk_peak = f_lk[np.argmax(p_lk)]
        f_fft_peak = f_fft[np.argmax(p_fft)]
        f_mt_peak = f_mt[np.argmax(p_mt)]
        
        print(f_ls_peak)
        print(f_lk_peak)
        print(f_fft_peak)
        print(f_mt_peak)

        # -------------------------
        # 2. PHASE FOLDING
        # -------------------------
        def fold(f):
            return (self.t * f) % 1

        phase_ls = fold(f_ls_peak)
        phase_lk = fold(f_lk_peak)
        phase_fft = fold(f_fft_peak)
        phase_mt = fold(f_mt_peak)

        y = self.y_original if self.y_original is not None else self.y
        y = np.asarray(y)
        y = (y - np.nanmean(y)) / np.nanstd(y) # normalisation, to enable comparison

        # sort helper
        def sort(phase):
            idx = np.argsort(phase)
            return phase[idx], y[idx]

        phase_ls, y_ls = sort(phase_ls)
        phase_lk, y_lk = sort(phase_lk)
        phase_fft, y_fft = sort(phase_fft)
        phase_mt, y_mt = sort(phase_mt)

        # -------------------------
        # 3. PLOT
        # -------------------------
        fig, ax = plt.subplots(2, 2, figsize=(12, 6.0))
        
        plt.subplots_adjust(
            hspace=0.28,   # more vertical breathing room
            wspace=0.2,
            top=0.88        # reserves space for suptitle
        )

        bx, by = binned_phase_curve(phase_ls, y_ls)
        order = np.argsort(phase_ls)
        phase_sorted = phase_ls[order]
        y_sorted = y_ls[order]
        y_smooth = gaussian_filter1d(y_sorted, sigma=10)
        rms_ls = np.std(y_ls)
        
        # LS
        ax[0,0].scatter(phase_ls, y_ls, s=3, alpha=0.25, color = "black")
        ax[0,0].set_title(
            f"Lomb–Scargle ($f_{{0}} = {f_ls_peak:.5f}$ d$^{{-1}}$)",
            fontstyle='italic'
        )
        
        ax[0,0].scatter(phase_ls, y_ls, s=3, alpha=0.25, color="black", label="Data")

        add_trend(ax[0,0], phase_ls, y_ls, "red", "LS")
        #ax[0,0].legend()
        #ax[0,0].legend(fontsize=8, frameon=False)
        
        #leg = ax[0,0].legend(
         #   loc="best",
         #   frameon=True,
         #   facecolor="lightgrey",
         #   edgecolor="black",
         #   framealpha=0.85,
         #   fontsize=8,
         #   fancybox=True
        #)
        
        #leg.set_zorder(10)

        # Lightkurve
        ax[0,1].scatter(phase_lk, y_lk, s=4, alpha=0.5, color="blue")
        ax[0,1].set_title(
            f"Lightkurve ($f_{{0}} = {f_lk_peak:.5f}$ d$^{{-1}}$)",
            fontstyle='italic'
        )
        
        ax[0,1].scatter(
            phase_lk, y_lk,
            s=3, alpha=0.25,
            color="blue",
            label="Data" 
        )

        add_trend(ax[0,1], phase_lk, y_lk, "orange", "LK")
        #ax[0,1].legend()
        #ax[0,1].legend(fontsize=8, frameon=False)
        
        #leg = ax[0,1].legend(
            #loc="best",
            #frameon=True,
            #facecolor="lightgrey",
            #edgecolor="black",
            #framealpha=0.85,
            #fontsize=8,
            #fancybox=True
        #)
        
        #leg.set_zorder(10)


        # FFT
        ax[1,0].scatter(phase_fft, y_fft, s=4, alpha=0.5, color="red")
        ax[1,0].set_title(
            f"FFT ($f_{{0}} = {f_fft_peak:.5f}$ d$^{{-1}}$)",
            fontstyle='italic'
        )
        
        ax[1,0].scatter(phase_fft, y_fft, s=3, alpha=0.25, color="darkred", label="Data")

        add_trend(ax[1,0], phase_fft, y_fft, "gold", "FFT")
        #ax[1,0].legend()
        #ax[1,0].legend(fontsize=8, frameon=False)
        
        #leg = ax[1,0].legend(
          #  loc="best",
          #  frameon=True,
          #  facecolor="lightgrey",
          #  edgecolor="black",
          #  framealpha=0.85,
          #  fontsize=8,
          #  fancybox=True
        #)
        
        #leg.set_zorder(10)

        # Multitaper
        ax[1,1].scatter(phase_mt, y_mt, s=4, alpha=0.5, color="green")
        ax[1,1].set_title(
            f"Multitaper ($f_{{0}} = {f_mt_peak:.5f}$ d$^{{-1}}$)",
            fontstyle='italic'
        )
        
        ax[1,1].scatter(phase_mt, y_mt, s=3, alpha=0.25, color="green", label="Data")

        add_trend(ax[1,1], phase_mt, y_mt, "purple", "MT")
        #ax[1,1].legend()
        #ax[1,1].legend(fontsize=8, frameon=False)
        
        #leg = ax[1,1].legend(
         #   loc="best",
         #   frameon=True,
         #   facecolor="lightgrey",
         #   edgecolor="black",
         #   framealpha=0.85,
         #   fontsize=8,
         #   fancybox=True
        #)

        #leg.set_zorder(10)

        # -------------------------
        # 4. GLOBAL FORMATTING
        # -------------------------
        for a in ax.flat:
            a.set_xlim(0, 1)
            a.set_xlabel("Phase", labelpad=3)
            a.set_ylabel("Magnitude", labelpad=3)
            a.invert_yaxis()
            a.tick_params(axis='both', labelsize=9)

        fig.suptitle(
            f"{self.filename} — Phase-fold comparison (main frequency)",
            fontsize=14,
            fontweight="bold",
            y=0.98
        )

        plt.tight_layout()
        
        plt.savefig("beautiful.jpg", dpi=150, bbox_inches="tight")

        plt.show()
