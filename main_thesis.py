from models_thesis import Thesis
import numpy as np
from astropy.timeseries import LombScargle
import os
import matplotlib.pyplot as plt

def main():

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.labelsize": 12,
        "axes.titlesize": 13
    })

    OUTPUT_DIR = "results_figures"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    analysis = Thesis()

    # -------------------------
    # LOAD DATA
    # -------------------------
    filename = "333.85683 +6.82261_sector82_['TESS-SPOC'].txt"
    analysis.load_data(filename)

    clean_name = analysis.filename

    # -------------------------
    # 1. TIME SERIES
    # -------------------------
    analysis.plot_time_series(
        savepath=f"{OUTPUT_DIR}/{clean_name}_time_series.png"
    )

    # -------------------------
    # 2. LOMB-SCARGLE
    # -------------------------
    freq, power = analysis.lomb_scargle_spectrum()

    f_peak, p_peak = analysis.find_dominant_frequency(freq, power)

    period = 1.0 / f_peak

    print(f"Period: {period:.6f} days")
    print(f"Dominant frequency: {f_peak:.6f} cycles/day")
    print(f"Peak power: {p_peak:.6f}")

    analysis.plot_spectrum(
        freq,
        power,
        f_peak,
        savepath=f"{OUTPUT_DIR}/{clean_name}_spectrum.png"
    )

    analysis.plot_rich_spectrum()

    analysis.plot_spectrum_lk(
        savepath=f"{OUTPUT_DIR}/{clean_name}_lightkurve_spectrum.png"
    )
    
    # -------------------------
    # 2.5 FFT SPECTRUM
    # -------------------------
    analysis.plot_fft_spectrum(
        savepath=f"{OUTPUT_DIR}/{clean_name}_fft_spectrum.png"
    )

    # -------------------------
    # 3. SINGLE SINE FIT
    # -------------------------
    A1, f1, phi1, C1, A_err, f_err, phi_err = analysis.fit_sine(f_peak)

    # result = analysis.fit_sine(f_peak)
    # print(result)
    
    analysis.plot_fit(
        f_peak, A1, phi1, C1,
        savepath=f"{OUTPUT_DIR}/{clean_name}_sine_fit.png"
    )

    analysis.plot_data_with_model(
        f_peak, A1, phi1, C1,
        savepath=f"{OUTPUT_DIR}/{clean_name}_time_series_and_sine_fit_zoom_out.png"
    )

    P = 1.0 / f_peak
    t0 = np.median(analysis.t)

    analysis.plot_data_with_model(
        f_peak, A1, phi1, C1,
        xlim=(t0 - 1.5*P, t0 + 1.5*P),
        ylim=(np.mean(analysis.y) - 3*np.std(analysis.y),
              np.mean(analysis.y) + 3*np.std(analysis.y)),
        savepath=f"{OUTPUT_DIR}/{clean_name}_time_series_and_sine_fit_zoom_in.png"
    )
    
    analysis.plot_data_with_model(
        f_peak, A1, phi1, C1,
        xlim=(t0 - 1.5*P, t0 + 1.5*P),
        ylim=(np.mean(analysis.y) - 2*np.std(analysis.y),
              np.mean(analysis.y) + 2*np.std(analysis.y)),
        savepath=f"{OUTPUT_DIR}/{clean_name}_zoom_mid.png"
    )
    
    analysis.plot_data_with_model(
        f_peak, A1, phi1, C1,
        xlim=(t0 - 0.5*P, t0 + 0.5*P),
        ylim=(np.mean(analysis.y) - 10*np.median(analysis.y_err),
              np.mean(analysis.y) + 10*np.median(analysis.y_err)),
        savepath=f"{OUTPUT_DIR}/{clean_name}_zoom_error_scale.png"
    )

    # -------------------------
    # 4. PREWHITENING
    # -------------------------
    freqs_pw, amps_pw, phases_pw, offsets_pw, freq_err, amp_err, phase_err, _ = \
        analysis.prewhiten(n_freqs=40)
    snr_pw = np.array(amps_pw) / np.array(amp_err)
    
    print("\nRESULTS:")
    print("ID  Freq ±σf              Amp ±σA               Phase ±σφ            SNR")

    for i in range(len(freqs_pw)):
        print(
            f"{i+1:02d}  "
            
            f"{freqs_pw[i]:.6f} ± {freq_err[i]:.6f}   "
            f"{amps_pw[i]:.6f} ± {amp_err[i]:.6f}   "
            f"{phases_pw[i]:.6f} ± {phase_err[i]:.6f} "
            
            f"{snr_pw[i]:.2f}"
        )

    analysis.plot_multisine_reconstruction(
        freqs_pw,
        amps_pw,
        phases_pw,
        C=offsets_pw[-1],
        savepath=f"{OUTPUT_DIR}/{clean_name}_multisine_reconstruction.png"
    )

    # -------------------------
    # 5. MULTI-SINE FIT
    # -------------------------
    freq_ls, power_ls = analysis.lomb_scargle_spectrum()

    freqs = [f_peak * k for k in range(1, 4)]  # harmonics

    print("Frequencies used in global fit:", freqs)

    A_list, freqs_fit, phi_list, C_multi, rms, snr, errors, result = analysis.fit_multi_sine(freqs)

    for i in range(len(freqs_fit)):
        print(
            f"{i+1:02d}  F={freqs_fit[i]:.6f}  "
            f"A={A_list[i]:.6f}  "
            f"P={phi_list[i]:.6f}"
        )

    # -------------------------
    # 6. MODEL + RESIDUALS
    # -------------------------
    model = analysis.multi_sine_model(analysis.t, result.x, freqs)
    residual = analysis.y - model

    ls = LombScargle(analysis.t, residual)
    freq_res = np.linspace(0.01, 10, 20000)
    power_res = ls.power(freq_res)

    analysis.plot_spectrum(
        freq_res,
        power_res,
        savepath=f"{OUTPUT_DIR}/{clean_name}_residual_spectrum.png"
    )

    analysis.plot_petersen_diagram(
        freqs,
        A_list,
        savepath=f"{OUTPUT_DIR}/{clean_name}_peterson.png"
    )
    # analysis.plot_phase_diagram(f_peak)
    # analysis.plot_phase_residual_diagram(f_peak, model)
    # analysis.plot_phase_diagram(f_peak)
    
    analysis.plot_phase_diagram_period04(
        f_peak,
        show_model=True,
        A=A1,
        phi=phi1,
        C=C1
    )
    
    analysis.plot_phase_folded_lightcurve(
        f_peak,
        bins=80,
        show_binned=True,
        show_errorbars=True,
        invert_y=True
    )
    
    analysis.plot_combined_phase_fold_4way(
        savepath=f"{OUTPUT_DIR}/{clean_name}_phase_fold_4way.png"
    )

    idx = np.argmax(power_res)
    f_res_peak = freq_res[idx]
    p_res_peak = power_res[idx]

    print(f"Residual peak: {f_res_peak:.6f}, power={p_res_peak:.6f}")
    print("Peak / median:", p_res_peak / np.median(power_res))

    # -------------------------
    # 7. FINAL REPORT
    # -------------------------
    print("\n=== GLOBAL FIT RESULTS ===")

    for i in range(len(freqs)):
        print(
            f"{i+1:02d}  F={freqs[i]:.6f}  "
            f"A={A_list[i]:.6f} ± {errors[2*i]:.6f}  "
            f"P={phi_list[i]:.6f} ± {errors[2*i+1]:.6f}"
        )

    print(f"\nOffset: {C_multi:.6f} ± {errors[-1]:.6f}")
    print(f"RMS error: {rms:.6e}")
    print(f"SNR: {snr:.2f}")

    # -------------------------
    # 8. DIAGNOSTICS
    # -------------------------
    analysis.plot_phase_folded(
        freqs[0],
        savepath=f"{OUTPUT_DIR}/{clean_name}_phase_fold.png"
    )

    analysis.plot_spectral_window(
        max_freq=10,
        savepath=f"{OUTPUT_DIR}/{clean_name}_spectral_window.png"
    )

    if len(freqs) >= 2:
        R21, sR21, R31, sR31, phi21, sphi21, phi31, sphi31 = \
            analysis.compute_fourier_params_with_errors(A_list, phi_list, errors)

        print("\n=== FOURIER PARAMETERS ===")
        print(f"R21 = {R21:.4f} ± {sR21:.4f}")
        print(f"R31 = {R31:.4f} ± {sR31:.4f}")
        print(f"phi21 = {phi21:.4f} ± {sphi21:.4f}")
        print(f"phi31 = {phi31:.4f} ± {sphi31:.4f}")

    periods = 1 / np.array(freqs)

    idx0 = np.argmax(A_list)
    P0 = periods[idx0]

    idx1 = np.argsort(A_list)[-2]
    P1 = periods[idx1]

    ratio = P1 / P0
    print(f"Period ratio = {ratio:.4f}")

    model = analysis.multi_sine_model(analysis.t, result.x, freqs)

    #analysis.plot_phase_residuals(freqs[0], model)

    #analysis.plot_phase_residuals(
     #   freqs[0],
       # model,
       # savepath=f"{OUTPUT_DIR}/{clean_name}_phase_residuals.png"
    #)
    
    #analysis.plot_phase_diagram(
     #   f_peak,
     #   A1,
     #   phi1,
     #   C1,
     #   savepath=f"{OUTPUT_DIR}/{clean_name}_phase_diagramNOT.png"
    #)

    # -------------------------
    # 2. SPECTRUM COMPARISON (4 METHODS)
    # -------------------------
    analysis.plot_combined_spectrum_4way(
        savepath_global=f"{OUTPUT_DIR}/{clean_name}_spectrum_global.png",
        savepath_zoom=f"{OUTPUT_DIR}/{clean_name}_spectrum_zoom.png",
        zoom_width=0.75
    )
    
    df = analysis.print_spectrum_table()
    
    print("\n=== FINAL SUMMARY ===")
    print(f"Main frequency: {f_peak:.6f}")
    print(f"Period: {1/f_peak:.6f} days")
    print(f"Number of modes: {len(freqs)}")
    print(f"SNR: {snr:.2f}")
    print(f"Total time span: {analysis.t.max() - analysis.t.min():.2f} days")

if __name__ == "__main__":
    main()
