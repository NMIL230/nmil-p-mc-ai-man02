# Full Analysis Report
Generated: 2025-10-01T08:32:37

## Configuration

```json
{
  "global_options": {
    "adjust_session_effect": true,
    "apply_validity_mask": true,
    "exclude_outliers": true,
    "exclude_repeats": false,
    "fig02_seed": null,
    "iqr_multiplier": 1.5,
    "outlier_method": "iqr_diff",
    "show_classic_threshold": true,
    "show_point_labels": false,
    "show_sort_metric": false,
    "survey_equivalence_bound": 0.5,
    "survey_metrics": [
      "minecraft_familiarity",
      "mouse_keyboard_comfort"
    ],
    "survey_pickle": "/Users/dommarticorena/Documents/nmil-p-mc-ai-man02/data/AMLEC_survey.pkl",
    "trial_stats_detail": false,
    "trial_stats_k": null,
    "use_gp_surface": true,
    "z_thresh": 3.0
  },
  "script_configs": {
    "Fig02": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    },
    "Fig03": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    },
    "Fig04": {
      "enabled": true,
      "extra_args": [],
      "overrides": {
        "stats_config": {
          "lines": [
            "ICC={icc}",
            "{icc_ci}",
            "R\u00b2={r2}",
            "p={pearson_p}",
            "BF10={bf10_value}",
            "n={n}"
          ]
        }
      }
    },
    "Fig05": {
      "enabled": true,
      "extra_args": [],
      "overrides": {
        "adjust_session_effect": false
      }
    },
    "Fig06": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    },
    "Fig07": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    },
    "FigS01": {
      "enabled": false,
      "extra_args": [],
      "overrides": {}
    },
    "FigS02": {
      "enabled": false,
      "extra_args": [],
      "overrides": {}
    },
    "SurveyAnalysis": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    },
    "TrialStats": {
      "enabled": true,
      "extra_args": [],
      "overrides": {}
    }
  }
}
```

## Results

### Figure02_Patterns (Fig02)
Representative pattern sets (Fig 02).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure02.py`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure02_Patterns

```text
Wrote representative pattern figure to manuscript_figures/Figure02_Patterns/Figure02_Patterns.png (seed=1695856744)
Reproduce with: python src/figures/MakeFigure02.py --seed 1695856744
```

### Figure03_Samples&Contours (Fig03)
Adaptive vs. Classic scatter and posterior (Fig 03).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure03.py --apply-validity-mask --exclude-outliers --outlier-method iqr_diff --iqr-multiplier 1.5 --z-thresh 3.0 --no-show-sort-metric --show-classic-threshold`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure03_Samples&Contours

```text
Validity-removed PIDs (2): AMLEC_018, AMLEC_022
Validity details (bounds [0.00, 18.0]):
- 018 [AMLEC_018]: Classic $\psi_{\theta}$=7.97, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
- 022 [AMLEC_022]: Classic $\psi_{\theta}$=5.04, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
Order: applied validity mask; proceeding to optional outlier exclusion.
Applied outlier exclusion on modality diffs (iqr_diff): removed 2/35 (bounds=-3.66..6.20); kept 33
Order: completed optional outlier exclusion; proceeding to adaptive scatter plotting on filtered set.
Order (sort=classic, K=3): AMLEC_027(target=-0.544, avg=-1.67, classic=16.5, theta@3=14.9), AMLEC_002_R1(target=-0.998, avg=-0.842, classic=15.0, theta@3=14.2), AMLEC_004_R1(target=-1.11, avg=-0.968, classic=14.0, theta@3=14.1), AMLEC_016(target=-1.22, avg=-1.12, classic=12.5, theta@3=12.2), AMLEC_018_R1(target=-0.879, avg=-0.775, classic=12.0, theta@3=13.7), AMLEC_021(target=-0.626, avg=-0.598, classic=12.0, theta@3=15.9), AMLEC_007_R1(target=-1.76, avg=-1.62, classic=9.49, theta@3=12.2), AMLEC_006_R1(target=-0.707, avg=-0.551, classic=8.97, theta@3=12.1), AMLEC_030(target=-0.790, avg=-0.675, classic=8.90, theta@3=9.14), AMLEC_028(target=-1.83, avg=-1.58, classic=8.50, theta@3=10.5), AMLEC_003(target=-0.697, avg=-0.649, classic=8.50, theta@3=8.68), AMLEC_008(target=-0.869, avg=-0.753, classic=8.50, theta@3=9.50), AMLEC_011(target=-1.02, avg=-0.820, classic=8.50, theta@3=8.42), AMLEC_020(target=-1.93, avg=-1.75, classic=8.07, theta@3=7.85), AMLEC_012(target=-0.537, avg=-0.420, classic=8.04, theta@3=9.63), AMLEC_009(target=-0.852, avg=-0.754, classic=8.04, theta@3=8.38), AMLEC_006(target=-0.723, avg=-0.794, classic=7.50, theta@3=9.80), AMLEC_024(target=-1.49, avg=-1.28, classic=7.50, theta@3=11.5), AMLEC_007(target=-2.07, avg=-1.86, classic=7.50, theta@3=9.05), AMLEC_004(target=-0.950, avg=-2.19, classic=7.50, theta@3=7.68), AMLEC_000(target=-2.51, avg=-2.31, classic=7.50, theta@3=8.07), AMLEC_015(target=-0.942, avg=-0.782, classic=7.50, theta@3=9.27), AMLEC_010_R1(target=-1.21, avg=-1.05, classic=7.50, theta@3=10.8), AMLEC_031(target=-0.834, avg=-0.711, classic=7.47, theta@3=8.44), AMLEC_025(target=-2.15, avg=-1.88, classic=7.24, theta@3=5.97), AMLEC_000_R1(target=-2.63, avg=-2.41, classic=7.03, theta@3=9.09), AMLEC_005(target=-0.511, avg=-0.449, classic=6.96, theta@3=7.47), AMLEC_019(target=-0.597, avg=-0.594, classic=6.96, theta@3=4.73), AMLEC_010(target=-1.27, avg=-1.16, classic=6.96, theta@3=9.59), AMLEC_023(target=-2.08, avg=-1.79, classic=6.95, theta@3=6.32), AMLEC_001(target=-0.565, avg=-0.537, classic=6.50, theta@3=5.86), AMLEC_002(target=-1.73, avg=-1.57, classic=6.50, theta@3=10.5), AMLEC_026(target=-2.46, avg=-2.26, classic=6.50, theta@3=8.31)
Wrote Figure03_Samples&Contours adaptive scatter + posterior50 figure to manuscript_figures/Figure03_Samples&Contours
Avg slope vs classic stats: n=33, r=0.155, p=3.90e-01, rho=0.147, rho_p=4.14e-01
Wrote supplemental avg-slope vs classic figure to manuscript_figures/Figure03_Samples&Contours
```

```text
/Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/utils/modeling/adaptive.py:658: RuntimeWarning: Mean of empty slice
  mean_surface = np.nanmean(stacked, axis=0)
/Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/utils/modeling/adaptive.py:476: RuntimeWarning: Mean of empty slice
  mean_zz = np.nanmean(np.stack(z_list, axis=0), axis=0)
```

### Figure04_Correlation (Fig04)
Classic vs Adaptive threshold parity (Fig 04).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure04.py --exclude-outliers --outlier-method iqr_diff --iqr-multiplier 1.5 --z-thresh 3.0 --adjust-session-effect --hide-labels --stats-config {"lines": ["ICC={icc}", "{icc_ci}", "R\u00b2={r2}", "p={pearson_p}", "BF10={bf10_value}", "n={n}"]}`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure04_Correlation

```text
Validity-removed PIDs (2): AMLEC_018, AMLEC_022
Validity details (bounds [0.00, 18.0]):
- 022 [AMLEC_022]: Classic $\psi_{\theta}$=5.04, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
- 018 [AMLEC_018]: Classic $\psi_{\theta}$=7.97, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
Applied validity mask: removed 2/37 subjects; kept 35
Global IQR filter: removed 2/35 diffs (bounds -3.66 to 6.20)
IQR-removed PIDs: AMLEC_014, AMLEC_016_R1
Estimated global session effect (second vs first): +0.486; removing from second-session $\psi_{\theta}$
Order: applied outlier exclusion BEFORE session-order adjustment and correlation.
ICC=0.755
95% CI [0.574, 0.868]
R²=0.678
p=3.96e-09
BF10=2.65e+06
n=33
Tests: ICC F-test (two-way random absolute agreement); Pearson correlation t-test (two-tailed, Fisher z CI); Spearman correlation t-test (two-tailed, Fisher z CI); Bayes factor BF10.
Wrote Figure04_Correlation correlation figure to manuscript_figures/Figure04_Correlation
```

### Figure05_OrderEffects (Fig05)
Mode order practice gains (Fig 05).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure05.py --exclude-outliers --outlier-method iqr_diff --iqr-multiplier 1.5 --z-thresh 3.0`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure05_OrderEffects

```text
Validity-removed PIDs (2): AMLEC_018, AMLEC_022
Validity details (bounds [0.00, 18.0]):
- 018 [AMLEC_018]: Classic $\psi_{\theta}$=7.97, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
- 022 [AMLEC_022]: Classic $\psi_{\theta}$=5.04, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
Applied validity mask: removed 2/37 subjects; kept 35
Order: no session-order adjustment; plotting raw values.
Figure05_OrderEffects outlier filter: removed 2/35 (iqr_diff; bounds=-4.46..5.45).

Figure05_OrderEffects $\Delta \psi_{\theta}$(Second − First Mode $\psi_{\theta}$) summary:
  n=14, mean=1.62, median=1.73, q1=0.332, q3=2.57, whiskers=(-0.244, 4.04), std=1.43, Effect size (mean Δ)=1.62; 95% CI [0.794, 2.45], One-sample t-test vs 0 (paired deltas): t(13)=4.23, p=9.83e-04, BF10=3.97e+01
  n=19, mean=-0.650, median=-0.338, q1=-1.72, q3=0.638, whiskers=(-3.87, 2.25), std=1.70, Effect size (mean Δ)=-0.650; 95% CI [-1.47, 0.171], One-sample t-test vs 0 (paired deltas): t(18)=-1.66, p=1.14e-01, BF10=7.60e-01
Tests: Paired-sample t-test on Δ (two-tailed); Bayes factor BF10.
Wrote Figure05_OrderEffects to manuscript_figures/Figure05_OrderEffects
```

### Figure06_AccuracyEvolution (Fig06)
RMSE evolution per sampling method (Fig 06).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure06.py --reject-outliers`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure06_AccuracyEvolution

```text
Validity-removed PIDs (2): AMLEC_018, AMLEC_022
Validity details (bounds [0.00, 18.0]):
- 018 [AMLEC_018]: Classic $\psi_{\theta}$=7.97, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
- 022 [AMLEC_022]: Classic $\psi_{\theta}$=5.04, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
Wrote Figure06_AccuracyEvolution to manuscript_figures/Figure06_AccuracyEvolution/Figure06_AccuracyEvolution.png

Figure06_AccuracyEvolution Active − Independent Staircase @30 samples:
  n=33
  Mean RMSE: Active=0.381 vs Independent=1.27
  Mean difference (Active−Independent)=-0.887 (SD=0.932, 95% CI [-1.22, -0.556])
  Paired t(32)=-5.47, p=5.12e-06, BF10=3.85e+03, dz=-0.951
  Std difference (Active−Independent)=-0.710 (ratio=0.254)
  Pitman-Morgan t(31)=-10.5, p=1.04e-11, BF10=5.68e+08
  BF10 equivalence on means (threshold 3.00): reached at sample 66 with BF10=2.96e+00
  BF10 equivalence on stds (threshold 3.00): reached at sample 3 with BF10=2.21e+00
  Final mean BF10 (sample 101)=1.83e+00
  Final std BF10 (sample 101)=8.22e+02
```

```text
/Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/utils/modeling/adaptive.py:658: RuntimeWarning: Mean of empty slice
  mean_surface = np.nanmean(stacked, axis=0)
```

### Figure07_ContourAccuracy (Fig07)
All-method contour overlays (Fig 07).

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/figures/MakeFigure07.py --reject-outliers`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_figures/Figure07_ContourAccuracy

```text
Validity-removed PIDs (2): AMLEC_018, AMLEC_022
Validity details (bounds [0.00, 18.0]):
- 018 [AMLEC_018]: Classic $\psi_{\theta}$=7.97, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
- 022 [AMLEC_022]: Classic $\psi_{\theta}$=5.04, Adaptive $\psi_{\theta}$=NA | Adaptive psi_theta non-finite
Wrote Figure07_ContourAccuracy to manuscript_figures/Figure07_ContourAccuracy/Figure07_ContourAccuracy.png
```

```text
/Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/utils/modeling/adaptive.py:658: RuntimeWarning: Mean of empty slice
  mean_surface = np.nanmean(stacked, axis=0)
```

### SurveyAnalysis
Survey correlates vs classic performance analysis.

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/analysis/survey_data_and_performance.py --04-exclude-outliers --04-outlier-method iqr_diff --04-iqr-multiplier 1.5 --04-z-thresh 3.0 --survey-metric minecraft_familiarity mouse_keyboard_comfort --equivalence-bound 0.5 --survey-pickle /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/data/AMLEC_survey.pkl`
- Exit code: 0
- Outputs: /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/manuscript_analysis/Survey Data and Performance

```text
Loaded survey subset from data/survey/Active+Machine+Learning+for+Evaluating+Cognition_September+26,+2025_14.17.csv
Fig 04 alignment: retained 33/35 participants after validity/outlier filtering (unique base IDs 27).
Performance data available for 27/27 Fig 04-aligned participants.
Excluded 4 numeric responses for metric 'minecraft_familiarity' not present in the Fig 04 subset.
Wrote Minecraft familiarity (self-reported, Q3.1) vs classic $\psi_{\theta}$ figure to manuscript_analysis/Survey Data and Performance/Survey Data and Performance - Minecraft familiarity self-reported Q3 1.png
Parsed 25/29 numeric survey responses for metric 'minecraft_familiarity'.
Paired-sample Pearson correlation (two-tailed) (Minecraft familiarity (self-reported, Q3.1) vs classic $\psi_{\theta}$): r=0.200; 95% CI [-0.212, 0.551]; R²=0.0399; t(23)=0.977; p=3.39e-01; n=25
Power to detect |r| ≥ 0.500: 0.755
Bayes factor BF10 ≈ 4.58e-01 (favoring H₀)
TOST equivalence test (bound ±0.500): z_lower=3.53, z_upper=-1.63, df=∞, p_lower=2.11e-04, p_upper=5.19e-02, 90% CI [-0.147, 0.503], n=25 → fails equivalence
Tests: Pearson correlation t-test (two-tailed, Fisher z CI); Bayes factor BF10; TOST equivalence (two one-sided z-tests).
Excluded 4 numeric responses for metric 'mouse_keyboard_comfort' not present in the Fig 04 subset.
Wrote Mouse & keyboard gaming comfort (self-reported) vs classic $\psi_{\theta}$ figure to manuscript_analysis/Survey Data and Performance/Survey Data and Performance - Mouse keyboard gaming comfort self-reported.png
Parsed 27/31 numeric survey responses for metric 'mouse_keyboard_comfort'.
Paired-sample Pearson correlation (two-tailed) (Mouse & keyboard gaming comfort (self-reported) vs classic $\psi_{\theta}$): r=-0.176; 95% CI [-0.521, 0.219]; R²=0.0310; t(25)=-0.894; p=3.80e-01; n=27
Power to detect |r| ≥ 0.500: 0.792
Bayes factor BF10 ≈ 4.14e-01 (favoring H₀)
TOST equivalence test (bound ±0.500): z_lower=1.82, z_upper=-3.56, df=∞, p_lower=3.44e-02, p_upper=1.83e-04, 90% CI [-0.473, 0.157], n=27 → passes equivalence
Tests: Pearson correlation t-test (two-tailed, Fisher z CI); Bayes factor BF10; TOST equivalence (two one-sided z-tests).
```

### TrialStats
Classic vs Adaptive trial-count summary.

- Command: `/opt/homebrew/Caskroom/miniconda/base/envs/nmil-mc-ai-amlec/bin/python /Users/dommarticorena/Documents/nmil-p-mc-ai-man02/src/analysis/report_trial_stats.py --outlier-filter --outlier-method iqr_diff --iqr-multiplier 1.5 --z-thresh 3.0`
- Exit code: 0

```text
Classic: pids=33, total=466, mean=14.12, std=4.17, range=(9, 26)
Adaptive: pids=33, total=990, mean=30.00, std=0.00, range=(30, 30)
Outlier filter (iqr_diff) removed 2/35; kept 33
```
