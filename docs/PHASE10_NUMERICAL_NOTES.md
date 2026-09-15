# Phase 10 numerical and browser corrections

## Observed failures, not assumed causes

The original historical exercise completed all three predeclared datasets. Subsequent fresh requests rejected SPY and IWM under exact floating-point OHLC comparisons. Diagnostic evidence from workflow run 34933065467 established the largest relative bound discrepancies as 1.28603e-16 (SPY, two rows) and 1.12646e-16 (IWM, one row). Those magnitudes are below double-precision machine epsilon.

OHLC comparisons now tolerate at most **8 * numpy.finfo(float).eps** relative to the largest positive price in the row, approximately 1.78e-15. The input prices are not clipped, filled, dropped, or rewritten. Larger violations still raise an error with count, date, and magnitude. The data manifest exposes the tolerance and the count of accepted floating-point discrepancies. Regression tests retain one-ULP discrepancies exactly while rejecting larger inconsistencies. This is a numerical comparison tolerance, not permission to repair bad market data.

The browser then successfully loaded the synthetic dataset, ran development, and opened the holdout. Its final visibility assertion failed because the explicit rerun reset the visible tab to Development while the holdout had already been evaluated. The page now gives the tabs a persistent widget key and state tracking, so the selected holdout tab survives the rerun. The browser workflow still requires the holdout confirmation text to be visible; it is not weakened to accept a hidden result.

Neither correction changes the candidate strategies, selection metric, training/test dates, exposure, fees, slippage, or accounting engine. First-run unfavorable historical results remain recorded in PR #14. Fresh downloads are separate snapshots, not replacements for that original evidence.

Reference for stateful tabs: https://docs.streamlit.io/develop/api-reference/layout/st.tabs
