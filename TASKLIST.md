<tasklist>
  <task status="COMPLETED">
    <id>1</id>
    <title>Project Setup and Boilerplate</title>
    <description>Create initial directories, setup pyproject.toml, setup.py, Cython bindings, and requirements.txt.</description>
  </task>
  <task status="COMPLETED">
    <id>2</id>
    <title>PyTorch GPU Acceleration Kernels</title>
    <description>Implement compute_ND_NUDFT, compute_Fast_ND_NUDFT, and covariance_compensation with CUDA/MPS/CPU backends.</description>
  </task>
  <task status="COMPLETED">
    <id>3</id>
    <title>Cython Performance CPU Kernels</title>
    <description>Implement optimized, multi-threaded C/C++ loops in Cython with OpenMP for parallel direct NUDFT calculation on the CPU.</description>
  </task>
  <task status="COMPLETED">
    <id>4</id>
    <title>Scikit-Learn NufiImputer Core</title>
    <description>Build the NufiImputer class supporting fit, transform, fit_transform, NaN masking, and smooth derivative-preserving infilling.</description>
  </task>
  <task status="COMPLETED">
    <id>5</id>
    <title>ETL & DataFrame Wrappers</title>
    <description>Implement wrappers for Pandas DataFrames, MultiIndexes, and CuDF compatibility for seamless pipeline integration.</description>
  </task>
  <task status="COMPLETED">
    <id>6</id>
    <title>LaTeX Whitepaper Formulation</title>
    <description>Create nufi/paper/main.tex and document the mathematical specifications of the NUDFT imputer, derivative properties, and LDLᵀ covariance compensation.</description>
  </task>
  <task status="COMPLETED">
    <id>7</id>
    <title>Advanced Solvers & Imputation</title>
    <description>Implement L2/Tikhonov regularization, Conjugate Gradient (CG) iterative solver, stochastic multiple imputation, and GCV auto-tuning.</description>
  </task>
  <task status="COMPLETED">
    <id>8</id>
    <title>Agent-Native Layer</title>
    <description>Implement high-level zero-config entrypoint (impute_dataframe), JSON diagnostics (SNR, spectral entropy), and interactive plot_diagnostics. Ensure robust, append-only transformation logging with quality enforcement exception raising, and Git/DVC style snapshot-based version tracking and reversion.</description>
  </task>
  <task status="COMPLETED">
    <id>9</id>
    <title>Empirical Benchmark Suite</title>
    <description>Create test scripts comparing nufi against GP, MICE, and Splines over RMSE, covariance error, and run-time.</description>
  </task>
  <task status="IN_PROGRESS">
    <id>10</id>
    <title>Validation, Tests & Device Benchmarking</title>
    <description>Perform end-to-end integration tests, verify mathematical properties, and ensure 100% of tests are passing.</description>
  </task>
</tasklist>
