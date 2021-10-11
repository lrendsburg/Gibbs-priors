Code for the AISTATS22 submission ``Discovering Inductive Bias with Gibbs Priors: A Diagnostic Tool for Approximate Bayesian Inference''
by Luca Rendsburg, Agustinus Kristiadi, Philipp Hennig, Ulrike von Luxburg.

Packages:
- Install numpyro with ``pip install numpyro''
- Install tikzplotlib with ``conda install -c conda-forge tikzplotlib''
- For everything else see requirements.txt



Source files are in the folder src/

Data for the toy example are stored in the folder data/

Data from the experiment section are in stored in the folder res/ and are created by exp/run_abc.ipynb and exp/run_volatility.ipynb

Figures of the paper are created in the exp/fig_* files:
    fig_intro.ipynb : intro_schematic_norm.pdf (Figure 1)
    fig_toy_example.ipynb : example_priors.pdf (Figure 3a) and example_posteriors.pdf (Figure 3b)
    fig_abc.ipynb : prior_vs_laplace.tex (Figure 4)
    fig_volatility.ipynb : timeseries_summary.pdf (Figure 5)
