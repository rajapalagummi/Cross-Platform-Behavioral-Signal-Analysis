# ContentPulse — Content Consumption Intelligence & Cross-Platform Signal Analysis

> Two-part content analytics platform: causal inference on content diversity's effect on streaming retention (Part 1) and cross-platform podcast signal comparison across Spotify, YouTube, and Apple Podcasts using D3.js interactive dashboards (Part 2).

---

## Abstract

ContentPulse addresses two questions at the intersection of content analytics and causal inference:

**Part 1:** Does content diversity (measured as genre entropy per listening session) causally increase 7-day user return rate? Three estimators are applied: Difference-in-Differences (primary), Propensity Score Matching (robustness), and Bayesian Beta-Binomial A/B test (uncertainty quantification). Parallel trends assumption is tested before DiD estimation.

**Part 2:** Do podcast content signals — episode frequency, engagement, content volume, genre diversity — differ systematically across Spotify, YouTube, and Apple Podcasts for the same topic category? KS-tests with Cohen's d effect sizes quantify distributional differences. A D3.js interactive dashboard renders platform fingerprints as radar charts with animated transitions.

---

## System Formulation

**Genre Entropy (Shannon):**
$$H(s) = -\sum_{g \in G} p_g \log_2 p_g$$

**Difference-in-Differences:**
$$\hat{\tau}_{DiD} = (\bar{Y}_{treat,post} - \bar{Y}_{treat,pre}) - (\bar{Y}_{ctrl,post} - \bar{Y}_{ctrl,pre})$$

**Bayesian A/B (Beta-Binomial):**
$$p_{treat} \sim \text{Beta}(1 + s_t, 1 + n_t - s_t), \quad p_{ctrl} \sim \text{Beta}(1 + s_c, 1 + n_c - s_c)$$
$$\delta = p_{treat} - p_{ctrl}, \quad P(\delta > 0) = \text{posterior probability treatment superior}$$

**Platform Fingerprint (HHI-based diversity):**
$$\text{HHI} = \sum_g \left(\frac{n_g}{N}\right)^2, \quad \text{Diversity} = 1 - \text{HHI}$$

---

## Architecture

```
Part 1 — Causal Analysis
Synthetic streaming data (200-1000 users, 90-365 days)
  → Session construction (30-min gap rule)
  → Genre entropy + lag features
  → DiD estimator (parallel trends tested)
  → Propensity Score Matching (robustness)
  → Bayesian Beta-Binomial A/B test
  → MLflow experiment tracking
  → D3.js: return rate bars, entropy histogram, posterior density

Part 2 — Cross-Platform Comparison
  → Apple Podcasts: iTunes Search API (free, no auth)
  → YouTube: Data API v3 (free, cached locally)
  → Spotify: public charts + RSS (free)
  → Normalized metrics across platforms
  → KS-test + Cohen's d distributional comparison
  → Platform fingerprint (5 dimensions)
  → D3.js: radar chart, animated bar comparison, scatter plots

FastAPI backend serves both dashboards at localhost:8000
```

---

## Key Results (populated after run)

| Stage | Metric | Value |
|---|---|---|
| Part 1 | DiD Estimate | TBD |
| Part 1 | DiD p-value | TBD |
| Part 1 | P(treatment > control) | TBD |
| Part 1 | High diversity return rate | TBD |
| Part 1 | Low diversity return rate | TBD |
| Part 2 | Platforms compared | 3 (Spotify, YouTube, Apple) |
| Part 2 | KS-test significant pairs | TBD |

---

## Stack

| Component | Technology |
|---|---|
| Causal Inference | DiD, PSM, Bayesian Beta-Binomial (PyMC) |
| Statistical Tests | SciPy KS-test, Cohen's d, parallel trends |
| Data Engineering | Pandas, session construction pipeline |
| Visualization | D3.js v7 (radar, posterior density, histograms) |
| APIs | iTunes Search API, YouTube Data API v3, Spotify RSS |
| Backend | FastAPI, Jinja2 |
| Experiment Tracking | MLflow |
| Infrastructure | Docker, Git |

---

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: add YouTube API key to .env
echo "YOUTUBE_API_KEY=your_key_here" > .env

# Run demo
python main.py --mode demo

# Start dashboard
python main.py --mode api
# Open http://localhost:8000

# Run tests
pytest tests/ -v
```

---

## References

Angrist, J.D. & Pischke, J.S. (2008). *Mostly Harmless Econometrics*. Princeton University Press.

Rosenbaum, P.R. & Rubin, D.B. (1983). The central role of the propensity score in observational studies for causal effects. *Biometrika*, 70(1), 41–55.

Gelman, A. & Hill, J. (2006). *Data Analysis Using Regression and Multilevel/Hierarchical Models*. Cambridge University Press.
