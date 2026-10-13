# Assignment 1 Report

- **Name**: Brissa James
- **Student ID**: 19800
- **Email**: bjames.ieu2022@student.ie.edu
- **Group**: BBADBA 5B

## Dataset

**UCI Bank Marketing (`bank-additional-full.csv`)**, Moro, Cortez & Rita (2014),
downloaded from the UCI ML Repository and committed in `data/`.

- One row = one phone call made by a Portuguese bank's call centre during a
  term-deposit campaign, May 2008 → November 2010, **in chronological order**.
- 41,188 rows × 21 columns: client profile (age, job, marital, education,
  default/housing/personal loan), contact info (channel, month, day of week),
  campaign history (`campaign` = calls in this campaign, `pdays`, `previous`,
  `poutcome`), five monthly macro-economic indicators, `duration` of the call
  and the target `y` (did the client subscribe?).
- It is larger than the ~20k-row guideline (5.6 MB), which I kept because the
  chronological split needs the full period: the later periods are where the
  interesting behaviour is.
- Picked because it has a clear business decision attached, a **leakage trap**
  (`duration`), a sentinel value (`pdays = 999`), and **strong drift over time**
  (the subscription rate goes from ~3% in mid-2008 to ~46% in late 2010), so the
  framing actually changes how the pipeline has to be built.

## Business / real-life framing

**Decision supported:** before the next campaign wave, the call-centre manager
must decide *which clients on the list to phone*. Each call costs money
(assumed **€8**: ~6 min of loaded agent time + telephony); each subscription
brings an expected net margin (assumed **€50**). The model gives each client a
probability of subscribing; we call when that probability ≥ a threshold.

What this framing forces in the pipeline:

1. **Target and features must be known *before* dialling.** `duration` is only
   known after the call ends (and a 0-second call is a guaranteed "no"), so it
   is dropped. The ablation shows why it is tempting: it lifts validation
   ROC-AUC from 0.70 to 0.90, which a deployed model could never achieve.
2. **Chronological split, not random.** The model will always score *future*
   waves. The file is ordered in time, so I split by row order: first 70%
   train, next 15% validation, last 15% test. A random split would put late-2010
   calls in training and hide the drift. Positive rate per split: **5.6% / 10.7%
   / 38.5%**, i.e. the test period looks very different from training. This is
   the realistic situation and I kept it rather than shuffling it away.
3. **Metric for the decision threshold = campaign profit**, not accuracy or F1:
   `profit = TP·(50−8) − FP·8`. Missing a subscriber costs margin, a useless
   call costs €8, so the costs are asymmetric. In theory the break-even
   probability is cost/value = **0.16**. The threshold is chosen on
   *validation* by maximising profit and then applied unchanged to test.
4. **Metric for model selection (regularisation) = validation PR-AUC.** With
   ~6–10% positives and a "who to call first" use case, ranking quality on the
   positive class matters more than accuracy. I also report precision among
   the top 10% scored clients (what a capacity-limited call centre would get).

## Data preparation & feature engineering

All statistics (category levels, means/stds) are fitted on the **training
split only** in `pipeline.Preprocessor`, then applied to val/test/app inputs.
Numerics are standardised; categoricals are one-hot encoded with the first
level as reference (unseen levels → reference). 45 features in total.

| Decision | Why | Evidence (val PR-AUC; final = 0.269) |
|---|---|---|
| Drop `duration` | Leakage: unknown before the call | with it: 0.510 (not deployable) |
| Drop the 5 macro indicators (euribor, CPI, confidence, employment) | They have **one value per month for every client**, so they can't rank clients within a wave, only identify the period. In test the euribor (0.63–1.29) is entirely outside the training range (1.41–5.04), and the sign of their relation to `y` flips (e.g. `cons.price.idx` univariate AUC 0.39 in train vs 0.57 in test). | with them: val 0.251, **test ROC-AUC drops to 0.49 (worse than random)** |
| `pdays = 999` → `prev_contacted` flag | 999 is a sentinel ("never contacted"), not 999 days | — |
| Drop `days_since_prev` (my first idea) | I originally filled "never" with 0, which made never-contacted clients look like "contacted yesterday". That was a mistake and it hurt. | with it: 0.246 |
| `log1p(campaign)`, `previous` capped at 3 | Right-skewed counts (max 56 calls); limits the influence of outliers | neutral: raw versions give 0.269 too |
| `age_band` (≤25, 26–35, 36–50, 51–60, 60+) on top of `age` | Response vs. age is U-shaped (students and retirees respond much more: 43.7% for 60+ in train) and a single linear term can't model that | without: 0.264 |
| Keep `month` | Known before calling (the wave's month is planned). It captures that big May waves target low-quality lists and small Mar/Oct/Dec waves target better ones. | without: 0.223 |
| Drop `day_of_week` | Small and unstable effect | with it: 0.270, a negligible difference, so I kept the simpler model |
| `default = "yes"` merged into `"unknown"` | Only 3 rows; a dummy for 3 rows is noise | — |
| Keep `"unknown"` as its own category | Missingness is informative (e.g. unknown default status) | — |

What didn't work: my engineered `pdays`/`campaign` transforms brought no gain
over the raw columns for a linear model (raw 999 vs. 0–27 already behaves like
a binary flag after scaling), and `days_since_prev` made things worse. The
biggest gains came from **removing** things (macro indicators, `duration`)
rather than adding them.

## Modeling: three implementations, one model

**Model:** L2-regularised **logistic regression** (binary target: subscribe
yes/no). All three implementations minimise *the same objective*:

- sklearn: `0.5·‖w‖² + C·Σ logloss`. Dividing by `C·n` gives
- PyTorch: `mean(BCE) + (λ/2)·‖w‖²` with **λ = 1/(C·n)**, bias not penalised
  (as in sklearn).

**Tuning (done on validation only):**
- `C` ∈ {0.0003 … 1} → **C = 0.003** (λ = 0.0116). Strong regularisation wins
  because of drift: smaller, more conservative coefficients transfer better to
  future periods.
- Manual PyTorch (full-batch gradient descent with autograd, `w -= lr·grad`,
  3000 epochs): lr ∈ {0.1, 0.5, 1, 2}, picked by lowest final training
  objective → **lr = 1.0**. lr = 0.1 had not converged after 3000 epochs.
- `nn.Module` + `torch.optim.Adam` (full batch, 1500 epochs): lr ∈ {0.01, 0.05,
  0.1} → **lr = 0.1**. lr = 0.01 had not fully converged.

**Results** (threshold 0.07, chosen on validation):

| Test set (last 15%, rate 38.5%) | ROC-AUC | PR-AUC | Log loss | Precision top 10% | Calls | Profit (€) |
|---|---|---|---|---|---|---|
| scikit-learn | 0.6985 | 0.5886 | 0.9694 | 0.705 | 3,067 | 54,964 |
| Manual PyTorch loop | 0.6984 | 0.5885 | 0.9690 | 0.705 | 3,070 | 54,890 |
| nn.Module + optim | 0.6984 | 0.5885 | 0.9690 | 0.705 | 3,070 | 54,890 |
| Baseline (train base rate) | 0.5000 | 0.3845 | 1.1457 | 0.102* | 0 | 0 |
| Call everyone | | | | | 6,179 | **69,368** |

| Validation set (rate 10.7%) | ROC-AUC | PR-AUC | Log loss | Precision top 10% | Calls | Profit (€) |
|---|---|---|---|---|---|---|
| scikit-learn | 0.6958 | 0.2685 | 0.3396 | 0.323 | 883 | 5,136 |
| Manual PyTorch loop | 0.6958 | 0.2681 | 0.3396 | 0.324 | 883 | 5,136 |
| nn.Module + optim | 0.6958 | 0.2681 | 0.3396 | 0.324 | 883 | 5,136 |
| Baseline | 0.5000 | 0.1065 | 0.3588 | 0.088* | 0 | 0 |
| Call everyone | | | | | 6,178 | −16,524 |

\*The constant baseline has no ranking, so its "top 10%" is just the first rows.

**Do the three agree? Yes.** The final objective is 0.211477 for all three
(differences of about 2·10⁻⁷). The largest coefficient difference from sklearn
is 0.002, the largest probability difference on test is 0.001, and 99.8% of
call/no-call decisions are identical. The remaining tiny gap is float32
(PyTorch) vs. float64 (sklearn) plus the iterative optimisers stopping at
slightly different points. **It wasn't like this at first:** with C = 0.1 and
the macro features, the manual loop's coefficients differed by up to 0.47 even
though the objective matched to 4·10⁻⁵. The macro indicators are highly
collinear, which makes the loss surface very flat along some directions, so
different optimisers stop at different points on it. Removing the collinear
features and using stronger regularisation (which makes the problem better
conditioned) made all three converge to the same solution.

**Business reading:** on validation the model is clearly useful. It turns
"call everyone" (−€16.5k) into +€5.1k by calling 14% of the list. On test,
however, **calling everyone beats the model** (€69k vs. €55k). With a 38.5%
base rate almost every client is above the 16% break-even, so any filtering
loses money. The model still *ranks* well on test (top-10% precision 70.5% vs.
38.5% base rate, lift ≈ 1.8×), but its *probabilities* are badly
under-calibrated: it was trained in a 5.6% world, as the calibration plot in
the dashboard shows. A threshold chosen on probabilities therefore does not
transfer across periods.

## Limitations & next steps

1. **Prior shift / calibration.** The base rate changes by ~7× between train
   and test. Fix: before each wave, recalibrate the intercept to the recent
   observed rate (`logit p' = logit p + log(odds_new / odds_train)`), or retrain
   on a rolling window of the most recent months.
2. **The threshold should be set as a capacity (top-k%), not as a probability**,
   because ranking survives drift and calibration doesn't. With more time I'd
   frame the decision as "the call centre can make N calls this wave; who are
   the top N?"
3. **Cost and value numbers are assumptions** (€8 / €50). The dashboard lets
   you change them; the real ones should come from finance.
4. **Selection bias.** The data only contains clients the bank chose to call,
   so the model has never seen clients outside the bank's targeting rules. The
   late-2010 lists were probably pre-filtered, which would also explain part of
   the jump in base rate.
5. **Month ≠ seasonality.** `month` partly encodes "which campaign type", not
   just season. September never appears in training, so it falls back to the
   reference level.
6. **Linear model limits.** Interactions (e.g. retired × cellular) would need
   explicit feature crosses. A tree model could be tested as a benchmark,
   outside this assignment's scope.
7. **One test window.** A single chronological split gives one estimate.
   Walk-forward validation over several windows would show how stable the
   0.70 AUC really is.

## Generative AI use disclosure

I used CLAUDE to help me with the code and also when lost on what to do next or how t set up some stuff.
