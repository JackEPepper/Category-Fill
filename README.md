# Category Fill

Fill in a categorical column that's blank on some rows (e.g. industry sector on 10–20% of a
commercial loan book) using XGBoost. It also shows which other columns actually predict it, and how
far you can trust each guess.

## Quick start

**Windows:** double-click `run_app.bat`. **Mac/Linux:** `./run_app.sh`.
The first run installs dependencies into `.venv` (about a minute). After that, a browser tab opens.

1. **Drop your file** (CSV, Excel or Parquet) into the sidebar. A synthetic 20k-loan sample loads
   if you haven't dropped a file yet.
2. **Tab 0 · Population** (optional): take loans out of the analysis with rules. Each rule picks one
   column and either **excludes** the loans holding the values you choose, or **keeps only** those loans.
   For numeric columns, a rule takes a range instead, and any rule can also match blanks. Stack one
   rule per column (loan type, state, amount range…). A loan stays in only if it passes every rule.
   Excluded loans aren't used for training and aren't filled. They stay in the download marked
   `excluded`. **Save these rules** downloads a `.json` you can load next time, or pass to the CLI.
3. **Column to fill** is picked for you: the category column whose missing share is closest to 15%. Change it if it's wrong.
4. **Tab 1 · Data** shows each column's type, how much is blank, how many distinct values it has, and the target's class mix.
5. **Tab 2 · Features** shows how strongly each column is associated with the target, and flags columns to leave out
   (IDs, constants, no signal, or blank on the same rows as the target). The recommended set is
   pre-selected; edit it as you like.
6. **Tab 3 · Model readout:** press **Train model**. You get:
   - test accuracy compared with a "guess the most common class" baseline, plus top-3 accuracy and macro F1
   - accuracy at each confidence level, and how many blank rows you'd fill at each cut-off
   - precision, recall and F1 for each category, and a confusion matrix
   - which columns matter (the accuracy lost when each column is shuffled), plus an optional test
     that finds the smallest set of columns that does the job
7. **Tab 4 · Filled data:** set a minimum confidence, then download the filled file as CSV, the
   predictions as CSV, or a full Excel report.

The filled file is your original data plus these columns:

| column | meaning |
|---|---|
| `<target>_filled` | original value, or the prediction for blank rows |
| `<target>_source` | `original` / `predicted` / `blank (low confidence)` / `excluded` |
| `<target>_confidence` | model probability for the predicted class (blank rows only) |
| `<target>_2nd_choice` | runner-up class |

## Command line

```bash
pip install -r requirements.txt
python -m category_fill loans.xlsx --target industry_sector --subset-curve --out results/
```

The command prints the full readout and writes `results/filled_data.csv`, `blank_row_predictions.csv`
and `report.xlsx`. Useful flags: `--exclude loan_id borrower_name`, `--categorical branch_no`,
`--missing-tokens Unknown TBD`, `--min-confidence 0.8`, `--balance-classes`.
Run `python -m category_fill -h` for all of them.

Population rules on the command line (repeat a flag for each column; `|` separates values;
`a..b` is a numeric range with either end optional; `<blank>` matches empty cells):

```bash
python -m category_fill loans.xlsx --target industry_sector \
  --exclude-rows "loan_type=SBA 7(a)|Line of Credit" \
  --exclude-rows "loan_amount=..250000" \
  --keep-rows "state=TX|CA|FL"
# or reuse rules saved from the app:
python -m category_fill loans.xlsx --rules population_rules.json
```

## How it works

- **Cleaning** (`category_fill/data.py`): blanks such as `N/A`, `null` and `#N/A` become missing.
  `$1,250,000`, `4.5%` and `(300)` become numbers. Date columns are split into year, month and
  days-since-1970. Codes with leading zeros stay text.
- **Screening** (`analysis.py`): measures each column's association with the target using
  bias-corrected Cramér's V (numeric columns are cut into deciles first). It also checks whether the
  blank-target rows look different from the labelled ones. If they do, the blanks aren't random and
  the test score may be optimistic.
- **Model** (`model.py`): XGBoost with native categorical support, so no one-hot encoding is needed.
  Labelled rows are split three ways, each stratified by class: train, validation (to choose the
  number of trees by early stopping) and test (only for the readout). The final model is then refit
  on every labelled row and predicts the blanks. Classes with fewer than 10 labelled rows (adjustable)
  are reported but not predicted.

## Tips for real loan data

- Exclude anything recorded *after* or *because of* the category (e.g. a sector-specific
  pricing grid). It inflates the test score without helping on the blank rows.
- A near-deterministic feature (flagged in tab 2) is fine if it's a real upstream field, such as a
  NAICS code predicting the sector. Check that it's populated on the blank rows (the
  "missing % on blank-target rows" column).
- If precision on rare categories matters more than overall accuracy, turn on
  **Balance classes** in Model settings.
- For hierarchical codes (NAICS/SIC), adding a 2- or 3-digit prefix column before upload often helps.

## Development

```bash
python -m category_fill.sample sample_data/sample_loans.csv   # regenerate the sample
python -m pytest -q
```
