"""Score the PDF Q&A pipeline against a 50-question golden set for ISLR (1st edition, 441-page PDF).

Run:    python run_eval.py data/islr.pdf
Options:
  --no-judge   skip the LLM judge (reports retrieval hit rate only)
  --limit N    run only the first N questions
  --sleep S    seconds to wait between questions (default 4, helps with Groq rate limits)
"""
import argparse
import csv
import os
import time

from rag import config as cfg
from rag.engine import load_or_build_index, build_chain, ask, flush_traces

# (question, golden answer, PDF pages where the answer is found)
GOLDEN = [
    ("What is the difference between supervised and unsupervised learning?",
     "Supervised: each observation has predictors and a response, and the goal is to predict or understand the response. Unsupervised: only inputs and no response; the goal is to find structure such as clusters.",
     [41, 43]),
    ("What is the difference between prediction and inference in statistical learning?",
     "Prediction estimates Y for new X, treating f as a black box and caring about accuracy. Inference tries to understand how Y is affected by each predictor, so the form of f matters.",
     [34, 35, 36]),
    ("What is irreducible error?",
     "The error term epsilon, which cannot be predicted from X, so it cannot be reduced by estimating f better; it sets an upper bound on prediction accuracy.",
     [33, 34]),
    ("What are the three components of the expected test MSE?",
     "The variance of the estimate f-hat, its squared bias, and the variance of the error term (irreducible error).",
     [50]),
    ("What is the bias-variance trade-off?",
     "Higher flexibility lowers bias but raises variance. Test error depends on both, so the best method balances them, giving a U-shaped test MSE curve.",
     [51, 52]),
    ("How do training MSE and test MSE change as model flexibility increases?",
     "Training MSE keeps decreasing, while test MSE first falls and then rises (U shape) because very flexible models overfit.",
     [45, 46, 47]),
    ("What does the Bayes classifier do?",
     "It assigns each observation to the class with the highest conditional probability Pr(Y=j | X=x0), giving the lowest possible test error rate.",
     [52, 53]),
    ("What is the Bayes error rate?",
     "The error rate of the Bayes classifier, which is the lowest possible test error rate; it is analogous to the irreducible error.",
     [53, 54]),
    ("How does the K-nearest neighbors classifier work?",
     "It finds the K training points closest to x0, estimates each class probability as the fraction of those neighbors in that class, and picks the class with the highest probability.",
     [54]),
    ("What is the difference between parametric and non-parametric methods?",
     "Parametric methods assume a functional form for f and estimate a fixed set of parameters. Non-parametric methods make no explicit assumption about the form of f, are more flexible, but need many more observations.",
     [36, 37, 38, 39]),
    ("What is the residual sum of squares (RSS)?",
     "The sum of squared differences between observed and fitted responses, RSS = sum of (y_i - y_hat_i)^2; least squares minimizes it.",
     [77]),
    ("What does the R-squared statistic measure?",
     "The proportion of variability in the response that is explained by the regression, between 0 and 1.",
     [83, 84, 85]),
    ("What is the F-statistic used for in multiple linear regression?",
     "It tests whether at least one predictor is related to the response (null hypothesis: all coefficients are zero).",
     [90, 91, 92]),
    ("What is the variance inflation factor and when is it a problem?",
     "VIF measures collinearity of a predictor with the others. The smallest value is 1, and a VIF above about 5 or 10 indicates problematic collinearity.",
     [116]),
    ("What is an interaction term in linear regression?",
     "A product of predictors (X1*X2) added to the model so the effect of one predictor depends on the value of another (synergy).",
     [96, 97]),
    ("How are qualitative predictors with two levels included in a regression model?",
     "By creating a dummy variable that is 1 for one level and 0 for the other and using it as a predictor.",
     [99, 100]),
    ("What is heteroscedasticity and how can it show up?",
     "Non-constant variance of the error terms. It shows up as a funnel shape in the residual plot; transforming the response with log or square root can help.",
     [110, 111]),
    ("What is the difference between an outlier and a high leverage point?",
     "An outlier has an unusual response y_i given x_i, while a high leverage point has an unusual predictor value x_i.",
     [112, 113, 114]),
    ("Why is linear regression generally not appropriate for a qualitative response with more than two classes?",
     "Coding classes as numbers implies an ordering and equal spacing that usually does not exist, so different codings give different models.",
     [144, 145, 146, 147]),
    ("What does the logistic function do in logistic regression?",
     "It models p(X) = e^(b0+b1X) / (1 + e^(b0+b1X)), which always stays between 0 and 1, so it gives valid probabilities.",
     [147]),
    ("What are odds and log-odds (logit) in logistic regression?",
     "Odds are p(X)/(1-p(X)); the log of the odds is the logit, which is linear in X in the logistic model.",
     [147]),
    ("What assumptions does linear discriminant analysis make?",
     "Observations in each class come from a Gaussian distribution with a class-specific mean and a covariance (variance) common to all classes.",
     [154, 155, 156, 157, 158]),
    ("What are sensitivity and specificity?",
     "Sensitivity is the percentage of true positives correctly identified; specificity is the percentage of true negatives correctly identified.",
     [162, 163]),
    ("What is an ROC curve and what does the AUC summarize?",
     "The ROC curve plots true positive rate against false positive rate across thresholds; the area under it (AUC) summarizes overall performance, higher is better.",
     [163, 164]),
    ("How does quadratic discriminant analysis differ from LDA?",
     "QDA lets each class have its own covariance matrix, giving a quadratic boundary; it is more flexible, so LDA is better with few training observations and QDA with many.",
     [164, 165, 166]),
    ("What are the drawbacks of the validation set approach?",
     "The test error estimate can be highly variable depending on the split, and it tends to overestimate test error because fewer observations are used for training.",
     [193, 194]),
    ("What is leave-one-out cross-validation?",
     "Each observation is used once as the validation set while the model is fit on the other n-1; the n errors are averaged.",
     [193, 194, 195]),
    ("What is k-fold cross-validation and what values of k are typically used?",
     "Split the data into k folds, hold out each fold once while fitting on the rest, and average the k errors. Typically k = 5 or 10.",
     [196, 197]),
    ("How does the bias-variance trade-off apply to choosing k in k-fold cross-validation?",
     "k-fold CV with k = 5 or 10 often gives more accurate test error estimates than LOOCV because LOOCV has higher variance; k-fold accepts a little bias for lower variance.",
     [198, 199, 200]),
    ("What is the bootstrap used for?",
     "To quantify the uncertainty of an estimator or method by repeatedly sampling with replacement from the original data.",
     [202, 203, 204]),
    ("How does best subset selection work?",
     "Fit every possible model for each number of predictors, pick the best of each size, then choose among sizes using cross-validation, Cp, AIC, BIC or adjusted R^2.",
     [220, 221]),
    ("How does forward stepwise selection work?",
     "Start with no predictors and add one at a time, each time adding the predictor giving the greatest improvement, then pick the best model in the sequence.",
     [222, 223]),
    ("Which criteria can be used to estimate test error when comparing models of different sizes?",
     "Cp, AIC, BIC and adjusted R^2, or direct estimates using validation or cross-validation.",
     [226, 227, 228]),
    ("What does ridge regression do?",
     "It minimizes RSS plus lambda times the sum of squared coefficients, shrinking coefficients toward zero but not exactly to zero.",
     [230, 231, 232]),
    ("What is the key difference between the lasso and ridge regression?",
     "The lasso uses an L1 penalty that can set coefficients exactly to zero, so it does variable selection; ridge keeps all predictors.",
     [234, 235, 236]),
    ("What is principal components regression?",
     "Least squares regression that uses the first M principal components as predictors; the components are found without using the response (unsupervised).",
     [245, 246, 247, 248]),
    ("How does partial least squares differ from principal components regression?",
     "PLS is supervised: it also uses the response when finding the new features, unlike PCR.",
     [252, 253]),
    ("What is a regression spline?",
     "A piecewise polynomial fit in regions separated by knots, constrained to join smoothly at the knots.",
     [285, 286, 288]),
    ("What is a smoothing spline?",
     "The function that minimizes RSS plus a roughness penalty, lambda times the integral of the squared second derivative, so it fits well but stays smooth.",
     [292, 293]),
    ("What are generalized additive models (GAMs)?",
     "An extension of multiple linear regression that allows non-linear functions of each predictor while keeping the model additive.",
     [297, 298]),
    ("What is recursive binary splitting in decision trees?",
     "A top-down, greedy approach that repeatedly splits the predictor space at the cut point giving the greatest reduction in RSS (or impurity).",
     [321, 322, 323]),
    ("How is a decision tree pruned?",
     "Grow a large tree, then use cost complexity (weakest link) pruning with a tuning parameter alpha to get subtrees, choosing alpha by cross-validation.",
     [323, 324]),
    ("What is bagging?",
     "Bootstrap aggregation: fit a tree to each of B bootstrap samples and average the predictions (or majority vote), which reduces variance.",
     [331, 332]),
    ("How do random forests differ from bagging?",
     "At each split only a random subset of m predictors is considered (about the square root of p), which decorrelates the trees.",
     [332, 333, 334]),
    ("How does boosting work for decision trees?",
     "Trees are grown sequentially, each fit to the residuals of the current model rather than the outcome, and added with a shrinkage parameter so the model learns slowly.",
     [335, 336, 337]),
    ("What is the Gini index used for in classification trees?",
     "A measure of node purity (total variance across classes); it is small when a node has mostly one class.",
     [327, 328, 329]),
    ("What is the maximal margin classifier?",
     "The separating hyperplane that is farthest from the training observations (largest margin); it depends only on the support vectors closest to it.",
     [353, 354, 355, 356]),
    ("How does K-means clustering work?",
     "Randomly assign observations to K clusters, then repeatedly compute each cluster centroid and reassign each observation to the closest centroid until assignments stop changing; it finds a local optimum.",
     [403, 404, 405]),
    ("How does hierarchical clustering work?",
     "It builds a dendrogram bottom-up, starting with each observation as its own cluster and repeatedly fusing the two closest clusters; cutting the dendrogram picks the number of clusters.",
     [406, 407, 408, 409, 410]),
    ("What is principal components analysis?",
     "A method that finds a low-dimensional representation: the first principal component is the direction of largest variance, and each later one is uncorrelated with earlier ones and captures the most remaining variance.",
     [388, 389, 390, 391]),
]

JUDGE_PROMPT = """You are grading an answer against a reference answer.

Question: {q}
Reference answer: {gold}
Model answer: {ans}

The model answer is CORRECT if it contains the key facts of the reference answer and does not contradict it. Extra detail is fine. If it says it does not know, or that the topic is not in the document, it is INCORRECT.
Reply with exactly one line: CORRECT or INCORRECT, then " - " and a short reason."""


def with_retry(fn, tries=5):
    for n in range(tries):
        try:
            return fn()
        except Exception as e:
            msg = str(e).lower()
            transient = any(k in msg for k in ("rate", "429", "timeout", "overloaded", "503"))
            if n == tries - 1 or not transient:
                raise
            wait = 15 * (n + 1)
            print(f"      rate limit or timeout, waiting {wait}s...")
            time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sleep", type=float, default=4.0)
    args = ap.parse_args()

    cfg.check_keys()
    print("Loading index...")
    chain = build_chain(load_or_build_index(args.pdf))

    judge = None
    if not args.no_judge:
        from langchain_groq import ChatGroq
        judge = ChatGroq(model=os.getenv("JUDGE_MODEL", cfg.GROQ_MODEL), temperature=0)

    items = GOLDEN[: args.limit] if args.limit else GOLDEN
    rows, hits, judged, correct = [], 0, 0, 0
    hit_ok = hit_n = miss_ok = miss_n = 0
    pdf_name = os.path.basename(args.pdf)

    for i, (q, gold, pages) in enumerate(items, 1):
        try:
            result, _ = with_retry(lambda: ask(chain, q, session_id="eval", pdf_name=pdf_name))
            ans = result["answer"].strip()
            got = sorted({d.metadata.get("page", 0) + 1 for d in result["docs"]})
        except Exception as e:
            ans, got = f"ERROR: {e}", []

        hit = any(p in pages for p in got)
        hits += hit

        verdict, reason = "SKIPPED", ""
        if judge and not ans.startswith("ERROR"):
            try:
                out = with_retry(lambda: judge.invoke(
                    JUDGE_PROMPT.format(q=q, gold=gold, ans=ans))).content.strip()
                first = out.splitlines()[0] if out else ""
                up = first.upper()
                verdict = "INCORRECT" if "INCORRECT" in up else ("CORRECT" if "CORRECT" in up else "UNCLEAR")
                reason = first.split("-", 1)[1].strip() if "-" in first else ""
            except Exception as e:
                verdict, reason = "UNCLEAR", str(e)[:120]

        if verdict in ("CORRECT", "INCORRECT"):
            judged += 1
            ok = verdict == "CORRECT"
            correct += ok
            if hit:
                hit_n += 1
                hit_ok += ok
            else:
                miss_n += 1
                miss_ok += ok

        print(f"{i:>2}. {'HIT ' if hit else 'MISS'}  {verdict:<9}  {q[:62]}")
        rows.append({"id": i, "question": q, "golden_answer": gold,
                     "golden_pages": " ".join(map(str, pages)),
                     "retrieved_pages": " ".join(map(str, got)),
                     "retrieval_hit": hit, "answer": ans,
                     "verdict": verdict, "reason": reason})
        time.sleep(args.sleep)

    n = len(items)
    print("\n" + "=" * 52)
    print(f"Questions run:        {n}")
    print(f"Retrieval hit rate:   {hits}/{n} = {hits / n:.0%}  (a retrieved chunk came from a golden page)")
    if judged:
        print(f"Answer accuracy:      {correct}/{judged} = {correct / judged:.0%}  (judge: {os.getenv('JUDGE_MODEL', cfg.GROQ_MODEL)})")
        if hit_n:
            print(f"  when retrieval hit: {hit_ok}/{hit_n} = {hit_ok / hit_n:.0%}")
        if miss_n:
            print(f"  when retrieval miss:{miss_ok}/{miss_n} = {miss_ok / miss_n:.0%}")
    print("=" * 52)

    with open("eval_results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("Saved eval_results.csv")
    flush_traces()


if __name__ == "__main__":
    main()