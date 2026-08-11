# AI Decision Guidelines

**Purpose of this document**

This is an internal system specification for the multi-agent triage pipeline — not a policy document, and it must never be placed in the RAG corpus or cited as a retrieved policy in the justification trail. Policy answers "is this return eligible." This document answers "how does our system decide, given policy eligibility plus behavioral/fraud signals, what verdict to reach."

If the Orchestrator ever cites a threshold from this document as if it were a policy clause, that is a grounding failure and should be treated as a bug.

---

## 0. Trained Model Status (Risk Scoring Agent)

**Model:** LightGBM, trained on the team's synthetic dataset.
**Current performance:** 76% accuracy · F1 (Legitimate) = 0.78 · F1 (Fraud) = 0.73.

**Confirmed feature schema** (from trained model's feature importance output — this supersedes earlier placeholder column names elsewhere in this doc and in the RAG/mock-doc set):

`account_age_days, discount_pct, order_value_percentile, seller_return_rate, seller_age_days, seller_customer_frequency, days_since_last_return, order_value, user_avg_order_value, review_count, days_since_delivered, return_to_order_ratio, seller_rating, price, total_orders_at_time, days_left_to_return, image_uploaded, shared_device_flag, category_base_defect, reason_category, total_returns_at_time, phone_verified, category, return_frequency_score, seller_repeat_ratio, weekend_return_ratio, is_prepaid, high_value_return_ratio, returns_last_90d, same_category_return_ratio, return_type, email_verified, requires_manual_verification, returns_last_30d, within_return_window, is_non_returnable`

**Top predictive features** (by importance, descending): `account_age_days`, `discount_pct`, `order_value_percentile`, `seller_return_rate`, `seller_age_days`, `seller_customer_frequency`, `days_since_last_return`, `order_value`, `user_avg_order_value`, `review_count`.

**Scope correction:** an earlier version of this document flagged Seller-Buyer Collusion as a known limitation, reasoning that per-order data likely couldn't capture seller-side relational signals. That reasoning no longer holds — the trained model already includes `seller_return_rate`, `seller_age_days`, `seller_rating`, `seller_repeat_ratio`, and `seller_customer_frequency` as features. Section 1 below has been corrected accordingly.

**⚠️ Calibration status — read before using Section 2's thresholds:**
The Risk Score bands in Section 2 (< 0.30 / 0.30–0.70 / > 0.70) were set as directional placeholders before this model existed. They have **not** been validated against this model's actual precision/recall-at-threshold curve. 76% accuracy with F1-fraud of 0.73 is workable but not so strong that these exact cutoffs should be trusted blind — in particular, the "> 0.70 → eligible for Auto-Reject" band deserves scrutiny once threshold-level precision/recall numbers are available (not just aggregate accuracy/F1), since that's what your Fraud Recall @ controlled FPR metric actually depends on. **Action item: pull precision/recall at several candidate thresholds (e.g., 0.5, 0.6, 0.7, 0.8) from Rushil and re-set Section 2 bands against real numbers before the pipeline is finalized.**

---

## 1. Fraud Scenario → Signal Map

Ground truth scenario coverage for synthetic data generation, eval sets, and Risk/Text agent feature design.

| Scenario | Label | Key Signals | Primary Detecting Agent |
|---|---|---|---|
| Genuine Defect | Legitimate | Reason plausible for category; account history unremarkable; optional damage photo consistent with claim | Text Reasoning Agent (plausibility), Risk Agent (low score) |
| Changed Mind | Legitimate | "Doesn't fit"/"don't need it" on an eligible category, within window, normal return frequency | Data Agent (normal ratio), RAG (within window) |
| Wardrobing | Fraud | High-value lifestyle item; purchase adjacent to an event (wedding season, festival date proximity); "doesn't fit"/"didn't like it" reason; possible signs of use in image; first return on account; often paired with a high discount | Text Reasoning Agent + Risk Agent — maps to `order_value_percentile`, `high_value_return_ratio`, `discount_pct`; see rule R9 in Section 6a for the discount-driven sub-signal |
| Empty Box Claim | Fraud | Delivery scan/weight confirms shipped; vague or insistent reason; often a **sudden claim from an otherwise old, trusted account** (contradiction between account age and claim novelty) | Data Agent (`account_age_days` vs. `days_since_last_return`) + Orchestrator (contradiction flag) — see rules R1, R4 in Section 6a |
| Serial Returner | Fraud | Return-to-order ratio > 40%; multiple SKUs with low review counts; pattern spans categories | Risk Agent (primary) — maps to `return_to_order_ratio`, `return_frequency_score`, `returns_last_30d`, `returns_last_90d`, `same_category_return_ratio`; see rule R5 in Section 6a for the related window-timing signal |
| Item Swap | Fraud | Returned item's IMEI/serial/barcode does not match original order | RAG (Identity Verification clause) is the primary detection path for this scenario — the policy-level identifier check governs the verdict directly; no dedicated Section 6a rule is needed on top of it |
| Seller-Buyer Collusion | Fraud | Statistically anomalous — repeated returns tied to one seller-buyer pair, refund-without-return patterns, abnormal approval velocity | Risk Agent — **confirmed in scope**: the trained model includes `seller_return_rate`, `seller_age_days`, `seller_rating`, `seller_repeat_ratio`, and `seller_customer_frequency` as features. See rules R7 and R8 in Section 6a. Retain in Failure Analysis only if the eval set can't construct enough labeled collusion cases to actually test it. |
| Borderline / Ambiguous (required 3rd bucket) | Escalate (ground truth) | E.g. old account's first-ever fraud-pattern-adjacent claim; return window expired by 1 day; category is policy-grey-area | N/A — correct behavior is abstention, not detection |

---

## 2. Risk Score Thresholds (Orchestrator logic, not policy)

⚠️ **Not yet calibrated against the trained model.** These bands are directional placeholders — see Section 0 for current model status and the required calibration action item. Do not treat the specific numbers below as final until validated against real precision/recall-at-threshold data.

These bands are tunable system parameters, set by your team based on the Precision/Recall tradeoff you choose to optimize for (see evaluation criteria: Fraud Recall @ controlled FPR, target ≤5% wrongful rejection rate).

| Fraud Risk Score | Behavior |
|---|---|
| < 0.30 | Low risk — eligible for Auto-Approve if policy conditions (window, category, evidence) are also satisfied |
| 0.30 – 0.70 | Ambiguous — Escalate to Human by default |
| > 0.70 | High risk — eligible for Auto-Reject **only if** a confirmed policy violation or hard fraud signal also applies (see Section 4); high score alone should escalate, not auto-reject, unless corroborated |

**Important:** a high risk score alone should not auto-reject. Auto-Reject requires risk score + a corroborating hard signal (confirmed identity mismatch, confirmed outside-window with no exception, confirmed policy violation). This is your main lever against the False-Positive Trap named in the problem statement.

---

## 3. AI Confidence Bands (Orchestrator's own output confidence, distinct from Risk Score)

| Orchestrator Confidence | Behavior |
|---|---|
| > 95% | Auto-Decision (Approve or Reject) permitted, provided Risk Score band above also supports it |
| 70% – 95% | Proceed to full policy validation pass before finalizing; do not short-circuit |
| < 70% | Escalate to Human regardless of Risk Score |

---

## 4. Escalation Matrix

Escalate to Human if **any** of the following are true:

- Risk Score is in the 0.30–0.70 band
- Required evidence is missing or unreadable (see Evidence and Reason Consistency policy)
- Policy ambiguity remains after applying precedence (see Policy Precedence clause)
- Order value > ₹50,000 and supporting evidence is incomplete (see High-Value Order Verification policy)
- Image or text evidence is inconsistent with the claim (see Evidence and Reason Consistency policy)
- Orchestrator Confidence < 70%
- Any confirmed hard-fraud signal (identity mismatch, empty-box contradiction) is present **without** a second corroborating signal — single uncorroborated hard signals escalate rather than auto-reject, to protect Precision

Auto-Reject requires **at least one confirmed policy violation or hard fraud signal**, not risk score alone.

Auto-Approve requires: Risk Score < 0.30 **and** all applicable RAG-retrieved policy conditions satisfied **and** Orchestrator Confidence > 95%.

---

## 5. Evidence Priority (for Orchestrator synthesis, not a policy ranking)

When agent outputs disagree or must be weighed, the Orchestrator should weight inputs in this order:

1. RAG-retrieved policy (eligibility is a hard gate — no ML score overrides a clear policy ineligibility)
2. Confirmed product/order metadata (identifiers, delivery scans, timestamps)
3. ML Risk Score + SHAP feature attributions
4. Customer-uploaded evidence (images, documentation)
5. Account/behavioral history (Data Agent features)
6. Text Reasoning Agent's plausibility/contradiction assessment

This ordering exists so the justification trail has a deterministic way to explain precedence when signals conflict (e.g., low risk score but a confirmed policy ineligibility — policy wins).

---

## 6. Contradiction, Behavioral, and Override Rules

### 6a. Contradiction and Behavioral Rules (Text Reasoning / Data / Risk Agent inputs)

These are specific, checkable rules that feed the Text Reasoning Agent (contradiction detection) and Risk Agent (behavioral scoring). They operationalize the general "Evidence and Reason Consistency" and "Repeat Returns" policy principles into concrete checks. As with all rules in this document, they are system logic — never cite these as policy in the justification trail; cite the underlying policy clause instead, and use these rules to explain *how* the system evaluated it.

**R1 — Contradictory Claim (delivery proof)**
IF customer reason indicates non-delivery ("never received it") AND a delivery scan/proof-of-delivery record exists
→ Escalate (do not auto-reject; delivery proof can itself be disputed, e.g. wrong address, doorstep theft — this needs human judgment)

**R2 — Contradictory Claim (attribute mismatch)**
IF customer reason states a specific product attribute (colour, size, model) AND uploaded image or order record confirms the correct attribute was delivered
→ Escalate. Image-based attribute contradictions are heuristic, not certain, so this is not sufficient for auto-reject on its own — a human reviewer should confirm before a final verdict.

**R3 — Multiple High-Value Items, Same Customer**
IF a customer submits returns for ≥3 units of the same high-value item (e.g., 5 iPhones) in a single batch or short window
→ Route to Fraud Review (Escalate); treat as a distinct signal from ordinary Serial Returner pattern, since it suggests resale/collusion rather than personal-use dissatisfaction

**R4 — Sudden Behavior Change**
IF historical return ratio < 5% AND returns in the last 30 days ≥ 5
→ Escalate. This is the "sudden spike from a previously quiet account" pattern — do not resolve it with the Trusted Customer Override (Section 6b), since the override requires a *sustained* low return ratio, and a sudden spike is the exact pattern that override should not paper over.

**R5 — Return Window Manipulation**
IF a customer submits multiple returns, each timed exactly on the last eligible day of that item's return window
→ Increase Risk Score (Risk Agent feature; pattern suggests deliberate testing of window limits, but is not itself sufficient for Escalate or Reject without corroboration)

**R6 — New Account, High-Value Return**
IF `account_age_days` is low (new account, exact cutoff to be set once account-age distribution is known) AND the associated order falls in a high `order_value_percentile` band
→ Increase Risk Score; do not auto-escalate or auto-reject on this alone. A new account's first order can legitimately be high-value (e.g., a gift purchase); this is a contributing signal, not a standalone trigger. Corroborate against R3, R4, or the High-Value Order Verification policy gate before escalating.

**R7 — High Seller Fraud Rate**
IF `seller_return_rate` for the seller on this order exceeds a defined threshold (to be calibrated against the seller-level distribution, not assumed)
→ Increase Risk Score. A high seller-level return rate does not itself indict the customer's specific claim, but it lowers the bar for what counts as a corroborating signal elsewhere in this rule set (e.g., makes R3 or R5 more likely to justify Escalate when combined with this).

**R8 — Shared Device Across Accounts**
IF `shared_device_flag` is set (the device associated with this return has also been used by other customer accounts)
→ Escalate. This is a relational signal most relevant to Seller-Buyer Collusion and account-farming patterns (see Section 1) — it indicates the claim should not be evaluated as an isolated single-customer case, and a human reviewer should check for linked-account abuse before a verdict is finalized.

**R9 — Discount-Driven High-Value Return**
IF `discount_pct` on the order is high AND the item falls in a Lifestyle/high-value category AND the stated reason matches a Wardrobing-pattern reason ("doesn't fit," "didn't like it," "not needed")
→ Increase Risk Score; treat as a Wardrobing sub-signal, not a separate scenario. Deep-discount lifestyle purchases followed by a fit/preference-based return combine two independently weak signals into a stronger one — heavily discounted items are cheaper to "trial and return" for an event, which is the core Wardrobing pattern. This directly closes the earlier gap where `discount_pct` (the model's 2nd-most-important feature) had no rule mapping it to any of the 7 scenarios.

### 6b. Auto-Approve Override for Trusted Accounts

IF account age > 3 years AND historical return ratio < 5% (sustained, not just current period) AND no prior confirmed fraud on account AND order value < ₹10,000
AND the only flag present is a **minor** inconsistency (defined narrowly as: a Text Reasoning Agent plausibility flag with no corroborating Risk Agent or Data Agent signal, and no R1/R3 trigger)
→ Auto-Approve is permitted despite the flag, to reduce false-positive rejection of high-trust customers.

This override does **not** apply if:
- Any hard signal from Section 6a (R1, R3) is present
- The current request is part of a sudden spike (R4)
- Order value ≥ ₹10,000
- Risk Score ≥ 0.30 (i.e., this override sits inside the existing < 0.30 Auto-Approve band from Section 2 — it does not create a new path to bypass that band, it only permits a minor inconsistency flag to be overridden within that band)

### 6c. Policy-Boundary Escalation Example

The general Escalation Matrix (Section 4) already routes policy ambiguity to Escalate. A frequent concrete instance of this: return window expired by a single day, combined with a trusted account profile. Per Section 4's ambiguity rule, this should Escalate rather than Auto-Reject — a hard one-day cutoff enforced against a long-standing customer is a legitimate case for human discretion, not an automated rejection.

## 7. Audit Trail Requirements

Every decision output must include, at minimum:

- Verdict (Auto-Approve / Auto-Reject / Escalate to Human)
- Retrieved policy clause(s) actually applied, with source document
- ML Risk Score and top SHAP feature contributions
- Evidence considered (behavioral features, text reasoning flags, image flags if applicable)
- Final reasoning chain connecting the above to the verdict, in human-readable form
- If Escalated: a summarized case dossier for the human reviewer

This satisfies the Explainability Faithfulness rubric (0–3 scale) — a "3" requires grounded signals **and** a correctly cited policy clause, not just a plausible-sounding explanation.
