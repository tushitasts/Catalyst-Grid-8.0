"""
Rule Engine Agent — Evaluates heuristic fraud-detection rules (R0–R6).
Hard rules (R0) force immediate rejection. Soft rules (R1–R6) contribute
to a normalised score that feeds into the combined risk score.
"""
import settings


class RuleResult:
    """Single triggered rule with metadata."""

    def __init__(self, rule_id: str, rule_name: str,
                 description: str, score: float):
        self.rule_id = rule_id
        self.rule_name = rule_name
        self.description = description
        self.score = score

    def to_dict(self) -> dict:
        return {
            'rule_id': self.rule_id,
            'rule_name': self.rule_name,
            'description': self.description,
            'score': self.score,
        }


class RuleEngineAgent:
    """Tool: Evaluates business rules and returns triggered rules + score."""

    def run(self, case_data: dict) -> dict:
        raw = case_data['raw']
        fv = case_data['feature_vector'].iloc[0]   # single row as Series
        triggered: list[RuleResult] = []

        # ── HARD RULES (policy violations -> immediate auto-reject) ────────────

        # R0 — Non-returnable product
        if int(raw.get('is_non_returnable', 0)) == 1:
            triggered.append(RuleResult(
                "R0_NON_RETURNABLE",
                "Non-Returnable Product",
                f"Product category '{raw.get('category')}' is marked non-returnable. "
                f"Return policy does not allow returns for this item.",
                settings.RULE_WEIGHTS["R0_NON_RETURNABLE"],
            ))

        # R0 — Outside return window
        if int(raw.get('within_return_window', 1)) == 0:
            triggered.append(RuleResult(
                "R0_OUT_OF_WINDOW",
                "Outside Return Window",
                f"Return requested outside the "
                f"{raw.get('return_window_days', '?')}-day return window. "
                f"Days since delivery: {raw.get('days_since_delivered', '?')}.",
                settings.RULE_WEIGHTS["R0_OUT_OF_WINDOW"],
            ))

        # ── SOFT RULES (contribute to risk score) ─────────────────────────────
        
        acct_age = int(raw.get('account_age_days', 9999))
        order_val = float(raw.get('order_value', 0))
        cat = raw.get('category', '')
        ovr = float(fv.get('order_value_ratio', 0) if 'order_value_ratio' in fv.index else 0)
        disc = float(raw.get('discount_pct', 0))
        reason = raw.get('reason_category', '')
        ret_30d = float(fv.get('returns_last_30d', 0) if 'returns_last_30d' in fv.index else 0)
        total_rets = int(raw.get('total_returns_at_time', 0))
        ret_ratio = float(fv.get('return_to_order_ratio', 0) if 'return_to_order_ratio' in fv.index else 0)
        days_left = int(raw.get('days_left_to_return', 999))
        seller_rr = float(raw.get('seller_return_rate', 0))
        shared_dev = int(raw.get('shared_device_flag', 0))

        # R1 — Delivery proof contradiction (Simulated tabular proxy)
        if reason == "not_delivered" and float(raw.get('days_since_delivered', -1)) >= 0:
            triggered.append(RuleResult("R1", "Delivery Proof Contradiction", "Customer claims non-delivery but scan exists.", settings.RULE_WEIGHTS["R1"]))

        # R2 — Attribute mismatch (Normally image/text based, placeholder for tabular proxy)
        pass 

        # R3 — Multiple High-Value Items
        if ret_30d >= settings.MULTIPLE_RETURNS_THRESHOLD and ovr > settings.HIGH_VALUE_RATIO_THRESHOLD:
            triggered.append(RuleResult("R3", "Multiple High-Value Returns", f"{int(ret_30d)} recent returns with high value ratio.", settings.RULE_WEIGHTS["R3"]))

        # R4 — Sudden Behavior Change
        if ret_ratio < 0.05 and ret_30d >= 5:
            triggered.append(RuleResult("R4", "Sudden Behavior Change", "Historically trusted account with sudden spike in returns.", settings.RULE_WEIGHTS["R4"]))

        # R5 — Return Window Manipulation
        if days_left == 0 and total_rets >= settings.MULTIPLE_RETURNS_THRESHOLD:
            triggered.append(RuleResult("R5", "Return Window Manipulation", "Repeatedly returning items on the last eligible day.", settings.RULE_WEIGHTS["R5"]))

        # R6 — New Account, High-Value Return
        if acct_age < settings.NEW_ACCOUNT_AGE_THRESHOLD and order_val > settings.HIGH_VALUE_THRESHOLD:
            triggered.append(RuleResult("R6", "New Account, High-Value", "New account placing unusually high-value order.", settings.RULE_WEIGHTS["R6"]))

        # R7 — High Seller Fraud Rate
        if seller_rr > 0.15:
            triggered.append(RuleResult("R7", "High Seller Fraud Rate", f"Seller has a suspiciously high return rate ({seller_rr:.1%}).", settings.RULE_WEIGHTS["R7"]))

        # R8 — Shared Device
        if shared_dev == 1:
            triggered.append(RuleResult("R8", "Shared Device Across Accounts", "Device linked to multiple accounts (potential farming/collusion).", settings.RULE_WEIGHTS["R8"]))

        # R9 — Discount-Driven High-Value (Wardrobing)
        if cat in ("Clothing", "Footwear", "Beauty") and ovr >= 1.0 and disc > 0 and reason in ("not_fit", "quality_issue"):
            triggered.append(RuleResult("R9", "Discount-Driven High-Value Return", "Discounted lifestyle item returned for fit/preference.", settings.RULE_WEIGHTS["R9"]))

        # R10 & R11 — (Missing from current tabular schema)
        pass

        # ── Aggregation ──────────────────────────────────────────────────────
        hard_rule_triggered = any(r.rule_id.startswith("R0_") for r in triggered)

        # Normalised score (soft rules only for the numeric score)
        soft_ids = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]
        soft_total = sum(r.score for r in triggered if not r.rule_id.startswith("R0_"))
        soft_max = sum(settings.RULE_WEIGHTS.get(rid, 0) for rid in soft_ids)
        normalized_score = min(soft_total / soft_max, 1.0) if soft_max > 0 else 0.0

        # If a hard rule triggered, force to 1.0
        if hard_rule_triggered:
            normalized_score = 1.0

        return {
            'triggered_rules': triggered,
            'rule_score': normalized_score,
            'hard_rule_triggered': hard_rule_triggered,
            'total_rules_checked': len(soft_ids) + 2,   # +2 for the R0 rules
            'rules_triggered_count': len(triggered),
        }
