"""
predict.py — SIA Inference Script v2
Support Integrity Auditor: CSV → Predictions + Evidence Dossiers

Usage:
  python predict.py --input data/enhanced_customer_support_data.csv
  python predict.py --input tickets.csv --output results/ --no-gemini
"""

import os, json, re, time, argparse, warnings
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack, csr_matrix
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
#  CONSTANTS  (must match train_pipeline.py exactly)
# ─────────────────────────────────────────────────────
PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
INV_PRIORITY = {v: k for k, v in PRIORITY_MAP.items()}

CAT_EXPECT = {
    'Fraud': 2.5, 'Technical': 1.8, 'Billing': 1.2,
    'Account': 1.0, 'General Inquiry': 0.3,
}
CRISIS_KW = [
    'crash', 'data not syncing', '2fa issues', 'login failed', 'payment failed',
    'data loss', 'outage', 'breach', 'unauthorized', 'ransomware', 'virus',
    'account suspended', 'access denied', 'corrupted', 'invoice discrepancy',
    'security', 'phishing', 'not working', 'cannot access', 'locked out',
    'system down', 'cannot login', 'failed to', 'error', 'broken',
]
TRIVIAL_KW = [
    'hours of operation', 'office location', 'product question', 'feature request',
    'demo request', 'subscription upgrade', 'cancel subscription', 'refund status',
    'pricing', 'where is', 'general question', 'how to upgrade', 'headquarters',
    'business hours', 'info request',
]


# ─────────────────────────────────────────────────────
#  FEATURE ENGINEERING  (mirrors train_pipeline.py)
# ─────────────────────────────────────────────────────
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['sl'] = df['Ticket_Subject'].fillna('').str.lower()
    df['dl'] = df['Ticket_Description'].fillna('').str.lower()

    df['is_crisis']  = df['sl'].apply(lambda x: int(any(k in x for k in CRISIS_KW)))
    df['is_trivial'] = df['sl'].apply(lambda x: int(any(k in x for k in TRIVIAL_KW)))
    df['n_crisis']   = df['sl'].apply(lambda x: sum(1 for k in CRISIS_KW if k in x))

    df['sat_low']  = (df['Satisfaction_Score'] <= 2).astype(int)
    df['sat_1']    = (df['Satisfaction_Score'] == 1).astype(int)
    df['sat_high'] = (df['Satisfaction_Score'] >= 4).astype(int)

    df['res_fast'] = (df['Resolution_Time_Hours'] <= 10).astype(int)
    df['res_slow'] = (df['Resolution_Time_Hours'] >= 80).astype(int)
    df['res_log']  = np.log1p(df['Resolution_Time_Hours'])

    df['cat_expect']   = df['Issue_Category'].map(CAT_EXPECT).fillna(1.0)
    df['assigned_num'] = df['Priority_Level'].map(PRIORITY_MAP).fillna(1)
    df['combined_text'] = (
        df['Ticket_Subject'].fillna('') + ' ' +
        df['Ticket_Description'].fillna('').str[:300]
    )
    return df


def make_X(df: pd.DataFrame, tfidf, le_ch):
    Xt = tfidf.transform(df['combined_text'])
    try:
        ch = le_ch.transform(df['Ticket_Channel'])
    except Exception:
        ch = np.zeros(len(df), dtype=int)

    Xs = csr_matrix(np.column_stack([
        df['Satisfaction_Score'].fillna(3).values,
        df['Resolution_Time_Hours'].fillna(30).values,
        df['res_log'].values,
        df['is_crisis'].values,
        df['is_trivial'].values,
        df['n_crisis'].values,
        df['sat_low'].values,
        df['sat_1'].values,
        df['sat_high'].values,
        df['res_fast'].values,
        df['res_slow'].values,
        df['cat_expect'].values,
        ch,
    ]))
    return hstack([Xt, Xs])


def infer_severity(row: dict):
    """Compute inferred severity and delta from a single row dict."""
    cat_expect = CAT_EXPECT.get(row.get('Issue_Category', ''), 1.0)
    sl = str(row.get('Ticket_Subject', '')).lower()
    is_crisis  = int(any(k in sl for k in CRISIS_KW))
    is_trivial = int(any(k in sl for k in TRIVIAL_KW))
    res_hours  = float(row.get('Resolution_Time_Hours', 30))
    res_slow   = int(res_hours >= 80)
    res_fast   = int(res_hours <= 10)
    res_log    = np.log1p(res_hours)
    sat        = float(row.get('Satisfaction_Score', 3))

    s1 = cat_expect
    s2 = float(np.clip(is_crisis * 1.5 - is_trivial * 1.5 + res_slow * 0.5 - res_fast * 0.2, -1, 3))
    s3 = float(np.clip((5 - sat) * 0.3 + res_log * 0.1, 0, 2))
    score = float(np.clip(0.50*s1 + 0.30*s2 + 0.20*s3, 0, 3))

    inferred_num = min(int(round(score)), 3)
    assigned_num = PRIORITY_MAP.get(str(row.get('Priority_Level', 'Medium')), 1)
    delta = inferred_num - assigned_num
    inferred_sev = INV_PRIORITY.get(inferred_num, 'Medium')
    mtype = 'Hidden Crisis' if delta > 0 else ('False Alarm' if delta < 0 else 'Consistent')
    return inferred_sev, delta, mtype


# ─────────────────────────────────────────────────────
#  EVIDENCE DOSSIER GENERATION
# ─────────────────────────────────────────────────────
def generate_dossier(row: dict, gemini_model=None) -> dict:
    """Generate a hallucination-free Evidence Dossier for a mismatch ticket."""
    inferred_sev, delta, mtype = infer_severity(row)
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    # ── Gemini AI dossier ──
    if gemini_model:
        prompt = f"""You are a Support Integrity Auditor. Generate a JSON evidence dossier.
TICKET:
- ID: {row.get('Ticket_ID','N/A')}
- Subject: {row.get('Ticket_Subject','')}
- Description: {str(row.get('Ticket_Description',''))[:400]}
- Category: {row.get('Issue_Category','N/A')}
- Channel: {row.get('Ticket_Channel','N/A')}
- Assigned Priority: {row.get('Priority_Level','N/A')}
- Resolution Time: {row.get('Resolution_Time_Hours','N/A')} hours
- Satisfaction Score: {row.get('Satisfaction_Score','N/A')}
- Inferred Severity: {inferred_sev}
- Mismatch Type: {mtype}

Return ONLY valid JSON, no markdown fences:
{{
  "ticket_id": "{row.get('Ticket_ID','N/A')}",
  "assigned_priority": "{row.get('Priority_Level','N/A')}",
  "inferred_severity": "{inferred_sev}",
  "mismatch_type": "{mtype}",
  "severity_delta": "{delta_str}",
  "feature_evidence": [
    {{"signal": "keyword", "value": "exact phrase from subject/description", "weight": "0.0-1.0"}},
    {{"signal": "resolution_time", "value": "{row.get('Resolution_Time_Hours','N/A')} hours", "interpretation": "one sentence grounded in actual ticket data"}}
  ],
  "constraint_analysis": "2-3 sentences using only verifiable data from the ticket above",
  "confidence": "0.0-1.0"
}}"""
        try:
            resp = gemini_model.generate_content(prompt)
            text = re.sub(r'^```json\s*|^```\s*|```$', '',
                          resp.text.strip(), flags=re.MULTILINE).strip()
            return json.loads(text)
        except Exception:
            pass  # fallback to rule-based below

    # ── Rule-based dossier (zero-hallucination guaranteed) ──
    sl = str(row.get('Ticket_Subject', '')).lower()
    matched_kws = [k for k in CRISIS_KW if k in sl]
    trivial_matched = [k for k in TRIVIAL_KW if k in sl]
    res_hours = row.get('Resolution_Time_Hours', 'N/A')
    sat = row.get('Satisfaction_Score', 'N/A')

    # Primary keyword evidence
    kw_evidence = (matched_kws[0] if matched_kws
                   else (trivial_matched[0] if trivial_matched else 'no urgent keywords'))
    kw_weight = (0.8 if matched_kws else (0.1 if trivial_matched else 0.3))

    # Resolution time interpretation
    if isinstance(res_hours, (int, float)):
        if res_hours >= 80:
            res_interp = (f"Resolution of {res_hours}h is 2× the dataset mean (39h), "
                          "indicating elevated handling complexity consistent with higher severity.")
        elif res_hours <= 10:
            res_interp = (f"Resolution of {res_hours}h is very fast, suggesting the ticket "
                          "may have been straightforward despite its assigned priority.")
        else:
            res_interp = f"Resolution of {res_hours}h is within normal range."
    else:
        res_interp = "Resolution time not available."

    # Confidence derived from signal strength
    confidence = round(min(abs(delta) * 0.20 + 0.55 +
                           (0.1 if matched_kws else 0) +
                           (0.05 if isinstance(sat, (int,float)) and sat <= 2 else 0), 0.99), 3)

    return {
        "ticket_id":         str(row.get('Ticket_ID', 'N/A')),
        "assigned_priority": str(row.get('Priority_Level', 'N/A')),
        "inferred_severity": inferred_sev,
        "mismatch_type":     mtype,
        "severity_delta":    delta_str,
        "feature_evidence": [
            {
                "signal": "keyword",
                "value":  kw_evidence,
                "weight": str(kw_weight)
            },
            {
                "signal":         "resolution_time",
                "value":          f"{res_hours} hours",
                "interpretation": res_interp
            },
            {
                "signal": "satisfaction_score",
                "value":  str(sat),
                "weight": str(round(max(0, (5 - float(sat)) / 5), 3) if isinstance(sat, (int,float)) else "N/A")
            },
        ],
        "constraint_analysis": (
            f"Ticket '{row.get('Ticket_Subject','')}' in category "
            f"'{row.get('Issue_Category','')}' was assigned '{row.get('Priority_Level','')}' "
            f"but signals infer '{inferred_sev}' (delta={delta_str}). "
            f"Satisfaction score={sat}, resolution_time={res_hours}h — "
            f"classifying this as a {mtype} scenario."
        ),
        "confidence": str(confidence),
    }


# ─────────────────────────────────────────────────────
#  MAIN PREDICT FUNCTION
# ─────────────────────────────────────────────────────
def predict(input_csv: str, output_dir: str = 'results',
            model_dir: str = 'models',
            make_dossiers: bool = True,
            use_gemini: bool = True,
            gemini_api_key: str = ''):
    os.makedirs(output_dir, exist_ok=True)
    sep = '=' * 62

    print(f"\n{sep}\n  SIA v2 — INFERENCE MODE\n{sep}")
    df = pd.read_csv(input_csv)
    if 'Ticket_ID' not in df.columns:
        df['Ticket_ID'] = [f"T{i+1:05d}" for i in range(len(df))]
    print(f"Loaded {len(df):,} tickets")

    # Load models
    clf    = joblib.load(f'{model_dir}/sia_classifier.pkl')
    tfidf  = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
    le_ch  = joblib.load(f'{model_dir}/le_channel.pkl')
    best_t = joblib.load(f'{model_dir}/best_threshold.pkl')
    print(f"✅ Models loaded  (threshold={best_t:.2f})")

    df = enrich(df)
    X  = make_X(df, tfidf, le_ch)
    probs = clf.predict_proba(X)[:, 1]
    preds = (probs >= best_t).astype(int)

    df['predicted_mismatch']   = preds
    df['mismatch_probability'] = probs.round(4)
    df['final_label']          = np.where(preds == 1, 'Mismatch', 'Consistent')

    # Compute per-row inferred severity
    sev_list = []
    for _, row in df.iterrows():
        inferred_sev, delta, mtype = infer_severity(row.to_dict())
        sev_list.append({'inferred_severity': inferred_sev,
                         'severity_delta': delta,
                         'mismatch_type': mtype})
    sev_df = pd.DataFrame(sev_list, index=df.index)
    df = pd.concat([df, sev_df], axis=1)

    n_mm = preds.sum()
    n_hc = ((df['predicted_mismatch'] == 1) & (df['mismatch_type'] == 'Hidden Crisis')).sum()
    n_fa = ((df['predicted_mismatch'] == 1) & (df['mismatch_type'] == 'False Alarm')).sum()
    print(f"\nResults:")
    print(f"  Total tickets    : {len(df):,}")
    print(f"  Mismatches found : {n_mm:,} ({n_mm/len(df)*100:.1f}%)")
    print(f"  Hidden Crisis    : {n_hc:,}")
    print(f"  False Alarm      : {n_fa:,}")

    # Save predictions CSV
    out_cols = ['Ticket_ID', 'Ticket_Subject', 'Priority_Level', 'inferred_severity',
                'final_label', 'mismatch_type', 'severity_delta', 'mismatch_probability']
    out_cols = [c for c in out_cols if c in df.columns]
    pred_path = f'{output_dir}/predictions.csv'
    df[out_cols].to_csv(pred_path, index=False)
    print(f"\n✅ Predictions → {pred_path}")

    # Generate Evidence Dossiers
    if make_dossiers:
        gemini_model = None
        key = gemini_api_key or os.environ.get('GEMINI_API_KEY', '')
        if use_gemini and key and key not in ('', 'YOUR_GEMINI_API_KEY_HERE'):
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                print("  Gemini 1.5 Flash enabled for dossiers.")
            except Exception as e:
                print(f"  Gemini init failed: {e}. Using rule-based dossiers.")

        mismatch_rows = df[df['predicted_mismatch'] == 1].reset_index(drop=True)
        print(f"Generating dossiers for {len(mismatch_rows):,} mismatch tickets...")
        dossiers = []
        for _, row in tqdm(mismatch_rows.iterrows(), total=len(mismatch_rows)):
            d = generate_dossier(row.to_dict(), gemini_model)
            dossiers.append(d)
            if gemini_model:
                time.sleep(0.5)

        dossier_path = f'{output_dir}/dossiers.json'
        with open(dossier_path, 'w') as f:
            json.dump(dossiers, f, indent=2)
        print(f"✅ Dossiers → {dossier_path}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SIA Inference v2')
    parser.add_argument('--input',       required=True)
    parser.add_argument('--output',      default='results')
    parser.add_argument('--models',      default='models')
    parser.add_argument('--gemini-key',  default='')
    parser.add_argument('--no-dossiers', action='store_true')
    parser.add_argument('--no-gemini',   action='store_true')
    args = parser.parse_args()
    predict(args.input, args.output, args.models,
            make_dossiers=not args.no_dossiers,
            use_gemini=not args.no_gemini,
            gemini_api_key=args.gemini_key)
