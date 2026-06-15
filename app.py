"""
app.py — SIA Streamlit Web App v2
Support Integrity Auditor — Priority Mismatch Dashboard

Run: streamlit run app.py
"""

import os, json, re, time, warnings
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from scipy.sparse import hstack, csr_matrix
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIA — Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    text-align: center;
}
.hero-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; }
.hero-header p  { color: #a8b4c8; font-size: 0.95rem; margin: 0.5rem 0 0; }

.kpi-card {
    background: #1a1f2e; border-radius: 12px; padding: 1.2rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.07); margin-bottom: 0.5rem;
}
.kpi-card .label { color: #8892a4; font-size: 0.78rem; text-transform: uppercase;
                   letter-spacing: 0.08em; margin-bottom: 0.3rem; }
.kpi-card .value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; line-height: 1; }
.kpi-card .sub   { color: #6b7a8d; font-size: 0.78rem; margin-top: 0.2rem; }

.badge-mismatch { background: #e74c3c; color: #fff; padding: 3px 10px;
                  border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-consistent{ background: #27ae60; color: #fff; padding: 3px 10px;
                   border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-crisis   { background: #c0392b; color: #fff; padding: 3px 10px;
                  border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.badge-alarm    { background: #e67e22; color: #fff; padding: 3px 10px;
                  border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

.dossier-box {
    background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
    padding: 1.2rem; font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem; color: #c9d1d9; overflow-x: auto;
}

.metric-pass { color: #27ae60; font-weight: 600; }
.metric-fail { color: #e74c3c; font-weight: 600; }

.stTabs [data-baseweb="tab-list"] {
    background: #0e1117; border-radius: 8px; padding: 4px;
}
.stTabs [data-baseweb="tab"] { color: #8892a4; }
.stTabs [aria-selected="true"] { color: #e2e8f0; background: #1a1f2e; border-radius: 6px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
#  CONSTANTS  (must match train_pipeline.py)
# ─────────────────────────────────────────────────────
PRIORITY_MAP = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
INV_PRIORITY = {v: k for k, v in PRIORITY_MAP.items()}
PRIORITY_ORDER = ['Low', 'Medium', 'High', 'Critical']

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
#  MODEL LOADING
# ─────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        clf    = joblib.load('models/sia_classifier.pkl')
        tfidf  = joblib.load('models/tfidf_vectorizer.pkl')
        le_ch  = joblib.load('models/le_channel.pkl')
        best_t = joblib.load('models/best_threshold.pkl')
        return clf, tfidf, le_ch, best_t
    except Exception as e:
        st.error(f"❌ Model files not found. Run `python train_pipeline.py` first.\n\n`{e}`")
        return None, None, None, None


# ─────────────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────────────
def enrich_single(d: dict) -> dict:
    sl = str(d.get('Ticket_Subject', '')).lower()
    d['is_crisis']  = int(any(k in sl for k in CRISIS_KW))
    d['is_trivial'] = int(any(k in sl for k in TRIVIAL_KW))
    d['n_crisis']   = sum(1 for k in CRISIS_KW if k in sl)
    sat = float(d.get('Satisfaction_Score', 3))
    res = float(d.get('Resolution_Time_Hours', 30))
    d['sat_low']  = int(sat <= 2)
    d['sat_1']    = int(sat == 1)
    d['sat_high'] = int(sat >= 4)
    d['res_fast'] = int(res <= 10)
    d['res_slow'] = int(res >= 80)
    d['res_log']  = float(np.log1p(res))
    d['cat_expect'] = CAT_EXPECT.get(d.get('Issue_Category', ''), 1.0)
    d['assigned_num'] = PRIORITY_MAP.get(d.get('Priority_Level', 'Medium'), 1)
    d['combined_text'] = (str(d.get('Ticket_Subject', '')) + ' ' +
                          str(d.get('Ticket_Description', ''))[:300])
    return d


def infer_severity(d: dict):
    s1 = d['cat_expect']
    s2 = float(np.clip(d['is_crisis']*1.5 - d['is_trivial']*1.5 +
                        d['res_slow']*0.5 - d['res_fast']*0.2, -1, 3))
    s3 = float(np.clip((5 - float(d.get('Satisfaction_Score',3)))*0.3 +
                        d['res_log']*0.1, 0, 2))
    score = float(np.clip(0.50*s1 + 0.30*s2 + 0.20*s3, 0, 3))
    inferred_num = min(int(round(score)), 3)
    delta = inferred_num - d['assigned_num']
    inferred_sev = INV_PRIORITY.get(inferred_num, 'Medium')
    mtype = 'Hidden Crisis' if delta > 0 else ('False Alarm' if delta < 0 else 'Consistent')
    return inferred_sev, delta, mtype, score, s1, s2, s3


def make_X_single(d: dict, tfidf, le_ch):
    Xt = tfidf.transform([d['combined_text']])
    try:
        ch = le_ch.transform([d.get('Ticket_Channel', 'Chat')])[0]
    except Exception:
        ch = 0
    Xs = csr_matrix([[
        d.get('Satisfaction_Score', 3),
        d.get('Resolution_Time_Hours', 30),
        d['res_log'], d['is_crisis'], d['is_trivial'], d['n_crisis'],
        d['sat_low'], d['sat_1'], d['sat_high'],
        d['res_fast'], d['res_slow'], d['cat_expect'], ch,
    ]])
    return hstack([Xt, Xs])


# ─────────────────────────────────────────────────────
#  DOSSIER GENERATION
# ─────────────────────────────────────────────────────
def generate_dossier_rule(d: dict, inferred_sev: str, delta: int, mtype: str) -> dict:
    sl = d['combined_text'].lower()
    kws = [k for k in CRISIS_KW if k in sl]
    triv = [k for k in TRIVIAL_KW if k in sl]
    res = d.get('Resolution_Time_Hours', 'N/A')
    sat = d.get('Satisfaction_Score', 'N/A')
    kw_val = kws[0] if kws else (triv[0] if triv else 'no critical keyword detected')
    kw_w = 0.8 if kws else (0.1 if triv else 0.3)
    delta_str = f"+{delta}" if delta > 0 else str(delta)

    if isinstance(res, (int, float)):
        if res >= 80:
            ri = f"Resolution of {res}h is 2× the 39h dataset mean, indicating high complexity."
        elif res <= 10:
            ri = f"Fast resolution of {res}h suggests the ticket was straightforward."
        else:
            ri = f"Resolution of {res}h is within normal range."
    else:
        ri = "Resolution time data unavailable."

    confidence = round(min(abs(delta)*0.20 + 0.55 +
                           (0.1 if kws else 0) +
                           (0.05 if isinstance(sat,(int,float)) and sat <= 2 else 0), 0.99), 3)
    return {
        "ticket_id":         str(d.get('Ticket_ID', 'T-MANUAL')),
        "assigned_priority": str(d.get('Priority_Level', 'N/A')),
        "inferred_severity": inferred_sev,
        "mismatch_type":     mtype,
        "severity_delta":    delta_str,
        "feature_evidence": [
            {"signal": "keyword",          "value": kw_val, "weight": str(kw_w)},
            {"signal": "resolution_time",  "value": f"{res} hours", "interpretation": ri},
            {"signal": "satisfaction",     "value": str(sat),
             "weight": str(round(max(0,(5-float(sat))/5),3) if isinstance(sat,(int,float)) else "N/A")},
        ],
        "constraint_analysis": (
            f"Ticket '{d.get('Ticket_Subject','')}' in category "
            f"'{d.get('Issue_Category','')}' was assigned '{d.get('Priority_Level','')}' "
            f"but signals infer '{inferred_sev}' (Δ={delta_str}). "
            f"Satisfaction={sat}, resolution={res}h — classified as {mtype}."
        ),
        "confidence": str(confidence),
    }


def generate_dossier_gemini(d: dict, inferred_sev: str, delta: int,
                            mtype: str, gemini_model) -> dict:
    delta_str = f"+{delta}" if delta > 0 else str(delta)
    prompt = f"""You are a Support Integrity Auditor. Generate a JSON evidence dossier.
TICKET:
- ID: {d.get('Ticket_ID','T-MANUAL')}
- Subject: {d.get('Ticket_Subject','')}
- Description: {str(d.get('Ticket_Description',''))[:400]}
- Category: {d.get('Issue_Category','N/A')}
- Channel: {d.get('Ticket_Channel','N/A')}
- Assigned Priority: {d.get('Priority_Level','N/A')}
- Resolution Time: {d.get('Resolution_Time_Hours','N/A')} hours
- Satisfaction Score: {d.get('Satisfaction_Score','N/A')}
- Inferred Severity: {inferred_sev}
- Mismatch Type: {mtype}

Return ONLY valid JSON (no markdown):
{{
  "ticket_id": "{d.get('Ticket_ID','T-MANUAL')}",
  "assigned_priority": "{d.get('Priority_Level','N/A')}",
  "inferred_severity": "{inferred_sev}",
  "mismatch_type": "{mtype}",
  "severity_delta": "{delta_str}",
  "feature_evidence": [
    {{"signal": "keyword", "value": "exact phrase from subject/description", "weight": "0.0-1.0"}},
    {{"signal": "resolution_time", "value": "{d.get('Resolution_Time_Hours','N/A')} hours", "interpretation": "one sentence grounded in the ticket data"}}
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
        return generate_dossier_rule(d, inferred_sev, delta, mtype)


# ─────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 SIA — Support Integrity Auditor")
    st.caption("AI-powered priority mismatch detection for CRM tickets")
    st.divider()
    gemini_key = st.text_input("🔑 Gemini API Key", type="password",
                               placeholder="Optional — for AI dossiers",
                               help="Get a free key at aistudio.google.com")
    st.divider()
    st.markdown("**Quick Start**")
    st.code("python train_pipeline.py\nstreamlit run app.py", language="bash")
    st.divider()

    # Show metrics if available
    if os.path.exists('output/metrics.json'):
        with open('output/metrics.json') as f:
            m = json.load(f)
        st.markdown("**Model Metrics**")
        def _m(label, val, threshold):
            icon = "✅" if val >= threshold else "❌"
            st.markdown(f"{icon} **{label}**: `{val:.4f}`")
        _m("Accuracy", m.get('accuracy',0), 0.83)
        _m("Macro F1", m.get('macro_f1',0), 0.82)
        _m("Consistent Recall", m.get('recall_consistent',0), 0.78)
        _m("Mismatch Recall",   m.get('recall_mismatch',0),   0.78)


# ─────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
  <h1>🔍 Support Integrity Auditor</h1>
  <p>Semantics-driven, evidence-grounded priority mismatch detection for CRM support tickets</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["  🎫 Single Ticket Audit  ",
                             "  📦 Batch CSV Audit  ",
                             "  📊 Dashboard  "])


# ═════════════════════════════════════════════════════
#  TAB 1 — SINGLE TICKET
# ═════════════════════════════════════════════════════
with tab1:
    st.subheader("Single Ticket Audit")
    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        ticket_id = c1.text_input("Ticket ID", value="T-00001")
        category  = c2.selectbox("Issue Category",
                                  ['Technical','Billing','Account','General Inquiry','Fraud'])
        subject = st.text_input("Ticket Subject",
                                 value="Login failed - Cannot access production account")
        desc    = st.text_area("Ticket Description",
                                value="The application crashes every time I try to log in. This is affecting all 50 users in our team. Production is down and we are losing revenue.",
                                height=100)
        c3, c4, c5, c6 = st.columns(4)
        priority      = c3.selectbox("Assigned Priority", PRIORITY_ORDER, index=1)
        channel       = c4.selectbox("Channel", ['Chat','Email','Web Form'])
        resolution_h  = c5.number_input("Resolution Time (hours)", 1, 120, 40)
        satisfaction  = c6.slider("Satisfaction Score", 1, 5, 3)
        submitted = st.form_submit_button("🚀 Audit Ticket", type="primary", use_container_width=True)

    if submitted:
        clf, tfidf, le_ch, best_t = load_models()
        if clf is None:
            st.stop()

        ticket = {
            'Ticket_ID': ticket_id, 'Ticket_Subject': subject,
            'Ticket_Description': desc, 'Issue_Category': category,
            'Priority_Level': priority, 'Ticket_Channel': channel,
            'Resolution_Time_Hours': resolution_h, 'Satisfaction_Score': satisfaction,
        }
        ticket = enrich_single(ticket)
        X_s    = make_X_single(ticket, tfidf, le_ch)
        prob   = clf.predict_proba(X_s)[0][1]
        pred   = int(prob >= best_t)
        inferred_sev, delta, mtype, score, s1, s2, s3 = infer_severity(ticket)

        # ── Result banner ──
        if pred == 1:
            color = "#e74c3c" if mtype == 'Hidden Crisis' else "#e67e22"
            icon  = "🚨" if mtype == 'Hidden Crisis' else "⚠️"
            st.markdown(f"""
            <div style="background:{color}22;border-left:4px solid {color};
                        border-radius:8px;padding:1rem 1.2rem;margin:1rem 0;">
              <strong style="color:{color};font-size:1.1rem;">{icon} MISMATCH DETECTED — {mtype}</strong><br>
              <span style="color:#ccc;">Assigned: <b>{priority}</b> → Inferred: <b>{inferred_sev}</b>
              &nbsp;|&nbsp; Δ={'+'+str(delta) if delta>0 else str(delta)}
              &nbsp;|&nbsp; Confidence: {prob:.1%}</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#27ae6022;border-left:4px solid #27ae60;
                        border-radius:8px;padding:1rem 1.2rem;margin:1rem 0;">
              <strong style="color:#27ae60;font-size:1.1rem;">✅ CONSISTENT — Assigned priority aligns with inferred severity</strong><br>
              <span style="color:#ccc;">Assigned: <b>{priority}</b> = Inferred: <b>{inferred_sev}</b>
              &nbsp;|&nbsp; Confidence: {1-prob:.1%}</span>
            </div>""", unsafe_allow_html=True)

        # ── Signal breakdown ──
        st.markdown("**Signal Breakdown**")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Mismatch Prob.", f"{prob:.1%}")
        sc2.metric("Category Signal", f"{s1:.2f}", help="Domain-based expected severity")
        sc3.metric("Urgency Signal",  f"{s2:.2f}", help="Keyword + resolution time")
        sc4.metric("Satisfaction Signal", f"{s3:.2f}", help="Customer satisfaction proxy")

        fig = go.Figure(go.Bar(
            x=['Category (w=0.50)', 'Urgency+Time (w=0.30)', 'Satisfaction (w=0.20)'],
            y=[s1, max(s2, 0), s3],
            marker_color=['#6c63ff', '#3498db', '#2ecc71'],
            text=[f"{v:.2f}" for v in [s1, max(s2,0), s3]],
            textposition='outside',
        ))
        fig.update_layout(
            title="Signal Contributions to Inferred Severity",
            height=280, yaxis_range=[0, 3.5],
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e2e8f0',
            yaxis_title="Signal Score (0–3)",
        )
        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            # ── Dossier ──
            st.markdown("**📄 Evidence Dossier**")
            gemini_model = None
            if gemini_key:
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                except Exception:
                    pass

            with st.spinner("Generating evidence dossier..."):
                if gemini_model:
                    dossier = generate_dossier_gemini(ticket, inferred_sev, delta, mtype, gemini_model)
                else:
                    dossier = generate_dossier_rule(ticket, inferred_sev, delta, mtype)

            st.markdown(f'<div class="dossier-box"><pre>{json.dumps(dossier, indent=2)}</pre></div>',
                        unsafe_allow_html=True)
            st.download_button("⬇️ Download Dossier JSON",
                               json.dumps(dossier, indent=2),
                               f"dossier_{ticket_id}.json", "application/json")


# ═════════════════════════════════════════════════════
#  TAB 2 — BATCH CSV
# ═════════════════════════════════════════════════════
with tab2:
    st.subheader("Batch Ticket Audit via CSV Upload")
    st.info(
        "Required columns: `Ticket_Subject`, `Ticket_Description`, `Priority_Level`, "
        "`Issue_Category`, `Ticket_Channel`, `Resolution_Time_Hours`, `Satisfaction_Score`"
    )

    uploaded = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded:
        df_batch = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(df_batch):,}** tickets")
        st.dataframe(df_batch.head(5), use_container_width=True)

        clf, tfidf, le_ch, best_t = load_models()
        if clf and st.button("🚀 Run Batch Audit", type="primary"):
            results = []
            progress = st.progress(0)
            status   = st.empty()

            for i, row in df_batch.iterrows():
                d = row.to_dict()
                if 'Ticket_ID' not in d:
                    d['Ticket_ID'] = f"T{i+1:05d}"
                d = enrich_single(d)
                X_s  = make_X_single(d, tfidf, le_ch)
                prob = clf.predict_proba(X_s)[0][1]
                pred = int(prob >= best_t)
                inferred_sev, delta, mtype, *_ = infer_severity(d)
                results.append({
                    'Ticket_ID':        d['Ticket_ID'],
                    'Subject':          d.get('Ticket_Subject',''),
                    'Assigned_Priority':d.get('Priority_Level',''),
                    'Inferred_Severity':inferred_sev,
                    'Prediction':       'Mismatch' if pred else 'Consistent',
                    'Mismatch_Type':    mtype if pred else '—',
                    'Severity_Delta':   (f"+{delta}" if delta>0 else str(delta)) if pred else '0',
                    'Confidence':       round(prob, 4),
                })
                progress.progress((i + 1) / len(df_batch))
                if i % 100 == 0:
                    status.caption(f"Processing ticket {i+1}/{len(df_batch)}...")

            status.empty()
            res_df = pd.DataFrame(results)
            n_mm = (res_df['Prediction'] == 'Mismatch').sum()
            st.success(f"✅ Audit complete — **{n_mm:,}** mismatches found out of **{len(res_df):,}** tickets.")

            # Summary cards
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Mismatches", n_mm)
            c2.metric("Hidden Crisis", (res_df['Mismatch_Type'] == 'Hidden Crisis').sum())
            c3.metric("False Alarm",   (res_df['Mismatch_Type'] == 'False Alarm').sum())

            st.dataframe(res_df, use_container_width=True)
            st.download_button("⬇️ Download Results CSV",
                               res_df.to_csv(index=False),
                               "sia_results.csv", "text/csv")


# ═════════════════════════════════════════════════════
#  TAB 3 — DASHBOARD
# ═════════════════════════════════════════════════════
with tab3:
    st.subheader("Priority Mismatch Dashboard")

    try:
        df_proc = pd.read_csv('data/processed_tickets.csv')
        st.success(f"Loaded processed dataset — **{len(df_proc):,}** tickets")

        # ── KPIs ──
        total    = len(df_proc)
        n_mm     = int(df_proc['is_mismatch'].sum())
        n_hc     = int(((df_proc['is_mismatch']==1)&(df_proc['mismatch_type']=='Hidden Crisis')).sum())
        n_fa     = int(((df_proc['is_mismatch']==1)&(df_proc['mismatch_type']=='False Alarm')).sum())
        n_ok     = int((df_proc['is_mismatch']==0).sum())
        mm_rate  = n_mm / total

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Mismatches",  f"{n_mm:,}",  f"{mm_rate:.1%} of tickets")
        k2.metric("🚨 Hidden Crisis",  f"{n_hc:,}",  "Under-prioritized")
        k3.metric("⚠️ False Alarm",    f"{n_fa:,}",  "Over-prioritized")
        k4.metric("✅ Consistent",      f"{n_ok:,}",  "Correctly assigned")

        st.divider()

        # ── Row 1: Mismatch type pie + severity delta heatmap ──
        c1, c2 = st.columns(2)
        with c1:
            mdf = df_proc[df_proc['is_mismatch']==1]
            fig1 = px.pie(
                mdf, names='mismatch_type',
                title='Mismatch Type Distribution',
                color_discrete_map={'Hidden Crisis': '#e74c3c', 'False Alarm': '#e67e22'},
                hole=0.42,
            )
            fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            if 'Issue_Category' in df_proc.columns and 'Ticket_Channel' in df_proc.columns:
                pivot = (df_proc.groupby(['Issue_Category','Ticket_Channel'])
                         ['severity_delta'].mean().unstack().fillna(0))
                fig2 = px.imshow(
                    pivot, title='Avg Severity Delta Heatmap (Category × Channel)',
                    color_continuous_scale='RdYlGn_r', text_auto='.2f', aspect='auto',
                    zmin=-2, zmax=2,
                )
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
                st.plotly_chart(fig2, use_container_width=True)

        # ── Row 2: Assigned vs Inferred + mismatch rate by category ──
        c3, c4 = st.columns(2)
        with c3:
            fig3 = px.histogram(
                df_proc, x='Priority_Level', color='inferred_severity',
                barmode='group', title='Assigned vs Inferred Priority Distribution',
                category_orders={
                    'Priority_Level': PRIORITY_ORDER,
                    'inferred_severity': PRIORITY_ORDER,
                },
                color_discrete_map={
                    'Low':'#3498db','Medium':'#2ecc71','High':'#e67e22','Critical':'#e74c3c'
                },
            )
            fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0')
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            cat_mm = (df_proc.groupby('Issue_Category')['is_mismatch']
                      .mean().reset_index()
                      .rename(columns={'is_mismatch':'Mismatch Rate'})
                      .sort_values('Mismatch Rate', ascending=True))
            fig4 = px.bar(
                cat_mm, x='Mismatch Rate', y='Issue_Category', orientation='h',
                title='Mismatch Rate by Category',
                color='Mismatch Rate', color_continuous_scale='Oranges',
                text=cat_mm['Mismatch Rate'].apply(lambda v: f"{v:.1%}"),
            )
            fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e2e8f0',
                               coloraxis_showscale=False)
            st.plotly_chart(fig4, use_container_width=True)

        # ── Top Mismatch Tickets ──
        st.subheader("🚨 Top Mismatch Tickets")
        top_cols = ['Ticket_ID','Ticket_Subject','Priority_Level','inferred_severity',
                    'mismatch_type','severity_delta']
        top_cols = [c for c in top_cols if c in df_proc.columns]
        top_mm = df_proc[df_proc['is_mismatch']==1].sort_values(
            'severity_delta', key=abs, ascending=False).head(15)
        st.dataframe(top_mm[top_cols].reset_index(drop=True), use_container_width=True)

        # ── Verification metrics ──
        if os.path.exists('output/metrics.json'):
            st.divider()
            st.subheader("📊 Verification Metrics")
            with open('output/metrics.json') as f:
                m = json.load(f)

            m1, m2, m3, m4 = st.columns(4)

            def _metric_card(col, label, val, threshold, suffix=''):
                icon = "✅" if val >= threshold else "❌"
                col.metric(f"{icon} {label}", f"{val:.4f}{suffix}",
                           f"≥ {threshold} required")

            _metric_card(m1, "Accuracy",   m.get('accuracy',0),          0.83)
            _metric_card(m2, "Macro F1",   m.get('macro_f1',0),          0.82)
            _metric_card(m3, "Consistent Recall", m.get('recall_consistent',0), 0.78)
            _metric_card(m4, "Mismatch Recall",   m.get('recall_mismatch',0),   0.78)

            if m.get('passes_verification'):
                st.success("✅ ALL VERIFICATION THRESHOLDS MET — Submission Valid!")
            else:
                st.error("❌ One or more thresholds not met. Retrain or tune the model.")

            # Ablation table
            if 'ablation' in m:
                st.subheader("Signal Ablation Table")
                abl = m['ablation']
                abl_df = pd.DataFrame([
                    {'Signal': 'Category (Sig1)', 'Mismatch Rate': abl.get('sig1_category_mismatch_rate','N/A')},
                    {'Signal': 'Urgency+Time (Sig2)', 'Mismatch Rate': abl.get('sig2_urgency_time_mismatch_rate','N/A')},
                    {'Signal': 'Satisfaction (Sig3)', 'Mismatch Rate': abl.get('sig3_satisfaction_mismatch_rate','N/A')},
                    {'Signal': 'Sig1↔Sig2 Agreement', 'Mismatch Rate': abl.get('sig1_vs_sig2_agreement','N/A')},
                    {'Signal': 'Sig1↔Sig3 Agreement', 'Mismatch Rate': abl.get('sig1_vs_sig3_agreement','N/A')},
                    {'Signal': 'Sig2↔Sig3 Agreement', 'Mismatch Rate': abl.get('sig2_vs_sig3_agreement','N/A')},
                ])
                st.dataframe(abl_df, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.warning(
            "No processed data found. Run `python train_pipeline.py` first, then refresh.\n\n"
            "```bash\npython train_pipeline.py --data data/enhanced_customer_support_data.csv\n```"
        )
