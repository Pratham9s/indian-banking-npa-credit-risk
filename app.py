"""
Indian Banking NPA Credit Risk Dashboard
Author: Pratham | Krea University MBA
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Banking NPA Credit Risk",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f9f9f9; }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 18px 22px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card h2 { margin: 0; font-size: 2rem; color: #c0392b; }
    .metric-card p  { margin: 4px 0 0; font-size: 0.85rem; color: #555; }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        border-left: 4px solid #c0392b;
        padding-left: 10px;
        margin: 24px 0 12px;
    }
    .ecl-box {
        background: #fff8f0;
        border: 1px solid #f0a500;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
    }
    .ai-response {
        background: #f0f4ff;
        border-left: 4px solid #2980b9;
        border-radius: 6px;
        padding: 16px 20px;
        margin-top: 12px;
        line-height: 1.7;
    }
    .risk-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_npa():
    base = os.path.dirname(__file__)
    # Try root first, then subfolders
    for path in [
        os.path.join(base, "npa_cleaned.csv"),
        os.path.join(base, "data", "processed", "npa_cleaned.csv"),
    ]:
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError("npa_cleaned.csv not found")

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    # Try root first, then models subfolder
    for folder in [base, os.path.join(base, "models")]:
        xgb_path = os.path.join(folder, "xgb_model.pkl")
        if os.path.exists(xgb_path):
            xgb    = joblib.load(xgb_path)
            scaler = joblib.load(os.path.join(folder, "scaler.pkl"))
            feat   = joblib.load(os.path.join(folder, "feature_cols.pkl"))
            return xgb, scaler, feat
    raise FileNotFoundError("Model files not found")

try:
    npa = load_npa()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.sidebar.error(f"NPA data not found: {e}")

try:
    xgb_model, scaler, feature_cols = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False
    st.sidebar.warning(f"Model files not found: {e}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/RBI_logo.svg/200px-RBI_logo.svg.png", width=80)
st.sidebar.title("🏦 NPA Credit Risk")
st.sidebar.caption("Indian Banking Sector Analysis")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Macro Overview", "🤖 AI Event Explainer", "🎯 Credit Scorecard", "📋 ECL Framework"]
)

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_KEY:
    GEMINI_KEY = st.sidebar.text_input("Gemini API Key", type="password", placeholder="Paste your key here")

st.sidebar.markdown("---")
st.sidebar.markdown("**Project:** Indian Banking NPA\n\n**Author:** Pratham | Krea University MBA\n\n**Stack:** Python · XGBoost · SHAP · Streamlit")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — MACRO OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Macro Overview":
    st.title("Indian Banking Sector — NPA Macro Analysis")
    st.caption("Source: RBI Handbook of Statistics on Indian Economy 2024-25 | Coverage: FY1997–FY2025")

    if not DATA_OK:
        st.error("Place `npa_cleaned.csv` in `data/processed/` and restart.")
        st.stop()

    all_scb = npa[npa['bank_group'] == 'All SCBs'].dropna(subset=['gnpa_pct_advances'])
    psu     = npa[npa['bank_group'] == 'Public Sector'].dropna(subset=['gnpa_pct_advances'])
    pvt     = npa[npa['bank_group'] == 'Private Sector'].dropna(subset=['gnpa_pct_advances'])
    fgn     = npa[npa['bank_group'] == 'Foreign Banks'].dropna(subset=['gnpa_pct_advances'])

    # ── KPI cards ──
    col1, col2, col3, col4 = st.columns(4)
    peak_row = all_scb.loc[all_scb['gnpa_pct_advances'].idxmax()]
    latest   = all_scb.iloc[-1]

    with col1:
        st.markdown('<div class="metric-card"><h2>14.6%</h2><p>PSU GNPA Peak (FY2018)</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{peak_row["gnpa_pct_advances"]:.1f}%</h2><p>All SCB GNPA Peak ({int(peak_row["year_int"])+1})</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{latest["gnpa_pct_advances"]:.1f}%</h2><p>All SCB GNPA Latest</p></div>', unsafe_allow_html=True)
    with col4:
        drop = peak_row["gnpa_pct_advances"] - latest["gnpa_pct_advances"]
        st.markdown(f'<div class="metric-card"><h2>↓{drop:.1f}pp</h2><p>GNPA Decline Since Peak</p></div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Chart 1: All SCB Trend ──
    st.markdown('<div class="section-header">Gross NPA % — All Scheduled Commercial Banks (1997–2025)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f8f8')
    ax.plot(all_scb['year_int'], all_scb['gnpa_pct_advances'], color='#c0392b', lw=2.5, marker='o', ms=4, label='Gross NPA %')
    ax.plot(all_scb['year_int'], all_scb['nnpa_pct_advances'], color='#e67e22', lw=2, ls='--', marker='s', ms=3, label='Net NPA %')
    ax.axvspan(2015, 2018, alpha=0.12, color='red', label='NPA Crisis (2015–18)')
    for yr, lbl, col in [(2008,'GFC\n2008','steelblue'), (2016,'IBC\n2016','purple'), (2020,'COVID\n2020','gray')]:
        ax.axvline(yr, color=col, ls=':', alpha=0.7, lw=1.5)
        ax.text(yr+0.2, all_scb['gnpa_pct_advances'].max()*0.88, lbl, fontsize=8, color=col)
    peak = all_scb.loc[all_scb['gnpa_pct_advances'].idxmax()]
    ax.annotate(f"Peak: {peak['gnpa_pct_advances']:.1f}%",
                xy=(peak['year_int'], peak['gnpa_pct_advances']),
                xytext=(peak['year_int']-4, peak['gnpa_pct_advances']+0.8),
                arrowprops=dict(arrowstyle='->', color='black'), fontsize=9, fontweight='bold')
    ax.set_xlabel('Fiscal Year Start'); ax.set_ylabel('NPA as % of Advances')
    ax.legend(); ax.grid(alpha=0.3); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.set_xticks(all_scb['year_int'][::3]); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # ── Chart 2: PSU vs Private ──
    st.markdown('<div class="section-header">PSU vs Private vs Foreign Banks — GNPA % Comparison</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(13, 4.5))
    fig2.patch.set_facecolor('white'); ax2.set_facecolor('#f8f8f8')
    for df_g, lbl, col in [(psu,'Public Sector','#e74c3c'), (pvt,'Private Sector','#2980b9'), (fgn,'Foreign Banks','#27ae60')]:
        ax2.plot(df_g['year_int'], df_g['gnpa_pct_advances'], label=lbl, color=col, lw=2.2, marker='o', ms=3)
    ax2.axvspan(2015, 2018, alpha=0.1, color='red')
    ax2.axvline(2016, color='purple', ls=':', alpha=0.6, lw=1.5)
    ax2.text(2016.2, 0.5, 'IBC 2016', fontsize=8, color='purple')
    ax2.set_xlabel('Fiscal Year Start'); ax2.set_ylabel('Gross NPA %')
    ax2.legend(); ax2.grid(alpha=0.3); ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False)
    ax2.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # ── Key observations ──
    st.markdown('<div class="section-header">Key Findings</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.info("**NPA Crisis (2015–2018):** RBI's Asset Quality Review (AQR) in 2015–16 forced banks to reclassify hidden stressed assets. PSU banks bore the brunt — GNPA peaked at 14.6% in FY2018 vs 4% for private banks.")
        st.success("**Recovery (2018–2024):** IBC 2016 gave banks a legal resolution mechanism. Combined with recapitalisation of PSU banks (₹3.5 lakh crore), GNPA fell to 2.7% by FY2024 — a 17-year low.")
    with c2:
        st.warning("**COVID Anomaly (FY2021):** RBI's loan moratorium (Mar–Aug 2020) froze NPA recognition. Reported GNPA of 7.3% understates true stress — restructured assets peaked at ~5% of advances.")
        st.info("**Structural Gap:** PSU banks consistently carried 3–4x higher NPAs than private banks throughout the cycle — reflecting differences in lending discipline, governance, and borrower mix.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — AI EVENT EXPLAINER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Event Explainer":
    st.title("AI-Powered NPA Event Explainer")
    st.caption("Select a year — Gemini AI explains what drove NPA movement using macro context and its own knowledge of Indian banking history.")

    if not DATA_OK:
        st.error("NPA data not loaded. Check file path.")
        st.stop()

    all_scb = npa[npa['bank_group'] == 'All SCBs'].dropna(subset=['gnpa_pct_advances']).sort_values('year_int')
    psu_df  = npa[npa['bank_group'] == 'Public Sector'].dropna(subset=['gnpa_pct_advances']).set_index('year_int')
    pvt_df  = npa[npa['bank_group'] == 'Private Sector'].dropna(subset=['gnpa_pct_advances']).set_index('year_int')

    # Year slider
    available_years = sorted(all_scb['year_int'].unique())
    selected_year = st.select_slider(
        "Select Fiscal Year (year shown = FY start, e.g. 2017 = FY2017-18)",
        options=available_years,
        value=2017
    )

    # Show data for selected year
    row = all_scb[all_scb['year_int'] == selected_year].iloc[0]
    prev_rows = all_scb[all_scb['year_int'] < selected_year]
    prev_row  = prev_rows.iloc[-1] if not prev_rows.empty else None

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fiscal Year", f"FY{selected_year}-{str(selected_year+1)[-2:]}")
    with col2:
        delta = f"{row['gnpa_pct_advances'] - prev_row['gnpa_pct_advances']:+.2f}pp" if prev_row is not None else "N/A"
        st.metric("All SCB GNPA %", f"{row['gnpa_pct_advances']:.2f}%", delta=delta)
    with col3:
        psu_val = psu_df.loc[selected_year, 'gnpa_pct_advances'] if selected_year in psu_df.index else "N/A"
        st.metric("PSU GNPA %", f"{psu_val:.2f}%" if isinstance(psu_val, float) else psu_val)
    with col4:
        pvt_val = pvt_df.loc[selected_year, 'gnpa_pct_advances'] if selected_year in pvt_df.index else "N/A"
        st.metric("Private GNPA %", f"{pvt_val:.2f}%" if isinstance(pvt_val, float) else pvt_val)

    st.markdown("---")

    # Mini sparkline for context
    fig_s, ax_s = plt.subplots(figsize=(13, 2.5))
    fig_s.patch.set_facecolor('white'); ax_s.set_facecolor('#f8f8f8')
    ax_s.plot(all_scb['year_int'], all_scb['gnpa_pct_advances'], color='#c0392b', lw=2)
    ax_s.axvline(selected_year, color='#2980b9', lw=2.5, ls='--', label=f'Selected: FY{selected_year}')
    ax_s.fill_between(all_scb['year_int'], all_scb['gnpa_pct_advances'], alpha=0.08, color='#c0392b')
    ax_s.set_ylabel('GNPA %'); ax_s.grid(alpha=0.3)
    ax_s.spines['top'].set_visible(False); ax_s.spines['right'].set_visible(False)
    ax_s.legend(fontsize=9); ax_s.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig_s)

    st.markdown("---")

    # AI Explainer
    if not GEMINI_KEY:
        st.warning("Enter your Gemini API key in the sidebar to use the AI explainer.")
    else:
        if st.button("🤖 Explain This Year's NPA Movement", type="primary", use_container_width=True):
            # Build context
            yoy_change = row['gnpa_pct_advances'] - prev_row['gnpa_pct_advances'] if prev_row is not None else 0
            direction  = "increased" if yoy_change > 0 else "decreased"
            psu_context = f"PSU GNPA: {psu_val:.2f}%" if isinstance(psu_val, float) else ""
            pvt_context = f"Private GNPA: {pvt_val:.2f}%" if isinstance(pvt_val, float) else ""

            prompt = f"""You are a senior credit risk analyst specialising in Indian banking.

Analyse NPA movement for Indian SCBs in FY{selected_year}-{str(selected_year+1)[-2:]}.

DATA:
- All SCB GNPA: {row['gnpa_pct_advances']:.2f}%
- YoY change: {yoy_change:+.2f}pp ({direction})
- Previous year: {prev_row['gnpa_pct_advances']:.2f}%
- {psu_context}
- {pvt_context}

Respond in exactly this structure, each section 3-4 lines:

1. HEADLINE VERDICT
2. MAJOR EVENTS & SHOCKS (2-3 specific events with NPA impact)
3. RBI & GOVERNMENT POLICY ACTIONS (2-3 specific policies/circulars)
4. PSU vs PRIVATE DIVERGENCE
5. FORWARD IMPLICATION

Be specific — name actual policies, dates, sectors. No filler language."""

            with st.spinner("Gemini is analysing this year's macro context..."):
                try:
                    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
                    headers = {"Content-Type": "application/json"}
                    payload = {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 8192}
                    }
                    resp = requests.post(f"{url}?key={GEMINI_KEY}", headers=headers, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data['candidates'][0]['content']['parts'][0]['text']
                    st.markdown(f'<div class="ai-response">{answer}</div>', unsafe_allow_html=True)
                    st.caption(f"Generated by Gemini 2.0 Flash | Data: RBI Handbook of Statistics 2024-25")
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code == 429:
                        st.warning("Rate limit hit — please wait 30 seconds and try again. Gemini free tier allows ~15 requests/minute.")
                    else:
                        st.error(f"API error: {e.response.status_code} — check your API key.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**How it works:** Your actual RBI NPA data is passed to Gemini as context. The AI then draws on its knowledge of Indian banking history, RBI policy actions, global macro events, and sectoral dynamics to explain the NPA movement — no hardcoding of events.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CREDIT SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Credit Scorecard":
    st.title("Credit Risk Scorecard — Applicant PD Predictor")
    st.caption("Upload a CSV with full bureau data for accurate prediction, or use manual entry for an indicative score.")

    if not MODEL_OK:
        st.warning("""
        **Model files not found.** Place the following in a `models/` folder next to `app.py`:
        - `xgb_model.pkl`
        - `scaler.pkl`  
        - `feature_cols.pkl`
        Then restart the app.
        """)
        st.stop()

    st.markdown('<div class="section-header">Risk Band Reference</div>', unsafe_allow_html=True)
    demo_data = {
        'Score Band': ['P1 — Prime', 'P2 — Near-Prime', 'P3 — Sub-Prime', 'P4 — High Risk'],
        'PD Range': ['< 2%', '2% – 8%', '8% – 20%', '> 20%'],
        'ECL Stage': ['Stage 1', 'Stage 1', 'Stage 2', 'Stage 3'],
        'Provisioning': ['12-month ECL', '12-month ECL', 'Lifetime ECL', 'Lifetime ECL (credit-impaired)'],
        'Decision': ['✅ Approve', '✅ Approve with monitoring', '⚠️ Refer for review', '❌ Decline']
    }
    st.dataframe(pd.DataFrame(demo_data), use_container_width=True, hide_index=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["📁 CSV Upload (Full Model — Accurate)", "✏️ Manual Entry (Indicative Only)"])

    with tab1:
        st.markdown("""
        **How to use:**
        - Upload a CSV with one or more applicant rows containing all 85 bureau features
        - Use your `Unseen_Dataset.xlsx` converted to CSV as a sample input
        - Model runs on complete data → accurate PD scores and risk bands
        """)

        template_df  = pd.DataFrame(columns=feature_cols)
        template_csv = template_df.to_csv(index=False)
        st.download_button("⬇️ Download CSV Template (85 features)", template_csv, "applicant_template.csv", "text/csv")

        uploaded_file = st.file_uploader("Upload Applicant CSV", type=["csv"])

        if uploaded_file:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success(f"Loaded {len(input_df)} applicant(s) — {input_df.shape[1]} columns detected")

                enc_map = {
                    'MARITALSTATUS':   {'Married': 1, 'Single': 2, 'Divorced': 0},
                    'EDUCATION':       {'Graduate': 0, 'Post-Graduate': 1, 'SSC': 2, '12TH': 3, 'OTHERS': 4},
                    'GENDER':          {'M': 1, 'F': 0},
                    'last_prod_enq2':  {'PL': 3, 'AL': 0, 'HL': 2, 'CC': 1, 'others': 4},
                    'first_prod_enq2': {'PL': 3, 'AL': 0, 'HL': 2, 'CC': 1, 'others': 4},
                }

                proc_df = input_df.copy()
                for col, mapping in enc_map.items():
                    if col in proc_df.columns:
                        proc_df[col] = proc_df[col].map(mapping).fillna(0)

                for col in feature_cols:
                    if col not in proc_df.columns:
                        proc_df[col] = 0
                proc_df = proc_df[feature_cols].fillna(0)

                pd_scores   = xgb_model.predict_proba(proc_df)[:, 1]
                risk_scores = 1 - pd_scores

                def assign_band(rs):
                    if rs < 0.02:   return 'P1 — Prime', 'Stage 1', '✅ Approve'
                    elif rs < 0.08: return 'P2 — Near-Prime', 'Stage 1', '✅ Approve with monitoring'
                    elif rs < 0.20: return 'P3 — Sub-Prime', 'Stage 2', '⚠️ Refer for review'
                    else:           return 'P4 — High Risk', 'Stage 3', '❌ Decline'

                bands     = [assign_band(r) for r in risk_scores]
                result_df = input_df.copy()
                result_df['PD (%)']        = (risk_scores * 100).round(2)
                result_df['Creditworthy (%)'] = (pd_scores * 100).round(2)
                result_df['Risk Band']     = [b[0] for b in bands]
                result_df['ECL Stage']     = [b[1] for b in bands]
                result_df['Decision']      = [b[2] for b in bands]

                st.markdown('<div class="section-header">Prediction Results</div>', unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                approve_count = len([b for b in bands if b[0] in ['P1 — Prime', 'P2 — Near-Prime']])
                review_count  = len([b for b in bands if 'P3' in b[0]])
                decline_count = len([b for b in bands if 'P4' in b[0]])
                with c1: st.metric("Total Applicants", len(result_df))
                with c2: st.metric("✅ Approve", approve_count)
                with c3: st.metric("⚠️ Review", review_count)
                with c4: st.metric("❌ Decline", decline_count)

                display_cols = [c for c in ['AGE', 'GENDER', 'NETMONTHLYINCOME', 'Credit_Score', 'EDUCATION', 'MARITALSTATUS'] if c in result_df.columns]
                display_cols += ['PD (%)', 'Creditworthy (%)', 'Risk Band', 'ECL Stage', 'Decision']
                st.dataframe(result_df[display_cols], use_container_width=True, hide_index=True)

                band_counts = pd.Series([b[0] for b in bands]).value_counts()
                fig_b, ax_b = plt.subplots(figsize=(8, 3.5))
                fig_b.patch.set_facecolor('white')
                colors_b = {'P1 — Prime': '#27ae60', 'P2 — Near-Prime': '#2ecc71', 'P3 — Sub-Prime': '#e67e22', 'P4 — High Risk': '#e74c3c'}
                bar_colors = [colors_b.get(b, 'gray') for b in band_counts.index]
                ax_b.bar(band_counts.index, band_counts.values, color=bar_colors, edgecolor='white', width=0.5)
                for bar, val in zip(ax_b.patches, band_counts.values):
                    ax_b.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, str(val), ha='center', fontweight='bold')
                ax_b.set_ylabel('Count'); ax_b.set_title('Risk Band Distribution', fontweight='bold')
                ax_b.grid(axis='y', alpha=0.3)
                ax_b.spines['top'].set_visible(False); ax_b.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_b)

                csv_out = result_df[display_cols].to_csv(index=False)
                st.download_button("⬇️ Download Results CSV", csv_out, "risk_assessment_results.csv", "text/csv")

            except Exception as e:
                st.error(f"Error processing file: {e}")

    with tab2:
        st.info("⚠️ Manual entry uses ~18 of 85 model features. Results are indicative only — upload a full bureau CSV for accurate assessment.")

        with st.form("scorecard_form"):
            c1, c2, c3 = st.columns(3)

            with c1:
                st.markdown("**Personal Info**")
                age       = st.slider("Age", 21, 65, 35)
                income    = st.number_input("Net Monthly Income (₹)", 10000, 500000, 50000, step=5000)
                education = st.selectbox("Education", ["Graduate", "Post-Graduate", "SSC", "12TH", "OTHERS"])
                gender    = st.selectbox("Gender", ["M", "F"])
                marital   = st.selectbox("Marital Status", ["Married", "Single", "Divorced"])
                time_empr = st.slider("Years with Current Employer", 0, 30, 3)

            with c2:
                st.markdown("**Credit Behaviour**")
                credit_score = st.slider("CIBIL Credit Score", 300, 900, 720)
                missed_pmnt  = st.slider("Total Missed Payments", 0, 30, 0)
                tot_tl       = st.slider("Total Trade Lines", 0, 30, 5)
                active_tl    = st.slider("Active Trade Lines", 0, 20, 3)
                tot_tl_l6m   = st.slider("New Trade Lines (Last 6M)", 0, 10, 1)
                num_deliq    = st.slider("Times Delinquent (Ever)", 0, 20, 0)
                num_30dpd    = st.slider("Times 30+ DPD", 0, 15, 0)
                num_60dpd    = st.slider("Times 60+ DPD", 0, 10, 0)

            with c3:
                st.markdown("**Loan & Enquiry Details**")
                loan_type  = st.selectbox("Product Enquired", ["PL", "AL", "HL", "CC", "others"])
                loan_amount= st.number_input("Loan Amount Requested (₹)", 10000, 10000000, 500000, step=10000)
                cc_util    = st.slider("Credit Card Utilization %", 0, 100, 30)
                pl_util    = st.slider("Personal Loan Utilization %", 0, 100, 20)
                enq_l6m    = st.slider("Enquiries Last 6 Months", 0, 15, 2)
                enq_l12m   = st.slider("Enquiries Last 12 Months", 0, 20, 3)
                cc_flag    = st.selectbox("Has Credit Card?", [1, 0])
                pl_flag    = st.selectbox("Has Personal Loan?", [1, 0])

            submitted = st.form_submit_button("🔍 Get Indicative Risk Score", type="primary", use_container_width=True)

        if submitted:
            input_dict = {col: 0 for col in feature_cols}
            input_dict['AGE']                 = age
            input_dict['NETMONTHLYINCOME']     = income
            input_dict['Credit_Score']         = credit_score
            input_dict['Tot_Missed_Pmnt']      = missed_pmnt
            input_dict['Total_TL']             = tot_tl
            input_dict['Tot_Active_TL']        = active_tl
            input_dict['Total_TL_opened_L6M']  = tot_tl_l6m
            input_dict['num_times_delinquent'] = num_deliq
            input_dict['num_times_30p_dpd']    = num_30dpd
            input_dict['num_times_60p_dpd']    = num_60dpd
            input_dict['CC_utilization']       = cc_util
            input_dict['PL_utilization']       = pl_util
            input_dict['enq_L6m']              = enq_l6m
            input_dict['enq_L12m']             = enq_l12m
            input_dict['CC_Flag']              = cc_flag
            input_dict['PL_Flag']              = pl_flag
            input_dict['Time_With_Curr_Empr']  = time_empr * 12
            input_dict['EDUCATION']            = {'Graduate': 0, 'Post-Graduate': 1, 'SSC': 2, '12TH': 3, 'OTHERS': 4}.get(education, 0)
            input_dict['GENDER']               = {'M': 1, 'F': 0}.get(gender, 1)
            input_dict['MARITALSTATUS']        = {'Married': 1, 'Single': 2, 'Divorced': 0}.get(marital, 1)
            input_dict['last_prod_enq2']       = {'PL': 3, 'AL': 0, 'HL': 2, 'CC': 1, 'others': 4}.get(loan_type, 4)
            input_dict['first_prod_enq2']      = input_dict['last_prod_enq2']

            pred_df    = pd.DataFrame([input_dict])[feature_cols]
            pd_score   = xgb_model.predict_proba(pred_df)[0][1]
            risk_score = 1 - pd_score

            if risk_score < 0.02:   band, stage, decision = "P1 — Prime", "Stage 1", "✅ Approve"
            elif risk_score < 0.08: band, stage, decision = "P2 — Near-Prime", "Stage 1", "✅ Approve with monitoring"
            elif risk_score < 0.20: band, stage, decision = "P3 — Sub-Prime", "Stage 2", "⚠️ Refer for review"
            else:                   band, stage, decision = "P4 — High Risk", "Stage 3", "❌ Decline"

            st.markdown("---")
            st.markdown("### Indicative Assessment")
            r1, r2, r3, r4 = st.columns(4)
            with r1: st.metric("Probability of Default", f"{risk_score*100:.1f}%")
            with r2: st.metric("Creditworthiness", f"{pd_score*100:.1f}%")
            with r3: st.metric("Risk Band", band.split("—")[0].strip())
            with r4: st.metric("ECL Stage", stage)

            lti = loan_amount / income if income > 0 else 0
            st.caption(f"Loan: ₹{loan_amount:,.0f} | Income: ₹{income:,.0f}/mo | LTI Ratio: {lti:.1f}x")
            st.markdown(f'''
            <div class="ecl-box">
            <b>Decision:</b> {decision} &nbsp;|&nbsp; <b>Band:</b> {band} &nbsp;|&nbsp; <b>ECL Stage:</b> {stage} &nbsp;|&nbsp; <b>PD:</b> {risk_score*100:.2f}%
            </div>''', unsafe_allow_html=True)
            st.markdown(f"""
            **RBI ECL Implications:**
            - **Stage:** {stage}
            - **Provisioning:** {"12-month ECL" if "Stage 1" in stage else "Lifetime ECL"}
            - **SICR Trigger:** {"None — performing asset" if "Stage 1" in stage else "Significant increase in credit risk detected"}
            - **Decision:** {decision}

            ⚠️ *Indicative score using 18 of 85 model features. Upload full bureau CSV for accurate assessment.*
            *Per RBI ECL Final Directions, April 2026 (effective April 2027)*
            """)



elif page == "📋 ECL Framework":
    st.title("RBI Expected Credit Loss (ECL) Framework")
    st.caption("Final Directions issued April 27, 2026 | Effective April 1, 2027 | Aligned with IFRS 9")

    # Timeline
    st.markdown('<div class="section-header">Regulatory Timeline</div>', unsafe_allow_html=True)
    timeline = {
        "Jan 2023": "RBI Discussion Paper on ECL released — public consultation begins",
        "Oct 2025": "Draft Directions issued — detailed ECL rules proposed for SCBs",
        "Apr 2026": "Final Directions issued — framework confirmed",
        "Apr 2027": "ECL goes live — all SCBs (excl. RRBs, SFBs, Payments Banks)",
        "Mar 2031": "Full compliance deadline — 4-year glide path ends",
    }
    for date, event in timeline.items():
        st.markdown(f"**{date}** — {event}")

    st.markdown("---")
    st.markdown('<div class="section-header">Three-Stage Classification</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="ecl-box" style="border-color:#27ae60; background:#f0fff4">
        <b style="color:#27ae60">Stage 1 — Performing</b><br><br>
        No significant increase in credit risk since origination.<br><br>
        <b>Provision:</b> 12-month ECL<br>
        <b>PD horizon:</b> Next 12 months<br>
        <b>Maps to:</b> P1, P2 (our scorecard)
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="ecl-box" style="border-color:#e67e22; background:#fff8f0">
        <b style="color:#e67e22">Stage 2 — Under-performing</b><br><br>
        Significant Increase in Credit Risk (SICR) detected since origination.<br><br>
        <b>Provision:</b> Lifetime ECL<br>
        <b>PD horizon:</b> Full remaining tenor<br>
        <b>Maps to:</b> P3 (our scorecard)
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="ecl-box" style="border-color:#e74c3c; background:#fff5f5">
        <b style="color:#e74c3c">Stage 3 — Credit-Impaired</b><br><br>
        Asset is credit-impaired — NPA equivalent (90+ DPD or unlikely to pay).<br><br>
        <b>Provision:</b> Lifetime ECL on net basis<br>
        <b>PD horizon:</b> Full remaining tenor<br>
        <b>Maps to:</b> P4 (our scorecard)
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">ECL Formula & Component Mapping</div>', unsafe_allow_html=True)

    st.latex(r"ECL = PD \times LGD \times EAD \times DF")

    mapping = pd.DataFrame({
        'Component': ['PD — Probability of Default', 'LGD — Loss Given Default', 'EAD — Exposure at Default', 'DF — Discount Factor'],
        'This Project': ['XGBoost model output (0–1 score)', 'Not modelled (requires workout data)', 'Outstanding balance (approximated)', 'Risk-free rate / EIR'],
        'RBI Requirement': ['Point-in-time PD; calibrate to TTC', 'Based on observed recoveries, collateral', 'Include CCF for off-balance sheet', 'Effective Interest Rate method'],
        'Status': ['✅ Built', '⬜ Placeholder', '⬜ Approximated', '⬜ Placeholder']
    })
    st.dataframe(mapping, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Macro Stress Test — ECL Sensitivity to GNPA Scenarios</div>', unsafe_allow_html=True)

    base_gnpa   = 2.7
    base_ecl_mn = 12.4  # illustrative base from Layer 3 simulation
    alpha       = 0.5

    scenarios = {
        'Base (FY2024)': 2.7,
        'Mild Stress (+1.5pp)': 4.2,
        'Moderate Stress (+3pp)': 5.7,
        'Severe Stress (FY2020 level)': 8.2
    }

    stress_data = []
    for scen, gnpa in scenarios.items():
        scalar = 1 + alpha * (gnpa - base_gnpa) / base_gnpa
        ecl    = base_ecl_mn * scalar
        stress_data.append({'Scenario': scen, 'GNPA %': gnpa, 'PD Scalar': round(scalar, 3), 'Portfolio ECL (₹ Mn)': round(ecl, 2)})

    stress_df = pd.DataFrame(stress_data)
    st.dataframe(stress_df, use_container_width=True, hide_index=True)

    fig_s, ax_s = plt.subplots(figsize=(9, 4))
    fig_s.patch.set_facecolor('white'); ax_s.set_facecolor('#f8f8f8')
    colors_s = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c']
    bars = ax_s.bar(stress_df['Scenario'], stress_df['Portfolio ECL (₹ Mn)'], color=colors_s, edgecolor='white', width=0.5)
    for bar, val in zip(bars, stress_df['Portfolio ECL (₹ Mn)']):
        ax_s.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2, f'₹{val:.1f}Mn', ha='center', fontsize=9, fontweight='bold')
    ax_s.set_ylabel('ECL Provision (₹ Mn)'); ax_s.set_title('ECL Sensitivity — GNPA Stress Scenarios', fontweight='bold')
    ax_s.tick_params(axis='x', rotation=15); ax_s.grid(axis='y', alpha=0.3)
    ax_s.spines['top'].set_visible(False); ax_s.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_s)

    st.caption("Formula: PD_scalar = 1 + α × (GNPA_scenario − GNPA_base) / GNPA_base | α = 0.5 (illustrative) | Recalibrate using historical loss data")

    st.markdown("---")
    st.markdown('<div class="section-header">Model Validation Checklist (RBI ECL Requirements)</div>', unsafe_allow_html=True)

    checklist = pd.DataFrame({
        'Requirement': ['Discriminatory power (AUC > 0.75)', 'Calibration (predicted vs actual PD)', 'Feature explainability', 'Out-of-sample validation', 'Challenger / benchmark model', 'Model documentation', 'Population Stability Index (PSI)', 'Backtesting with observed defaults'],
        'Status': ['✅ LR: 0.957 | XGB: 0.999', '✅ Classification report + confusion matrix', '✅ SHAP summary + waterfall plots', '✅ Unseen dataset (100 applicants)', '✅ Logistic Regression vs XGBoost', '✅ README + Layer 3 write-up', '⬜ Requires longitudinal data', '⬜ Requires time-series default history']
    })
    st.dataframe(checklist, use_container_width=True, hide_index=True)
