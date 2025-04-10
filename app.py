import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from codetogether import detect_fraud

# Custom page settings
st.set_page_config(
    page_title="Pension Fraud Detector",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# Sidebar - Navigation
st.sidebar.image("senior-care-logo--removebg-preview.png", width=80)
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Check Individual", "📁 Upload New Data"])

# App header
st.markdown("<h1 style='color:#0E76A8;'>🕵️ Pension Fraud Detection System</h1>", unsafe_allow_html=True)
st.markdown("An intelligent tool to identify individuals receiving multiple pensions using anomaly detection and clustering techniques.")

# Load individual-level flagged data
@st.cache_data
def load_flagged_data():
    return pd.read_csv("clustered_pension_dataset.csv")

# Load cluster-level suspicious summary
@st.cache_data
def load_cluster_data():
    return pd.read_excel("suspicious_clusters_report.xlsx")

df = load_flagged_data()
cluster_df = load_cluster_data()

# Section: Dashboard
if section == "📊 Dashboard":
    st.subheader("📈 Suspicious Patterns Detected")

    tab1, tab2 = st.tabs(["🧠 Cluster Overview", "🧍 Individual Overview"])

    # Cluster-Level Visualization
    with tab1:
        st.markdown("### 🚩 Anomalous Clusters (Detected by Isolation Forest)")

        st.metric("Total Suspicious Clusters", cluster_df['cluster_dbscan'].nunique())
        fig = px.scatter(cluster_df,
                         x="entry_count",
                         y="amount",
                         color="scheme",
                         size="amount",
                         hover_name="cluster_dbscan",
                         title="Suspicious Clusters Overview",
                         color_continuous_scale="Turbo")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(cluster_df, use_container_width=True)

    # Individual-Level Visualization
    with tab2:
        st.markdown("### 🧍 Flagged Individuals")

        st.metric("Total Flagged Individuals", len(df[df['Fraud_Flag'] == 'Suspicious']))

        if 'benefit_count_per_person' in df.columns and 'benefit_amount_total' in df.columns:
            fig = px.scatter(df[df['Fraud_Flag'] == 'Suspicious'],
                             x="benefit_count_per_person",
                             y="benefit_amount_total",
                             color="cluster_dbscan",
                             size="benefit_amount_total",
                             hover_name="Name",
                             title="Suspicious Individuals",
                             color_continuous_scale="OrRd")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df[df['Fraud_Flag'] == 'Suspicious'].head(10), use_container_width=True)

# Section: Individual Checker
elif section == "🔍 Check Individual":
    st.subheader("🔎 Check if an Individual is Receiving Multiple Pensions")

    with st.form("fraud_check_form"):
        name_input = st.text_input("Enter Full Name:")
        region_input = st.text_input("Enter Region:")
        aadhaar_input = st.text_input("Enter Aadhaar/SSN:")
        dob_input = st.text_input("Enter Date of Birth (YYYY-MM-DD) :")
        submitted = st.form_submit_button("🔍 Check")

    if submitted:
        matches = df.copy()

        if name_input:
            matches = matches[matches['Name'].str.contains(name_input, case=False, na=False)]

        if region_input:
            matches = matches[matches['Region'].str.contains(region_input, case=False, na=False)]

        if aadhaar_input:
            matches = matches[matches['Aadhaar/SSN'].astype(str).str.contains(aadhaar_input, na=False)]

        if dob_input:
            matches = matches[matches['DOB'].astype(str).str.contains(dob_input, na=False)]

        if matches.empty:
            st.info("ℹ️ This individual is not registered in the system.")

        else:
            grouped = matches.groupby(['Name', 'DOB', 'Aadhaar/SSN', 'Address'])
            for (name, dob, aadhaar, address), group in grouped:
                total_amt = group['Amount'].sum()
                if (group['Fraud_Flag'] == 'Suspicious').any():
                    scheme_details = "".join([
                        f"<div style='color:white; padding-left:10px;'>• <strong>{row['Pension Scheme']}</strong> under the <strong>{row['Pension Type']}</strong> scheme – ₹{row['Amount']:.2f}</div>"
                        for _, row in group.iterrows()
                    ])
                    st.markdown(
                        f"""<div style='background-color:#B22222; padding:30px 40px; border-radius:15px; width:90%; margin:auto;'>
                        <h4 style='color:white;'>🚨 Fraud Alert</h4>
                        <p style='color:white; font-size:18px;'>🕵️ Individual: <strong>{name}</strong><br>
                        📄 Status: <strong style='color:white;'>Flagged for Fraud</strong></p>
                        {scheme_details}
                        <br><div style='color:white; font-size:16px;'><strong>Total Benefit Amount:</strong> ₹{total_amt:.2f}</div>
                        </div>""",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""<div style='background-color:#e0f7e9;padding:20px;border-radius:10px'>
                            <h4 style='color:green;'>✅ No Fraud Detected</h4>
                            <p style='color:green;'>The individual <strong>{name}</strong>, born on <strong>{dob}</strong>, residing at 
                            <strong>{address}</strong> with Aadhaar/SSN <strong>{aadhaar}</strong> is registered and appears to be receiving pension benefits legitimately.</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
                    for _, row in group.iterrows():
                        st.markdown(
                            f"""<div style='color:green; padding-left:10px;'>• <strong>{row['Pension Scheme']}</strong> under the <strong>{row['Pension Type']}</strong> scheme – ₹{row['Amount']:.2f}</div>""",
                            unsafe_allow_html=True
                        )
                    st.markdown(
                        f"""<br><div style='color:green; font-size:16px;'><strong>Total Benefit Amount:</strong> ₹{total_amt:.2f}</div>""",
                        unsafe_allow_html=True
                    )

# Section: Upload (optional functionality)
elif section == "📁 Upload New Data":
    st.subheader("📤 Upload and Analyze a New Pension Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully. Processing...")

        required_cols = {"Name", "DOB", "Aadhaar/SSN", "Address", "Pension Scheme", "Pension Type", "Amount", "Region"}
        if not required_cols.issubset(user_df.columns):
            st.error(f"❌ Missing required columns: {required_cols - set(user_df.columns)}")
        else:
            try:
                flagged_df, cluster_summary = detect_fraud(user_df)
                suspicious_df = flagged_df[flagged_df['Fraud_Flag'] == 'Suspicious']

                st.markdown("### 🔍 Results of Fraud Detection")
                st.metric("Flagged Records", len(suspicious_df))

                if not suspicious_df.empty:
                    st.dataframe(suspicious_df, use_container_width=True)
                    csv = suspicious_df.to_csv(index=False).encode("utf-8")
                    st.download_button("📥 Download Flagged Results", data=csv, file_name="flagged_individuals.csv", mime="text/csv")
                else:
                    st.success("✅ No fraudulent patterns detected in the uploaded data.")
            except Exception as e:
                st.error(f"❌ Error processing data: {e}")