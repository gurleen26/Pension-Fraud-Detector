# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load flagged suspicious data
# try:
#     flagged = pd.read_csv("flagged_cases.csv")
# except FileNotFoundError:
#     st.error("⚠️ 'flagged_cases.csv' not found. Please make sure you have run the main analysis script.")
#     st.stop()

# # Streamlit Page Config
# st.set_page_config(page_title="Pension Fraud Monitor", layout="wide")
# st.title("🔍 Pension Scheme Monitoring Dashboard")

# # Input for searching suspicious names or IDs
# search_input = st.text_input("Search by Name or Aadhaar/SSN")

# # Filter suspicious data
# if search_input:
#     filtered = flagged[
#         flagged['Name'].str.contains(search_input, case=False, na=False) |
#         flagged['Aadhaar/SSN'].astype(str).str.contains(search_input, na=False)
#     ]
# else:
#     filtered = flagged.copy()

# # Show filtered data
# st.subheader("Suspicious Clusters Identified")
# st.dataframe(filtered)

# # Show cluster distribution
# st.subheader("Cluster Distribution")
# if 'cluster' in flagged.columns:
#     cluster_counts = flagged['cluster'].value_counts().sort_index()
#     st.bar_chart(cluster_counts)
# else:
#     st.write("Cluster data not available in 'flagged_cases.csv'.")

# # Download option
# st.download_button("Download Flagged Cases", data=flagged.to_csv(index=False), file_name="flagged_cases.csv")
# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt

# # Load flagged suspicious data
# try:
#     flagged = pd.read_csv("flagged_cases.csv")
# except FileNotFoundError:
#     st.error("⚠️ 'flagged_cases.csv' not found. Please make sure you ran the notebook to create this file.")
#     st.stop()

# # Sidebar filters
# st.sidebar.header("Search Beneficiaries")
# name_query = st.sidebar.text_input("Search by Name")
# aadhaar_query = st.sidebar.text_input("Search by Aadhaar/SSN")

# # Filter data
# filtered = flagged.copy()
# if name_query:
#     filtered = filtered[filtered['Name'].str.contains(name_query, case=False, na=False)]
# if aadhaar_query:
#     filtered = filtered[filtered['Aadhaar/SSN'].astype(str).str.contains(aadhaar_query, na=False)]

# # Main Dashboard
# st.title("🔍 Pension Scheme Monitoring Dashboard")
# st.subheader(f"Search Results for Name: '{name_query}'" if name_query else "Suspicious Clusters Identified")
# st.dataframe(filtered)

# # Cluster Distribution
# if 'cluster' in flagged.columns:
#     st.subheader("Cluster Distribution")
#     cluster_counts = flagged['cluster'].value_counts().sort_index()
#     st.bar_chart(cluster_counts)

# # Download button
# st.download_button("📥 Download Flagged Cases", data=filtered.to_csv(index=False), file_name="flagged_cases_filtered.csv")
# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # Custom page settings
# st.set_page_config(
#     page_title="Pension Fraud Detector",
#     page_icon="🕵️‍♂️",
#     layout="wide"
# )

# # Sidebar - Navigation
# st.sidebar.image("https://img.icons8.com/emoji/96/spy.png", width=80)
# st.sidebar.title("Navigation")
# section = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Check Individual", "📁 Upload New Data"])

# # App header
# st.markdown("<h1 style='color:#0E76A8;'>🕵️ Pension Fraud Detection System</h1>", unsafe_allow_html=True)
# st.markdown("An intelligent tool to identify individuals receiving multiple pensions using anomaly detection and clustering techniques.")

# # Upload or load existing data
# @st.cache_data
# def load_data():
#     return pd.read_csv("flagged_cases.csv")

# df = load_data()

# # Section: Dashboard
# if section == "📊 Dashboard":
#     st.subheader("📈 Overview of Suspicious Clusters")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.metric("Total Flagged Records", len(df))
#         st.metric("Suspicious Clusters", df['cluster_dbscan'].nunique())

#     with col2:
#         fig = px.scatter(df,
#     x="benefit_count_per_person",
#     y="benefit_amount_total",
#     color="cluster_dbscan",
#     size="benefit_amount_total",
#     hover_name="Name",
#     title="Suspicious Records",
#     color_continuous_scale="OrRd"
# )

#         st.plotly_chart(fig, use_container_width=True)

#     st.dataframe(df.head(10), use_container_width=True)

# # Section: Individual Checker
# elif section == "🔍 Check Individual":
#     st.subheader("🔎 Check if an Individual is Receiving Multiple Pensions")

#     user_name = st.text_input("Enter Full Name:")
#     region = st.text_input("Enter Region (optional):")

#     if user_name:
#         matches = df[df['Name'].str.contains(user_name, case=False, na=False)]

#         if region:
#             matches = matches[matches['Region'].str.contains(region, case=False, na=False)]

#         if not matches.empty:
#             st.success(f"✅ Found {len(matches)} flagged record(s) matching '{user_name}'")
#             st.dataframe





# import streamlit as st
# import pandas as pd
# import plotly.express as px
# from PIL import Image

# # Custom page settings
# st.set_page_config(
#     page_title="Pension Fraud Detector",
#     page_icon="🕵️‍♂️",
#     layout="wide"
# )

# # Sidebar - Navigation
# st.sidebar.image("senior-care-logo--removebg-preview.png", width=80)
# st.sidebar.title("Navigation")
# section = st.sidebar.radio("Go to", ["📊 Dashboard", "🔍 Check Individual", "📁 Upload New Data"])

# # App header
# st.markdown("<h1 style='color:#0E76A8;'>🕵️ Pension Fraud Detection System</h1>", unsafe_allow_html=True)
# st.markdown("An intelligent tool to identify individuals receiving multiple pensions using anomaly detection and clustering techniques.")

# # Load individual-level flagged data
# @st.cache_data
# def load_flagged_data():
#     return pd.read_csv("flagged_cases.csv")

# # Load cluster-level suspicious summary
# @st.cache_data
# def load_cluster_data():
#     return pd.read_excel("suspicious_clusters_report.xlsx")

# df = load_flagged_data()
# cluster_df = load_cluster_data()

# # Section: Dashboard
# if section == "📊 Dashboard":
#     st.subheader("📈 Suspicious Patterns Detected")

#     tab1, tab2 = st.tabs(["🧠 Cluster Overview", "🧍 Individual Overview"])

#     # Cluster-Level Visualization
#     with tab1:
#         st.markdown("### 🚩 Anomalous Clusters (Detected by Isolation Forest)")

#         st.metric("Total Suspicious Clusters", cluster_df['cluster_dbscan'].nunique())
#         fig = px.scatter(cluster_df,
#                          x="entry_count",
#                          y="amount",
#                          color="scheme",
#                          size="amount",
#                          hover_name="cluster_dbscan",
#                          title="Suspicious Clusters Overview",
#                          color_continuous_scale="Turbo")
#         st.plotly_chart(fig, use_container_width=True)

#         st.dataframe(cluster_df, use_container_width=True)

#     # Individual-Level Visualization
#     with tab2:
#         st.markdown("### 🧍 Flagged Individuals")

#         st.metric("Total Flagged Individuals", len(df))

#         fig = px.scatter(df,
#                          x="benefit_count_per_person",
#                          y="benefit_amount_total",
#                          color="cluster_dbscan",
#                          size="benefit_amount_total",
#                          hover_name="Name",
#                          title="Suspicious Individuals",
#                          color_continuous_scale="OrRd")
#         st.plotly_chart(fig, use_container_width=True)

#         st.dataframe(df.head(10), use_container_width=True)

# # Section: Individual Checker
# elif section == "🔍 Check Individual":
#     st.subheader("🔎 Check if an Individual is Receiving Multiple Pensions")

#     with st.form("fraud_check_form"):
#         name_input = st.text_input("Enter Full Name:")
#         region_input = st.text_input("Enter Region:")
#         aadhaar_input = st.text_input("Enter Aadhaar/SSN:")
#         dob_input = st.text_input("Enter Date of Birth (YYYY-MM-DD) :")
#         submitted = st.form_submit_button("🔍 Check")

#     if submitted:
#         matches = df.copy()

#         if name_input:
#             matches = matches[matches['Name'].str.contains(name_input, case=False, na=False)]

#         if region_input:
#             matches = matches[matches['Region'].str.contains(region_input, case=False, na=False)]

#         if aadhaar_input:
#             matches = matches[matches['Aadhaar/SSN'].astype(str).str.contains(aadhaar_input, na=False)]

#         if dob_input:
#             matches = matches[matches['DOB'].astype(str).str.contains(dob_input, na=False)]

#         if not matches.empty:
#             for _, row in matches.iterrows():
#                 statement = f"""
#                     <div style='background-color:#B22222;padding:20px;border-radius:10px'>
#                         <h4 style='color:white;'>🚨 Fraud Alert</h4>
#                         <p style='color:white;'>The individual <strong>{row['Name']}</strong>, born on <strong>{row['DOB']}</strong>, residing at 
#                         <strong>{row['Address']}</strong> with Aadhaar/SSN <strong>{row['Aadhaar/SSN']}</strong>, has been flagged for receiving multiple pensions.</p>
#                         <p style='color:white;'>They are receiving: <strong>{row['Pension Scheme']}</strong> under the <strong>{row['Pension Type']}</strong> scheme.</p>
#                         <p style='color:white;'>Total Benefit Amount: ₹{row['benefit_amount_total']}</p>
#                     </div>
#                 """
#                 st.markdown(statement, unsafe_allow_html=True)
#             st.dataframe(matches, use_container_width=True)
#         else:
#             st.warning("⚠️ No flagged records found for the entered details.")




# # Section: Upload (optional functionality)
# elif section == "📁 Upload New Data":
#     st.subheader("📤 Upload New Pension Dataset (Coming Soon...)")
#     st.info("This section will allow admins to upload new data for fraud detection. Stay tuned!")



import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

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
    return pd.read_csv("flagged_cases.csv")

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

        st.metric("Total Flagged Individuals", len(df))

        fig = px.scatter(df,
                         x="benefit_count_per_person",
                         y="benefit_amount_total",
                         color="cluster_dbscan",
                         size="benefit_amount_total",
                         hover_name="Name",
                         title="Suspicious Individuals",
                         color_continuous_scale="OrRd")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df.head(10), use_container_width=True)

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

        if not matches.empty:
            grouped = matches.groupby(['Name', 'DOB', 'Aadhaar/SSN', 'Address'])

            for (name, dob, aadhaar, address), group in grouped:
                total_amt = group['Amount'].sum()

                st.markdown(
                    f"""
                    <div style='background-color:#B22222;padding:20px;border-radius:10px'>
                        <h4 style='color:white;'>🚨 Fraud Alert</h4>
                        <p style='color:white;'>The individual <strong>{name}</strong>, born on <strong>{dob}</strong>, residing at 
                        <strong>{address}</strong> with Aadhaar/SSN <strong>{aadhaar}</strong>, has been flagged for receiving multiple pensions.</p>
                        <p style='color:white;'><strong>They are receiving:</strong></p>
                    """,
                    unsafe_allow_html=True
                )

                for _, row in group.iterrows():
                    st.markdown(
                        f"""<div style='color:white; padding-left:10px;'>• <strong>{row['Pension Scheme']}</strong> under the <strong>{row['Pension Type']}</strong> scheme – ₹{row['Amount']:.2f}</div>""",
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""<br><div style='color:white; font-size:16px;'><strong>Total Benefit Amount:</strong> ₹{total_amt:.2f}</div></div>""",
                    unsafe_allow_html=True
                )


        else:
                    # ✅ Not fraudulent
            st.success("✅ This individual is only receiving one pension. No fraud detected.")

# Section: Upload (optional functionality)
elif section == "📁 Upload New Data":
    st.subheader("📤 Upload and Analyze a New Pension Dataset")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        st.success("✅ File uploaded successfully. Processing...")

        # --- Minimal required columns check ---
        required_cols = {"Name", "DOB", "Aadhaar/SSN", "Address", "Pension Scheme", "Pension Type", "Amount", "Region"}
        if not required_cols.issubset(user_df.columns):
            st.error(f"❌ Missing required columns: {required_cols - set(user_df.columns)}")
        else:
            # --- Add benefit count and total benefit per person ---
            user_df["Aadhaar/SSN"] = user_df["Aadhaar/SSN"].astype(str)
            person_group = user_df.groupby("Aadhaar/SSN").agg({
                "Amount": "sum",
                "Pension Scheme": "nunique"
            }).rename(columns={"Amount": "benefit_amount_total", "Pension Scheme": "benefit_count_per_person"})

            user_df = user_df.merge(person_group, on="Aadhaar/SSN")

            # --- Run clustering and anomaly detection ---
            from sklearn.cluster import DBSCAN
            from sklearn.ensemble import IsolationForest

            features = user_df[["benefit_count_per_person", "benefit_amount_total"]].drop_duplicates()

            db = DBSCAN(eps=0.9, min_samples=2)
            user_df["cluster_dbscan"] = db.fit_predict(features)

            iso = IsolationForest(contamination=0.05, random_state=42)
            user_df["anomaly"] = iso.fit_predict(features)

            # --- Flag suspicious (clustered + anomalous) ---
            suspicious_df = user_df[(user_df["cluster_dbscan"] != -1) & (user_df["anomaly"] == -1)]

            st.markdown("### 🔍 Results of Fraud Detection")
            st.metric("Flagged Records", len(suspicious_df))

            if not suspicious_df.empty:
                st.dataframe(suspicious_df, use_container_width=True)
                csv = suspicious_df.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download Flagged Results", data=csv, file_name="flagged_individuals.csv", mime="text/csv")
            else:
                st.success("✅ No fraudulent patterns detected in the uploaded data.")
