import streamlit as st
import pandas as pd
import io
import re
from rapidfuzz import fuzz
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OrdinalEncoder
import numpy as np

st.set_page_config(page_title="Data Cleaning App", layout="wide")
st.title("Spreadsheet Cleaner App")
st.caption("Automatically detect issues, fix errors and impute missing values")

@st.cache_data(show_spinner=False)
def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    return pd.read_csv(uploaded_file)

def normalize_spaces(s):
    return re.sub(r"\s+", " ", str(s)).strip()

def normalize_name(s):
    s = normalize_spaces(str(s))
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return " ".join(w.capitalize() for w in s.split())

def is_valid_email(email):
    return isinstance(email, str) and "@" in email and "." in email.split("@")[-1]

def generate_email_from_name(name):
    parts = re.split(r"\s+", name.strip())
    if len(parts) == 1:
        user = parts[0].lower()
    elif len(parts) >= 2:
        user = f"{parts[0].lower()}.{parts[-1].lower()}"
    else:
        return ""
    return f"{user}@company.com"

def clean_email(email):
    email = str(email).strip().lower().replace(" ", "")
    if email == "" or not is_valid_email(email):
        return email
    local, domain = email.split("@", 1)
    if not domain.endswith(".com"):
        domain = "company.com"
    return f"{local}@{domain}"

def detect_missing_summary(df):
    missing_cols = df.isna().sum()
    return missing_cols[missing_cols > 0]

def highlight_invalid(s):
    styles = []
    for col, value in s.items():
        text = str(value).strip()
        invalid = False
        if col == "Email" and not is_valid_email(text):
            invalid = True
        elif col == "Name" and not re.match(r"^[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(\s[A-ZÇĞİÖŞÜ][a-zçğıöşü]+)+$", text):
            invalid = True
        elif col == "Phone" and (text != "" and (not re.match(r"^\+?\d[\d\-\s]{5,}$", text))):
            invalid = True
        elif col in ["Location", "Education", "Job_Title", "Gender"] and text == "":
            invalid = True
        elif col == "Salary" and (text == "" or text == "nan" or float(text) == 0.0):
            invalid = True
        styles.append("color:red;font-weight:bold;" if invalid else "")
    return styles

def find_near_duplicate_pairs(df, threshold=95):
    combined = [" ".join(map(str, row)) for _, row in df.astype(str).fillna("").iterrows()]
    pairs = []
    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            score = fuzz.ratio(combined[i].lower(), combined[j].lower())
            if score >= threshold and score < 100:
                pairs.append({"Group": len(pairs) + 1, "i1": i, "i2": j, "Similarity": round(score, 2)})
    return pairs

def build_downloads(df, log_df):
    csv = df.to_csv(index=False).encode("utf-8")
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cleaned")
    bio.seek(0)
    log_csv = log_df.to_csv(index=False).encode("utf-8")
    return csv, bio, log_csv

uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded:
    df = read_file(uploaded)
    df = df.rename(columns=lambda x: x.strip().title())
    total_rows = len(df)

    st.subheader("1. Data Profiling")
    missing_cols = detect_missing_summary(df)
    dup_count = len(df[df.duplicated(keep=False)])
    st.write(f"Total rows: {total_rows}")
    st.write(f"Total missing cells: {int(df.isna().sum().sum())} | Duplicate rows: {dup_count}")
    if not missing_cols.empty:
        miss_df = pd.DataFrame(list(missing_cols.items()), columns=["Column", "Missing Count"])
        html_table = miss_df.to_html(index=False, justify="center")
        html_table = html_table.replace('<th>', '<th style="text-align:center;">')
        html_table = html_table.replace('<td>', '<td style="text-align:center;">')
        st.markdown(html_table, unsafe_allow_html=True)

    if "clean_df" not in st.session_state:
        st.session_state.clean_df = df.copy()

    st.subheader("2. Near-Duplicate Detection")
    pairs = find_near_duplicate_pairs(st.session_state.clean_df, threshold=95)
    if len(pairs) > 0:
        st.info(f"{len(pairs)} near-duplicate pairs detected.")
        for p in pairs:
            if p["i1"] not in st.session_state.clean_df.index or p["i2"] not in st.session_state.clean_df.index:
                continue
            r1 = st.session_state.clean_df.loc[p["i1"]]
            r2 = st.session_state.clean_df.loc[p["i2"]]
            st.markdown(f"<p style='color:#FFB347;font-weight:bold;'>Group {p['Group']} — Similarity: {p['Similarity']}%</p>", unsafe_allow_html=True)
            for idx, row in [(p["i1"], r1), (p["i2"], r2)]:
                c1, c2 = st.columns([10, 1])
                with c1:
                    st.markdown(f"<div style='background:#2e2e2e;padding:8px;border-radius:6px;color:#eee;font-family:monospace;'>{' | '.join(map(str, row.values))}</div>", unsafe_allow_html=True)
                with c2:
                    if st.button("Delete", key=f"del_{p['Group']}_{idx}"):
                        if idx in st.session_state.clean_df.index:
                            st.session_state.clean_df = st.session_state.clean_df.drop(index=idx)
                            st.rerun()
            st.markdown("---")
    else:
        st.success("No near-duplicate rows detected.")

    st.subheader("3. Preview Invalid Rows")
    def has_red(row):
        return any("red" in style for style in highlight_invalid(row))
    invalid_rows = df[df.apply(has_red, axis=1)]
    styled = invalid_rows.style.apply(highlight_invalid, axis=1)
    if invalid_rows.empty:
        st.success("No invalid data found.")
    else:
        def fmt_salary(x):
            try:
                xi = int(float(x))
                return f"{xi:,}"
            except:
                return x
        st.dataframe(styled.format({"Salary": fmt_salary}))

    st.subheader("4. Fix the Data")
    if st.button("Fix the Data"):
        log_records = []

        fixed_name_count = 0
        if "Name" in df.columns:
            for i, v in df["Name"].items():
                cleaned = normalize_name(v)
                if cleaned != v:
                    log_records.append([i, "Name", v, cleaned, "normalize", "title_spacing"])
                    df.at[i, "Name"] = cleaned
                    fixed_name_count += 1

        fixed_email_count = 0
        if "Email" in df.columns:
            for i, v in df["Email"].items():
                old = str(v)
                if not is_valid_email(old):
                    new_email = generate_email_from_name(df.at[i, "Name"])
                    log_records.append([i, "Email", old, new_email, "fix_email", "invalid"])
                    df.at[i, "Email"] = new_email
                    fixed_email_count += 1
                else:
                    cleaned = clean_email(old)
                    if cleaned != old:
                        log_records.append([i, "Email", old, cleaned, "fix_email", "standardize"])
                        df.at[i, "Email"] = cleaned
                        fixed_email_count += 1

        before = len(df)
        df = df.drop_duplicates()
        removed_duplicates = before - len(df)

        imputed_count = 0
        if "Salary" in df.columns:
            df["Salary"] = df["Salary"].replace(0, np.nan)

            needed_cols = ["Education", "Experience", "Job_Title", "Age", "Salary"]
            available = [c for c in needed_cols if c in df.columns]

            sub = df[available].copy()

            cat_cols = [c for c in ["Education", "Job_Title"] if c in sub.columns]
            num_cols = [c for c in ["Experience", "Age", "Salary"] if c in sub.columns]

            for col in cat_cols:
                sub[col] = sub[col].astype(str)

            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            for col in cat_cols:
                sub[col] = enc.fit_transform(sub[[col]])

            missing_mask = sub["Salary"].isna()

            if missing_mask.any():
                imputer = IterativeImputer(random_state=42, estimator=BayesianRidge())
                imputed = imputer.fit_transform(sub)
                sub_imputed = pd.DataFrame(imputed, columns=sub.columns, index=sub.index)

                predicted_values = sub_imputed.loc[missing_mask, "Salary"].values
                df.loc[missing_mask, "Salary"] = predicted_values

                for idx, val in zip(df[missing_mask].index, predicted_values):
                    log_records.append([idx, "Salary", "", val, "impute", "iterative"])
                imputed_count = len(predicted_values)

        log_df = pd.DataFrame(log_records, columns=["row_id", "field", "old_value", "new_value", "action_type", "reason"])

        st.success("Data cleaning completed.")
        st.write(f"{fixed_name_count} names normalized")
        st.write(f"{fixed_email_count} emails fixed")
        st.write(f"{removed_duplicates} exact duplicates removed")
        st.write(f"{imputed_count} salaries imputed")

        def fmt_salary_int(x):
            try:
                xi = int(float(x))
                return f"{xi:,}"
            except:
                return x

        csv_bytes, xlsx_bytes, log_bytes = build_downloads(df, log_df)
        
        st.subheader("5. Cleaned Results")
        st.dataframe(df.style.format({"Salary": fmt_salary_int}))

        sp1, c1, sp2, c2, sp3, c3 = st.columns([0.2, 0.5, 0.1, 0.5, 0.1, 0.5])

        with c1:
            st.download_button("Download Cleaned CSV", data=csv_bytes, file_name="cleaned.csv", mime="text/csv")

        with c2:
            st.download_button("Download Excel", data=xlsx_bytes, file_name="cleaned.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        with c3:
            st.download_button("Download Log CSV", data=log_bytes, file_name="cleaning_log.csv", mime="text/csv")        
