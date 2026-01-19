import re
import unicodedata
from typing import Optional

import pandas as pd
import streamlit as st


APP_TITLE = "Serving Girl Availability Review"
BASE_CSV_PATH_DEFAULT = "/mnt/data/Serving base with allocated directors.csv"


def normalize_name(s: str) -> str:
    """
    Normalize names so comparisons are robust:
    - strip whitespace
    - collapse repeated spaces
    - remove accents
    - lowercase
    """
    if s is None:
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.lower().strip()


@st.cache_data(ttl=20, show_spinner=False)
def load_serving_base(csv_path: str) -> pd.DataFrame:
    # Your file is semicolon-delimited
    df = pd.read_csv(csv_path, sep=";", dtype=str).fillna("")
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]
    required = {"Director", "Serving Girl"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {sorted(required)}. Found: {list(df.columns)}")
    df["Director_norm"] = df["Director"].apply(normalize_name)
    df["ServingGirl_norm"] = df["Serving Girl"].apply(normalize_name)
    return df


def get_sheet_df_from_gspread(
    spreadsheet_id: str,
    worksheet_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Reads the Google Sheet into a DataFrame.
    Uses a Service Account JSON stored in st.secrets["gcp_service_account"].
    """
    import gspread
    from google.oauth2.service_account import Credentials

    if "gcp_service_account" not in st.secrets:
        raise RuntimeError(
            "Missing st.secrets['gcp_service_account']. "
            "Add your Google service account JSON into .streamlit/secrets.toml."
        )

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly",
    ]
    creds = Credentials.from_service_account_info(dict(st.secrets["gcp_service_account"]), scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(worksheet_name) if worksheet_name else sh.sheet1

    # Assumes first row is headers
    records = ws.get_all_records()
    df = pd.DataFrame(records)
    # Ensure string columns don’t become NaN
    df = df.fillna("")
    df.columns = [str(c).strip() for c in df.columns]
    return df


@st.cache_data(ttl=20, show_spinner=False)
def load_google_sheet(
    spreadsheet_id: str,
    worksheet_name: Optional[str],
    name_col: str,
    reason_col: str,
) -> pd.DataFrame:
    df = get_sheet_df_from_gspread(spreadsheet_id, worksheet_name)

    if name_col not in df.columns:
        raise ValueError(f"Name column '{name_col}' not found in sheet. Found: {list(df.columns)}")
    if reason_col not in df.columns:
        raise ValueError(f"Reason column '{reason_col}' not found in sheet. Found: {list(df.columns)}")

    df["name_norm"] = df[name_col].apply(normalize_name)
    df["reason_clean"] = df[reason_col].astype(str).fillna("").apply(lambda x: x.strip())
    return df


def compute_status(sheet_df: pd.DataFrame, serving_girl_name: str) -> dict:
    """
    Returns:
      {"status": "Done"/"Review"/"Not submitted", "reason": "..."}
    """
    sg_norm = normalize_name(serving_girl_name)
    match = sheet_df[sheet_df["name_norm"] == sg_norm]

    if match.empty:
        return {"status": "Not submitted", "reason": ""}

    # If multiple matches, take the first (you can improve this by using latest timestamp if you have one)
    reason = str(match.iloc[0]["reason_clean"]).strip()
    if reason == "":
        return {"status": "Done", "reason": ""}
    return {"status": "Review", "reason": reason}


def badge(status: str) -> str:
    # Simple text badges (no dependency on extra libraries)
    if status == "Done":
        return "✅ Done"
    if status == "Review":
        return "⚠️ Review"
    return "❌ Not submitted"


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    # ---- Configuration (from secrets, with sensible defaults) ----
    sheet_id = st.secrets.get("sheet", {}).get("spreadsheet_id", "")
    worksheet_name = st.secrets.get("sheet", {}).get("worksheet_name", "")
    sheet_name_col = st.secrets.get("sheet", {}).get("name_column", "Serving Girl")
    sheet_reason_col = st.secrets.get("sheet", {}).get("reason_column", "Reason")
    base_csv_path = st.secrets.get("base", {}).get("csv_path", BASE_CSV_PATH_DEFAULT)

    with st.expander("⚙️ Settings (read-only)", expanded=False):
        st.write("**Serving base CSV path:**", base_csv_path)
        st.write("**Google Sheet ID:**", sheet_id if sheet_id else "(not set)")
        st.write("**Worksheet name:**", worksheet_name if worksheet_name else "(default: first sheet)")
        st.write("**Sheet name column:**", sheet_name_col)
        st.write("**Sheet reason column:**", sheet_reason_col)

    # ---- Load base mapping ----
    try:
        base_df = load_serving_base(base_csv_path)
    except Exception as e:
        st.error(f"Could not load serving base CSV: {e}")
        st.stop()

    directors = sorted(base_df["Director"].unique(), key=lambda x: normalize_name(x))

    st.subheader("Select Director")
    director = st.selectbox("Director", directors, index=0)

    # Refresh button (clears caches)
    cols = st.columns([1, 4])
    with cols[0]:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    # ---- Load Google Sheet ----
    if not sheet_id:
        st.warning(
            "Google Sheet is not configured yet. Add your Sheet ID to `.streamlit/secrets.toml` "
            "under `[sheet] spreadsheet_id = \"...\"`."
        )
        st.stop()

    try:
        sheet_df = load_google_sheet(
            spreadsheet_id=sheet_id,
            worksheet_name=worksheet_name if worksheet_name else None,
            name_col=sheet_name_col,
            reason_col=sheet_reason_col,
        )
    except Exception as e:
        st.error(f"Could not load Google Sheet: {e}")
        st.stop()

    # ---- Filter serving girls for this director ----
    director_norm = normalize_name(director)
    girls_df = base_df[base_df["Director_norm"] == director_norm].copy()
    girls = sorted(girls_df["Serving Girl"].tolist(), key=lambda x: normalize_name(x))

    st.subheader(f"Serving Girls for {director}")
    if not girls:
        st.info("No serving girls found for this director in the serving base file.")
        st.stop()

    # ---- Compute statuses ----
    results = []
    for g in girls:
        r = compute_status(sheet_df, g)
        results.append((g, r["status"], r["reason"]))

    done_count = sum(1 for _, s, _ in results if s == "Done")
    review_count = sum(1 for _, s, _ in results if s == "Review")
    not_count = sum(1 for _, s, _ in results if s == "Not submitted")

    m1, m2, m3 = st.columns(3)
    m1.metric("Done", done_count)
    m2.metric("Review", review_count)
    m3.metric("Not submitted", not_count)

    st.divider()

    # ---- Display list with feedback ----
    for name, status, reason in results:
        row = st.container()
        c1, c2 = row.columns([3, 2])
        with c1:
            st.write(f"**{name}**")
        with c2:
            st.write(badge(status))

        if status == "Review" and reason:
            st.caption(f"Reason: {reason}")

        st.write("")  # spacing


if __name__ == "__main__":
    main()
