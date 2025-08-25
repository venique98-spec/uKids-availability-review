import pandas as pd
import numpy as np
import re
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Constants & Helpers
# ──────────────────────────────────────────────────────────────────────────────

MONTH_ALIASES = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
YES_SET = {"yes", "y", "true", "available"}

# <<< NEW: Allowed roles for Directors
DIRECTOR_ALLOWED_ROLES = {
    "Babies Leader Age 1",
    "Babies Leader Age 2",
    "Pre-School Leader Age 3",
    "Pre-School Leader Age 4",
    "Pre-School Leader Age 5",
    "Elementary Leader Age 6",
    "Elementary Leader Age 7",
    "Elementary Leader Age 8",
    "uGroup Age 9",
    "uGroup Age 10",
    "uGroup Age 11",
    "Special Needs",
    "Brooklyn Babies Leader",
    "Brooklyn Pre-school Leader",
}

EXTRA_GROUP_A_ROWS = [
    "Setup",
    "Info Desk",
    "Sound",
]

EXTRA_GROUP_B_ROWS = [
    "Helping Ninja & Check in",
    "Hall Support",
]

# ──────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ──────────────────────────────────────────────────────────────────────────────

def base_max_for_person(flags):
    """Directors only serve once per service. Everyone else: 1 by default."""
    if flags.get("has_D", False):
        return 1
    return 1

def is_ukids_leader(flags):
    return flags.get("has_BL") or flags.get("has_PL") or flags.get("has_EL") or flags.get("has_SL")

# ──────────────────────────────────────────────────────────────────────────────
# Data Prep
# ──────────────────────────────────────────────────────────────────────────────

def build_long_df(people_df: pd.DataFrame, name_col: str, role_cols, codes_col: str = None):
    """Reshape form responses into long format (person, role, priority)."""
    records = []
    role_codes = {}

    for _, r in people_df.iterrows():
        person = str(r[name_col]).strip()
        if not person or person.lower() == "nan":
            continue

        # parse codes
        flags = {"has_D": False, "has_BL": False, "has_PL": False, "has_EL": False, "has_SL": False, "raw": ""}
        if codes_col and codes_col in people_df.columns:
            raw = str(r.get(codes_col, "") or "")
            flags["raw"] = raw
            toks = re.findall(r"[A-Za-z]+", raw.upper())
            for t in toks:
                if t == "D":
                    flags["has_D"] = True
                elif t == "BL":
                    flags["has_BL"] = True
                elif t == "PL":
                    flags["has_PL"] = True
                elif t == "EL":
                    flags["has_EL"] = True
                elif t == "SL":
                    flags["has_SL"] = True
        role_codes[person] = flags

        # iterate role columns
        for role in role_cols:
            pr = pd.to_numeric(r[role], errors="coerce")
            if pd.isna(pr):
                continue
            pr = int(round(pr))
            if pr >= 1:
                # <<< CHANGED: Director restrictions
                if flags["has_D"]:
                    if pr != 1 or role not in DIRECTOR_ALLOWED_ROLES:
                        continue
                records.append({"person": person, "role": role, "priority": pr})

    return pd.DataFrame(records), role_codes

# ──────────────────────────────────────────────────────────────────────────────
# Scheduling Logic
# ──────────────────────────────────────────────────────────────────────────────

def assign_roles(service_dates, long_df, role_codes, availability):
    """Assign priority 1 roles first, then extras."""
    schedule_cells = defaultdict(list)
    assign_count = defaultdict(int)

    # Step 1: Assign priority 1 roles
    for d in service_dates:
        df1 = long_df[long_df.priority == 1]
        for role in sorted(df1.role.unique()):
            cands = df1[df1.role == role].person.tolist()
            cands = [p for p in cands if availability.get(p, {}).get(d, False)]
            cands = [p for p in cands if assign_count.get(p, 0) < base_max_for_person(role_codes[p])]
            if not cands:
                continue
            cands.sort(key=lambda nm: assign_count.get(nm, 0))
            chosen = cands[0]
            schedule_cells[(role, d)].append(chosen)
            assign_count[chosen] += 1

    # Step 2: Extras Group A
    assign_extra_group_a(service_dates, schedule_cells, assign_count, availability, role_codes)

    # Step 3: Extras Group B
    served_p1_people = {p for (role, d), names in schedule_cells.items() for p in names}
    assign_extra_group_b(service_dates, schedule_cells, assign_count, availability, role_codes, served_p1_people)

    return schedule_cells

def assign_extra_group_a(service_dates, schedule_cells, assign_count, availability, role_codes):
    for d in service_dates:
        assigned_today = set(nm for (rn, dd), names in schedule_cells.items() if dd == d for nm in names)
        for row_name in EXTRA_GROUP_A_ROWS:
            key = (row_name, d)
            if len(schedule_cells[key]) >= 1:
                continue
            cands = []
            for p in availability.keys():
                flags = role_codes.get(p, {})
                # <<< NEW: Directors cannot serve in Group A
                if flags.get("has_D", False):
                    continue
                if not availability.get(p, {}).get(d, False):
                    continue
                if p in assigned_today:
                    continue
                base = base_max_for_person(flags)
                if assign_count.get(p, 0) >= base + 1:
                    continue
                cands.append(p)
            cands.sort(key=lambda nm: assign_count.get(nm, 0))
            if cands:
                chosen = cands[0]
                schedule_cells[key].append(chosen)
                assign_count[chosen] += 1

def assign_extra_group_b(service_dates, schedule_cells, assign_count, availability, role_codes, served_p1_people):
    for d in service_dates:
        assigned_today = set(nm for (rn, dd), names in schedule_cells.items() if dd == d for nm in names)
        for row_name in EXTRA_GROUP_B_ROWS:
            key = (row_name, d)
            if len(schedule_cells[key]) >= 1:
                continue
            cands = []
            for p in availability.keys():
                flags = role_codes.get(p, {})
                # <<< NEW: Directors cannot serve in Group B
                if flags.get("has_D", False):
                    continue
                if p not in served_p1_people:
                    continue
                if not availability.get(p, {}).get(d, False):
                    continue
                if p in assigned_today:
                    continue
                base = base_max_for_person(flags)
                if assign_count.get(p, 0) >= base:
                    continue
                if "Helping Ninja & Check in" in row_name and not is_ukids_leader(flags):
                    continue
                cands.append(p)
            cands.sort(key=lambda nm: assign_count.get(nm, 0))
            if cands:
                chosen = cands[0]
                schedule_cells[key].append(chosen)
                assign_count[chosen] += 1

# ──────────────────────────────────────────────────────────────────────────────
# Example usage (adjust this for your dataframes)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example: load CSVs here
    # people_df = pd.read_csv("Untitled form (Responses).csv")
    # roles_df = pd.read_csv("Serving Positions VC weergawe.csv")
    pass
