"""
DETECT data stream missingness report.

For every subject, checks which data streams they have, how many days (or
surveys/visits) were expected, how many were observed, and where the gaps are.
Writes an Excel workbook to Missingness_Report/ next to this script.

Expected window, per subject per stream: from the subject's first observed
record to their last observed record in that stream, inclusive. Data from
2024-01-01 to 2024-12-31 expects 366 days. Enrollment and device dates are
not used.
  Daily sensor streams  every day in that span
  SMART                 every calendar month in that span
  HUF                   every week (Mon to Sun) in that span
  UDS3 / Amyloid        counted, no expected window

Usage:
    python missingness_report.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "Missingness_Report"
CTX = ROOT / "Sensor_Data" / "_CONTEXT_FILES"
CTX_TAG = "2025-07-16"

PATHS = {
    "enroll": CTX / f"Study_Subject-Enrollment_Dates_{CTX_TAG}" / f"DETECT_Subject-Enrollment_Dates_Through_{CTX_TAG}.csv",
    "home_subj": CTX / f"Study_Home-Subject_Dates_{CTX_TAG}" / f"DETECT_Home-Subject_Dates_Through_{CTX_TAG}.csv",
    "smart": ROOT / "SMART" / "DETECT_smart_cleaned.csv",
    "huf": ROOT / "HUF" / "DETECT_huf_var_2026-07-28.csv",
    "uds3": ROOT / "Clinical_Data" / "detect_uds3_full_2.csv",
    "amyloid": ROOT / "Clinical_Data" / "DETECT-AD_Enrolled_Amyloid Status_PET_SUVR_QUEST_CENTILOID_20250116.xlsx",
    "uds3_dict": ROOT / "Clinical_Data" / "detect_dd 1.xlsx",
    "apoe": ROOT / "Clinical_Data" / "DETECT_APOE.xlsx",
    "emfit": ROOT / "Sensor_Data" / "Emfit_Data" / "Summary" / "Emfit_Summary_Data.csv",
    "nyce": ROOT / "Sensor_Data" / "NYCE_Date_Processed" / "nyce_daily_presence.csv",
    "scale": ROOT / "Sensor_Data" / "Scale_Data" / "Scale_Raw_Data_DETECT_2025-07-16.csv",
    "worktime": ROOT / "Sensor_Data" / "Worktime_Data" / "Processed_Daily" / "WorkTime_Processed_Daily_Data_DETECT_2025-07-16.csv",
    "watch_sleep": ROOT / "Sensor_Data" / "Watch_Data" / "Combined" / "Watch_Sleep_Data_Combined_DETECT_2025-07-16.csv",
    "watch_steps": ROOT / "Sensor_Data" / "Watch_Data" / "Combined" / "Watch_Daily_Steps_Combined_DETECT_2025-07-16.csv",
}

DAILY_STREAMS = ["Watch_Sleep", "Watch_Steps", "Emfit", "Scale", "NYCE", "Worktime"]
SURVEY_STREAMS = {"SMART": "month", "HUF": "week"}
ALL_STREAMS = DAILY_STREAMS + list(SURVEY_STREAMS) + ["UDS3", "Amyloid", "APOE"]

# Yearly clinical assessments: UDS3 data-dictionary form name -> report label.
# An assessment counts as done at a visit if any of its form's columns has a value.
YEARLY_ASSESSMENTS = {
    "nacc_uds_c2_moca": "MoCA",
    "nacc_uds3_c2_neuropsych_battery_required": "UDS3 neuropsych battery (C2)",
    "cerad_word_list_task": "CERAD word list",
    "digit_span_sequencing": "Digit span sequencing",
    "stroop": "Stroop",
    "waisr_digit_symbol": "WAIS-R digit symbol",
    "waisr_block_design": "WAIS-R block design",
    "adco_visual_reproduction_scoring_summary": "Visual reproduction",
    "fcsrt": "FCSRT",
    "adco_mcirs": "MCIRS",
    "modified_updrs_adsc_version": "Modified UPDRS",
    "nacc_uds3_b4_cdr_dementia_staging_instrument_plus": "CDR",
    "nacc_uds3_b9_clinician_judgement_of_symptoms_requi": "Clinician judgment (B9)",
    "nacc_uds3_d1_clinician_diagnosis_required": "Clinician diagnosis (D1)",
}
VISITS = ["Initial", "FU 1", "FU 2"]
ONE_TIME = ["Amyloid PET", "QUEST blood test", "APOE genotype"]

GAP_FLAG_DAYS = 7  # a run of missing days at least this long counts as a "long gap"


# ---------------------------------------------------------------- helpers

def to_day(s):
    return pd.to_datetime(s, errors="coerce").dt.normalize()


def gap_stats(observed_dates):
    """Longest run, and number of long runs, of missing days between observed days."""
    days = np.array(sorted(observed_dates), dtype="datetime64[D]")
    gaps = np.diff(days).astype(int) - 1  # missing days between consecutive observations
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return 0, 0
    return int(gaps.max()), int((gaps >= GAP_FLAG_DAYS).sum())


# ---------------------------------------------------------------- loaders

def load_context():
    enroll = pd.read_csv(PATHS["enroll"]).rename(columns={"sub_id": "subid"})
    enroll["enroll_start"] = to_day(enroll["enroll_start"])
    enroll["enroll_end"] = to_day(enroll["enroll_end"])

    # Home-Subject dates are still needed to attribute home-level scale data to subjects
    hs = pd.read_csv(PATHS["home_subj"]).rename(columns={"sub_id": "subid", "home_id": "homeid"})
    hs["start"], hs["end"] = to_day(hs["start_date"]), to_day(hs["end_date"])
    return enroll, hs


def subject_home_days(hs):
    rows = [pd.DataFrame({"subid": r["subid"], "homeid": r["homeid"],
                          "date": pd.date_range(r["start"], r["end"])})
            for r in hs.dropna(subset=["start", "end"]).to_dict("records")]
    return pd.concat(rows, ignore_index=True).drop_duplicates()


def observed_daily(hs):
    """Return {stream: DataFrame(subid, date)} of days with usable data, plus extra flags."""
    obs, extras = {}, {}

    ws = pd.read_csv(PATHS["watch_sleep"], usecols=["subid", "date", "device"])
    wt = pd.read_csv(PATHS["watch_steps"], usecols=["subid", "date", "steps", "device"])
    wt = wt[wt["steps"] > 0]
    for name, df in [("Watch_Sleep", ws), ("Watch_Steps", wt)]:
        df = df.assign(date=to_day(df["date"]))
        obs[name] = df[["subid", "date"]].drop_duplicates()
        dev = df.groupby("subid")["device"].agg(
            device1_days=lambda s: (s == 1).sum(), device2_days=lambda s: (s == 2).sum())
        extras[name] = dev

    em = pd.read_csv(PATHS["emfit"], usecols=["subid", "date"])
    obs["Emfit"] = em.assign(date=to_day(em["date"])).drop_duplicates()

    ny = pd.read_csv(PATHS["nyce"], usecols=["subid", "date", "nyce_outage_day", "num_people_in_home"])
    ny = ny.dropna(subset=["subid"]).assign(subid=lambda d: d["subid"].astype(int),
                                            date=lambda d: to_day(d["date"]))
    obs["NYCE"] = ny.loc[ny["nyce_outage_day"] == 0, ["subid", "date"]].drop_duplicates()
    extras["NYCE"] = ny.groupby("subid").agg(
        outage_days=("nyce_outage_day", "sum"),
        multi_person_home_days=("num_people_in_home", lambda s: int((s > 1).sum())))

    # Scale is recorded per home; attribute each weigh-in to everyone living there that day
    sc = pd.read_csv(PATHS["scale"], usecols=["homeid", "date", "weight"])
    sc = sc.dropna(subset=["weight"]).assign(date=lambda d: to_day(d["date"]))
    sc = sc[["homeid", "date"]].drop_duplicates()
    shd = subject_home_days(hs)
    sc_sub = sc.merge(shd, on=["homeid", "date"])
    residents = shd.groupby(["homeid", "date"])["subid"].nunique().rename("n_res").reset_index()
    sc_sub = sc_sub.merge(residents, on=["homeid", "date"])
    obs["Scale"] = sc_sub[["subid", "date"]].drop_duplicates()
    extras["Scale"] = sc_sub.groupby("subid").agg(
        shared_home_days=("n_res", lambda s: int((s > 1).sum())))
    extras["_scale_unmapped_home_days"] = len(sc) - len(sc.merge(shd[["homeid", "date"]].drop_duplicates(),
                                                                on=["homeid", "date"]))

    wk = pd.read_csv(PATHS["worktime"], usecols=["subid", "date"])
    obs["Worktime"] = wk.assign(date=to_day(wk["date"])).drop_duplicates()
    return obs, extras


def load_surveys():
    s = pd.read_csv(PATHS["smart"], usecols=["subid", "StartDate", "Finished"], low_memory=False)
    h = pd.read_csv(PATHS["huf"], usecols=["subid", "StartDate", "Finished"], low_memory=False)
    out = {}
    for name, df in [("SMART", s), ("HUF", h)]:
        df = df.assign(date=to_day(df["StartDate"]),
                       finished=df["Finished"].astype(str).str.lower().isin(["1", "true", "1.0"]))
        out[name] = df[["subid", "date", "finished"]]
    return out


def load_clinical():
    """Return per-subject UDS3 summary, one-time results, and a visit x assessment table."""
    u = pd.read_csv(PATHS["uds3"], low_memory=False)
    u["visit_date"] = pd.to_datetime(u["visit_date"], errors="coerce")
    uds = u.groupby("subid").agg(
        uds3_n_visits=("detect_visit", "size"),
        uds3_visits=("detect_visit", lambda s: ";".join(s.astype(str))),
        uds3_first_visit=("visit_date", "min"),
        uds3_last_visit=("visit_date", "max"))

    dd = pd.read_excel(PATHS["uds3_dict"]).iloc[:, :2]
    dd.columns = ["var", "form"]
    form_cols = dd[dd["var"].isin(u.columns)].groupby("form")["var"].apply(list)
    visits = u[["subid", "detect_visit", "visit_date"]].copy()
    for form, label in YEARLY_ASSESSMENTS.items():
        visits[label] = u[form_cols[form]].notna().any(axis=1)

    a = pd.read_excel(PATHS["amyloid"]).rename(columns={"sub_id": "subid"})
    quest = [c for c in a.columns if c.startswith("QUEST")][0]
    amy = pd.DataFrame({
        "subid": a["subid"],
        "amyloid_has_pet": a["suvr"].notna() | a["centiloid"].notna(),
        "amyloid_read": a["clinical amyloid (+/-) read"],
        "amyloid_centiloid": a["centiloid"],
        "amyloid_has_quest": a[quest].notna(),
        "amyloid_quest_ratio": a[quest],
        "amyloid_group": a["group"],
    }).drop_duplicates("subid").set_index("subid")

    ap = pd.read_excel(PATHS["apoe"]).rename(columns={"Record ID": "subid"})
    apoe = (ap.dropna(subset=["APOE Status"]).drop_duplicates("subid")
              .set_index("subid")[["APOE Status"]].rename(columns={"APOE Status": "apoe_status"}))
    return uds, amy, apoe, visits


def clinical_tables(visits, amy, apoe, subjects):
    """Per-assessment overview and per-subject clinical availability."""
    labels = list(YEARLY_ASSESSMENTS.values())
    n_attended = visits.groupby("subid").size()

    # Per subject: visits with each assessment, written as "done/attended"
    done = visits.groupby("subid")[labels].sum().astype(int)
    by_subj = pd.DataFrame(index=pd.Index(subjects, name="subid"))
    by_subj["uds3_visits_attended"] = n_attended.reindex(by_subj.index).fillna(0).astype(int)
    by_subj["visits"] = visits.groupby("subid")["detect_visit"].agg(
        lambda s: ";".join(v for v in VISITS if v in set(s))).reindex(by_subj.index)
    for lab in labels:
        d = done[lab].reindex(by_subj.index)
        by_subj[lab] = [f"{int(x)}/{n}" if n else "-" for x, n in zip(d.fillna(0), by_subj["uds3_visits_attended"])]
    by_subj["Amyloid PET"] = by_subj.index.map(
        lambda s: amy.at[s, "amyloid_read"] if s in amy.index and amy.at[s, "amyloid_has_pet"] else "-")
    by_subj["amyloid_centiloid"] = amy["amyloid_centiloid"].reindex(by_subj.index)
    by_subj["QUEST blood test"] = by_subj.index.map(
        lambda s: round(amy.at[s, "amyloid_quest_ratio"], 3) if s in amy.index and amy.at[s, "amyloid_has_quest"] else "-")
    by_subj["APOE genotype"] = apoe["apoe_status"].reindex(by_subj.index).fillna("-")

    def missing_list(row):
        miss = []
        n = row["uds3_visits_attended"]
        if n == 0:
            miss.append("all yearly assessments (no UDS3 visit)")
        else:
            miss += [f"{lab} (missing at {n - int(row[lab].split('/')[0])} of {n} visits)"
                     for lab in labels if int(row[lab].split("/")[0]) < n]
        miss += [o for o in ONE_TIME if row[o] == "-"]
        return "; ".join(miss)
    by_subj["missing_clinical"] = by_subj.apply(missing_list, axis=1)

    # Per assessment overview
    rows = []
    attended_by_visit = visits["detect_visit"].value_counts()
    for lab in labels:
        v = visits[visits[lab]]
        row = {"assessment": lab, "type": "yearly (UDS3 visit)",
               "subjects_with_any": v["subid"].nunique(),
               "subjects_without": len(subjects) - v["subid"].nunique()}
        for vis in VISITS:
            row[f"subjects_at_{vis.replace(' ', '')}"] = int((v["detect_visit"] == vis).sum())
            row[f"missing_at_{vis.replace(' ', '')}"] = int(attended_by_visit.get(vis, 0) - (v["detect_visit"] == vis).sum())
        row["visits_with_it"] = len(v)
        row["visits_attended"] = len(visits)
        row["pct_of_attended_visits"] = round(100 * len(v) / len(visits), 1)
        rows.append(row)
    pet = amy[amy["amyloid_has_pet"]]
    one_time = {
        "Amyloid PET": (pet.index, ", ".join(f"{k}: {n}" for k, n in pet["amyloid_read"].value_counts().items())),
        "QUEST blood test": (amy.index[amy["amyloid_has_quest"]], ""),
        "APOE genotype": (apoe.index, ", ".join(f"{k}: {n}" for k, n in apoe["apoe_status"].value_counts().items())),
    }
    for lab, (idx, breakdown) in one_time.items():
        rows.append({"assessment": lab, "type": "one-time",
                     "subjects_with_any": len(set(idx) & set(subjects)),
                     "subjects_without": len(set(subjects) - set(idx)),
                     "result_breakdown": breakdown})
    overview = pd.DataFrame(rows)

    visit_counts = (by_subj["uds3_visits_attended"].value_counts().sort_index()
                    .rename_axis("uds3_visits_attended").reset_index(name="n_subjects"))
    return overview, by_subj, visit_counts


# ---------------------------------------------------------------- report building

def daily_detail(stream, observed, extras):
    rows = []
    for sub, dates in observed.groupby("subid")["date"]:
        o = set(dates)
        first, last = min(o), max(o)
        expected = (last - first).days + 1
        longest, n_long = gap_stats(o)
        rows.append({
            "subid": sub, "stream": stream, "unit": "day",
            "has_data": True,
            "first_date": first, "last_date": last,
            "expected": expected,
            "observed": len(o),
            "missing": expected - len(o),
            "pct_complete": round(100 * len(o) / expected, 1),
            "longest_gap_days": longest,
            f"gaps_{GAP_FLAG_DAYS}plus_days": n_long,
        })
    df = pd.DataFrame(rows)
    if stream in extras:
        df = df.merge(extras[stream], left_on="subid", right_index=True, how="left")
    return df


def survey_detail(stream, unit, surveys):
    df = surveys[stream]
    freq = "M" if unit == "month" else "W-SUN"
    df = df.assign(period=df["date"].dt.to_period(freq))
    rows = []
    for sub, s in df.groupby("subid"):
        done = s[s["finished"]]
        row = {"subid": sub, "stream": stream, "unit": unit, "has_data": True,
               "n_responses": len(s), "n_unfinished": int((~s["finished"]).sum())}
        if len(done):
            # Expected periods run from the first to the last finished survey
            first, last = done["date"].min(), done["date"].max()
            expected = len(pd.period_range(first, last, freq=freq))
            got = done["period"].nunique()
            row.update({
                "first_date": first, "last_date": last,
                "expected": expected, "observed": got, "missing": expected - got,
                "pct_complete": round(100 * got / expected, 1),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_coverage(obs, surveys):
    """Number of distinct subjects contributing data to each stream, by calendar month."""
    frames = []
    for name, df in {**obs, **{k: v[v["finished"]] for k, v in surveys.items()}}.items():
        m = df.assign(month=df["date"].dt.to_period("M").astype(str))
        frames.append(m.groupby("month")["subid"].nunique().rename(name))
    return pd.concat(frames, axis=1).fillna(0).astype(int).sort_index()


def main():
    enroll, hs = load_context()
    observed, extras = observed_daily(hs)
    surveys = load_surveys()
    uds, amy, apoe, visits = load_clinical()

    detail = pd.concat(
        [daily_detail(s, observed[s], extras) for s in DAILY_STREAMS]
        + [survey_detail(s, u, surveys) for s, u in SURVEY_STREAMS.items()],
        ignore_index=True)

    # Subject universe: enrolled subjects plus anyone appearing in any stream
    all_subs = (set(enroll["subid"]) | set(detail["subid"]) | set(uds.index) | set(amy.index)
                | set(apoe.index))
    subj = pd.DataFrame({"subid": sorted(int(s) for s in all_subs)})
    subj = subj.merge(enroll[["subid", "enroll_start", "enroll_end"]], on="subid", how="left")
    subj["in_enrollment_file"] = subj["enroll_start"].notna()
    subj["enrolled_days"] = (subj["enroll_end"] - subj["enroll_start"]).dt.days + 1
    homes = hs.groupby("subid")["homeid"].agg(lambda s: ";".join(map(str, sorted(set(s)))))
    subj["homeids"] = subj["subid"].map(homes)

    has = (detail.pivot_table(index="subid", columns="stream", values="has_data", aggfunc="any")
           .reindex(subj["subid"]).astype("boolean").fillna(False).astype(bool))
    has["UDS3"] = has.index.isin(uds.index)
    has["Amyloid"] = has.index.isin(amy.index)
    has["APOE"] = has.index.isin(apoe.index)
    has = has[ALL_STREAMS]
    pct = (detail.pivot_table(index="subid", columns="stream", values="pct_complete")
           .reindex(subj["subid"])[DAILY_STREAMS + list(SURVEY_STREAMS)])

    availability = subj.set_index("subid").join(has.replace({True: "Y", False: "-"}))
    availability["n_streams"] = has.sum(axis=1)
    availability["missing_streams"] = has.apply(lambda r: ", ".join(r.index[~r]), axis=1)
    availability = availability.join(uds).join(amy).join(apoe)
    clin_overview, clin_subj, clin_visits = clinical_tables(visits, amy, apoe, subj["subid"].tolist())

    completeness = subj.set_index("subid")[["in_enrollment_file", "enrolled_days"]].join(pct)

    # Stream-level overview
    enrolled = set(enroll["subid"])
    overview = []
    for s in ALL_STREAMS:
        col = has[s]
        d = detail[detail["stream"] == s]
        row = {
            "stream": s,
            "unit": d["unit"].iloc[0] if len(d) else "record",
            "subjects_with_data": int(col.sum()),
            "enrolled_with_data": int(col[col.index.isin(enrolled)].sum()),
            "enrolled_without_data": int((~col[col.index.isin(enrolled)]).sum()),
            "with_data_not_in_enrollment": int(col[~col.index.isin(enrolled)].sum()),
        }
        if len(d):
            e = d[d["expected"] > 0]
            row.update({
                "total_expected": int(e["expected"].sum()),
                "total_observed": int(e["observed"].sum()),
                "total_missing": int(e["missing"].sum()),
                "overall_pct_complete": round(100 * e["observed"].sum() / max(e["expected"].sum(), 1), 1),
                "median_subject_pct": e["pct_complete"].median(),
                "subjects_under_50pct": int((e["pct_complete"] < 50).sum()),
                "median_subject_span": e["expected"].median(),
                "first_date": d["first_date"].min(), "last_date": d["last_date"].max(),
            })
        overview.append(row)
    overview = pd.DataFrame(overview)

    # How often each combination of streams occurs
    combos = (has.apply(lambda r: " + ".join(r.index[r]) or "(none)", axis=1)
              .value_counts().rename_axis("streams_present").reset_index(name="n_subjects"))
    combos["n_streams"] = combos["streams_present"].map(
        lambda c: 0 if c == "(none)" else c.count("+") + 1)

    # Pairwise overlap: subjects having both stream A and stream B
    hv = has.astype(int)
    overlap = hv.T.dot(hv)

    notes = pd.DataFrame({"note": [
        f"Generated from {ROOT}",
        "expected = days from the subject's first observed date to last observed date in that stream, inclusive (2024-01-01 to 2024-12-31 = 366). Enrollment and device dates are not used.",
        "observed = distinct days with data in that span. missing = expected - observed. pct_complete = observed / expected x 100.",
        "Because the span is set by the data, the first and last days are always observed, and a subject who stopped early is not counted as missing after they stopped.",
        "Watch_Steps counts a day only if steps > 0. Watch combined file uses device 2 (Pulse HR) when present.",
        "Emfit: multiple sessions on one date count as one day.",
        "Scale is recorded per home. A weigh-in is credited to every subject living in that home that day (Home-Subject dates); see shared_home_days.",
        f"Scale home-days with a weigh-in but no resident on record (dropped): {extras['_scale_unmapped_home_days']}.",
        "NYCE: outage days (nyce_outage_day = 1) count as missing.",
        "SMART: expected = calendar months from first to last finished survey. A month counts as observed if it has at least one finished survey.",
        "HUF: expected = weeks (Mon to Sun) from first to last finished survey. A week counts as observed if it has at least one finished survey.",
        f"longest_gap_days = longest run of missing days between two observed days. gaps_{GAP_FLAG_DAYS}plus_days counts runs of {GAP_FLAG_DAYS}+ days.",
        "Enrollment dates are shown on Subject_Availability for reference only.",
        "UDS3, Amyloid and APOE have no expected window; availability and visit counts only.",
        "Clinical_Overview / Clinical_By_Subject split UDS3 into its yearly assessments (MoCA, CDR, neuropsych tests, diagnosis) using the UDS3 data dictionary form names. An assessment counts as done at a visit if any of its form's fields has a value.",
        "Yearly assessment missingness is measured against UDS3 visits the subject attended (Initial, FU 1, FU 2). Cells read done/attended, e.g. 2/3 = done at 2 of 3 visits.",
        "One-time measures: Amyloid PET (suvr or centiloid present), QUEST amyloid blood test, APOE genotype (DETECT_APOE.xlsx).",
    ]})

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"DETECT_Missingness_Report_{CTX_TAG}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl", datetime_format="yyyy-mm-dd") as xw:
        notes.to_excel(xw, sheet_name="README", index=False)
        overview.to_excel(xw, sheet_name="Stream_Overview", index=False)
        availability.to_excel(xw, sheet_name="Subject_Availability")
        completeness.to_excel(xw, sheet_name="Subject_Pct_Complete")
        detail.sort_values(["subid", "stream"]).to_excel(xw, sheet_name="Subject_Stream_Detail", index=False)
        clin_overview.to_excel(xw, sheet_name="Clinical_Overview", index=False)
        clin_subj.to_excel(xw, sheet_name="Clinical_By_Subject")
        clin_visits.to_excel(xw, sheet_name="Clinical_Visit_Counts", index=False)
        combos.to_excel(xw, sheet_name="Stream_Combinations", index=False)
        overlap.to_excel(xw, sheet_name="Stream_Overlap")
        monthly_coverage(observed, surveys).to_excel(xw, sheet_name="Monthly_Subjects")
        for ws in xw.sheets.values():
            ws.freeze_panes = "B2"
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
    detail.to_csv(OUT_DIR / f"DETECT_Missingness_Detail_{CTX_TAG}.csv", index=False)

    print(overview[["stream", "subjects_with_data", "enrolled_without_data", "overall_pct_complete",
                    "median_subject_pct"]].to_string(index=False))
    print()
    print(clin_overview[["assessment", "subjects_with_any", "subjects_without"]].to_string(index=False))
    print(f"\nSubjects: {len(subj)} total, {subj['in_enrollment_file'].sum()} in enrollment file")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
