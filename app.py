"""CivicPulse AI — output-grounded Streamlit dashboard."""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).parent; DATA = ROOT / "data" / "civicpulse_dataset.csv"; OUT = ROOT / "outputs"; MODELS = ROOT / "models"
st.set_page_config("CivicPulse AI", "CP", layout="wide", initial_sidebar_state="expanded")

def stamp(p): return p.stat().st_mtime_ns if p.exists() else 0
@st.cache_data(show_spinner="Loading verified project data...")
def load_data(_):
    if not DATA.exists(): return None, "Dataset cache is missing. Run the notebook to regenerate it."
    try:
        cols=["unique_key","created_date","agency","agency_name","complaint_type","descriptor","borough","status","latitude","longitude","resolution_hours"]
        d=pd.read_csv(DATA,usecols=lambda c:c in cols,low_memory=False); d.created_date=pd.to_datetime(d.created_date,errors="coerce")
        d.borough=d.borough.fillna("Unknown").replace({"Unspecified":"Unknown"}); d.agency_name=d.agency_name.fillna(d.agency).fillna("Unknown")
        return d,None
    except Exception as e: return None,f"Could not load project data: {e}"
@st.cache_data(show_spinner=False)
def csv(path,_):
    try: return pd.read_csv(path) if Path(path).exists() else pd.DataFrame()
    except Exception: return pd.DataFrame()
@st.cache_resource(show_spinner=False)
def models(_,__):
    try: return joblib.load(MODELS/"nlp_model.joblib"),joblib.load(MODELS/"tfidf_vectorizer.joblib")
    except Exception: return None,None

data,error=load_data(stamp(DATA)); metrics=csv(OUT/"final_metrics.csv",stamp(OUT/"final_metrics.csv")); ups=csv(OUT/"urban_problem_scores.csv",stamp(OUT/"urban_problem_scores.csv")); hotspots=csv(OUT/"hotspot_summary.csv",stamp(OUT/"hotspot_summary.csv")); forecast=csv(OUT/"prediction_results.csv",stamp(OUT/"prediction_results.csv"))
def metric(name,default="Not available"):
    if metrics.empty: return default
    x=metrics.loc[metrics.Metric.astype(str).eq(name),"Value"]
    return str(x.iloc[0]) if not x.empty else default

with st.sidebar:
    st.markdown("## CivicPulse AI"); st.caption("AI-Powered Urban Problem Intelligence Engine")
    light=st.toggle("Light mode"); page=st.radio("Workspace",["Overview","Data Explorer","Problem Analytics","Temporal Trends","NLP Classification","Hotspots","Urban Problem Score","Forecast","Methodology"])
    if data is not None:
        dates=data.created_date.dropna(); dr=st.date_input("Created date",(dates.min().date(),dates.max().date()),dates.min().date(),dates.max().date())
        borough_values=sorted(data.loc[data.borough.ne("Unknown"),"borough"].unique()); borough=st.multiselect("Borough",borough_values,borough_values)
        agency_values=sorted(data.agency_name.unique()); agency=st.multiselect("Agency",agency_values,agency_values)
    st.caption("AICTE | IBM SkillsBuild 2026 · BharatCares\n\nAuthor: Riya Mandal · Local execution only")

bg,panel,ink,muted=("#f6f8fb","#fff","#102a43","#52606d") if light else ("#0b1220","#121c2d","#e6edf7","#9fb0c3")
st.markdown(
    f"""
    <style>
    .stApp {{
        background: {bg};
        color: {ink};
    }}

    .block-container {{
        padding-top: 1.5rem;
        max-width: 1500px;
    }}

    .ey {{
        font-size: .72rem;
        letter-spacing: .12em;
        font-weight: 700;
        color: #38bdf8;
    }}

    .h {{
        font-size: 2rem;
        font-weight: 750;
        color: {ink};
        margin: .15rem 0;
    }}

    .s {{
        color: {muted};
        margin-bottom: 1rem;
    }}

    .card {{
        background: {panel};
        border: 1px solid #28405e;
        border-top: 3px solid #38bdf8;
        border-radius: 14px;
        padding: 1rem;
        min-height: 110px;
    }}

    .l {{
        font-size: .75rem;
        color: {muted};
        text-transform: uppercase;
    }}

    .v {{
        font-size: 1.6rem;
        font-weight: 750;
        color: {ink};
    }}

    .c {{
        font-size: .82rem;
        color: {muted};
    }}
    </style>
    """,
    unsafe_allow_html=True
)
def title(ey,title,subtitle): st.markdown(f'<div class="ey">{ey}</div><div class="h">{title}</div><div class="s">{subtitle}</div>',unsafe_allow_html=True)
if error: st.error(error); st.stop()
view=data[data.created_date.dt.date.between(*dr)] if len(dr)==2 else data
view=view[view.borough.isin(borough)&view.agency_name.isin(agency)]; template="plotly_white" if light else "plotly_dark"
def download(label,df,name): st.download_button(label,df.to_csv(index=False).encode(),name,"text/csv")

if page=="Overview":
    title("CITY OPERATIONS CONTROL TOWER","CivicPulse AI","Observed NYC 311 service-request patterns in a reproducible controlled public-data batch.")
    facts=[("Requests analyzed",f"{len(view):,}","Active filters"),("Date window",f"{view.created_date.min():%d %b}–{view.created_date.max():%d %b}","Created-date range"),("Boroughs covered",str(view.borough[view.borough.ne('Unknown')].nunique()),"Reported coverage"),("NLP accuracy",metric("Accuracy"),"Description-disjoint holdout")]
    for col,(a,b,c) in zip(st.columns(4),facts): col.markdown(f'<div class="card"><div class="l">{a}</div><div class="v">{b}</div><div class="c">{c}</div></div>',unsafe_allow_html=True)
    a,b=st.columns(2); cats=view.complaint_type.value_counts().head(10).sort_values().reset_index(name="Requests"); monthly=view.set_index("created_date").resample("ME").size().reset_index(name="Requests")
    a.plotly_chart(px.bar(cats,x="Requests",y="complaint_type",orientation="h",color="Requests",color_continuous_scale="Teal",template=template,title="Leading reported problem categories"),use_container_width=True)
    b.plotly_chart(px.line(monthly,x="created_date",y="Requests",markers=True,template=template,title="Monthly service-request volume"),use_container_width=True)
    with st.expander("Scope and limitations"): st.write("Reporting volume is not verified defect severity. UPS is a project-defined index, not an official government rating. Patterns are descriptive and do not establish causality.")
elif page=="Data Explorer":
    title("DATA EXPLORER","Search service requests","Paginated records use all active global filters."); q=st.text_input("Search complaint type, descriptor, or borough")
    if q: view=view[view[["complaint_type","descriptor","borough"]].fillna("").astype(str).apply(lambda s:s.str.contains(q,case=False,regex=False)).any(axis=1)]
    n=st.selectbox("Rows per page",[25,50,100],index=1); p=st.number_input("Page",1,max(1,int(np.ceil(len(view)/n))),1)
    st.dataframe(view[["unique_key","created_date","borough","agency_name","complaint_type","descriptor","status"]].iloc[(p-1)*n:p*n],use_container_width=True,hide_index=True); download("Download filtered records",view,"civicpulse_filtered.csv")
elif page=="Problem Analytics":
    title("COMPLAINT INTELLIGENCE","What is being reported?","Comparisons reflect active global filters."); a,b=st.columns(2)
    cats=view.complaint_type.value_counts().head(15).sort_values().reset_index(name="Requests"); ag=view.agency_name.value_counts().head(12).reset_index(name="Requests")
    a.plotly_chart(px.bar(cats,x="Requests",y="complaint_type",orientation="h",color="Requests",color_continuous_scale="Teal",template=template,title="Top categories"),use_container_width=True); b.plotly_chart(px.bar(ag,x="agency_name",y="Requests",color="Requests",color_continuous_scale="Teal",template=template,title="Agency workload"),use_container_width=True)
elif page=="Temporal Trends":
    title("TEMPORAL PATTERNS","When are requests observed?","Descriptive trends—not causal claims."); daily=view.set_index("created_date").resample("D").size().reset_index(name="Requests")
    st.plotly_chart(px.line(daily,x="created_date",y="Requests",template=template,title="Daily request volume"),use_container_width=True)
    order=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]; days=view.created_date.dt.day_name().value_counts().reindex(order).reset_index(name="Requests")
    st.plotly_chart(px.bar(days,x="created_date",y="Requests",color="Requests",color_continuous_scale="Teal",template=template,title="Requests by day of week"),use_container_width=True)
elif page=="NLP Classification":
    title("COMPLAINT INTELLIGENCE","NLP classification",f"TF-IDF + Logistic Regression · Accuracy {metric('Accuracy')} · Macro F1 {metric('Macro F1-Score')}"); text=st.text_area("Complaint description","Large deep pothole on the roadway causing damage to passing cars")
    model,vector=models(stamp(MODELS/"nlp_model.joblib"),stamp(MODELS/"tfidf_vectorizer.joblib"))
    if st.button("Classify complaint",type="primary"):
        if model is None: st.warning("Trained model files are unavailable. Run the notebook to regenerate them.")
        elif not text.strip(): st.info("Enter a complaint description.")
        else:
            prob=model.predict_proba(vector.transform([text]))[0]; rank=pd.DataFrame({"Category":model.classes_,"Probability":prob*100}).nlargest(5,"Probability"); st.success(f"Predicted category: {rank.iloc[0].Category} ({rank.iloc[0].Probability:.1f}% model confidence)"); st.plotly_chart(px.bar(rank.sort_values("Probability"),x="Probability",y="Category",orientation="h",template=template,title="Top model probabilities"),use_container_width=True)
    st.caption("Predictions are limited to selected training categories; model confidence is not operational certainty.")
elif page == "Hotspots":
    title(
        "SPATIAL INTELLIGENCE",
        "Reported request concentrations",
        f"DBSCAN detected {metric('Detected Clusters')} clusters in the project sample; settings are dataset-specific."
    )

    if hotspots.empty:
        st.warning(
            "Hotspot output is unavailable. Run the notebook to generate it."
        )

    else:
        hotspot_map = hotspots.copy()

        # Convert coordinates and counts to numeric
        hotspot_map["avg_latitude"] = pd.to_numeric(
            hotspot_map["avg_latitude"],
            errors="coerce"
        )

        hotspot_map["avg_longitude"] = pd.to_numeric(
            hotspot_map["avg_longitude"],
            errors="coerce"
        )

        hotspot_map["complaint_count"] = pd.to_numeric(
            hotspot_map["complaint_count"],
            errors="coerce"
        )

        # Remove invalid records
        hotspot_map = hotspot_map.dropna(
            subset=[
                "avg_latitude",
                "avg_longitude",
                "complaint_count"
            ]
        ).copy()

        # Keep valid NYC coordinates
        hotspot_map = hotspot_map[
            hotspot_map["avg_latitude"].between(40.45, 40.95)
            &
            hotspot_map["avg_longitude"].between(-74.30, -73.65)
        ].copy()

        if hotspot_map.empty:

            st.error(
                "DBSCAN clusters were found, but no valid NYC coordinates "
                "are available for visualization."
            )

        else:

            # Fixed marker size so every cluster is clearly visible
            hotspot_map["marker_size"] = 18

            fig = px.scatter_map(
                hotspot_map,
                lat="avg_latitude",
                lon="avg_longitude",
                color="primary_borough",
                size="marker_size",
                size_max=22,
                hover_name="primary_complaint",
                hover_data=[
                    "avg_latitude",
                    "avg_longitude",
                    "complaint_count",
                    "primary_borough"
                ],
                center={
                    "lat": 40.7128,
                    "lon": -74.0060
                },
                zoom=9,
                height=560,
                map_style="open-street-map",
                title="DBSCAN Cluster Centroids"
            )

            # Make markers highly visible
            fig.update_traces(
                marker={
                    "size": 18,
                    "opacity": 1
                }
            )

            fig.update_layout(
                margin={
                    "l": 0,
                    "r": 0,
                    "t": 55,
                    "b": 0
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.success(
                f"{len(hotspot_map)} DBSCAN cluster centroids plotted successfully."
            )

            st.dataframe(
                hotspot_map.head(30),
                use_container_width=True,
                hide_index=True
            )

            download(
                "Download hotspot summary",
                hotspots,
                "civicpulse_hotspots.csv"
            )
elif page=="Urban Problem Score":
    title("PRIORITIZATION INDEX","Urban Problem Score","UPS = 0.40 volume + 0.25 recent growth + 0.35 resolution. It is an explainable project-defined comparison index.")
    if ups.empty: st.warning("UPS output is unavailable. Run the notebook to generate it.")
    else: st.plotly_chart(px.bar(ups.sort_values("urban_problem_score"),x="urban_problem_score",y="borough",orientation="h",color="urban_problem_score",color_continuous_scale="Teal",text="urban_problem_score",template=template,title="UPS by borough"),use_container_width=True); st.dataframe(ups,use_container_width=True,hide_index=True); download("Download UPS table",ups,"urban_problem_scores.csv")
elif page=="Forecast":
    title("PREDICTIVE ANALYTICS","Daily request-volume estimate",f"Chronological holdout · MAE {metric('Mean Absolute Error (MAE)')} · RMSE {metric('Root Mean Squared Error (RMSE)')} · R² {metric('R² Score')}.")
    if forecast.empty: st.warning("Forecast output is unavailable. Run the notebook to generate it.")
    else:
        fig=go.Figure([go.Scatter(x=forecast.Date,y=forecast.Actual,mode="lines+markers",name="Actual"),go.Scatter(x=forecast.Date,y=forecast.Predicted,mode="lines+markers",name="Predicted",line={"dash":"dash"})]); fig.update_layout(template=template,height=460,title="Actual vs predicted daily requests",xaxis_title="Date",yaxis_title="Requests"); st.plotly_chart(fig,use_container_width=True); st.info(f"Next observed-day estimate: {metric('Estimated Requests')} requests. This is an estimate, not a guarantee."); download("Download forecast results",forecast,"civicpulse_forecast.csv")
else:
    title("METHODS & GOVERNANCE","About CivicPulse AI","Reproducible analytical workflow for a controlled NYC 311 public-data batch."); st.markdown("**Pipeline:** collection → quality checks → EDA → NLP classification → DBSCAN → UPS → chronological Random Forest forecasting → grounded insights."); st.markdown("**Data source:** NYC OpenData, 311 Service Requests from 2020 to Present (`erm2-nwe9`). Local execution only; no deployment URL is claimed.")
