"""Streamlit Community Cloud entrypoint for the hosted demo branch."""

from pathlib import Path
import runpy

import streamlit as st

APP = Path(__file__).resolve().parent / "demo" / "app.py"
runpy.run_path(str(APP), run_name="__main__")

# Hosted visual layer: industrial instrumentation rather than the teal/navy
# security-dashboard language used elsewhere in the portfolio.
st.markdown(
    """
<style>
:root{--graphite:#171817;--graphite-2:#22231f;--steel:#a7aaa3;--cream:#e8e3d7;--amber:#e2a63b;--amber-soft:#f4d28a;--line:#3b3b34;}
[data-testid="stAppViewContainer"]{background:linear-gradient(180deg,#171817 0%,#1c1d1a 100%)!important;color:var(--cream)!important;}
[data-testid="stSidebar"]{background:#24241f!important;border-right:1px solid #45443b!important;}
[data-testid="stSidebar"] *{color:#d8d2c5!important;}
.block-container{max-width:1320px!important;padding-top:1.8rem!important;}
h1,h2,h3,h4{color:#eee8dc!important;font-family:'Arial Narrow',Arial,sans-serif!important;letter-spacing:.015em!important;}
.hero{background:linear-gradient(110deg,#27271f,#1e1f1c)!important;border:1px solid #555142!important;border-left:5px solid var(--amber)!important;border-radius:5px!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.025)!important;}
.hero h1{font-size:2rem!important;text-transform:uppercase!important;letter-spacing:.045em!important;}
.hero p{color:#bcb6a9!important;max-width:900px!important;}
.pill{border:1px solid #625d4f!important;background:#302f28!important;color:#e7c277!important;border-radius:3px!important;text-transform:uppercase!important;letter-spacing:.05em!important;font-size:.7rem!important;}
.explain-card{background:#22231f!important;border:1px solid #414139!important;border-radius:4px!important;box-shadow:none!important;min-height:108px!important;}
.explain-kicker{color:#e2a63b!important;font-family:monospace!important;letter-spacing:.12em!important;}
.explain-card h4{color:#e8e3d7!important;}.explain-card p,.section-hint,.small-note{color:#aaa69b!important;}
.risk-card{background:repeating-linear-gradient(-45deg,#24251f,#24251f 12px,#26271f 12px,#26271f 24px)!important;border:1px solid #565246!important;border-top:4px solid #e2a63b!important;border-radius:4px!important;}
[data-testid="stMetric"]{background:#22231f!important;border:1px solid #414139!important;border-radius:3px!important;padding:12px!important;}
[data-testid="stMetricLabel"]{color:#aaa69b!important;text-transform:uppercase!important;letter-spacing:.045em!important;font-size:.74rem!important;}
[data-testid="stMetricValue"]{color:#f0eadf!important;font-family:ui-monospace,SFMono-Regular,Menlo,monospace!important;}
.stButton>button{background:#38362f!important;color:#f1e7d5!important;border:1px solid #5a5549!important;border-radius:3px!important;text-transform:uppercase!important;letter-spacing:.035em!important;}
.stButton>button:hover{background:#4a4435!important;border-color:#e2a63b!important;color:#fff5df!important;}
[data-baseweb="select"]>div, textarea,input{background:#22231f!important;color:#e9e2d5!important;border:1px solid #4a4940!important;border-radius:3px!important;}
[data-testid="stFileUploader"] section{background:#20211d!important;border-color:#575348!important;border-radius:3px!important;}
[data-testid="stExpander"]{background:#20211d!important;border:1px solid #414139!important;border-radius:3px!important;}
hr{border-color:#3d3d35!important;}
[data-testid="stDataFrame"]{border:1px solid #44433a!important;border-radius:2px!important;overflow:hidden!important;}
[data-testid="stCaptionContainer"]{color:#9f9b91!important;}
/* Plotly canvases should feel like instrument panels. */
.js-plotly-plot .plotly .bg{fill:#1f201c!important;}.js-plotly-plot text{fill:#b9b4a8!important;}.js-plotly-plot .gridlayer path{stroke:#35362f!important;}
</style>
""",
    unsafe_allow_html=True,
)
