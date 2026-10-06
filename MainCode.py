```python
import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KPAX Weekly Investment Engine",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# VERSION
# ============================================================

VERSION = "KPAX V2.0"


# ============================================================
# DEFAULT WEIGHTS
# ============================================================

DEFAULT_WEIGHTS = {
    "kpax": 0.35,
    "kpax_fv": 0.40,
    "dip": 0.15,
    "risk": 0.10,
}


# ============================================================
# DEFAULT KPAX UNIVERSE
# ============================================================

DEFAULT_TICKERS = {
    "NVDA": "NVIDIA",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "GOOGL": "Alphabet",
    "ASML": "ASML",
    "TSM": "TSMC",
    "005930.KS": "Samsung Electronics",
    "000660.KS": "SK Hynix",
    "MU": "Micron",
    "MRVL": "Marvell",
    "AMD": "AMD",
    "INTC": "Intel",

    "SAP.DE": "SAP",
    "BMW.DE": "BMW",
    "ALV.DE": "Allianz",
    "IFX.DE": "Infineon",
    "ENR.DE": "Siemens Energy",
    "AIR.PA": "Airbus",

    "NVO": "Novo Nordisk",
    "PFE": "Pfizer",

    "KO": "Coca-Cola",
    "PEP": "PepsiCo",
    "MCD": "McDonald's",
    "NFLX": "Netflix",
    "NKE": "Nike",

    "EQIX": "Equinix",
    "VRT": "Vertiv",
    "VST": "Vistra",
    "IREN": "IREN",

    "1810.HK": "Xiaomi",
    "NOK": "Nokia",
    "BYDDF": "BYD",
    "TCEHY": "Tencent",
    "5802.T": "Sumitomo Electric",
}


# ============================================================
# SESSION STATE
# ============================================================

if "kpax_tickers" not in st.session_state:
    st.session_state.kpax_tickers = DEFAULT_TICKERS.copy()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum=0, maximum=100):

    if value is None or pd.isna(value):
        return np.nan

    return max(
        minimum,
        min(maximum, float(value))
    )


def safe(value, default=0):

    if value is None or pd.isna(value):
        return default

    return float(value)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_metric(value, minimum, maximum):

    if value is None or pd.isna(value):
        return 50.0

    if maximum == minimum:
        return 50.0

    score = (
        (float(value) - minimum)
        / (maximum - minimum)
        * 100
    )

    return clamp(score)


def normalize_growth(value):

    if value is None or pd.isna(value):
        return 50.0

    value = float(value)

    score = 50 + value * 1.25

    return clamp(score)


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    roic,
    gross_margin,
    fcf_margin,
    earnings_growth,
    revenue_growth,
    is_financial=False,
    roe=None,
    net_margin=None,
):

    if is_financial:

        roe_score = normalize_metric(
            roe, -5, 30
        )

        net_margin_score = normalize_metric(
            net_margin, -10, 40
        )

        earnings_score = normalize_growth(
            earnings_growth
        )

        revenue_score = normalize_growth(
            revenue_growth
        )

        score = (
            0.40 * roe_score
            + 0.30 * net_margin_score
            + 0.20 * earnings_score
            + 0.10 * revenue_score
        )

    else:

        roic_score = normalize_metric(
            roic, -5, 30
        )

        gross_margin_score = normalize_metric(
            gross_margin, 0, 80
        )

        fcf_margin_score = normalize_metric(
            fcf_margin, -20, 40
        )

        earnings_score = normalize_growth(
            earnings_growth
        )

        revenue_score = normalize_growth(
            revenue_growth
        )

        score = (
            0.30 * roic_score
            + 0.15 * gross_margin_score
            + 0.20 * fcf_margin_score
            + 0.20 * earnings_score
            + 0.15 * revenue_score
        )

    return clamp(score)


# ============================================================
# FUTURE SCORE
# ============================================================

def calculate_future_score(
    earnings_growth,
    revenue_growth,
    growth_trend,
    analyst_growth,
    fcf_profitability_trend,
    turnaround_potential,
):

    earnings_score = normalize_growth(
        earnings_growth
    )

    revenue_score = normalize_growth(
        revenue_growth
    )

    score = (
        0.30 * earnings_score
        + 0.20 * revenue_score
        + 0.15 * clamp(growth_trend)
        + 0.15 * clamp(analyst_growth)
        + 0.10 * clamp(fcf_profitability_trend)
        + 0.10 * clamp(turnaround_potential)
    )

    return clamp(score)


# ============================================================
# KPAX
# ============================================================

def calculate_kpax(
    quality_score,
    future_score
):

    return clamp(
        0.40 * quality_score
        + 0.60 * future_score
    )


# ============================================================
# FAIR VALUE
# ============================================================

def calculate_fair_value(
    analyst_target,
    fcf_fair_value,
    analyst_weight=0.60
):

    values = []

    if (
        analyst_target is not None
        and not pd.isna(analyst_target)
    ):
        values.append(
            ("analyst", float(analyst_target))
        )

    if (
        fcf_fair_value is not None
        and not pd.isna(fcf_fair_value)
    ):
        values.append(
            ("fcf", float(fcf_fair_value))
        )

    if not values:
        return np.nan

    if len(values) == 1:
        return values[0][1]

    analyst = values[0][1]
    fcf = values[1][1]

    return (
        analyst_weight * analyst
        + (1 - analyst_weight) * fcf
    )


# ============================================================
# FV UPSIDE
# ============================================================

def calculate_fv_upside(
    current_price,
    fair_value
):

    if (
        current_price is None
        or fair_value is None
        or pd.isna(current_price)
        or pd.isna(fair_value)
        or current_price <= 0
    ):
        return np.nan

    return (
        (fair_value - current_price)
        / current_price
        * 100
    )


# ============================================================
# KPAX-FV
# ============================================================

def calculate_kpax_fv_score(
    fv_upside
):

    if fv_upside is None or pd.isna(fv_upside):
        return np.nan

    u = float(fv_upside)

    if u >= 40:
        return 100

    if u >= 25:
        return 85 + (
            (u - 25) / 15
        ) * 15

    if u >= 15:
        return 75 + (
            (u - 15) / 10
        ) * 10

    if u >= 5:
        return 65 + (
            (u - 5) / 10
        ) * 10

    if u >= 0:
        return 50 + (
            u / 5
        ) * 15

    if u >= -15:
        return 35 + (
            (u + 15) / 15
        ) * 15

    if u >= -30:
        return 20 + (
            (u + 30) / 15
        ) * 15

    return max(
        0,
        20 + (
            (u + 30) / 30
        ) * 20
    )


# ============================================================
# DIP SCORE
# ============================================================

def calculate_dip_score(
    price_change,
    earnings_change=0,
    revenue_change=0,
    analyst_target_change=0
):

    price_change = safe(price_change)
    earnings_change = safe(earnings_change)
    revenue_change = safe(revenue_change)
    analyst_target_change = safe(
        analyst_target_change
    )

    fundamental_change = (
        0.50 * earnings_change
        + 0.25 * revenue_change
        + 0.25 * analyst_target_change
    )

    divergence = (
        fundamental_change
        - price_change
    )

    if price_change >= 0:
        return 30

    score = 30 + divergence

    return clamp(score)


# ============================================================
# RISK SCORE
# ============================================================

def calculate_risk_score(
    beta,
    debt_to_equity,
    volatility,
    earnings_volatility,
    balance_sheet_strength
):

    beta_score = 100 - normalize_metric(
        beta,
        0.5,
        2.0
    )

    debt_score = 100 - normalize_metric(
        debt_to_equity,
        0,
        250
    )

    volatility_score = 100 - normalize_metric(
        volatility,
        10,
        80
    )

    earnings_vol_score = 100 - normalize_metric(
        earnings_volatility,
        0,
        100
    )

    balance_score = clamp(
        balance_sheet_strength
    )

    score = (
        0.20 * beta_score
        + 0.20 * debt_score
        + 0.20 * volatility_score
        + 0.15 * earnings_vol_score
        + 0.25 * balance_score
    )

    return clamp(score)


# ============================================================
# INVESTMENT SCORE
# ============================================================

def calculate_investment_score(
    kpax,
    kpax_fv,
    dip,
    risk,
    weights
):

    return clamp(
        weights["kpax"] * kpax
        + weights["kpax_fv"] * kpax_fv
        + weights["dip"] * dip
        + weights["risk"] * risk
    )


# ============================================================
# PRICE ATTRACTIVENESS
# ============================================================

def calculate_price_attractiveness(
    fv_upside
):

    if fv_upside is None or pd.isna(fv_upside):
        return np.nan

    if fv_upside >= 40:
        return 100

    if fv_upside >= 25:
        return 90

    if fv_upside >= 15:
        return 80

    if fv_upside >= 5:
        return 70

    if fv_upside >= 0:
        return 60

    if fv_upside >= -15:
        return 45

    if fv_upside >= -30:
        return 30

    return 15


# ============================================================
# VERDICT
# ============================================================

def get_verdict(score):

    if score >= 82.5:
        return "🟢 STRONG BUY"

    if score >= 77.5:
        return "🟢 BUY"

    if score >= 70:
        return "🟡 HOLD / ACCUMULATE"

    if score >= 55:
        return "🟠 WATCH / REDUCE"

    return "🔴 AVOID"


# ============================================================
# SINGLE STOCK
# ============================================================

def calculate_stock(
    data,
    weights
):

    quality = calculate_quality_score(
        data.get("roic"),
        data.get("gross_margin"),
        data.get("fcf_margin"),
        data.get("earnings_growth"),
        data.get("revenue_growth"),
        data.get("is_financial", False),
        data.get("roe"),
        data.get("net_margin"),
    )

    future = calculate_future_score(
        data.get("earnings_growth"),
        data.get("revenue_growth"),
        data.get("growth_trend"),
        data.get("analyst_growth"),
        data.get("fcf_profitability_trend"),
        data.get("turnaround_potential"),
    )

    kpax = calculate_kpax(
        quality,
        future
    )

    fair_value = calculate_fair_value(
        data.get("analyst_target"),
        data.get("fcf_fair_value"),
    )

    fv_upside = calculate_fv_upside(
        data.get("current_price"),
        fair_value
    )

    kpax_fv = calculate_kpax_fv_score(
        fv_upside
    )

    dip = calculate_dip_score(
        data.get("price_change"),
        data.get("earnings_change"),
        data.get("revenue_change"),
        data.get("analyst_target_change"),
    )

    risk = calculate_risk_score(
        data.get("beta"),
        data.get("debt_to_equity"),
        data.get("volatility"),
        data.get("earnings_volatility"),
        data.get("balance_sheet_strength"),
    )

    investment_score = calculate_investment_score(
        kpax,
        kpax_fv,
        dip,
        risk,
        weights
    )

    return {
        "Ticker": data.get("ticker"),
        "Name": data.get("name"),

        "Quality": round(
            quality, 1
        ),

        "Future": round(
            future, 1
        ),

        "KPAX": round(
            kpax, 1
        ),

        "Kurs": data.get(
            "current_price"
        ),

        "Fair Value": round(
            fair_value, 2
        )
        if not pd.isna(fair_value)
        else np.nan,

        "FV Upside %": round(
            fv_upside, 1
        )
        if not pd.isna(fv_upside)
        else np.nan,

        "KPAX-FV": round(
            kpax_fv, 1
        )
        if not pd.isna(kpax_fv)
        else np.nan,

        "Dip": round(
            dip, 1
        ),

        "Risk": round(
            risk, 1
        ),

        "Investment Score": round(
            investment_score, 1
        ),

        "Price Attractiveness": round(
            calculate_price_attractiveness(
                fv_upside
            ),
            1
        ),

        "Verdict": get_verdict(
            investment_score
        ),
    }


# ============================================================
# PAGE HEADER
# ============================================================

st.title("📊 KPAX Weekly Investment Engine")

st.caption(
    f"{VERSION} | "
    "KPAX = Qualität + Zukunft | "
    "KPAX-FV = Bewertung"
)


# ============================================================
# SIDEBAR – WEIGHTS
# ============================================================

st.sidebar.header(
    "⚙️ KPAX Gewichtung"
)

kpax_weight = st.sidebar.number_input(
    "KPAX",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_WEIGHTS["kpax"],
    step=0.05,
)

fv_weight = st.sidebar.number_input(
    "KPAX-FV",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_WEIGHTS["kpax_fv"],
    step=0.05,
)

dip_weight = st.sidebar.number_input(
    "Dip",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_WEIGHTS["dip"],
    step=0.05,
)

risk_weight = st.sidebar.number_input(
    "Risk",
    min_value=0.0,
    max_value=1.0,
    value=DEFAULT_WEIGHTS["risk"],
    step=0.05,
)

weight_sum = (
    kpax_weight
    + fv_weight
    + dip_weight
    + risk_weight
)

if abs(weight_sum - 1.0) > 0.001:

    st.sidebar.error(
        f"Gewichtung = {weight_sum:.0%}. "
        "Bitte exakt 100 % einstellen."
    )

else:

    st.sidebar.success(
        "Gewichtung = 100 %"
    )


weights = {
    "kpax": kpax_weight,
    "kpax_fv": fv_weight,
    "dip": dip_weight,
    "risk": risk_weight,
}


# ============================================================
# STOCK UNIVERSE
# ============================================================

st.header("📋 Aktien-Universum")

st.write(
    f"**{len(st.session_state.kpax_tickers)} Aktien im Universum**"
)


# ============================================================
# ADD STOCKS
# ============================================================

col1, col2 = st.columns([4, 1])

with col1:

    new_tickers = st.text_input(
        "Weitere Aktien hinzufügen",
        placeholder="z. B. ORCL, AVGO, LVMH.PA",
        label_visibility="collapsed"
    )

with col2:

    add_button = st.button(
        "➕ Hinzufügen",
        use_container_width=True
    )


if add_button:

    if new_tickers.strip():

        added = []
        duplicates = []

        for ticker in new_tickers.split(","):

            ticker = ticker.strip().upper()

            if not ticker:
                continue

            if ticker in st.session_state.kpax_tickers:

                duplicates.append(ticker)

            else:

                st.session_state.kpax_tickers[
                    ticker
                ] = ticker

                added.append(ticker)

        if added:

            st.success(
                "Hinzugefügt: "
                + ", ".join(added)
            )

        if duplicates:

            st.info(
                "Bereits vorhanden: "
                + ", ".join(duplicates)
            )


# ============================================================
# STOCK SELECTION
# ============================================================

ticker_options = list(
    st.session_state.kpax_tickers.keys()
)

selected_tickers = st.multiselect(
    "Aktien für die Berechnung",
    options=ticker_options,
    default=ticker_options,
    format_func=lambda ticker:
        f"{ticker} – "
        f"{st.session_state.kpax_tickers[ticker]}"
)


# ============================================================
# REMOVE CUSTOM STOCKS
# ============================================================

with st.expander(
    "🗑️ Eigene Aktien entfernen"
):

    custom_tickers = [
        ticker
        for ticker in ticker_options
        if ticker not in DEFAULT_TICKERS
    ]

    if custom_tickers:

        remove_tickers = st.multiselect(
            "Auswahl",
            custom_tickers
        )

        if st.button(
            "Ausgewählte Aktien entfernen"
        ):

            for ticker in remove_tickers:

                if ticker in st.session_state.kpax_tickers:
                    del st.session_state.kpax_tickers[
                        ticker
                    ]

            st.rerun()

    else:

        st.info(
            "Keine eigenen Aktien vorhanden."
        )


# ============================================================
# DEMO / INPUT DATA
# ============================================================

def create_demo_data():

    data = []

    for ticker, name in st.session_state.kpax_tickers.items():

        # neutrale Ausgangswerte
        row = {
            "ticker": ticker,
            "name": name,

            "current_price": 100,

            "analyst_target": 110,
            "fcf_fair_value": 105,

            "roic": 15,
            "gross_margin": 40,
            "fcf_margin": 15,

            "earnings_growth": 10,
            "revenue_growth": 8,

            "growth_trend": 70,
            "analyst_growth": 70,
            "fcf_profitability_trend": 70,
            "turnaround_potential": 50,

            "price_change": -5,
            "earnings_change": 5,
            "revenue_change": 5,
            "analyst_target_change": 2,

            "beta": 1.0,
            "debt_to_equity": 50,
            "volatility": 25,
            "earnings_volatility": 20,

            "balance_sheet_strength": 75,

            "roe": 15,
            "net_margin": 10,

            "is_financial": (
                ticker == "ALV.DE"
            ),
        }

        data.append(row)

    return data


# ============================================================
# CALCULATION
# ============================================================

st.divider()

calculate_button = st.button(
    "🚀 KPAX-Wochenberechnung starten",
    type="primary",
    use_container_width=True
)


if calculate_button:

    if not selected_tickers:

        st.warning(
            "Bitte mindestens eine Aktie auswählen."
        )

    elif abs(weight_sum - 1.0) > 0.001:

        st.error(
            "Die Gewichtungen müssen exakt 100 % ergeben."
        )

    else:

        demo_data = create_demo_data()

        selected_data = [
            row
            for row in demo_data
            if row["ticker"] in selected_tickers
        ]

        results = []

        for row in selected_data:

            result = calculate_stock(
                row,
                weights
            )

            results.append(result)

        result_df = pd.DataFrame(
            results
        )

        result_df = result_df.sort_values(
            "Investment Score",
            ascending=False
        ).reset_index(drop=True)

        result_df.insert(
            0,
            "Rang",
            range(1, len(result_df) + 1)
        )

        st.header(
            "🏆 KPAX Wochenranking"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        # ----------------------------------------------------
        # TOP PICKS
        # ----------------------------------------------------

        st.subheader(
            "⭐ Top KPAX Picks"
        )

        top3 = result_df.head(3)

        cols = st.columns(
            len(top3)
        )

        for col, (_, row) in zip(
            cols,
            top3.iterrows()
        ):

            with col:

                st.metric(
                    row["Ticker"],
                    f'{row["Investment Score"]:.1f}',
                    row["Verdict"]
                )

                st.caption(
                    row["Name"]
                )


        # ----------------------------------------------------
        # CSV DOWNLOAD
        # ----------------------------------------------------

        csv = result_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 KPAX-Tabelle als CSV",
            csv,
            "kpax_weekly.csv",
            "text/csv",
            use_container_width=True
        )


# ============================================================
# FORMULAS
# ============================================================

with st.expander(
    "🧮 KPAX-Formeln"
):

    st.markdown("""
### KPAX

**Quality**

Normale Unternehmen:

`Quality = 30 % ROIC + 15 % Gross Margin + 20 % FCF Margin + 20 % Earnings Growth + 15 % Revenue Growth`

Financials:

`Quality = 40 % ROE + 30 % Net Margin + 20 % Earnings Growth + 10 % Revenue Growth`

**Future**

`Future = 30 % Earnings Growth + 20 % Revenue Growth + 15 % Growth Trend + 15 % Analyst Growth + 10 % FCF/Profitability Trend + 10 % Turnaround`

**KPAX**

`KPAX = 40 % Quality + 60 % Future`

---

### Fair Value

`Fair Value = 60 % Analyst Target + 40 % FCF Fair Value`

### FV-Upside

`FV-Upside = (Fair Value - Kurs) / Kurs × 100`

### Investment Score

Standard:

`35 % KPAX + 40 % KPAX-FV + 15 % Dip + 10 % Risk`

---

### Investment Score

| Score | Urteil |
|---:|---|
| ≥ 82,5 | 🟢 STRONG BUY |
| 77,5–82,4 | 🟢 BUY |
| 70–77,4 | 🟡 HOLD / ACCUMULATE |
| 55–69,9 | 🟠 WATCH / REDUCE |
| < 55 | 🔴 AVOID |
""")


# ============================================================
# INFO
# ============================================================

st.info(
    "Hinweis: Diese Version enthält bereits das komplette "
    "KPAX-Berechnungsmodell und das Aktien-Universum. "
    "Die Fundamentaldaten sind aktuell Demo-/Platzhalterwerte. "
    "Für die echte Wochenberechnung müssen diese im nächsten "
    "Schritt automatisch über yfinance bzw. weitere Datenquellen "
    "geladen werden."
)
```

### `requirements.txt`

Dazu gehört:

```txt
streamlit
pandas
numpy
yfinance
```

**Aber:** Ich würde diesen Stand noch **nicht als endgültigen KPAX bezeichnen**. Der entscheidende nächste Schritt ist die automatische Datenversorgung. Aktuell würde beispielsweise jede neu hinzugefügte Aktie zunächst mit den neutralen Demo-Werten `ROIC=15`, `Growth=10 %`, `Fair Value=+10 %` usw. bewertet.

Für deine eigentliche Anwendung sollte der Ablauf stattdessen sein:

**Ticker → yfinance → Fundamentaldaten → Analystenziel → Fair Value → KPAX → KPAX-FV → Dip → Risk → Investment Score → Ranking.**

Dann entspricht das wirklich unserer **wöchentlichen KPAX-Berechnung** und nicht nur einer Streamlit-Oberfläche dafür.
