import streamlit as st
import pandas as pd
import numpy as np


# ============================================================
# KPAX ENGINE
# ============================================================

APP_VERSION = "KPAX Weekly Engine V1.0"


# ------------------------------------------------------------
# DEFAULT WEIGHTS
# ------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "kpax": 0.35,
    "kpax_fv": 0.40,
    "dip": 0.15,
    "risk": 0.10,
}


# ============================================================
# SCORE HELPERS
# ============================================================

def clamp(value, minimum=0, maximum=100):
    """Begrenzt einen Wert auf 0-100."""
    if value is None or pd.isna(value):
        return np.nan

    return max(minimum, min(maximum, float(value)))


def safe(value, default=0):
    """Ersetzt NaN/None durch default."""
    if value is None or pd.isna(value):
        return default

    return float(value)


# ============================================================
# 1. QUALITY SCORE
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
    """
    QUALITY SCORE

    Normale Unternehmen:
        ROIC             30 %
        Gross Margin     15 %
        FCF Margin       20 %
        Earnings Growth  20 %
        Revenue Growth   15 %

    Financials:
        ROE              40 %
        Net Margin       30 %
        Earnings Growth  20 %
        Revenue Growth   10 %

    Alle Einzelwerte werden zunächst auf 0-100 normalisiert.
    """

    if is_financial:

        roe_score = normalize_metric(roe, -5, 30)
        net_margin_score = normalize_metric(net_margin, -10, 40)
        earnings_score = normalize_growth(earnings_growth)
        revenue_score = normalize_growth(revenue_growth)

        score = (
            0.40 * roe_score
            + 0.30 * net_margin_score
            + 0.20 * earnings_score
            + 0.10 * revenue_score
        )

    else:

        roic_score = normalize_metric(roic, -5, 30)
        gross_margin_score = normalize_metric(gross_margin, 0, 80)
        fcf_margin_score = normalize_metric(fcf_margin, -20, 40)
        earnings_score = normalize_growth(earnings_growth)
        revenue_score = normalize_growth(revenue_growth)

        score = (
            0.30 * roic_score
            + 0.15 * gross_margin_score
            + 0.20 * fcf_margin_score
            + 0.20 * earnings_score
            + 0.15 * revenue_score
        )

    return clamp(score)


# ============================================================
# 2. FUTURE SCORE
# ============================================================

def calculate_future_score(
    earnings_growth,
    revenue_growth,
    growth_trend,
    analyst_growth,
    fcf_profitability_trend,
    turnaround_potential,
):
    """
    FUTURE SCORE

        Earnings Growth              30 %
        Revenue Growth               20 %
        Growth Trend                 15 %
        Analyst Growth / Target      15 %
        FCF / Profitability Trend    10 %
        Turnaround Potential         10 %
    """

    earnings_score = normalize_growth(earnings_growth)
    revenue_score = normalize_growth(revenue_growth)

    growth_trend = clamp(growth_trend)
    analyst_growth = clamp(analyst_growth)
    fcf_profitability_trend = clamp(fcf_profitability_trend)
    turnaround_potential = clamp(turnaround_potential)

    score = (
        0.30 * earnings_score
        + 0.20 * revenue_score
        + 0.15 * growth_trend
        + 0.15 * analyst_growth
        + 0.10 * fcf_profitability_trend
        + 0.10 * turnaround_potential
    )

    return clamp(score)


# ============================================================
# 3. KPAX
# ============================================================

def calculate_kpax(quality_score, future_score):
    """
    KPAX

        40 % Quality
        60 % Future
    """

    score = (
        0.40 * quality_score
        + 0.60 * future_score
    )

    return clamp(score)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_metric(value, minimum, maximum):
    """
    Lineare Normalisierung auf 0-100.
    """

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
    """
    Wachstumswert auf 0-100.

    Orientierung:

        <= -20 %  -> 0
        0 %       -> ~50
        +20 %     -> ~75
        +40 %     -> ~100

    Werte werden bewusst nicht beliebig über 100 hinaus
    laufen gelassen.
    """

    if value is None or pd.isna(value):
        return 50.0

    value = float(value)

    score = 50 + (value * 1.25)

    return clamp(score)


# ============================================================
# 4. FAIR VALUE
# ============================================================

def calculate_fair_value(
    analyst_target,
    fcf_fair_value,
    analyst_weight=0.60,
):
    """
    Fair Value

        60 % Analyst Target
        40 % FCF-based Fair Value
    """

    values = []

    if analyst_target is not None and not pd.isna(analyst_target):
        values.append(("analyst", float(analyst_target)))

    if fcf_fair_value is not None and not pd.isna(fcf_fair_value):
        values.append(("fcf", float(fcf_fair_value)))

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
# 5. FV UPSIDE
# ============================================================

def calculate_fv_upside(current_price, fair_value):
    """
    FV-Upside:

        (Fair Value - Kurs) / Kurs * 100
    """

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
# 6. KPAX-FV SCORE
# ============================================================

def calculate_kpax_fv_score(fv_upside):
    """
    KPAX-FV Bewertung gemäß unserer Bandlogik.

    > +40 %      -> 95-100
    +25 bis +40  -> 85-95
    +15 bis +25  -> 75-85
    +5 bis +15   -> 65-75
    0 bis +5      -> 50-65
    0 bis -15     -> 35-50
    -15 bis -30   -> 20-35
    < -30 %      -> 0-20
    """

    if fv_upside is None or pd.isna(fv_upside):
        return np.nan

    u = float(fv_upside)

    # Stark unterbewertet
    if u >= 40:
        return 100

    # 25-40 %
    if u >= 25:
        return 85 + ((u - 25) / 15) * 15

    # 15-25 %
    if u >= 15:
        return 75 + ((u - 15) / 10) * 10

    # 5-15 %
    if u >= 5:
        return 65 + ((u - 5) / 10) * 10

    # 0-5 %
    if u >= 0:
        return 50 + (u / 5) * 15

    # 0 bis -15 %
    if u >= -15:
        return 35 + ((u + 15) / 15) * 15

    # -15 bis -30 %
    if u >= -30:
        return 20 + ((u + 30) / 15) * 15

    # >30 % über Fair Value
    return max(
        0,
        20 + ((u + 30) / 30) * 20
    )


# ============================================================
# 7. DIP SCORE
# ============================================================

def calculate_dip_score(
    price_change,
    earnings_change=0,
    revenue_change=0,
    analyst_target_change=0,
):
    """
    DIP SCORE

    Idee:
    Ein starker Kursrückgang ist interessant, wenn sich
    die Fundamentaldaten deutlich weniger verschlechtert haben.

    Je größer die Differenz zwischen Kursentwicklung und
    Fundamentaldaten, desto höher der Dip Score.
    """

    price_change = safe(price_change)
    earnings_change = safe(earnings_change)
    revenue_change = safe(revenue_change)
    analyst_target_change = safe(analyst_target_change)

    # Fundamentale Entwicklung
    fundamental_change = (
        0.50 * earnings_change
        + 0.25 * revenue_change
        + 0.25 * analyst_target_change
    )

    # "Ungerechtfertigter" Kursrückgang
    divergence = fundamental_change - price_change

    # Kein wirklicher Dip
    if price_change >= 0:
        return 30

    # Mapping:
    # 0 Differenz   -> ~30
    # +20 Punkte    -> ~50
    # +40 Punkte    -> ~70
    # +60 Punkte    -> ~90
    # +80 Punkte    -> 100

    score = 30 + divergence

    return clamp(score)


# ============================================================
# 8. RISK SCORE
# ============================================================

def calculate_risk_score(
    beta,
    debt_to_equity,
    volatility,
    earnings_volatility,
    balance_sheet_strength,
):
    """
    RISK SCORE

    100 = sehr niedriges Risiko
    0   = sehr hohes Risiko
    """

    # Beta
    beta_score = 100 - normalize_metric(
        beta,
        0.5,
        2.0
    )

    # Verschuldung
    debt_score = 100 - normalize_metric(
        debt_to_equity,
        0,
        250
    )

    # Volatilität
    volatility_score = 100 - normalize_metric(
        volatility,
        10,
        80
    )

    # Gewinnvolatilität
    earnings_vol_score = 100 - normalize_metric(
        earnings_volatility,
        0,
        100
    )

    balance_score = clamp(balance_sheet_strength)

    score = (
        0.20 * beta_score
        + 0.20 * debt_score
        + 0.20 * volatility_score
        + 0.15 * earnings_vol_score
        + 0.25 * balance_score
    )

    return clamp(score)


# ============================================================
# 9. INVESTMENT SCORE
# ============================================================

def calculate_investment_score(
    kpax,
    kpax_fv,
    dip,
    risk,
    weights=None,
):
    """
    Investment Score

    DEFAULT:

        35 % KPAX
        40 % KPAX-FV
        15 % Dip
        10 % Risk

    Risk Score:
        100 = wenig Risiko
        daher direkt positiv gewichtet.
    """

    if weights is None:
        weights = DEFAULT_WEIGHTS

    score = (
        weights["kpax"] * kpax
        + weights["kpax_fv"] * kpax_fv
        + weights["dip"] * dip
        + weights["risk"] * risk
    )

    return clamp(score)


# ============================================================
# 10. PRICE ATTRACTIVENESS
# ============================================================

def calculate_price_attractiveness(fv_upside):
    """
    Ein einfaches, separates Preisattraktivitäts-Rating.
    """

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
# 11. VERDICT
# ============================================================

def get_verdict(score):
    """
    Unsere Investment-Score Schwellen.
    """

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
# 12. COMPLETE STOCK CALCULATION
# ============================================================

def calculate_stock(data, weights=None):
    """
    Führt die komplette KPAX-Berechnung für eine Aktie aus.
    """

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
        fair_value,
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
        weights,
    )

    return {
        "Ticker": data.get("ticker"),
        "Name": data.get("name"),

        "Quality": round(quality, 1),
        "Future": round(future, 1),
        "KPAX": round(kpax, 1),

        "Current Price": data.get("current_price"),
        "Fair Value": round(fair_value, 2)
        if not pd.isna(fair_value) else np.nan,

        "FV Upside %": round(fv_upside, 1)
        if not pd.isna(fv_upside) else np.nan,

        "KPAX-FV": round(kpax_fv, 1)
        if not pd.isna(kpax_fv) else np.nan,

        "Dip": round(dip, 1),
        "Risk": round(risk, 1),

        "Investment Score": round(
            investment_score,
            1
        ),

        "Price Attractiveness":
            round(
                calculate_price_attractiveness(
                    fv_upside
                ),
                1
            ),

        "Verdict":
            get_verdict(investment_score),
    }


# ============================================================
# 13. STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="KPAX Weekly Engine",
    page_icon="📊",
    layout="wide",
)

st.title("📊 KPAX Weekly Investment Engine")
st.caption(APP_VERSION)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("KPAX Gewichtung")

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
        f"Gewichtungen müssen 100 % ergeben. "
        f"Aktuell: {weight_sum:.0%}"
    )

weights = {
    "kpax": kpax_weight,
    "kpax_fv": fv_weight,
    "dip": dip_weight,
    "risk": risk_weight,
}


# ============================================================
# DEMO DATA
# ============================================================

st.subheader("Aktien")

default_data = pd.DataFrame([
    {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "current_price": 180,
        "analyst_target": 215,
        "fcf_fair_value": 205,
        "roic": 90,
        "gross_margin": 73,
        "fcf_margin": 45,
        "earnings_growth": 35,
        "revenue_growth": 30,
        "growth_trend": 90,
        "analyst_growth": 85,
        "fcf_profitability_trend": 95,
        "turnaround_potential": 70,
        "price_change": -8,
        "earnings_change": 15,
        "revenue_change": 12,
        "analyst_target_change": 5,
        "beta": 1.7,
        "debt_to_equity": 30,
        "volatility": 45,
        "earnings_volatility": 25,
        "balance_sheet_strength": 90,
        "is_financial": False,
    },
    {
        "ticker": "MSFT",
        "name": "Microsoft",
        "current_price": 500,
        "analyst_target": 550,
        "fcf_fair_value": 535,
        "roic": 35,
        "gross_margin": 69,
        "fcf_margin": 32,
        "earnings_growth": 15,
        "revenue_growth": 14,
        "growth_trend": 85,
        "analyst_growth": 80,
        "fcf_profitability_trend": 90,
        "turnaround_potential": 50,
        "price_change": -4,
        "earnings_change": 8,
        "revenue_change": 7,
        "analyst_target_change": 3,
        "beta": 1.0,
        "debt_to_equity": 45,
        "volatility": 25,
        "earnings_volatility": 15,
        "balance_sheet_strength": 95,
        "is_financial": False,
    },
    {
        "ticker": "BMW.DE",
        "name": "BMW",
        "current_price": 85,
        "analyst_target": 105,
        "fcf_fair_value": 100,
        "roic": 12,
        "gross_margin": 18,
        "fcf_margin": 8,
        "earnings_growth": -3,
        "revenue_growth": 2,
        "growth_trend": 55,
        "analyst_growth": 60,
        "fcf_profitability_trend": 60,
        "turnaround_potential": 75,
        "price_change": -18,
        "earnings_change": -5,
        "revenue_change": 2,
        "analyst_target_change": 4,
        "beta": 1.2,
        "debt_to_equity": 120,
        "volatility": 35,
        "earnings_volatility": 30,
        "balance_sheet_strength": 75,
        "is_financial": False,
    },
    {
        "ticker": "ALV.DE",
        "name": "Allianz",
        "current_price": 380,
        "analyst_target": 430,
        "fcf_fair_value": 420,
        "roe": 18,
        "net_margin": 10,
        "earnings_growth": 8,
        "revenue_growth": 6,
        "growth_trend": 75,
        "analyst_growth": 75,
        "fcf_profitability_trend": 80,
        "turnaround_potential": 40,
        "price_change": -6,
        "earnings_change": 5,
        "revenue_change": 4,
        "analyst_target_change": 2,
        "beta": 0.9,
        "debt_to_equity": 80,
        "volatility": 20,
        "earnings_volatility": 15,
        "balance_sheet_strength": 85,
        "is_financial": True,
    },
])


# ============================================================
# CALCULATE
# ============================================================

if st.button("🚀 KPAX berechnen", type="primary"):

    results = []

    for _, row in default_data.iterrows():

        result = calculate_stock(
            row.to_dict(),
            weights=weights,
        )

        results.append(result)

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "Investment Score",
        ascending=False
    )

    st.subheader("KPAX Wochenranking")

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# FORMULAS
# ============================================================

with st.expander("🧮 Mathematik des KPAX-Modells"):

    st.markdown("""
### KPAX

**Quality**

Normale Unternehmen:

`Quality = 30% ROIC + 15% Gross Margin + 20% FCF Margin + 20% Earnings Growth + 15% Revenue Growth`

Financials:

`Quality = 40% ROE + 30% Net Margin + 20% Earnings Growth + 10% Revenue Growth`

**Future**

`Future = 30% Earnings Growth + 20% Revenue Growth + 15% Growth Trend + 15% Analyst Growth + 10% FCF/Profitability Trend + 10% Turnaround`

**KPAX**

`KPAX = 40% Quality + 60% Future`

---

### Fair Value

`Fair Value = 60% Analyst Target + 40% FCF Fair Value`

### FV-Upside

`FV-Upside = (Fair Value - aktueller Kurs) / aktueller Kurs × 100`

### KPAX-FV

Der FV-Upside wird anschließend in einen Score von 0–100 überführt.

### Investment Score

Aktuelle Gewichtung:

`Investment Score = 35% KPAX + 40% KPAX-FV + 15% Dip + 10% Risk`

Dabei gilt:

**Risk = 100 → sehr geringes Risiko**

**Risk = 0 → sehr hohes Risiko**

### Grundidee

**KPAX = Qualität + Zukunft**

**KPAX-FV = Bewertung**

**Dip = Chance durch überproportionalen Kursrückgang**

**Risk = Risikopuffer**

Der aktuelle Kurs verändert **nicht** den KPAX selbst.
Er beeinflusst dagegen KPAX-FV, Dip und damit den Investment Score.
""")


# ============================================================
# VERDICT TABLE
# ============================================================

with st.expander("🎯 Investment-Score Schwellen"):

    verdict_table = pd.DataFrame({
        "Score": [
            "≥ 82.5",
            "77.5 – 82.4",
            "70 – 77.4",
            "55 – 69.9",
            "< 55",
        ],
        "Urteil": [
            "STRONG BUY",
            "BUY",
            "HOLD / ACCUMULATE",
            "WATCH / REDUCE",
            "AVOID",
        ],
    })

    st.table(verdict_table)
