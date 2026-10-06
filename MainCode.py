import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import math
from datetime import datetime


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KPAX Stock Ranking",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# VERSION
# ============================================================

VERSION = "KPAX V1.0"


# ============================================================
# DEFAULT KPAX WEIGHTS
# ============================================================
#
# Investment Score:
# KPAX      35 %
# KPAX-FV   40 %
# Dip       15 %
# Risk      10 %
#
# Summe = 100 %
# ============================================================

DEFAULT_WEIGHTS = {
    "kpax": 0.35,
    "kpax_fv": 0.40,
    "dip": 0.15,
    "risk": 0.10
}


# ============================================================
# DEFAULT STOCK UNIVERSE
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
    "MAIR": "Madison Air Solutions"
}


# ============================================================
# FINANCIAL COMPANIES
# ============================================================

FINANCIAL_TICKERS = {
    "ALV.DE"
}


# ============================================================
# SESSION STATE
# ============================================================

if "tickers" not in st.session_state:
    st.session_state.tickers = DEFAULT_TICKERS.copy()

if "results" not in st.session_state:
    st.session_state.results = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return np.nan

        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return np.nan
            value = value[0]

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def clamp(value, minimum=0, maximum=100):
    if pd.isna(value):
        return np.nan

    return max(minimum, min(maximum, value))


def normalize_metric(value, low, high):
    value = safe_float(value)

    if pd.isna(value):
        return np.nan

    if high == low:
        return 50.0

    return clamp((value - low) / (high - low) * 100)


def normalize_growth(growth):
    """
    0 % Wachstum = 50 Punkte
    +40 % = 100
    -40 % = 0
    """

    growth = safe_float(growth)

    if pd.isna(growth):
        return np.nan

    return clamp(50 + growth * 1.25)


def weighted_average(values, weights):
    valid = []

    for value, weight in zip(values, weights):
        if not pd.isna(value):
            valid.append((value, weight))

    if not valid:
        return np.nan

    total_weight = sum(weight for _, weight in valid)

    if total_weight == 0:
        return np.nan

    return sum(value * weight for value, weight in valid) / total_weight


def percentage_change(old, new):
    old = safe_float(old)
    new = safe_float(new)

    if pd.isna(old) or pd.isna(new) or old == 0:
        return np.nan

    return (new / old - 1) * 100


def first_valid(*values):
    for value in values:
        value = safe_float(value)

        if not pd.isna(value):
            return value

    return np.nan


# ============================================================
# DATA EXTRACTION
# ============================================================

def get_statement_value(df, possible_names):
    """
    Sucht eine Kennzahl robust in Yahoo-Finance-Statements.
    """

    if df is None or df.empty:
        return np.nan

    try:
        for name in possible_names:

            if name in df.index:
                row = df.loc[name]

                for value in row:
                    value = safe_float(value)

                    if not pd.isna(value):
                        return value

    except Exception:
        pass

    return np.nan


def get_annual_values(df, possible_names, max_values=5):
    """
    Holt die letzten verfügbaren Jahreswerte.
    """

    if df is None or df.empty:
        return []

    try:

        for name in possible_names:

            if name not in df.index:
                continue

            row = df.loc[name]

            values = []

            for value in row:

                value = safe_float(value)

                if not pd.isna(value):
                    values.append(value)

            return values[:max_values]

    except Exception:
        pass

    return []


# ============================================================
# PRICE DATA
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_price_data(ticker):

    try:

        stock = yf.Ticker(ticker)

        history = stock.history(
            period="1y",
            interval="1d",
            auto_adjust=False
        )

        if history is None or history.empty:
            return None

        return history

    except Exception:
        return None


# ============================================================
# FUNDAMENTAL DATA
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def load_fundamental_data(ticker):

    try:

        stock = yf.Ticker(ticker)

        info = {}

        try:
            info = stock.info
        except Exception:
            info = {}

        try:
            financials = stock.financials
        except Exception:
            financials = pd.DataFrame()

        try:
            balance_sheet = stock.balance_sheet
        except Exception:
            balance_sheet = pd.DataFrame()

        try:
            cashflow = stock.cashflow
        except Exception:
            cashflow = pd.DataFrame()

        return {
            "info": info,
            "financials": financials,
            "balance_sheet": balance_sheet,
            "cashflow": cashflow
        }

    except Exception:
        return {
            "info": {},
            "financials": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cashflow": pd.DataFrame()
        }


# ============================================================
# PRICE METRICS
# ============================================================

def calculate_price_metrics(history):

    if history is None or history.empty:
        return {
            "price": np.nan,
            "1W": np.nan,
            "1M": np.nan,
            "3M": np.nan,
            "6M": np.nan,
            "1Y": np.nan,
            "52W_High": np.nan,
            "52W_Low": np.nan,
            "volatility": np.nan
        }

    close = history["Close"].dropna()

    if close.empty:
        return {
            "price": np.nan,
            "1W": np.nan,
            "1M": np.nan,
            "3M": np.nan,
            "6M": np.nan,
            "1Y": np.nan,
            "52W_High": np.nan,
            "52W_Low": np.nan,
            "volatility": np.nan
        }

    current = safe_float(close.iloc[-1])

    def change_days(days):

        if len(close) <= days:
            return np.nan

        old = safe_float(close.iloc[-days - 1])

        return percentage_change(old, current)

    daily_returns = close.pct_change().dropna()

    if len(daily_returns) > 20:
        volatility = safe_float(
            daily_returns.std() * np.sqrt(252) * 100
        )
    else:
        volatility = np.nan

    return {
        "price": current,
        "1W": change_days(5),
        "1M": change_days(21),
        "3M": change_days(63),
        "6M": change_days(126),
        "1Y": change_days(252),
        "52W_High": safe_float(close.max()),
        "52W_Low": safe_float(close.min()),
        "volatility": volatility
    }


# ============================================================
# GROWTH CALCULATION
# ============================================================

def calculate_growth(series):

    if series is None or len(series) < 2:
        return np.nan

    values = [
        safe_float(x)
        for x in series
        if not pd.isna(safe_float(x))
    ]

    if len(values) < 2:
        return np.nan

    newest = values[0]
    oldest = values[-1]

    if oldest == 0:
        return np.nan

    years = len(values) - 1

    try:

        if newest > 0 and oldest > 0:

            cagr = (
                (newest / oldest) ** (1 / years) - 1
            ) * 100

            return cagr

        return percentage_change(oldest, newest)

    except Exception:
        return np.nan


# ============================================================
# EPS GROWTH
# ============================================================

def calculate_eps_growth(financials):

    eps_values = get_annual_values(
        financials,
        [
            "Diluted EPS",
            "Basic EPS",
            "Diluted EPS Growth"
        ]
    )

    return calculate_growth(eps_values)


# ============================================================
# REVENUE GROWTH
# ============================================================

def calculate_revenue_growth(financials):

    revenue_values = get_annual_values(
        financials,
        [
            "Total Revenue",
            "Operating Revenue",
            "Revenue"
        ]
    )

    return calculate_growth(revenue_values)


# ============================================================
# FCF
# ============================================================

def get_fcf(cashflow):

    return get_statement_value(
        cashflow,
        [
            "Free Cash Flow",
            "Free Cashflow"
        ]
    )


# ============================================================
# FCF MARGIN
# ============================================================

def calculate_fcf_margin(fcf, revenue):

    fcf = safe_float(fcf)
    revenue = safe_float(revenue)

    if pd.isna(fcf) or pd.isna(revenue) or revenue == 0:
        return np.nan

    return fcf / revenue * 100


# ============================================================
# ROIC
# ============================================================

def calculate_roic(info, balance_sheet, financials):

    try:

        roe = safe_float(info.get("returnOnEquity"))

        if not pd.isna(roe):
            return roe * 100

        net_income = get_statement_value(
            financials,
            ["Net Income", "Net Income Common Stockholders"]
        )

        equity = get_statement_value(
            balance_sheet,
            [
                "Stockholders Equity",
                "Total Equity Gross Minority Interest",
                "Common Stock Equity"
            ]
        )

        if (
            not pd.isna(net_income)
            and not pd.isna(equity)
            and equity != 0
        ):
            return net_income / equity * 100

    except Exception:
        pass

    return np.nan


# ============================================================
# DEBT / EQUITY
# ============================================================

def calculate_debt_to_equity(info, balance_sheet):

    debt_equity = safe_float(
        info.get("debtToEquity")
    )

    if not pd.isna(debt_equity):
        return debt_equity

    debt = get_statement_value(
        balance_sheet,
        [
            "Total Debt",
            "Long Term Debt",
            "Total Non Current Liabilities Net Minority Interest"
        ]
    )

    equity = get_statement_value(
        balance_sheet,
        [
            "Stockholders Equity",
            "Total Equity Gross Minority Interest",
            "Common Stock Equity"
        ]
    )

    if (
        not pd.isna(debt)
        and not pd.isna(equity)
        and equity != 0
    ):
        return debt / equity * 100

    return np.nan


# ============================================================
# BALANCE SHEET STRENGTH
# ============================================================

def calculate_balance_sheet_strength(
    debt_to_equity,
    cash,
    debt
):

    score = 50.0

    if not pd.isna(debt_to_equity):

        if debt_to_equity <= 20:
            score += 25

        elif debt_to_equity <= 50:
            score += 15

        elif debt_to_equity <= 100:
            score += 5

        elif debt_to_equity <= 200:
            score -= 10

        else:
            score -= 25

    if (
        not pd.isna(cash)
        and not pd.isna(debt)
    ):

        if cash > debt:
            score += 20

        elif cash > debt * 0.5:
            score += 10

        elif cash < debt * 0.2:
            score -= 10

    return clamp(score)


# ============================================================
# FAIR VALUE - FCF DCF
# ============================================================

def calculate_fcf_fair_value(
    fcf,
    shares,
    growth
):

    fcf = safe_float(fcf)
    shares = safe_float(shares)
    growth = safe_float(growth)

    if (
        pd.isna(fcf)
        or pd.isna(shares)
        or shares <= 0
        or fcf <= 0
    ):
        return np.nan

    fcf_per_share = fcf / shares

    if pd.isna(growth):
        growth = 8.0

    growth = clamp(growth, -5, 15)

    discount_rate = 0.09
    terminal_growth = 0.03

    value = 0.0

    for year in range(1, 6):

        future_fcf = (
            fcf_per_share
            * (1 + growth / 100) ** year
        )

        value += future_fcf / (
            (1 + discount_rate) ** year
        )

    terminal_fcf = (
        fcf_per_share
        * (1 + growth / 100) ** 5
        * (1 + terminal_growth)
    )

    terminal_value = (
        terminal_fcf
        / (discount_rate - terminal_growth)
    )

    terminal_value /= (
        (1 + discount_rate) ** 5
    )

    value += terminal_value

    return value


# ============================================================
# ANALYST TARGET
# ============================================================

def get_analyst_target(info):

    return first_valid(
        info.get("targetMeanPrice"),
        info.get("targetMedianPrice"),
        info.get("targetLowPrice")
    )


# ============================================================
# KPAX-FV SCORE
# ============================================================

def calculate_kpax_fv_score(upside):

    upside = safe_float(upside)

    if pd.isna(upside):
        return np.nan

    if upside >= 40:
        return 100

    if upside >= 25:
        return 85 + (upside - 25) / 15 * 15

    if upside >= 15:
        return 75 + (upside - 15) / 10 * 10

    if upside >= 5:
        return 65 + (upside - 5) / 10 * 10

    if upside >= 0:
        return 50 + upside / 5 * 15

    if upside >= -15:
        return 35 + (upside + 15) / 15 * 15

    if upside >= -30:
        return 20 + (upside + 30) / 15 * 15

    return max(
        0,
        20 + (upside + 30) / 30 * 20
    )


# ============================================================
# QUALITY SCORE
# ============================================================

def calculate_quality_score(
    ticker,
    roic,
    gross_margin,
    fcf_margin,
    eps_growth,
    revenue_growth,
    roe,
    net_margin
):

    if ticker in FINANCIAL_TICKERS:

        roe_score = normalize_metric(
            roe,
            0,
            30
        )

        margin_score = normalize_metric(
            net_margin,
            0,
            30
        )

        eps_score = normalize_growth(
            eps_growth
        )

        revenue_score = normalize_growth(
            revenue_growth
        )

        return weighted_average(
            [
                roe_score,
                margin_score,
                eps_score,
                revenue_score
            ],
            [
                0.40,
                0.30,
                0.20,
                0.10
            ]
        )

    roic_score = normalize_metric(
        roic,
        0,
        30
    )

    gross_score = normalize_metric(
        gross_margin,
        20,
        70
    )

    fcf_score = normalize_metric(
        fcf_margin,
        0,
        35
    )

    eps_score = normalize_growth(
        eps_growth
    )

    revenue_score = normalize_growth(
        revenue_growth
    )

    return weighted_average(
        [
            roic_score,
            gross_score,
            fcf_score,
            eps_score,
            revenue_score
        ],
        [
            0.30,
            0.15,
            0.20,
            0.20,
            0.15
        ]
    )


# ============================================================
# FUTURE SCORE
# ============================================================

def calculate_future_score(
    eps_growth,
    revenue_growth,
    growth_trend,
    analyst_growth,
    fcf_margin,
    turnaround
):

    eps_score = normalize_growth(
        eps_growth
    )

    revenue_score = normalize_growth(
        revenue_growth
    )

    trend_score = normalize_growth(
        growth_trend
    )

    analyst_score = normalize_growth(
        analyst_growth
    )

    fcf_score = normalize_metric(
        fcf_margin,
        0,
        35
    )

    turnaround_score = clamp(
        turnaround
    )

    return weighted_average(
        [
            eps_score,
            revenue_score,
            trend_score,
            analyst_score,
            fcf_score,
            turnaround_score
        ],
        [
            0.30,
            0.20,
            0.15,
            0.15,
            0.10,
            0.10
        ]
    )


# ============================================================
# KPAX
# ============================================================

def calculate_kpax(
    quality,
    future
):

    return weighted_average(
        [
            quality,
            future
        ],
        [
            0.40,
            0.60
        ]
    )


# ============================================================
# DIP SCORE
# ============================================================

def calculate_dip_score(
    price_6m,
    eps_growth,
    revenue_growth,
    analyst_upside
):

    price_6m = safe_float(price_6m)
    eps_growth = safe_float(eps_growth)
    revenue_growth = safe_float(revenue_growth)
    analyst_upside = safe_float(analyst_upside)

    if pd.isna(price_6m):
        return np.nan

    eps_component = (
        normalize_growth(eps_growth)
        if not pd.isna(eps_growth)
        else 50
    )

    revenue_component = (
        normalize_growth(revenue_growth)
        if not pd.isna(revenue_growth)
        else 50
    )

    analyst_component = (
        normalize_growth(analyst_upside)
        if not pd.isna(analyst_upside)
        else 50
    )

    fundamental_score = (
        eps_component * 0.50
        + revenue_component * 0.25
        + analyst_component * 0.25
    )

    fundamental_change = (
        fundamental_score - 50
    )

    divergence = (
        fundamental_change - price_6m
    )

    if price_6m >= 0:

        return 30.0

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

    beta_score = (
        100 - normalize_metric(
            beta,
            0.5,
            2.0
        )
        if not pd.isna(beta)
        else np.nan
    )

    debt_score = (
        100 - normalize_metric(
            debt_to_equity,
            0,
            250
        )
        if not pd.isna(debt_to_equity)
        else np.nan
    )

    volatility_score = (
        100 - normalize_metric(
            volatility,
            10,
            80
        )
        if not pd.isna(volatility)
        else np.nan
    )

    earnings_vol_score = (
        100 - normalize_metric(
            earnings_volatility,
            0,
            100
        )
        if not pd.isna(earnings_volatility)
        else np.nan
    )

    return weighted_average(
        [
            beta_score,
            debt_score,
            volatility_score,
            earnings_vol_score,
            balance_sheet_strength
        ],
        [
            0.20,
            0.20,
            0.20,
            0.15,
            0.25
        ]
    )


# ============================================================
# TURNAROUND POTENTIAL
# ============================================================

def calculate_turnaround(
    price_6m,
    eps_growth,
    revenue_growth,
    analyst_upside
):

    score = 50.0

    if not pd.isna(price_6m):

        if price_6m <= -30:
            score += 25

        elif price_6m <= -20:
            score += 20

        elif price_6m <= -10:
            score += 10

        elif price_6m >= 20:
            score -= 10

    if not pd.isna(eps_growth):

        if eps_growth > 20:
            score += 15

        elif eps_growth > 10:
            score += 10

        elif eps_growth < -20:
            score -= 15

    if not pd.isna(revenue_growth):

        if revenue_growth > 15:
            score += 10

        elif revenue_growth < -15:
            score -= 10

    if not pd.isna(analyst_upside):

        if analyst_upside > 30:
            score += 15

        elif analyst_upside > 15:
            score += 8

        elif analyst_upside < -20:
            score -= 10

    return clamp(score)


# ============================================================
# PRICE ATTRACTIVENESS
# ============================================================

def calculate_price_attractiveness(upside):

    upside = safe_float(upside)

    if pd.isna(upside):
        return np.nan

    if upside >= 40:
        return 100

    if upside >= 25:
        return 90

    if upside >= 15:
        return 80

    if upside >= 5:
        return 70

    if upside >= 0:
        return 60

    if upside >= -15:
        return 45

    if upside >= -30:
        return 30

    return 15


# ============================================================
# VERDICT
# ============================================================

def get_verdict(score):

    if pd.isna(score):
        return "NO DATA"

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
# SINGLE STOCK CALCULATION
# ============================================================

def calculate_stock(ticker, name):

    history = load_price_data(ticker)
    fundamental = load_fundamental_data(ticker)

    info = fundamental["info"]
    financials = fundamental["financials"]
    balance_sheet = fundamental["balance_sheet"]
    cashflow = fundamental["cashflow"]

    price_data = calculate_price_metrics(history)

    current_price = price_data["price"]

    if pd.isna(current_price):
        return None

    # --------------------------------------------------------
    # FUNDAMENTALS
    # --------------------------------------------------------

    revenue = get_statement_value(
        financials,
        [
            "Total Revenue",
            "Operating Revenue",
            "Revenue"
        ]
    )

    net_income = get_statement_value(
        financials,
        [
            "Net Income",
            "Net Income Common Stockholders"
        ]
    )

    fcf = get_fcf(cashflow)

    shares = first_valid(
        info.get("sharesOutstanding"),
        info.get("impliedSharesOutstanding")
    )

    gross_margin = safe_float(
        info.get("grossMargins")
    )

    if not pd.isna(gross_margin):
        gross_margin *= 100

    net_margin = safe_float(
        info.get("profitMargins")
    )

    if not pd.isna(net_margin):
        net_margin *= 100

    roe = safe_float(
        info.get("returnOnEquity")
    )

    if not pd.isna(roe):
        roe *= 100

    roic = calculate_roic(
        info,
        balance_sheet,
        financials
    )

    revenue_growth = calculate_revenue_growth(
        financials
    )

    eps_growth = calculate_eps_growth(
        financials
    )

    fcf_margin = calculate_fcf_margin(
        fcf,
        revenue
    )

    debt_to_equity = calculate_debt_to_equity(
        info,
        balance_sheet
    )

    beta = safe_float(
        info.get("beta")
    )

    # --------------------------------------------------------
    # CASH / DEBT
    # --------------------------------------------------------

    cash = get_statement_value(
        balance_sheet,
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Financial"
        ]
    )

    debt = get_statement_value(
        balance_sheet,
        [
            "Total Debt",
            "Long Term Debt"
        ]
    )

    balance_strength = calculate_balance_sheet_strength(
        debt_to_equity,
        cash,
        debt
    )

    # --------------------------------------------------------
    # ANALYST TARGET
    # --------------------------------------------------------

    analyst_target = get_analyst_target(info)

    analyst_upside = percentage_change(
        current_price,
        analyst_target
    )

    # --------------------------------------------------------
    # FCF FAIR VALUE
    # --------------------------------------------------------

    fcf_fair_value = calculate_fcf_fair_value(
        fcf,
        shares,
        eps_growth
    )

    # --------------------------------------------------------
    # FAIR VALUE
    # --------------------------------------------------------

    if (
        not pd.isna(analyst_target)
        and not pd.isna(fcf_fair_value)
    ):

        fair_value = (
            analyst_target * 0.60
            + fcf_fair_value * 0.40
        )

    elif not pd.isna(analyst_target):

        fair_value = analyst_target

    elif not pd.isna(fcf_fair_value):

        fair_value = fcf_fair_value

    else:

        fair_value = np.nan

    fv_upside = percentage_change(
        current_price,
        fair_value
    )

    # --------------------------------------------------------
    # KPAX-FV
    # --------------------------------------------------------

    kpax_fv = calculate_kpax_fv_score(
        fv_upside
    )

    # --------------------------------------------------------
    # QUALITY
    # --------------------------------------------------------

    quality = calculate_quality_score(
        ticker,
        roic,
        gross_margin,
        fcf_margin,
        eps_growth,
        revenue_growth,
        roe,
        net_margin
    )

    # --------------------------------------------------------
    # GROWTH TREND
    # --------------------------------------------------------

    growth_trend = np.nan

    try:

        revenue_values = get_annual_values(
            financials,
            [
                "Total Revenue",
                "Operating Revenue",
                "Revenue"
            ],
            max_values=4
        )

        if len(revenue_values) >= 3:

            recent_growth = percentage_change(
                revenue_values[1],
                revenue_values[0]
            )

            previous_growth = percentage_change(
                revenue_values[2],
                revenue_values[1]
            )

            growth_trend = (
                recent_growth - previous_growth
            )

    except Exception:
        pass

    # --------------------------------------------------------
    # ANALYST GROWTH
    # --------------------------------------------------------

    analyst_growth = first_valid(
        info.get("earningsGrowth"),
        info.get("revenueGrowth")
    )

    if not pd.isna(analyst_growth):
        analyst_growth *= 100

    # --------------------------------------------------------
    # TURNAROUND
    # --------------------------------------------------------

    turnaround = calculate_turnaround(
        price_data["6M"],
        eps_growth,
        revenue_growth,
        analyst_upside
    )

    # --------------------------------------------------------
    # FUTURE
    # --------------------------------------------------------

    future = calculate_future_score(
        eps_growth,
        revenue_growth,
        growth_trend,
        analyst_growth,
        fcf_margin,
        turnaround
    )

    # --------------------------------------------------------
    # KPAX
    # --------------------------------------------------------

    kpax = calculate_kpax(
        quality,
        future
    )

    # --------------------------------------------------------
    # DIP
    # --------------------------------------------------------

    dip = calculate_dip_score(
        price_data["6M"],
        eps_growth,
        revenue_growth,
        analyst_upside
    )

    # --------------------------------------------------------
    # EARNINGS VOLATILITY
    # --------------------------------------------------------

    earnings_volatility = np.nan

    try:

        eps_values = get_annual_values(
            financials,
            [
                "Diluted EPS",
                "Basic EPS"
            ],
            max_values=6
        )

        if len(eps_values) >= 3:

            growths = []

            for i in range(len(eps_values) - 1):

                old = eps_values[i + 1]
                new = eps_values[i]

                if (
                    not pd.isna(old)
                    and not pd.isna(new)
                    and old != 0
                ):

                    growths.append(
                        abs((new / old - 1) * 100)
                    )

            if growths:
                earnings_volatility = np.std(
                    growths
                )

    except Exception:
        pass

    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    risk = calculate_risk_score(
        beta,
        debt_to_equity,
        price_data["volatility"],
        earnings_volatility,
        balance_strength
    )

    # --------------------------------------------------------
    # INVESTMENT SCORE
    # --------------------------------------------------------

    investment_score = weighted_average(
        [
            kpax,
            kpax_fv,
            dip,
            risk
        ],
        [
            st.session_state.kpax_weight,
            st.session_state.fv_weight,
            st.session_state.dip_weight,
            st.session_state.risk_weight
        ]
    )

    # --------------------------------------------------------
    # PRICE ATTRACTIVENESS
    # --------------------------------------------------------

    price_attractiveness = calculate_price_attractiveness(
        fv_upside
    )

    # --------------------------------------------------------
    # DATA COMPLETENESS
    # --------------------------------------------------------

    required_fields = [
        current_price,
        revenue_growth,
        eps_growth,
        fcf_margin,
        roic,
        analyst_target,
        fair_value,
        kpax,
        kpax_fv,
        dip,
        risk
    ]

    available = sum(
        not pd.isna(x)
        for x in required_fields
    )

    completeness = (
        available / len(required_fields) * 100
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "Ticker": ticker,
        "Name": name,

        "Price": current_price,

        "1W %": price_data["1W"],
        "1M %": price_data["1M"],
        "3M %": price_data["3M"],
        "6M %": price_data["6M"],
        "1Y %": price_data["1Y"],

        "Analyst Target": analyst_target,
        "Konsenspotenzial %": analyst_upside,

        "FCF Fair Value": fcf_fair_value,
        "Fair Value": fair_value,
        "FV Upside %": fv_upside,

        "Quality": quality,
        "Future": future,
        "KPAX": kpax,

        "KPAX-FV": kpax_fv,

        "Dip": dip,
        "Risk": risk,

        "Price Attractiveness": price_attractiveness,

        "Investment Score": investment_score,

        "Verdict": get_verdict(
            investment_score
        ),

        "Revenue Growth %": revenue_growth,
        "EPS Growth %": eps_growth,
        "FCF Margin %": fcf_margin,

        "ROIC %": roic,
        "ROE %": roe,
        "Net Margin %": net_margin,

        "Debt/Equity %": debt_to_equity,
        "Beta": beta,
        "Volatility %": price_data["volatility"],

        "Turnaround": turnaround,

        "Data Completeness %": completeness
    }


# ============================================================
# INITIALIZE WEIGHTS
# ============================================================

if "kpax_weight" not in st.session_state:
    st.session_state.kpax_weight = DEFAULT_WEIGHTS["kpax"]

if "fv_weight" not in st.session_state:
    st.session_state.fv_weight = DEFAULT_WEIGHTS["kpax_fv"]

if "dip_weight" not in st.session_state:
    st.session_state.dip_weight = DEFAULT_WEIGHTS["dip"]

if "risk_weight" not in st.session_state:
    st.session_state.risk_weight = DEFAULT_WEIGHTS["risk"]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ KPAX Einstellungen")

st.sidebar.caption(
    f"{VERSION} | Live Yahoo Finance"
)


# ------------------------------------------------------------
# WEIGHTS
# ------------------------------------------------------------

st.sidebar.subheader("Investment Score Gewichtung")

kpax_pct = st.sidebar.number_input(
    "KPAX %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.kpax_weight * 100,
    step=5.0
)

fv_pct = st.sidebar.number_input(
    "KPAX-FV %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.fv_weight * 100,
    step=5.0
)

dip_pct = st.sidebar.number_input(
    "Dip %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.dip_weight * 100,
    step=5.0
)

risk_pct = st.sidebar.number_input(
    "Risk %",
    min_value=0.0,
    max_value=100.0,
    value=st.session_state.risk_weight * 100,
    step=5.0
)

weight_sum = (
    kpax_pct
    + fv_pct
    + dip_pct
    + risk_pct
)

if abs(weight_sum - 100) > 0.01:

    st.sidebar.error(
        f"Summe = {weight_sum:.1f}%"
    )

else:

    st.session_state.kpax_weight = kpax_pct / 100
    st.session_state.fv_weight = fv_pct / 100
    st.session_state.dip_weight = dip_pct / 100
    st.session_state.risk_weight = risk_pct / 100

    st.sidebar.success(
        "Gewichtung = 100%"
    )


if st.sidebar.button(
    "↩️ Standardgewichtung zurücksetzen",
    use_container_width=True
):

    st.session_state.kpax_weight = DEFAULT_WEIGHTS["kpax"]
    st.session_state.fv_weight = DEFAULT_WEIGHTS["kpax_fv"]
    st.session_state.dip_weight = DEFAULT_WEIGHTS["dip"]
    st.session_state.risk_weight = DEFAULT_WEIGHTS["risk"]

    st.rerun()


# ============================================================
# STOCK MANAGEMENT
# ============================================================

st.sidebar.subheader("📈 Aktien")

selected_tickers = st.sidebar.multiselect(
    "Aktien auswählen",
    options=list(st.session_state.tickers.keys()),
    default=list(st.session_state.tickers.keys())
)


# ------------------------------------------------------------
# ADD STOCKS
# ------------------------------------------------------------

new_tickers = st.sidebar.text_input(
    "Weitere Ticker hinzufügen",
    placeholder="z.B. AAPL, ORCL, AVGO"
)

if st.sidebar.button(
    "➕ Aktien hinzufügen",
    use_container_width=True
):

    if new_tickers.strip():

        additions = [
            x.strip().upper()
            for x in new_tickers.split(",")
            if x.strip()
        ]

        for ticker in additions:

            if ticker not in st.session_state.tickers:

                st.session_state.tickers[ticker] = ticker

        st.rerun()


# ------------------------------------------------------------
# REMOVE CUSTOM STOCKS
# ------------------------------------------------------------

custom_tickers = [
    ticker
    for ticker in st.session_state.tickers
    if ticker not in DEFAULT_TICKERS
]

if custom_tickers:

    remove_tickers = st.sidebar.multiselect(
        "Eigene Aktien entfernen",
        options=custom_tickers
    )

    if st.sidebar.button(
        "🗑️ Entfernen",
        use_container_width=True
    ):

        for ticker in remove_tickers:
            st.session_state.tickers.pop(
                ticker,
                None
            )

        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.title("📊 KPAX Stock Ranking")

st.markdown(
    """
    **KPAX = Quality + Future + Valuation + Dip + Risk**

    Live-Berechnung auf Basis von Yahoo-Finance-Daten.
    """
)


# ============================================================
# INFO
# ============================================================

with st.expander(
    "ℹ️ KPAX-Algorithmus anzeigen"
):

    st.markdown(
        """
        ### KPAX

        **Quality**
        - ROIC / ROE
        - Margen
        - FCF-Marge
        - EPS-Wachstum
        - Umsatzwachstum

        **Future**
        - EPS-Wachstum
        - Umsatzwachstum
        - Wachstumstrend
        - Analystenwachstum
        - FCF-/Profitabilität
        - Turnaround-Potenzial

        **KPAX**
        - 40 % Quality
        - 60 % Future

        **Fair Value**
        - 60 % Analystenziel
        - 40 % FCF-DCF

        **KPAX-FV**
        - bewertet das Verhältnis von Fair Value zu aktuellem Kurs

        **Dip**
        - Kursrückgang vs. fundamentale Entwicklung

        **Risk**
        - Beta
        - Verschuldung
        - Volatilität
        - Earnings Volatility
        - Bilanzqualität

        **Investment Score**
        - KPAX
        - KPAX-FV
        - Dip
        - Risk
        """
    )


# ============================================================
# RUN BUTTON
# ============================================================

run_analysis = st.button(
    "🚀 KPAX-Berechnung starten",
    type="primary",
    use_container_width=True
)


# ============================================================
# CALCULATION
# ============================================================

if run_analysis:

    if not selected_tickers:

        st.warning(
            "Bitte mindestens eine Aktie auswählen."
        )

    else:

        results = []

        progress = st.progress(0)

        status = st.empty()

        total = len(selected_tickers)

        for i, ticker in enumerate(selected_tickers):

            name = st.session_state.tickers.get(
                ticker,
                ticker
            )

            status.text(
                f"Berechne {ticker} – {name}"
            )

            try:

                result = calculate_stock(
                    ticker,
                    name
                )

                if result is not None:
                    results.append(result)

            except Exception as e:

                st.warning(
                    f"{ticker}: Berechnung fehlgeschlagen – {e}"
                )

            progress.progress(
                (i + 1) / total
            )

        status.empty()
        progress.empty()

        if results:

            df = pd.DataFrame(results)

            df = df.sort_values(
                "Investment Score",
                ascending=False
            ).reset_index(drop=True)

            df.insert(
                0,
                "Rank",
                range(1, len(df) + 1)
            )

            st.session_state.results = df

            st.success(
                f"{len(df)} Aktien erfolgreich berechnet."
            )

        else:

            st.error(
                "Keine verwertbaren Daten gefunden."
            )


# ============================================================
# DISPLAY RESULTS
# ============================================================

if st.session_state.results is not None:

    df = st.session_state.results.copy()

    st.subheader("🏆 KPAX Ranking")


    # --------------------------------------------------------
    # MAIN TABLE
    # --------------------------------------------------------

    display_columns = [
        "Rank",
        "Ticker",
        "Name",
        "Price",
        "KPAX",
        "KPAX-FV",
        "Dip",
        "Risk",
        "Investment Score",
        "Verdict",
        "Fair Value",
        "FV Upside %",
        "Konsenspotenzial %",
        "Price Attractiveness",
        "Data Completeness %"
    ]

    display_df = df[
        [
            col
            for col in display_columns
            if col in df.columns
        ]
    ].copy()


    # --------------------------------------------------------
    # FORMATTING
    # --------------------------------------------------------

    numeric_columns = [
        "Price",
        "KPAX",
        "KPAX-FV",
        "Dip",
        "Risk",
        "Investment Score",
        "Fair Value",
        "FV Upside %",
        "Konsenspotenzial %",
        "Price Attractiveness",
        "Data Completeness %"
    ]

    for col in numeric_columns:

        if col in display_df.columns:

            display_df[col] = display_df[col].round(1)


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=750
    )


    # ========================================================
    # TOP 5
    # ========================================================

    st.subheader("🥇 Top 5")

    top5 = df.head(5)

    cols = st.columns(5)

    for i, (_, row) in enumerate(top5.iterrows()):

        with cols[i]:

            st.metric(
                label=f"#{int(row['Rank'])} {row['Ticker']}",
                value=f"{row['Investment Score']:.1f}",
                delta=row["Verdict"]
            )

            st.caption(
                f"{row['Name']}"
            )

            if not pd.isna(row["Konsenspotenzial %"]):

                st.write(
                    f"Konsenspotenzial: "
                    f"{row['Konsenspotenzial %']:.1f}%"
                )

            if not pd.isna(row["FV Upside %"]):

                st.write(
                    f"FV-Upside: "
                    f"{row['FV Upside %']:.1f}%"
                )


    # ========================================================
    # DETAILED DATA
    # ========================================================

    st.subheader("🔎 Fundamentaldaten")

    detail_columns = [
        "Ticker",
        "Name",
        "Revenue Growth %",
        "EPS Growth %",
        "FCF Margin %",
        "ROIC %",
        "ROE %",
        "Net Margin %",
        "Debt/Equity %",
        "Beta",
        "Volatility %",
        "Turnaround",
        "1W %",
        "1M %",
        "3M %",
        "6M %",
        "1Y %",
        "52W_High",
        "52W_Low"
    ]

    detail_df = df[
        [
            col
            for col in detail_columns
            if col in df.columns
        ]
    ].copy()

    st.dataframe(
        detail_df.round(1),
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    st.subheader("📊 Score-Verteilung")

    chart_df = df[
        [
            "Ticker",
            "KPAX",
            "KPAX-FV",
            "Dip",
            "Risk",
            "Investment Score"
        ]
    ].copy()

    chart_df = chart_df.set_index(
        "Ticker"
    )

    st.bar_chart(
        chart_df
    )


    # ========================================================
    # CSV DOWNLOAD
    # ========================================================

    csv = df.to_csv(
        index=False,
        sep=";",
        decimal=","
    )

    st.download_button(
        label="⬇️ KPAX-Ergebnis als CSV herunterladen",
        data=csv,
        file_name=(
            f"KPAX_Ranking_"
            f"{datetime.now().strftime('%Y-%m-%d')}.csv"
        ),
        mime="text/csv",
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "KPAX Stock Ranking | "
    "Datenquelle: Yahoo Finance via yfinance | "
    "Fair Value enthält modellbasierte FCF-DCF-Komponente."
)
