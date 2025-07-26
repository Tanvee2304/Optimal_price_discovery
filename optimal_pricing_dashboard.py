
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load or simulate data
np.random.seed(42)
n = 1000
ages = np.random.randint(18, 65, size=n)
incomes = np.random.normal(50000, 15000, size=n).astype(int)
prices = np.random.uniform(10, 100, size=n).round(2)

def get_segment(income):
    if income < 40000:
        return "Price-Sensitive"
    elif income > 70000:
        return "Premium"
    return "Average"

segments = [get_segment(i) for i in incomes]

def purchase_prob(price, segment):
    base = 1 - (price / 120)
    if segment == "Price-Sensitive":
        return base - 0.2
    elif segment == "Premium":
        return base + 0.1
    return base

probs = [purchase_prob(p, s) for p, s in zip(prices, segments)]
purchased = np.random.binomial(1, np.clip(probs, 0, 1))

df = pd.DataFrame({
    "Age": ages,
    "Income": incomes,
    "Segment": segments,
    "Price_Offered": prices,
    "Purchased": purchased
})

# Encode segment
le = LabelEncoder()
df['Segment_Code'] = le.fit_transform(df['Segment'])

# Fit logistic regression
X = df[['Price_Offered', 'Segment_Code']]
y = df['Purchased']
model = LogisticRegression()
model.fit(X, y)

# Streamlit UI
st.title("📈 Optimal Pricing Dashboard")

segment_option = st.selectbox("Select Customer Segment", df['Segment'].unique())
segment_code = df[df['Segment'] == segment_option]['Segment_Code'].iloc[0]
cost = st.slider("Cost per Unit (₹)", min_value=10, max_value=90, value=30, step=1)
price_range = np.linspace(10, 100, 100)

# Prediction
X_sim = pd.DataFrame({
    'Price_Offered': price_range,
    'Segment_Code': [segment_code]*len(price_range)
})
purchase_probs = model.predict_proba(X_sim)[:, 1]
profits = (price_range - cost) * purchase_probs

# Optimal price
opt_idx = np.argmax(profits)
optimal_price = price_range[opt_idx]
max_profit = profits[opt_idx]

# Plot
st.subheader("📉 Demand & Profit Curve")

fig, ax = plt.subplots()
ax.plot(price_range, purchase_probs, label='Purchase Probability', color='blue')
ax.set_ylabel('Purchase Probability', color='blue')
ax.set_xlabel('Price (₹)')
ax2 = ax.twinx()
ax2.plot(price_range, profits, label='Expected Profit', color='green')
ax2.set_ylabel('Expected Profit (₹)', color='green')
ax.axvline(optimal_price, color='red', linestyle='--', label=f'Optimal Price = ₹{optimal_price:.2f}')
fig.tight_layout()
st.pyplot(fig)

st.success(f"🎯 Optimal Price for '{segment_option}' Segment: ₹{optimal_price:.2f}")
st.info(f"💰 Maximum Expected Profit per Unit: ₹{max_profit:.2f}")
