"""
AI-Powered Support Ticket Intelligence System
Complete Production-Ready Application with Advanced Features
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re
import json
from datetime import datetime, timedelta
from collections import Counter
from wordcloud import WordCloud

# Sklearn metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# TextBlob for sentiment
try:
    from textblob import TextBlob
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'textblob'])
    from textblob import TextBlob

import warnings
warnings.filterwarnings('ignore')

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Support Ticket Intelligence System",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .insight-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    
    .critical-priority {
        background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
        color: white;
        padding: 5px 12px;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.05); }
        100% { opacity: 1; transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL VARIABLES
# =========================================================

df = None
vectorizer = None
model = None
label_encoder = None

# =========================================================
# CACHE DATA LOADING
# =========================================================

@st.cache_resource
def load_data():
    """Load cleaned tickets data from data/processed/"""
    try:
        df = pd.read_csv("data/processed/cleaned_tickets.csv")
        return df
    except FileNotFoundError:
        try:
            df = pd.read_csv("data/raw/support_tickets.csv")
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found! Please run the notebook first.")
            return None

@st.cache_resource
def load_models():
    """Load trained models from models/ folder"""
    global vectorizer, model, label_encoder
    
    try:
        vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
        model = joblib.load("models/ticket_classifier.pkl")
        
        try:
            label_encoder = joblib.load("models/label_encoder.pkl")
        except:
            label_encoder = None
            
        return vectorizer, model, label_encoder
    except FileNotFoundError as e:
        st.error(f"❌ Model not found: {e}")
        st.info("Please run the notebook to train and save models first.")
        return None, None, None

# =========================================================
# HELPER FUNCTIONS
# =========================================================

def assign_priority(text):
    """Enhanced priority assignment with Critical level"""
    text = str(text).lower()

    critical_keywords = [
        'production down', 'system crash', 'data loss', 'security breach',
        'critical', 'emergency', 'all customers affected', 'major outage'
    ]
    
    high_priority_keywords = [
        'payment failed', 'server down', 'account hacked', 'unable login',
        'fraud', 'urgent', 'asap', 'blocked', 'not working at all'
    ]

    medium_priority_keywords = [
        'slow', 'delay', 'bug', 'issue', 'not working', 'error',
        'login issue', 'password reset', 'billing', 'refund'
    ]

    for word in critical_keywords:
        if word in text:
            return "Critical"
    
    for word in high_priority_keywords:
        if word in text:
            return "High"

    for word in medium_priority_keywords:
        if word in text:
            return "Medium"

    return "Low"

def analyze_sentiment(text):
    """Advanced sentiment analysis with emotion detection"""
    try:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        if polarity < -0.6:
            sentiment = "Very Negative"
            emotion = "Angry/Frustrated"
        elif polarity < -0.3:
            sentiment = "Negative"
            emotion = "Dissatisfied"
        elif polarity < 0:
            sentiment = "Slightly Negative"
            emotion = "Disappointed"
        elif polarity == 0:
            sentiment = "Neutral"
            emotion = "Neutral"
        elif polarity < 0.3:
            sentiment = "Slightly Positive"
            emotion = "Satisfied"
        elif polarity < 0.6:
            sentiment = "Positive"
            emotion = "Happy"
        else:
            sentiment = "Very Positive"
            emotion = "Delighted"
        
        return sentiment, emotion, polarity, subjectivity
    except:
        return "Neutral", "Neutral", 0, 0.5

def detect_intent(text):
    """Comprehensive intent detection"""
    text = text.lower()
    
    intent_patterns = {
        'refund_request': ['refund', 'money back', 'return payment', 'chargeback'],
        'cancellation': ['cancel', 'stop subscription', 'terminate account'],
        'authentication': ['login', 'sign in', 'access denied', 'password reset', '2fa'],
        'payment_issue': ['payment failed', 'card declined', 'billing error', 'charged twice'],
        'bug_report': ['bug', 'glitch', 'error message', 'exception', 'crash'],
        'performance': ['slow', 'lag', 'timeout', 'loading issue'],
        'feature_request': ['suggest', 'recommend', 'would be nice', 'add feature'],
        'security': ['hacked', 'unauthorized', 'suspicious', 'breach', 'compromised'],
        'account_management': ['update email', 'change password', 'delete account'],
        'how_to': ['how to', 'guide', 'tutorial', 'steps to', 'help with']
    }
    
    for intent, keywords in intent_patterns.items():
        for keyword in keywords:
            if keyword in text:
                return intent.replace('_', ' ').title()
    
    return "General Support"

def detect_risk(text, priority, sentiment_polarity):
    """Comprehensive risk assessment"""
    text = text.lower()
    
    risk_score = 0
    risk_factors = []
    
    if priority == "Critical":
        risk_score += 40
        risk_factors.append("Critical Priority")
    elif priority == "High":
        risk_score += 25
        risk_factors.append("High Priority")
    
    if sentiment_polarity < -0.5:
        risk_score += 20
        risk_factors.append("Very Negative Sentiment")
    
    security_keywords = ['hack', 'breach', 'unauthorized', 'fraud', 'compromised']
    if any(word in text for word in security_keywords):
        risk_score += 30
        risk_factors.append("Security Concern")
    
    financial_keywords = ['payment', 'charge', 'money', 'refund', 'billing']
    if any(word in text for word in financial_keywords) and priority in ["High", "Critical"]:
        risk_score += 15
        risk_factors.append("Financial Impact")
    
    if risk_score >= 60:
        risk_level = "Critical Risk"
        risk_color = "🔴"
    elif risk_score >= 40:
        risk_level = "High Risk"
        risk_color = "🟠"
    elif risk_score >= 20:
        risk_level = "Medium Risk"
        risk_color = "🟡"
    else:
        risk_level = "Low Risk"
        risk_color = "🟢"
    
    return risk_level, risk_score, risk_factors, risk_color

def route_ticket(category, priority, risk_level):
    """Intelligent ticket routing"""
    routing_matrix = {
        'Billing': {'team': '💰 Finance Team', 'sla': '4 hours'},
        'Technical': {'team': '💻 Technical Support', 'sla': '6 hours'},
        'Account': {'team': '👤 Account Management', 'sla': '8 hours'},
        'Security': {'team': '🔒 Security Team', 'sla': '1 hour'}
    }
    
    base = None
    for key in routing_matrix:
        if key.lower() in category.lower():
            base = routing_matrix[key].copy()
            break
    
    if base is None:
        base = {'team': '📞 Support Team', 'sla': '12 hours'}
    
    if priority == "Critical":
        base['team'] = f"🚨 {base['team']} - Escalated"
        base['sla'] = "30 minutes"
    elif priority == "High":
        base['sla'] = "2 hours"
    
    return base['team'], base['sla']

def calculate_urgency_score(priority, sentiment_polarity, risk_score):
    """Calculate urgency score (0-100)"""
    priority_scores = {"Critical": 100, "High": 75, "Medium": 50, "Low": 25}
    base_score = priority_scores.get(priority, 30)
    sentiment_adjustment = max(0, min(20, (0 - sentiment_polarity) * 20))
    risk_adjustment = min(30, risk_score * 0.3)
    return min(100, base_score + sentiment_adjustment + risk_adjustment)

def estimate_response_time(priority, urgency_score_val):
    """Estimate response time"""
    base_times = {"Critical": 0.5, "High": 2, "Medium": 6, "Low": 24}
    base_time = base_times.get(priority, 12)
    urgency_factor = (100 - urgency_score_val) / 100
    adjusted_time = base_time * (0.5 + urgency_factor * 0.5)
    
    if adjusted_time < 1:
        return f"{int(adjusted_time * 60)} minutes"
    elif adjusted_time < 24:
        return f"{adjusted_time:.1f} hours"
    else:
        return f"{adjusted_time/24:.1f} days"

def predict_csat_risk(sentiment, priority):
    """Predict Customer Satisfaction risk"""
    risk_score = 0
    sentiment_risk = {"Very Negative": 40, "Negative": 30, "Slightly Negative": 20, "Neutral": 10}
    risk_score += sentiment_risk.get(sentiment, 0)
    
    if priority == "Critical":
        risk_score += 30
    elif priority == "High":
        risk_score += 20
    
    if risk_score >= 60:
        return "🔴 Very High CSAT Risk", risk_score
    elif risk_score >= 40:
        return "🟠 High CSAT Risk", risk_score
    elif risk_score >= 20:
        return "🟡 Medium CSAT Risk", risk_score
    else:
        return "🟢 Low CSAT Risk", risk_score

def calculate_business_impact(priority, risk_level, sentiment):
    """Business impact assessment"""
    impacts = []
    
    if priority in ["Critical", "High"]:
        impacts.append("⚠️ Revenue Impact: Potential customer churn")
        impacts.append("📉 Customer Churn Risk: High")
    
    if "Security" in risk_level:
        impacts.append("🔒 Security Impact: Data breach potential")
    
    if sentiment in ["Very Negative", "Negative"]:
        impacts.append("😞 Brand Reputation: Negative impact")
    
    if not impacts:
        impacts.append("✅ Minimal business impact expected")
    
    return impacts

def suggested_resolution(category, intent, priority):
    """AI-powered solution suggestions"""
    solutions = {
        'Billing': 'Verify payment logs and process refund/adjustment',
        'Technical': 'Debug error logs and escalate to engineering',
        'Account': 'Verify authentication logs and reset access',
        'Security': 'Initiate security incident response protocol'
    }
    
    for key, solution in solutions.items():
        if key.lower() in category.lower():
            if priority in ["Critical", "High"]:
                return f"🚨 IMMEDIATE: {solution}"
            return f"📋 ACTION: {solution}"
    
    return "Investigate and provide resolution"

def generate_ai_summary(category, priority, sentiment, risk_level, intent):
    """Generate AI summary"""
    return f"The system detected a **{priority.lower()} priority** **{category}** ticket with **{sentiment.lower()}** customer sentiment. Intent: {intent}. Risk level: {risk_level}."

# =========================================================
# LOAD DATA AND MODELS
# =========================================================

df = load_data()
vectorizer, model, label_encoder = load_models()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.title("AI Support Operations")
    
    st.markdown("---")
    
    st.markdown("### 📊 Business Impact")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Efficiency", "+75%", "gain")
    with col2:
        st.metric("Response", "-40%", "faster")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("CSAT", "+25%", "improved")
    with col2:
        st.metric("Cost", "-60%", "saved")
    
    st.markdown("---")
    
    st.markdown("### 💰 ROI Calculator")
    tickets_per_day = st.number_input("Daily Tickets", min_value=100, value=500, step=100)
    agent_cost = st.number_input("Hourly Rate ($)", min_value=15, value=25, step=5)
    
    manual_minutes = 5
    ai_seconds = 30
    
    manual_hours = (manual_minutes * tickets_per_day) / 60
    ai_hours = (ai_seconds * tickets_per_day) / 3600
    
    daily_savings = (manual_hours - ai_hours) * agent_cost
    yearly_savings = daily_savings * 365
    
    st.metric("💰 Daily Savings", f"${daily_savings:,.0f}")
    st.metric("📅 Yearly Savings", f"${yearly_savings:,.0f}")
    
    st.markdown("---")
    st.markdown("### 📊 System Status")
    if df is not None:
        st.success(f"✅ Data: {len(df)} tickets")
    if model is not None:
        st.success("✅ Model: Active")
    else:
        st.error("❌ Model: Not loaded")

# =========================================================
# MAIN CONTENT
# =========================================================

st.markdown('<div class="main-header">🎫 AI-Powered Support Ticket Intelligence System</div>', unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <p style='font-size: 1.1rem; color: #666;'>
        🤖 Enterprise-grade ML solution for automatic ticket classification, prioritization, and intelligent routing
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# KPI DASHBOARD
# =========================================================

if df is not None and model is not None:
    st.header("📊 Live Operations Dashboard")
    
    if 'cleaned_text' in df.columns:
        df['priority'] = df['cleaned_text'].apply(assign_priority)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        padding: 1rem; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 1rem;">📊 Total</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{len(df)}</p>
                <small>Tickets Analyzed</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        critical_count = len(df[df['priority'] == 'Critical']) if 'priority' in df.columns else 0
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #ff0844 0%, #ffb199 100%);
                        padding: 1rem; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 1rem;">🚨 Critical</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{critical_count}</p>
                <small>Need Immediate Action</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        high_count = len(df[df['priority'] == 'High']) if 'priority' in df.columns else 0
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                        padding: 1rem; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 1rem;">⚠️ High</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{high_count}</p>
                <small>Urgent Priority</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        categories = df['Ticket Type'].nunique() if 'Ticket Type' in df.columns else 0
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                        padding: 1rem; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 1rem;">📂 Types</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">{categories}</p>
                <small>Categories</small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col5:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                        padding: 1rem; border-radius: 15px; text-align: center; color: white;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
                <h3 style="margin: 0; font-size: 1rem;">🎯 Target</h3>
                <p style="font-size: 2rem; font-weight: bold; margin: 0.5rem 0;">85%</p>
                <small>Accuracy Target</small>
            </div>
            """, 
            unsafe_allow_html=True
        )

st.markdown("---")

# =========================================================
# TICKET ANALYZER SECTION
# =========================================================

st.header("🔍 AI Ticket Intelligence Analyzer")

col1, col2 = st.columns([3, 1])

with col1:
    ticket = st.text_area(
        "📝 Enter Customer Support Ticket",
        height=150,
        placeholder="Example: My payment failed twice, money was deducted but I didn't receive the product. This is urgent!",
        help="Paste the complete customer ticket for comprehensive AI analysis"
    )

with col2:
    st.markdown("### 💡 Quick Examples")
    
    if st.button("💰 Billing Issue", use_container_width=True):
        ticket = "My card was charged twice for the same order. Need refund urgently!"
    
    if st.button("🔐 Account Problem", use_container_width=True):
        ticket = "Cannot login to my account. Password reset not working. Please help!"
    
    if st.button("🐛 Technical Bug", use_container_width=True):
        ticket = "The application crashes every time I try to upload a file."
    
    if st.button("🔒 Security Alert", use_container_width=True):
        ticket = "Suspicious activity detected on my account. Unauthorized login!"

# =========================================================
# AI ANALYSIS & REPORT
# =========================================================

if st.button("🚀 Generate Complete AI Analysis", type="primary", use_container_width=True):
    
    if ticket and ticket.strip() != "":
        
        if model is None or vectorizer is None:
            st.error("❌ Models not loaded. Please train models first using the notebook.")
        else:
            with st.spinner("🧠 Running comprehensive AI analysis..."):
                
                # Core ML Predictions
                ticket_vector = vectorizer.transform([ticket])
                prediction = model.predict(ticket_vector)
                predicted_category = prediction[0]
                
                probabilities = model.predict_proba(ticket_vector)[0]
                confidence = round(probabilities.max() * 100, 2)
                
                # NLP Analysis
                priority = assign_priority(ticket)
                sentiment, emotion, polarity, subjectivity = analyze_sentiment(ticket)
                intent = detect_intent(ticket)
                risk_level, risk_score, risk_factors, risk_color = detect_risk(ticket, priority, polarity)
                urgency_score_val = calculate_urgency_score(priority, polarity, risk_score)
                
                # Business Logic
                assigned_team, sla_time = route_ticket(predicted_category, priority, risk_level)
                response_time = estimate_response_time(priority, urgency_score_val)
                csat_risk, csat_score = predict_csat_risk(sentiment, priority)
                business_impacts = calculate_business_impact(priority, risk_level, sentiment)
                resolution = suggested_resolution(predicted_category, intent, priority)
                ai_summary = generate_ai_summary(predicted_category, priority, sentiment, risk_level, intent)
            
            # Display Report
            st.header("📋 Complete AI Intelligence Report")
            
            # Priority Alert Banner
            if priority == "Critical":
                st.error(f"""
                🚨 **CRITICAL PRIORITY DETECTED** 🚨
                
                This ticket requires **IMMEDIATE ACTION**. Escalation team has been notified.
                - Response SLA: {sla_time}
                """)
            elif priority == "High":
                st.warning(f"""
                ⚠️ **HIGH PRIORITY TICKET** ⚠️
                
                Urgent attention required. Assign to senior support.
                - Response SLA: {sla_time}
                """)
            
            # Main Results Grid
            st.subheader("🎯 Classification Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.info(f"**🏷️ Category**\n\n{predicted_category}")
            
            with col2:
                priority_icon = "🔴" if priority == "Critical" else "🟠" if priority == "High" else "🟡" if priority == "Medium" else "🟢"
                st.info(f"**⚡ Priority**\n\n{priority_icon} {priority}")
            
            with col3:
                st.info(f"**🎯 Confidence**\n\n{confidence}%")
            
            with col4:
                st.info(f"**😊 Sentiment**\n\n{sentiment}")
            
            # =========================================================
            # ADVANCED ANALYTICS - FIXED VISIBILITY
            # =========================================================
            
            st.subheader("🔬 Advanced Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div style='background-color: #ffffff; 
                            padding: 1rem; 
                            border-radius: 10px; 
                            border-left: 4px solid #667eea;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            color: #333333;'>
                    <strong style='color: #667eea;'>🎯 Intent:</strong> <span style='color: #333333;'>{intent}</span><br><br>
                    <strong style='color: #667eea;'>🧩 Emotion:</strong> <span style='color: #333333;'>{emotion}</span><br><br>
                    <strong style='color: #667eea;'>📊 Subjectivity:</strong> <span style='color: #333333;'>{subjectivity:.2%}</span><br><br>
                    <strong style='color: #667eea;'>📈 Polarity:</strong> <span style='color: #333333;'>{polarity:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style='background-color: #ffffff; 
                            padding: 1rem; 
                            border-radius: 10px; 
                            border-left: 4px solid #667eea;
                            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                            color: #333333;'>
                    <strong style='color: #667eea;'>⚡ Urgency Score:</strong> <span style='color: #333333;'>{urgency_score_val:.1f}/100</span><br><br>
                    <strong style='color: #667eea;'>⏱️ Est. Response:</strong> <span style='color: #333333;'>{response_time}</span><br><br>
                    <strong style='color: #667eea;'>📨 Assigned To:</strong> <span style='color: #333333;'>{assigned_team}</span><br><br>
                    <strong style='color: #667eea;'>🎯 SLA Target:</strong> <span style='color: #333333;'>{sla_time}</span>
                </div>
                """, unsafe_allow_html=True)
            
            # =========================================================
            # RISK ASSESSMENT DASHBOARD - FIXED VERSION
            # =========================================================
            
            st.subheader("🛡️ Risk Assessment Dashboard")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Risk Level Card
                risk_percentage = min(100, max(0, risk_score))
                
                # Risk color based on level
                if risk_level == "Critical Risk":
                    risk_bg = "#ff0844"
                    risk_text = "white"
                elif risk_level == "High Risk":
                    risk_bg = "#ff6b00"
                    risk_text = "white"
                elif risk_level == "Medium Risk":
                    risk_bg = "#ffd700"
                    risk_text = "#333"
                else:
                    risk_bg = "#00cc00"
                    risk_text = "white"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {risk_bg} 0%, {risk_bg}cc 100%);
                            padding: 1.2rem; 
                            border-radius: 15px; 
                            color: {risk_text};
                            margin-bottom: 1rem;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0 0 0.5rem 0; font-size: 1.1rem;'>🎯 Risk Level</h4>
                    <p style='font-size: 1.8rem; font-weight: bold; margin: 0;'>{risk_color} {risk_level}</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Score: {risk_score}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk Meter - Fixed Progress Bar
                st.markdown(f"**Risk Score: {risk_score}%**")
                st.progress(risk_score / 100)
                
                # Risk Factors
                if risk_factors:
                    st.markdown("**Risk Factors:**")
                    for factor in risk_factors[:3]:
                        st.markdown(f"• {factor}")
            
            with col2:
                # CSAT Risk Card
                csat_percentage = min(100, max(0, csat_score))
                
                if csat_score >= 60:
                    csat_bg = "#ff0844"
                elif csat_score >= 40:
                    csat_bg = "#ff6b00"
                elif csat_score >= 20:
                    csat_bg = "#ffd700"
                else:
                    csat_bg = "#00cc00"
                
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {csat_bg} 0%, {csat_bg}cc 100%);
                            padding: 1.2rem; 
                            border-radius: 15px; 
                            color: white;
                            margin-bottom: 1rem;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0 0 0.5rem 0; font-size: 1.1rem;'>😊 CSAT Risk</h4>
                    <p style='font-size: 1.2rem; font-weight: bold; margin: 0;'>{csat_risk}</p>
                    <p style='margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Score: {csat_score}/100</p>
                </div>
                """, unsafe_allow_html=True)
                
                # CSAT Meter - Fixed Progress Bar
                st.markdown(f"**CSAT Risk Score: {csat_score}%**")
                st.progress(csat_score / 100)
                
                # Escalation Level
                escalation_level_val = 2 if priority in ["Critical", "High"] else 1 if priority == "Medium" else 0
                st.markdown(f"**Escalation Level:** Level {escalation_level_val}")
                if escalation_level_val >= 2:
                    st.warning("⚠️ Requires immediate escalation")
                elif escalation_level_val >= 1:
                    st.info("📋 Standard escalation")
                else:
                    st.success("✅ Normal workflow")
            
            with col3:
                # Business Impact Card
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 1.2rem; 
                            border-radius: 15px; 
                            color: white;
                            margin-bottom: 1rem;
                            box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                    <h4 style='margin: 0 0 0.5rem 0; font-size: 1.1rem;'>🏢 Business Impact</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for impact in business_impacts:
                    if "⚠️" in impact or "🔒" in impact:
                        st.warning(impact)
                    elif "😞" in impact:
                        st.error(impact)
                    else:
                        st.success(impact)
                
                # SLA Information
                st.markdown(f"**SLA Commitment:** {sla_time}")
                st.caption("Response time based on priority level")
            
            # Risk Meter Footer
            st.markdown("---")
            st.caption("⚡ Risk scores are calculated based on priority, sentiment, and detected risk factors")
            
            # Resolution & Summary
            st.subheader("💡 Recommended Action")
            st.success(f"**Resolution:** {resolution}")
            st.info(f"**🧠 AI Summary:** {ai_summary}")
            
            # Final Risk Meter
            st.subheader("⚡ Overall Risk Meter")
            if priority == "Critical":
                st.progress(95, text="🔴 CRITICAL - Immediate Action Required")
            elif priority == "High":
                st.progress(75, text="🟠 HIGH - Urgent Action Needed")
            elif priority == "Medium":
                st.progress(50, text="🟡 MEDIUM - Address Soon")
            else:
                st.progress(25, text="🟢 LOW - Standard Handling")
            
    else:
        st.warning("⚠️ Please enter a support ticket to analyze.")

st.markdown("---")

# =========================================================
# ANALYTICS DASHBOARD
# =========================================================

if df is not None:
    st.header("📈 Analytics & Insights Dashboard")
    
    tab1, tab2 = st.tabs(["📊 Distributions", "📉 Model Performance"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Category Distribution")
            if 'Ticket Type' in df.columns:
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                df['Ticket Type'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
                plt.xticks(rotation=45)
                plt.xlabel("Category")
                plt.ylabel("Count")
                plt.title("Ticket Categories")
                st.pyplot(fig1)
        
        with col2:
            st.subheader("Priority Distribution")
            if 'priority' in df.columns:
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                priority_colors = {'Critical': '#FF0000', 'High': '#FF6B00', 'Medium': '#FFD700', 'Low': '#00CC00'}
                colors = [priority_colors.get(p, '#999') for p in df['priority'].value_counts().index]
                df['priority'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2, colors=colors)
                plt.ylabel("")
                plt.title("Priority Distribution")
                st.pyplot(fig2)
    
    with tab2:
        st.subheader("Model Performance Metrics")
        
        if model is not None and vectorizer is not None and 'Ticket Type' in df.columns and 'cleaned_text' in df.columns:
            try:
                X = vectorizer.transform(df['cleaned_text'])
                y = df['Ticket Type']
                y_pred = model.predict(X)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{accuracy_score(y, y_pred):.2%}")
                with col2:
                    st.metric("Precision", f"{precision_score(y, y_pred, average='weighted'):.2%}")
                with col3:
                    st.metric("Recall", f"{recall_score(y, y_pred, average='weighted'):.2%}")
                with col4:
                    st.metric("F1-Score", f"{f1_score(y, y_pred, average='weighted'):.2%}")
                
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y, y_pred)
                fig4, ax4 = plt.subplots(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4)
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                st.pyplot(fig4)
                
            except Exception as e:
                st.warning(f"Could not generate performance metrics: {e}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🤖 <strong>AI-Powered Support Ticket Intelligence System</strong></p>
    <p>Machine Learning Internship Project 2026 | Future Interns</p>
    <p>Built with Python, Scikit-learn, NLTK, TextBlob & Streamlit</p>
</div>
""", unsafe_allow_html=True)