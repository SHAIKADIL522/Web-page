import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import plotly.graph_objects as go
import time
import datetime
import numpy as np
import re
from streamlit_option_menu import option_menu
import uuid
import hashlib
import sqlite3
import os
import base64
from PIL import Image
import io

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Login'
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# Function to add background image
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_base64(base64_string, opacity=0.3):
    st.markdown(
        f"""
        <style>
        .stApp:before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{base64_string}");
            background-size: cover;
            background-position: center;
            opacity: {opacity};
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_placeholder_img(width=1200, height=800, text="Finance Background", bg_color="#052963", text_color="#ffffff"):
    """Generate a placeholder image with text"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64
    
    # Create a new image with the given background color
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Try to use a font or fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = d.textsize(text, font=font)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw the text
    d.text(position, text, fill=text_color, font=font)
    
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

# Background images for different pages
def set_background_for_page(page_name):
    bg_images = {
        'Login': "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?q=80&w=1200",
        'Register': "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?q=80&w=1200",
        'Dashboard': "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=1200",
        'Analyze': "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=1200",
        'History': "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=1200",
        'About': "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?q=80&w=1200",
        'Settings': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1200"
    }
    
    # If online image URLs don't work, use generated placeholders
    try:
        add_bg_from_url(bg_images.get(page_name, bg_images['Login']))
    except:
        # Fallback to generated placeholder
        img_base64 = get_placeholder_img(text=f"{page_name} Background")
        add_bg_from_base64(img_base64)

# Apply page background and overlay styles
def set_page_style():
    # Apply background image based on current page
    set_background_for_page(st.session_state['current_page'])
    
    # Add a semi-transparent overlay for better readability
    overlay_color = "rgba(0, 0, 0, 0.7)" if st.session_state['dark_mode'] else "rgba(255, 255, 255, 0.8)"
    text_color = "#FFFFFF" if st.session_state['dark_mode'] else "#000000"
    
    st.markdown(f"""
    <style>
    .block-container {{
        background-color: {overlay_color};
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem;
        color: {text_color};
    }}
    
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }}
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: {'#333333' if st.session_state['dark_mode'] else '#FFFFFF'};
        color: {'white' if st.session_state['dark_mode'] else 'black'};
        border-radius: 5px;
        border: 1px solid {'#555555' if st.session_state['dark_mode'] else '#DDDDDD'};
    }}
    
    h1, h2, h3 {{
        font-family: 'Arial', sans-serif;
        color: {'#FFFFFF' if st.session_state['dark_mode'] else '#052963'};
    }}
    
    .stSidebar {{
        background-color: {'#222222' if st.session_state['dark_mode'] else '#F0F2F6'};
    }}
    
    .css-1544g2n {{  /* This targets the sidebar */
        padding-top: 2rem;
    }}
    
    .fraud-card {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: {'#2d2d2d' if st.session_state['dark_mode'] else '#f9f9f9'};
    }}
    
    .metric-card {{
        background-color: {'#333333' if st.session_state['dark_mode'] else '#FFFFFF'};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 15px;
    }}
    
    .login-container {{
        max-width: 400px;
        margin: 0 auto;
        padding: 30px;
        background-color: {'rgba(40, 40, 40, 0.9)' if st.session_state['dark_mode'] else 'rgba(255, 255, 255, 0.9)'};
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# Database setup
def init_db():
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        created_at TIMESTAMP
    )
    ''')
    
    # Create search history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS search_history (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        query TEXT,
        result TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Add some default users for testing
    try:
        default_password = hashlib.sha256("password123".encode()).hexdigest()
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", 
                 (str(uuid.uuid4()), "admin", default_password, "admin@example.com", datetime.datetime.now()))
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", 
                 (str(uuid.uuid4()), "user", default_password, "user@example.com", datetime.datetime.now()))
    except:
        pass
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Authentication functions
def login_user(username, password):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    
    conn.close()
    
    if user:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['user_id'] = user[0]
        return True
    return False

def register_user(username, password, email):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                 (user_id, username, hashed_password, email, datetime.datetime.now()))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def save_search_history(user_id, query, result):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    search_id = str(uuid.uuid4())
    c.execute("INSERT INTO search_history VALUES (?, ?, ?, ?, ?)", 
             (search_id, user_id, query, result, datetime.datetime.now()))
    
    conn.commit()
    conn.close()

def get_user_search_history(user_id):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    c.execute("SELECT query, result, created_at FROM search_history WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    history = c.fetchall()
    
    conn.close()
    return history

# Load the saved T5 model and tokenizer
@st.cache_resource
def load_model():
    saved_model_path = r"./Model"
    saved_tokenizer_path = r"./Model"
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
        tokenizer = T5Tokenizer.from_pretrained(saved_tokenizer_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Sample transaction data (expanded with more entries)
@st.cache_data
def load_transaction_data():
    return pd.DataFrame({
        'transaction_id': range(1, 11),
        'type': ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'TRANSFER', 
                'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT', 'TRANSFER'],
        'amount': [420330.71, 50000.00, 10000.00, 25000.00, 100000.00,
                  75000.00, 5000.00, 1200.00, 30000.00, 250000.00],
        'nameOrig': ['C1868228472', 'C123456789', 'C987654321', 'C555555555', 'C777777777',
                    'C888999000', 'C111222333', 'C444555666', 'C000111222', 'C333444555'],
        'nameDest': ['M123456789', 'C888888888', 'C999999999', 'M444444444', 'C666666666',
                    'M555666777', 'C888999111', 'M222333444', 'C999888777', 'M111000999'],
        'oldbalanceOrg': [420330.71, 50000.00, 0.00, 25000.00, 100000.00,
                         75000.00, 0.00, 2000.00, 30000.00, 250000.00],
        'newbalanceOrig': [0.00, 0.00, 10000.00, 0.00, 0.00,
                          0.00, 5000.00, 800.00, 0.00, 0.00],
        'oldbalanceDest': [0.00, 10000.00, 0.00, 100000.00, 5000.00,
                          2000.00, 0.00, 50000.00, 10000.00, 0.00],
        'newbalanceDest': [420330.71, 60000.00, 0.00, 125000.00, 105000.00,
                          77000.00, 0.00, 51200.00, 40000.00, 250000.00],
        'isFraud': [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        'timestamp': [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S') 
                     for i in range(10)]
    })

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    try:
        device = next(model.parameters()).device
        inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return ["Error in prediction"]

def analyze_transaction(query, df):
    # Extract details from query using string matching
    transfer_match = re.search(r"transfer of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    cash_out_match = re.search(r"cash_out of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    
    # More general pattern matching
    amount_match = re.search(r"\$([\d\.]+)", query)
    account_match = re.search(r"(C\d+)", query)
    
    if transfer_match or cash_out_match:
        match = transfer_match or cash_out_match
        amount = float(match.group(1))
        account = match.group(2)
        
        # Find matching transactions
        result = df[(df['nameOrig'] == account) & (df['amount'] == amount)]
        return result
    elif amount_match and account_match:
        amount = float(amount_match.group(1))
        account = account_match.group(1)
        
        # Find matching transactions
        result = df[(df['nameOrig'] == account) & (abs(df['amount'] - amount) < 0.01)]
        return result
    
    return pd.DataFrame()

def visualize_transaction(transaction_data):
    if transaction_data.empty:
        return None
    
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())),
                color="blue"
            ),
            link=dict(
                source=[list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())).index(x) for x in transaction_data['nameOrig']],
                target=[list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())).index(x) for x in transaction_data['nameDest']],
                value=transaction_data['amount'],
                color=["red" if x == 1 else "blue" for x in transaction_data['isFraud']]
            )
        )
    ])
    
    fig.update_layout(
        title_text="Transaction Flow Visualization",
        font_size=12,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state['dark_mode'] else 'black')
    )
    
    return fig

# Page functions
def login_page():
    st.title("Fraud Detection System")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    if login_user(username, password):
                        st.success("Logged in successfully!")
                        st.session_state['current_page'] = 'Dashboard'
                        st.experimental_rerun()
                    else:
                        st.error("Invalid username or password")
            with col2:
                if st.button("Register"):
                    st.session_state['current_page'] = 'Register'
                    st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    st.title("Create Your Account")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("### Register")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Create Account"):
                if not username or not email or not password:
                    st.error("All fields are required")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(username, password, email):
                    st.success("Registration successful! You can now login.")
                    st.session_state['current_page'] = 'Login'
                    st.experimental_rerun()
                else:
                    st.error("Username already exists or registration failed")
            
            if st.button("Back to Login"):
                st.session_state['current_page'] = 'Login'
                st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def dashboard_page():
    st.title(f"Fraud Detection Dashboard")
    st.subheader(f"Welcome back, {st.session_state['username']}!")
    
    # Load transaction data
    df = load_transaction_data()
    
    # Dashboard metrics
    st.markdown("### Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Cases", len(df[df['isFraud'] == 1]))
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fraud_percentage = (len(df[df['isFraud'] == 1]) / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent transactions
    st.markdown("### Recent Transactions")
    
    # Add a search filter
    search_term = st.text_input("Search by Account ID")
    
    filtered_df = df
    if search_term:
        filtered_df = df[(df['nameOrig'].str.contains(search_term)) | (df['nameDest'].str.contains(search_term))]
    
    # Show recent transactions with pagination
    transactions_per_page = 5
    if len(filtered_df) > 0:
        page_number = st.selectbox("Page", options=range(1, (len(filtered_df) // transactions_per_page) + 2))
        start_idx = (page_number - 1) * transactions_per_page
        end_idx = min(start_idx + transactions_per_page, len(filtered_df))
        
        for i in range(start_idx, end_idx):
            transaction = filtered_df.iloc[i]
            
            # Create a transaction card
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    card_color = "red" if transaction['isFraud'] == 1 else "green"
                    st.markdown(f"""
                    <div class="fraud-card" style="border-left: 5px solid {card_color};">
                        <h4>Transaction #{transaction['transaction_id']}</h4>
                        <p><strong>Type:</strong> {transaction['type']}</p>
                        <p><strong>Amount:</strong> ${transaction['amount']:,.2f}</p>
                        <p><strong>From:</strong> {transaction['nameOrig']} → <strong>To:</strong> {transaction['nameDest']}</p>
                        <p><strong>Date:</strong> {transaction['timestamp']}</p>
                        <p style="color: {card_color}"><strong>Status:</strong> {"Fraudulent" if transaction['isFraud'] == 1 else "Valid"}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No transactions found")
    
    # Charts and visualizations
    st.markdown("### Transaction Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction by Type
        type_counts = df['type'].value_counts()
        fig_types = go.Figure(data=[
            go.Pie(labels=type_counts.index, 
                  values=type_counts.values,
                  hole=0.3)
        ])
        fig_types.update_layout(
            title="Transaction Types Distribution",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white' if st.session_state['dark_mode'] else 'black')
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # Fraud Amount by Transaction Type
        fraud_by_type = df[df['isFraud'] == 1].groupby('type')['amount'].sum().reset_index()
        fig_fraud_by_type = go.Figure(data=[
            go.Bar(x=fraud_by_type['type'], 
                  y=fraud_by_type['amount'],
                  marker_color='crimson')
        ])
        fig_fraud_by_type.update_layout(
            title="Fraudulent Amount by Transaction Type",
            height=400,
            yaxis_title="Amount ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white' if st.session_state['dark_mode'] else 'black')
        )
        st.plotly_chart(fig_fraud_by_type, use_container_width=True)
    
    # Time series chart for transactions over time
    transaction_by_time = df.copy()
    transaction_by_time['date'] = pd.to_datetime(transaction_by_time['timestamp'])
    transaction_by_time = transaction_by_time.sort_values('date')
    
    fig_time = go.Figure()
    
    # Add all transactions
    fig_time.add_trace(go.Scatter(
        x=transaction_by_time['date'],
        y=transaction_by_time['amount'],
        mode='lines+markers',
        name='All Transactions',
        line=dict(color='blue')
    ))
    
    # Add fraudulent transactions
    fraud_transactions = transaction_by_time[transaction_by_time['isFraud'] == 1]
    fig_time.add_trace(go.Scatter(
        x=fraud_transactions['date'],
        y=fraud_transactions['amount'],
        mode='markers',
        name='Fraudulent',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    fig_time.update_layout(
        title="Transaction Timeline",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state['dark_mode'] else 'black')
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

def analyze_page():
    st.title("Analyze Transactions")
    
    # Load model and data
    model, tokenizer = load_model()
    df = load_transaction_data()
    
    # User input
    st.markdown("### Enter Transaction Query")
    user_query = st.text_area(
        'Enter your query in natural language', 
        placeholder='Example: "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0."',
        height=100
    )
    
    # Add example queries for quick selection
    example_queries = [
        "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0.",
        "Please check whether the cash_out of $50000.00 from account C123456789 might be fraudulent. It started with a balance of $50000.00 and ended with $0.0.",
        "Is the transfer of $100000.00 from account C777777777 suspicious? It started with a balance of $100000.00 and ended with $0.0."
    ]
    
    selected_example = st.selectbox("Or select an example query:", [""] + example_queries)
    if selected_example:
        user_query = selected_example
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("Analyze Transaction")
    
    if analyze_button and user_query:
        with st.spinner("Analyzing transaction..."):
            # Add a small delay to simulate processing
            time.sleep(1)
            
            # Generate prediction using T5 model
            if model and tokenizer:
                predictions = generate_predictions(model, tokenizer, [user_query])
                prediction_result = predictions[0]
                st.info(f"Model Prediction: {prediction_result}")
            else:
                st.warning("Model not loaded. Using rule-based analysis instead.")
                prediction_result = "Unable to generate model prediction"
            
            # Analyze transaction
            result = analyze_transaction(user_query, df)
            
            # Save to history
            if st.session_state['logged_in']:
                save_search_history(st.session_state['user_id'], user_query, prediction_result)
                st.session_state['search_history'].append((user_query, prediction_result, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            if not result.empty:
                st.success("Transaction found in database!")
                
                # Display transaction details
                st.markdown("### Transaction Details:")
                for _, transaction in result.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown(f"**Transaction ID:** {transaction['transaction_id']}")
                        st.markdown(f"**Transaction Type:** {transaction['type']}")
                        st.markdown(f"**Amount:** ${transaction['amount']:,.2f}")
                        st.markdown(f"**Timestamp:** {transaction['timestamp']}")
                    with col2:
                        st.markdown(f"**Sender Account:** {transaction['nameOrig']}")
                        st.markdown(f"**Recipient Account:** {transaction['nameDest']}")
                        st.markdown(f"**Original Balance:** ${transaction['oldbalanceOrg']:,.2f}")
                        st.markdown(f"**New Balance:** ${transaction['newbalanceOrig']:,.2f}")
                    with col3:
                        import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import plotly.graph_objects as go
import time
import datetime
import numpy as np
import re
from streamlit_option_menu import option_menu
import uuid
import hashlib
import sqlite3
import os
import base64
from PIL import Image
import io

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = None
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Login'
if 'search_history' not in st.session_state:
    st.session_state['search_history'] = []
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# Function to add background image
def add_bg_from_url(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def add_bg_from_base64(base64_string, opacity=0.3):
    st.markdown(
        f"""
        <style>
        .stApp:before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-image: url("data:image/png;base64,{base64_string}");
            background-size: cover;
            background-position: center;
            opacity: {opacity};
            z-index: -1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

def get_placeholder_img(width=1200, height=800, text="Finance Background", bg_color="#052963", text_color="#ffffff"):
    """Generate a placeholder image with text"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    import base64
    
    # Create a new image with the given background color
    img = Image.new('RGB', (width, height), color=bg_color)
    d = ImageDraw.Draw(img)
    
    # Try to use a font or fallback to default
    try:
        font = ImageFont.truetype("arial.ttf", 64)
    except IOError:
        font = ImageFont.load_default()
    
    # Calculate text position
    text_width, text_height = d.textsize(text, font=font)
    position = ((width - text_width) // 2, (height - text_height) // 2)
    
    # Draw the text
    d.text(position, text, fill=text_color, font=font)
    
    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return img_str

# Background images for different pages
def set_background_for_page(page_name):
    bg_images = {
        'Login': "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?q=80&w=1200",
        'Register': "https://images.unsplash.com/photo-1579621970563-ebec7560ff3e?q=80&w=1200",
        'Dashboard': "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=1200",
        'Analyze': "https://images.unsplash.com/photo-1639322537228-f710d846310a?q=80&w=1200",
        'History': "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?q=80&w=1200",
        'About': "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?q=80&w=1200",
        'Settings': "https://images.unsplash.com/photo-1554224155-6726b3ff858f?q=80&w=1200"
    }
    
    # If online image URLs don't work, use generated placeholders
    try:
        add_bg_from_url(bg_images.get(page_name, bg_images['Login']))
    except:
        # Fallback to generated placeholder
        img_base64 = get_placeholder_img(text=f"{page_name} Background")
        add_bg_from_base64(img_base64)

# Apply page background and overlay styles
def set_page_style():
    # Apply background image based on current page
    set_background_for_page(st.session_state['current_page'])
    
    # Add a semi-transparent overlay for better readability
    overlay_color = "rgba(0, 0, 0, 0.7)" if st.session_state['dark_mode'] else "rgba(255, 255, 255, 0.8)"
    text_color = "#FFFFFF" if st.session_state['dark_mode'] else "#000000"
    
    st.markdown(f"""
    <style>
    .block-container {{
        background-color: {overlay_color};
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem;
        color: {text_color};
    }}
    
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: bold;
    }}
    
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {{
        background-color: {'#333333' if st.session_state['dark_mode'] else '#FFFFFF'};
        color: {'white' if st.session_state['dark_mode'] else 'black'};
        border-radius: 5px;
        border: 1px solid {'#555555' if st.session_state['dark_mode'] else '#DDDDDD'};
    }}
    
    h1, h2, h3 {{
        font-family: 'Arial', sans-serif;
        color: {'#FFFFFF' if st.session_state['dark_mode'] else '#052963'};
    }}
    
    .stSidebar {{
        background-color: {'#222222' if st.session_state['dark_mode'] else '#F0F2F6'};
    }}
    
    .css-1544g2n {{  /* This targets the sidebar */
        padding-top: 2rem;
    }}
    
    .fraud-card {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        background-color: {'#2d2d2d' if st.session_state['dark_mode'] else '#f9f9f9'};
    }}
    
    .metric-card {{
        background-color: {'#333333' if st.session_state['dark_mode'] else '#FFFFFF'};
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 15px;
    }}
    
    .login-container {{
        max-width: 400px;
        margin: 0 auto;
        padding: 30px;
        background-color: {'rgba(40, 40, 40, 0.9)' if st.session_state['dark_mode'] else 'rgba(255, 255, 255, 0.9)'};
        border-radius: 10px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }}
    </style>
    """, unsafe_allow_html=True)

# Database setup
def init_db():
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    # Create users table
    c.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        username TEXT UNIQUE,
        password TEXT,
        email TEXT,
        created_at TIMESTAMP
    )
    ''')
    
    # Create search history table
    c.execute('''
    CREATE TABLE IF NOT EXISTS search_history (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        query TEXT,
        result TEXT,
        created_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id)
    )
    ''')
    
    # Add some default users for testing
    try:
        default_password = hashlib.sha256("password123".encode()).hexdigest()
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", 
                 (str(uuid.uuid4()), "admin", default_password, "admin@example.com", datetime.datetime.now()))
        c.execute("INSERT OR IGNORE INTO users VALUES (?, ?, ?, ?, ?)", 
                 (str(uuid.uuid4()), "user", default_password, "user@example.com", datetime.datetime.now()))
    except:
        pass
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Authentication functions
def login_user(username, password):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    hashed_password = hashlib.sha256(password.encode()).hexdigest()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    
    conn.close()
    
    if user:
        st.session_state['logged_in'] = True
        st.session_state['username'] = username
        st.session_state['user_id'] = user[0]
        return True
    return False

def register_user(username, password, email):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    try:
        user_id = str(uuid.uuid4())
        hashed_password = hashlib.sha256(password.encode()).hexdigest()
        c.execute("INSERT INTO users VALUES (?, ?, ?, ?, ?)", 
                 (user_id, username, hashed_password, email, datetime.datetime.now()))
        conn.commit()
        conn.close()
        return True
    except:
        conn.close()
        return False

def save_search_history(user_id, query, result):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    search_id = str(uuid.uuid4())
    c.execute("INSERT INTO search_history VALUES (?, ?, ?, ?, ?)", 
             (search_id, user_id, query, result, datetime.datetime.now()))
    
    conn.commit()
    conn.close()

def get_user_search_history(user_id):
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    
    c.execute("SELECT query, result, created_at FROM search_history WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
    history = c.fetchall()
    
    conn.close()
    return history

# Load the saved T5 model and tokenizer
@st.cache_resource
def load_model():
    saved_model_path = r"./Model"
    saved_tokenizer_path = r"./Model"
    
    try:
        model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
        tokenizer = T5Tokenizer.from_pretrained(saved_tokenizer_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Sample transaction data (expanded with more entries)
@st.cache_data
def load_transaction_data():
    return pd.DataFrame({
        'transaction_id': range(1, 11),
        'type': ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'TRANSFER', 
                'CASH_OUT', 'CASH_IN', 'DEBIT', 'PAYMENT', 'TRANSFER'],
        'amount': [420330.71, 50000.00, 10000.00, 25000.00, 100000.00,
                  75000.00, 5000.00, 1200.00, 30000.00, 250000.00],
        'nameOrig': ['C1868228472', 'C123456789', 'C987654321', 'C555555555', 'C777777777',
                    'C888999000', 'C111222333', 'C444555666', 'C000111222', 'C333444555'],
        'nameDest': ['M123456789', 'C888888888', 'C999999999', 'M444444444', 'C666666666',
                    'M555666777', 'C888999111', 'M222333444', 'C999888777', 'M111000999'],
        'oldbalanceOrg': [420330.71, 50000.00, 0.00, 25000.00, 100000.00,
                         75000.00, 0.00, 2000.00, 30000.00, 250000.00],
        'newbalanceOrig': [0.00, 0.00, 10000.00, 0.00, 0.00,
                          0.00, 5000.00, 800.00, 0.00, 0.00],
        'oldbalanceDest': [0.00, 10000.00, 0.00, 100000.00, 5000.00,
                          2000.00, 0.00, 50000.00, 10000.00, 0.00],
        'newbalanceDest': [420330.71, 60000.00, 0.00, 125000.00, 105000.00,
                          77000.00, 0.00, 51200.00, 40000.00, 250000.00],
        'isFraud': [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        'timestamp': [(datetime.datetime.now() - datetime.timedelta(days=i)).strftime('%Y-%m-%d %H:%M:%S') 
                     for i in range(10)]
    })

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    try:
        device = next(model.parameters()).device
        inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True)
        return tokenizer.batch_decode(outputs, skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error generating prediction: {e}")
        return ["Error in prediction"]

def analyze_transaction(query, df):
    # Extract details from query using string matching
    transfer_match = re.search(r"transfer of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    cash_out_match = re.search(r"cash_out of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    
    # More general pattern matching
    amount_match = re.search(r"\$([\d\.]+)", query)
    account_match = re.search(r"(C\d+)", query)
    
    if transfer_match or cash_out_match:
        match = transfer_match or cash_out_match
        amount = float(match.group(1))
        account = match.group(2)
        
        # Find matching transactions
        result = df[(df['nameOrig'] == account) & (df['amount'] == amount)]
        return result
    elif amount_match and account_match:
        amount = float(amount_match.group(1))
        account = account_match.group(1)
        
        # Find matching transactions
        result = df[(df['nameOrig'] == account) & (abs(df['amount'] - amount) < 0.01)]
        return result
    
    return pd.DataFrame()

def visualize_transaction(transaction_data):
    if transaction_data.empty:
        return None
    
    fig = go.Figure(data=[
        go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())),
                color="blue"
            ),
            link=dict(
                source=[list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())).index(x) for x in transaction_data['nameOrig']],
                target=[list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())).index(x) for x in transaction_data['nameDest']],
                value=transaction_data['amount'],
                color=["red" if x == 1 else "blue" for x in transaction_data['isFraud']]
            )
        )
    ])
    
    fig.update_layout(
        title_text="Transaction Flow Visualization",
        font_size=12,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state['dark_mode'] else 'black')
    )
    
    return fig

# Page functions
def login_page():
    st.title("Fraud Detection System")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("### Login")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login"):
                    if login_user(username, password):
                        st.success("Logged in successfully!")
                        st.session_state['current_page'] = 'Dashboard'
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
            with col2:
                if st.button("Register"):
                    st.session_state['current_page'] = 'Register'
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    st.title("Create Your Account")
    
    with st.container():
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            st.markdown("### Register")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            if st.button("Create Account"):
                if not username or not email or not password:
                    st.error("All fields are required")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif register_user(username, password, email):
                    st.success("Registration successful! You can now login.")
                    st.session_state['current_page'] = 'Login'
                    st.rerun()
                else:
                    st.error("Username already exists or registration failed")
            
            if st.button("Back to Login"):
                st.session_state['current_page'] = 'Login'
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

def dashboard_page():
    st.title(f"Fraud Detection Dashboard")
    st.subheader(f"Welcome back, {st.session_state['username']}!")
    
    # Load transaction data
    df = load_transaction_data()
    
    # Dashboard metrics
    st.markdown("### Key Metrics Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Transactions", len(df))
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Amount", f"${df['amount'].sum():,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Fraud Cases", len(df[df['isFraud'] == 1]))
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        fraud_percentage = (len(df[df['isFraud'] == 1]) / len(df)) * 100
        st.metric("Fraud Rate", f"{fraud_percentage:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent transactions
    st.markdown("### Recent Transactions")
    
    # Add a search filter
    search_term = st.text_input("Search by Account ID")
    
    filtered_df = df
    if search_term:
        filtered_df = df[(df['nameOrig'].str.contains(search_term)) | (df['nameDest'].str.contains(search_term))]
    
    # Show recent transactions with pagination
    transactions_per_page = 5
    if len(filtered_df) > 0:
        page_number = st.selectbox("Page", options=range(1, (len(filtered_df) // transactions_per_page) + 2))
        start_idx = (page_number - 1) * transactions_per_page
        end_idx = min(start_idx + transactions_per_page, len(filtered_df))
        
        for i in range(start_idx, end_idx):
            transaction = filtered_df.iloc[i]
            
            # Create a transaction card
            with st.container():
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    card_color = "red" if transaction['isFraud'] == 1 else "green"
                    st.markdown(f"""
                    <div class="fraud-card" style="border-left: 5px solid {card_color};">
                        <h4>Transaction #{transaction['transaction_id']}</h4>
                        <p><strong>Type:</strong> {transaction['type']}</p>
                        <p><strong>Amount:</strong> ${transaction['amount']:,.2f}</p>
                        <p><strong>From:</strong> {transaction['nameOrig']} → <strong>To:</strong> {transaction['nameDest']}</p>
                        <p><strong>Date:</strong> {transaction['timestamp']}</p>
                        <p style="color: {card_color}"><strong>Status:</strong> {"Fraudulent" if transaction['isFraud'] == 1 else "Valid"}</p>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.info("No transactions found")
    
    # Charts and visualizations
    st.markdown("### Transaction Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Transaction by Type
        type_counts = df['type'].value_counts()
        fig_types = go.Figure(data=[
            go.Pie(labels=type_counts.index, 
                  values=type_counts.values,
                  hole=0.3)
        ])
        fig_types.update_layout(
            title="Transaction Types Distribution",
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white' if st.session_state['dark_mode'] else 'black')
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with col2:
        # Fraud Amount by Transaction Type
        fraud_by_type = df[df['isFraud'] == 1].groupby('type')['amount'].sum().reset_index()
        fig_fraud_by_type = go.Figure(data=[
            go.Bar(x=fraud_by_type['type'], 
                  y=fraud_by_type['amount'],
                  marker_color='crimson')
        ])
        fig_fraud_by_type.update_layout(
            title="Fraudulent Amount by Transaction Type",
            height=400,
            yaxis_title="Amount ($)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white' if st.session_state['dark_mode'] else 'black')
        )
        st.plotly_chart(fig_fraud_by_type, use_container_width=True)
    
    # Time series chart for transactions over time
    transaction_by_time = df.copy()
    transaction_by_time['date'] = pd.to_datetime(transaction_by_time['timestamp'])
    transaction_by_time = transaction_by_time.sort_values('date')
    
    fig_time = go.Figure()
    
    # Add all transactions
    fig_time.add_trace(go.Scatter(
        x=transaction_by_time['date'],
        y=transaction_by_time['amount'],
        mode='lines+markers',
        name='All Transactions',
        line=dict(color='blue')
    ))
    
    # Add fraudulent transactions
    fraud_transactions = transaction_by_time[transaction_by_time['isFraud'] == 1]
    fig_time.add_trace(go.Scatter(
        x=fraud_transactions['date'],
        y=fraud_transactions['amount'],
        mode='markers',
        name='Fraudulent',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    fig_time.update_layout(
        title="Transaction Timeline",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white' if st.session_state['dark_mode'] else 'black')
    )
    
    st.plotly_chart(fig_time, use_container_width=True)

def analyze_page():
    st.title("Analyze Transactions")
    
    # Load model and data
    model, tokenizer = load_model()
    df = load_transaction_data()
    
    # User input
    st.markdown("### Enter Transaction Query")
    user_query = st.text_area(
        'Enter your query in natural language', 
        placeholder='Example: "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0."',
        height=100
    )
    
    # Add example queries for quick selection
    example_queries = [
        "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0.",
        "Please check whether the cash_out of $50000.00 from account C123456789 might be fraudulent. It started with a balance of $50000.00 and ended with $0.0.",
        "Is the transfer of $100000.00 from account C777777777 suspicious? It started with a balance of $100000.00 and ended with $0.0."
    ]
    
    selected_example = st.selectbox("Or select an example query:", [""] + example_queries)
    if selected_example:
        user_query = selected_example
    
    col1, col2 = st.columns([1, 5])
    with col1:
        analyze_button = st.button("Analyze Transaction")
    
    if analyze_button and user_query:
        with st.spinner("Analyzing transaction..."):
            # Add a small delay to simulate processing
            time.sleep(1)
            
            # Generate prediction using T5 model
            if model and tokenizer:
                predictions = generate_predictions(model, tokenizer, [user_query])
                prediction_result = predictions[0]
                st.info(f"Model Prediction: {prediction_result}")
            else:
                st.warning("Model not loaded. Using rule-based analysis instead.")
                prediction_result = "Unable to generate model prediction"
            
            # Analyze transaction
            result = analyze_transaction(user_query, df)
            
            # Save to history
            if st.session_state['logged_in']:
                save_search_history(st.session_state['user_id'], user_query, prediction_result)
                st.session_state['search_history'].append((user_query, prediction_result, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
            
            if not result.empty:
                st.success("Transaction found in database!")
                
                # Display transaction details
                st.markdown("### Transaction Details:")
                for _, transaction in result.iterrows():
                    col1, col2, col3 = st.columns([2, 2, 1])
                    with col1:
                        st.markdown(f"**Transaction ID:** {transaction['transaction_id']}")
                        st.markdown(f"**Transaction Type:** {transaction['type']}")
                        st.markdown(f"**Amount:** ${transaction['amount']:,.2f}")
                        st.markdown(f"**Timestamp:** {transaction['timestamp']}")
                    with col2:
                        st.markdown(f"**Sender Account:** {transaction['nameOrig']}")
                        st.markdown(f"**Recipient Account:** {transaction['nameDest']}")
                        st.markdown(f"**Original Balance:** ${transaction['oldbalanceOrg']:,.2f}")
                        st.markdown(f"**New Balance:** ${transaction['newbalanceOrig']:,.2f}")
                    with col3:
                        fraud_status = "Fraudulent" if transaction['isFraud'] == 1 else "Valid"
                        st.markdown(f"**Status:** {fraud_status}")
                        status_color = "red" if transaction['isFraud'] == 1 else "green"
                        st.markdown(f"""
                        <div style="background-color: {status_color}; padding: 10px; border-radius: 5px; text-align: center; color: white;">
                            <h3>{fraud_status}</h3>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Visualize the transaction
                st.markdown("### Transaction Flow:")
                fig = visualize_transaction(result)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors analysis
                st.markdown("### Risk Analysis:")
                risk_factors = []
                
                if transaction['oldbalanceOrg'] > 0 and transaction['newbalanceOrig'] == 0:
                    risk_factors.append("Account completely emptied")
                
                if transaction['nameDest'].startswith('M'):
                    risk_factors.append("Transfer to merchant account")
                
                if transaction['amount'] > 50000:
                    risk_factors.append("High value transaction")
                
                for i, factor in enumerate(risk_factors):
                    st.markdown(f"- {factor}")
                
                if not risk_factors:
                    st.markdown("No significant risk factors identified.")
            else:
                st.error("Transaction not found in database. Please check the account ID and amount.")

def history_page():
    st.title("Search History")
    
    if not st.session_state['logged_in']:
        st.warning("Please log in to view your search history")
        return
    
    # Load history from database
    history = get_user_search_history(st.session_state['user_id'])
    
    if not history:
        st.info("No search history found")
    else:
        st.markdown(f"### Your Recent Searches ({len(history)} queries)")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["List View", "Table View"])
        
        with tab1:
            for query, result, timestamp in history:
                with st.expander(f"Query on {timestamp}"):
                    st.markdown("**Your query:**")
                    st.info(query)
                    st.markdown("**Model result:**")
                    st.success(result)
        
        with tab2:
            history_df = pd.DataFrame(history, columns=["Query", "Result", "Timestamp"])
            st.dataframe(history_df)
        
        # Export options
        st.markdown("### Export Options")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Export as CSV"):
                history_df = pd.DataFrame(history, columns=["Query", "Result", "Timestamp"])
                csv = history_df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="search_history.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)
        with col2:
            if st.button("Clear History"):
                if st.session_state['logged_in']:
                    conn = sqlite3.connect('fraud_detection_app.db')
                    c = conn.cursor()
                    c.execute("DELETE FROM search_history WHERE user_id = ?", (st.session_state['user_id'],))
                    conn.commit()
                    conn.close()
                    st.session_state['search_history'] = []
                    st.success("Search history cleared!")
                    st.experimental_rerun()

def about_page():
    st.title("About Fraud Detection System")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Our Mission
        
        Our fraud detection system aims to help financial institutions identify potentially fraudulent transactions in real-time, reducing losses and protecting customers.
        
        ### Key Features
        
        - Natural language query processing
        - Machine learning-based fraud detection
        - Transaction flow visualization
        - Historical transaction analysis
        - User-friendly dashboard
        
        ### Technology Stack
        
        - Python
        - Streamlit
        - PyTorch
        - Transformers (T5 model)
        - SQLite
        - Plotly
        """)
    
    with col2:
        st.markdown("""
        ### How It Works
        
        1. **Data Collection**: Transaction data is collected and preprocessed
        2. **Feature Engineering**: Key features are extracted from transaction data
        3. **Model Training**: Our T5 model is trained on labeled fraud data
        4. **Inference**: The model analyzes new transactions for fraud patterns
        5. **Visualization**: Results are presented in an intuitive dashboard
        
        ### Privacy & Security
        
        All data is encrypted and securely stored. We comply with financial regulations and prioritize customer privacy.
        
        ### Contact
        
        For more information, please contact us at:
        support@frauddetection.example.com
        """)
    
    # Team section
    st.markdown("### Our Team")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **John Smith**
        
        Chief Data Scientist
        
        Specializes in anomaly detection and machine learning algorithms.
        """)
    
    with col2:
        st.markdown("""
        **Jane Doe**
        
        Security Engineer
        
        Expert in cybersecurity and financial transaction systems.
        """)
    
    with col3:
        st.markdown("""
        **Michael Johnson**
        
        UX/UI Designer
        
        Creates intuitive interfaces for complex data systems.
        """)

def settings_page():
    st.title("Settings")
    
    if not st.session_state['logged_in']:
        st.warning("Please log in to access settings")
        return
    
    st.markdown("### Application Settings")
    
    # Theme settings
    st.markdown("#### Theme")
    dark_mode = st.toggle("Dark Mode", st.session_state['dark_mode'])
    if dark_mode != st.session_state['dark_mode']:
        st.session_state['dark_mode'] = dark_mode
        st.rerun()
    
    # Account settings
    st.markdown("#### Account Settings")
    
    # Get user data
    conn = sqlite3.connect('fraud_detection_app.db')
    c = conn.cursor()
    c.execute("SELECT username, email FROM users WHERE username = ?", (st.session_state['username'],))
    user_data = c.fetchone()
    conn.close()
    
    if user_data:
        username, email = user_data
        
        with st.form("update_profile"):
            st.subheader("Update Profile")
            new_email = st.text_input("Email", value=email)
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password (leave blank to keep current)", type="password")
            confirm_password = st.text_input("Confirm New Password", type="password")
            
            update_button = st.form_submit_button("Update Profile")
            
            if update_button:
                if current_password:
                    # Verify current password
                    hashed_current = hashlib.sha256(current_password.encode()).hexdigest()
                    
                    conn = sqlite3.connect('fraud_detection_app.db')
                    c = conn.cursor()
                    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", 
                             (st.session_state['username'], hashed_current))
                    user_id = c.fetchone()
                    
                    if user_id:
                        # Update email
                        if new_email != email:
                            c.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, user_id[0]))
                        
                        # Update password if provided
                        if new_password:
                            if new_password == confirm_password:
                                hashed_new = hashlib.sha256(new_password.encode()).hexdigest()
                                c.execute("UPDATE users SET password = ? WHERE id = ?", (hashed_new, user_id[0]))
                                st.success("Password updated successfully!")
                            else:
                                st.error("New passwords do not match")
                        
                        conn.commit()
                        conn.close()
                        st.success("Profile updated successfully!")
                    else:
                        conn.close()
                        st.error("Current password is incorrect")
                else:
                    st.error("Please enter your current password to make changes")
    
    # Notification preferences
    st.markdown("#### Notification Preferences")
    email_notifications = st.checkbox("Email Notifications for Fraud Alerts")
    mobile_notifications = st.checkbox("Mobile Notifications")
    
    if st.button("Save Notification Preferences"):
        st.success("Notification preferences saved")
    
    # App information
    st.markdown("#### Application Information")
    st.info("Fraud Detection System v1.0.0")
    st.info("© 2023 FraudGuard Technologies")
    
    # Logout button
    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.session_state['username'] = None
        st.session_state['current_page'] = 'Login'
        st.success("Logged out successfully")
        st.rerun()

# Main app
def main():
    # Initialize sidebar navigation when logged in
    if st.session_state['logged_in']:
        with st.sidebar:
            st.image("https://via.placeholder.com/150x80?text=FraudGuard", width=150)
            
            # Navigation using option menu
            selected = option_menu(
                "Navigation",
                ["Dashboard", "Analyze", "History", "About", "Settings"],
                icons=['house', 'search', 'clock-history', 'info-circle', 'gear'],
                menu_icon="list",
                default_index=0,
                orientation="vertical",
                styles={
                    "container": {"padding": "5!important", "background-color": "#f0f2f6" if not st.session_state['dark_mode'] else "#262730"},
                    "icon": {"color": "orange", "font-size": "25px"}, 
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee" if not st.session_state['dark_mode'] else "#333"},
                    "nav-link-selected": {"background-color": "#02ab21"},
                }
            )
            
            st.session_state['current_page'] = selected
            
            # Display current time
            st.markdown(f"**Current time:** {datetime.datetime.now().strftime('%H:%M:%S')}")
            st.markdown(f"**Logged in as:** {st.session_state['username']}")
    
    # Apply styles based on current page and theme
    set_page_style()
    
    # Render the appropriate page
    if st.session_state['current_page'] == 'Login':
        login_page()
    elif st.session_state['current_page'] == 'Register':
        register_page()
    elif st.session_state['current_page'] == 'Dashboard':
        dashboard_page()
    elif st.session_state['current_page'] == 'Analyze':
        analyze_page()
    elif st.session_state['current_page'] == 'History':
        history_page()
    elif st.session_state['current_page'] == 'About':
        about_page()
    elif st.session_state['current_page'] == 'Settings':
        settings_page()

if __name__ == "__main__":
    main()
                        





