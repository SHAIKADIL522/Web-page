import streamlit as st
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import plotly.graph_objects as go

# Load the saved T5 model and tokenizer
saved_model_path = r"./Model"
saved_tokenizer_path = r"./Model"

model = T5ForConditionalGeneration.from_pretrained(saved_model_path)
tokenizer = T5Tokenizer.from_pretrained(saved_tokenizer_path)

# Sample transaction data (you can replace this with your own dataset)
@st.cache_data
def load_transaction_data():
    return pd.DataFrame({
        'transaction_id': range(1, 6),
        'type': ['TRANSFER', 'CASH_OUT', 'CASH_IN', 'PAYMENT', 'TRANSFER'],
        'amount': [420330.71, 50000.00, 10000.00, 25000.00, 100000.00],
        'nameOrig': ['C1868228472', 'C123456789', 'C987654321', 'C555555555', 'C777777777'],
        'nameDest': ['M123456789', 'C888888888', 'C999999999', 'M444444444', 'C666666666'],
        'oldbalanceOrg': [420330.71, 50000.00, 0.00, 25000.00, 100000.00],
        'newbalanceOrig': [0.00, 0.00, 10000.00, 0.00, 0.00],
        'isFraud': [1, 0, 0, 0, 1]
    })

def generate_predictions(model, tokenizer, input_texts, max_length=128):
    device = next(model.parameters()).device
    inputs = tokenizer.batch_encode_plus(input_texts, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length, num_beams=4, early_stopping=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def analyze_transaction(query, df):
    # Extract details from query using string matching
    import re
    
    transfer_match = re.search(r"transfer of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    cash_out_match = re.search(r"cash_out of \$(.*?) from account (C.*?) might be fraudulent\. It started with a balance of \$(.*?) and ended with \$(.*?)\.", query, re.IGNORECASE)
    
    if transfer_match or cash_out_match:
        match = transfer_match or cash_out_match
        amount = float(match.group(1))
        account = match.group(2)
        
        # Find matching transactions
        result = df[(df['nameOrig'] == account) & (df['amount'] == amount)]
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
                source=[transaction_data['nameOrig'].tolist().index(x) for x in transaction_data['nameOrig']],
                target=[list(set(transaction_data['nameOrig'].tolist() + transaction_data['nameDest'].tolist())).index(x) for x in transaction_data['nameDest']],
                value=transaction_data['amount'],
                color=["red" if x == 1 else "blue" for x in transaction_data['isFraud']]
            )
        )
    ])
    
    fig.update_layout(
        title_text="Transaction Flow Visualization",
        font_size=12,
        height=600
    )
    
    return fig

# Streamlit UI
st.title('Financial Fraud Detection System')

# Sidebar for user input
st.sidebar.title('User Input')
user_query = st.sidebar.text_area(
    'Enter your query in natural language', 
    placeholder='Example: "Please check whether the transfer of $420330.71 from account C1868228472 might be fraudulent. It started with a balance of $420330.71 and ended with $0.0."'
)

if st.sidebar.button('Analyze Transaction'):
    if user_query:
        try:
            # Load transaction data
            df = load_transaction_data()
            
            # Generate prediction using T5 model
            predictions = generate_predictions(model, tokenizer, [user_query])
            st.write("Model Prediction:", predictions[0])
            
            # Analyze transaction
            result = analyze_transaction(user_query, df)
            
            if not result.empty:
                st.subheader("Transaction Details:")
                for _, transaction in result.iterrows():
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("Transaction Type:", transaction['type'])
                        st.write("Amount: $", transaction['amount'])
                        st.write("Sender Account:", transaction['nameOrig'])
                    with col2:
                        st.write("Recipient Account:", transaction['nameDest'])
                        st.write("Fraud Status:", "Fraudulent" if transaction['isFraud'] == 1 else "Not Fraudulent")
                        st.write("New Balance: $", transaction['newbalanceOrig'])
                
                # Visualize transaction
                fig = visualize_transaction(result)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Add additional visualizations
                st.subheader("Transaction Analytics")
                
                # Create three columns for different charts
                chart_col1, chart_col2, chart_col3 = st.columns(3)
                
                with chart_col1:
                    # Transaction Type Distribution
                    type_counts = df['type'].value_counts()
                    fig_types = go.Figure(data=[
                        go.Pie(labels=type_counts.index, 
                              values=type_counts.values,
                              hole=0.3)
                    ])
                    fig_types.update_layout(
                        title="Transaction Types Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_types, use_container_width=True)
                
                with chart_col2:
                    # Fraud vs Non-Fraud
                    fig_fraud = go.Figure(data=[
                        go.Bar(x=['Non-Fraudulent', 'Fraudulent'],
                              y=[len(df[df['isFraud'] == 0]), len(df[df['isFraud'] == 1])],
                              marker_color=['blue', 'red'])
                    ])
                    fig_fraud.update_layout(
                        title="Fraud vs Non-Fraud Transactions",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig_fraud, use_container_width=True)
                
                with chart_col3:
                    # Amount Distribution
                    fig_amount = go.Figure(data=[
                        go.Box(y=df['amount'],
                              name='Transaction Amounts',
                              marker_color='lightseagreen')
                    ])
                    fig_amount.update_layout(
                        title="Transaction Amount Distribution",
                        height=400
                    )
                    st.plotly_chart(fig_amount, use_container_width=True)
                
                # Summary statistics
                st.subheader("Transaction Summary")
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric("Total Transactions", len(df))
                    st.metric("Average Transaction Amount", f"${df['amount'].mean():.2f}")
                
                with stats_col2:
                    st.metric("Total Fraudulent Transactions", len(df[df['isFraud'] == 1]))
                    st.metric("Fraud Rate", f"{(len(df[df['isFraud'] == 1])/len(df)*100):.1f}%")

            else:
                st.warning("No matching transaction found.")
                
        except Exception as e:
            st.error(f"An error occurred: {e}")