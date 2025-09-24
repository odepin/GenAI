import streamlit as st
import matplotlib.pyplot as plt

# Import your scripts as modules
import src.no_agent as na
import src.openai_agent as oa

# Title of the app
st.title("Financial Analysis App")

# Dropdown to select the script
script_options = {
    "No Framework": na.run_analysis,  
    "OpenAI Agents": oa.run_analysis,      
}

selected_script_name = st.selectbox(
    "Select the agent framework to use:",
    options=list(script_options.keys())
)

selected_function = script_options[selected_script_name]

# Input for stock name
stock_name = st.text_input("Enter the stock symbol (e.g., AAPL, TSLA):")

col1, col2, col3 = st.columns([2,1,2])

with col2:
    analyze_button = st.button("Analyze")

# Run button
if analyze_button:
    if not stock_name:
        st.warning("Please enter a stock symbol to proceed.")
    else:
        try:
            warning = st.warning("Analyzing... Please wait.")

            # Call the selected function with the stock_name argument
            investment_thesis, hist = selected_function(stock_name)

            success = st.success(f"{stock_name} sucessfully analyzed with {selected_script_name}!")
            warning.empty()

            # Select 'Open' and 'Close' columns from the hist dataframe
            hist_selected = hist[['Open', 'Close']]
    
            # Create a new figure in matplotlib
            fig, ax = plt.subplots()
    
            # Plot the selected data
            hist_selected.plot(kind='line', ax=ax)
    
            # Set the title and labels
            ax.set_title(f"{stock_name} Stock Price")
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price")

            # Display the plot in Streamlit
            st.pyplot(fig)
    
            st.write("Investment Thesis / Recommendation:")
    
            st.markdown(investment_thesis, unsafe_allow_html=True)
            success.empty()
        except Exception as e:
            st.code(e.args)
