import os
import streamlit as st
import pandas as pd
import warnings 
import tensorflow as tf
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
warnings.filterwarnings('ignore')

from tensorflow.keras.models import load_model

#model = tf.keras.models.load_model(r'bestModel\bestmodel.h5') #for local development
model = tf.keras.models.load_model('bestmodel.h5') #for deployments

#tokenizing and stuff

tokenizer = Tokenizer()
#with open(r'C:\Users\User\Desktop\SA\tokenizer.pkl', 'rb') as tokenizer_file: #for local development
with open('tokenizer.pkl', 'rb') as tokenizer_file: #for deployment
    loaded_tokenizer = pickle.load(tokenizer_file)

# Defining regex patterns

urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|(www\.)[^ ]*)"
userPattern       = '@[^\s]+'
hashtagPattern    = '#[^\s]+'
alphaPattern      = "[^a-z0-9<>]"
sequencePattern   = r"(.)\1\1+"
seqReplacePattern = r"\1\1"

# Defining regex for emojis
smileemoji        = r"[8:=;]['`\-]?[)d]+"
sademoji          = r"[8:=;]['`\-]?\(+"
neutralemoji      = r"[8:=;]['`\-]?[\/|l*]"
lolemoji          = r"[8:=;]['`\-]?p+"

def preprocess_apply(text):

    text = text.lower()

    # Replace all URls with '<url>'
    text = re.sub(urlPattern,'<url>',text)
    # Replace @USERNAME to '<user>'.
    text = re.sub(userPattern,'<user>', text)
    
    # Replace 3 or more consecutive letters by 2 letter.
    text = re.sub(sequencePattern, seqReplacePattern, text)

    # Replace all emojis.
    text = re.sub(r'<3', '<heart>', text)
    text = re.sub(smileemoji, '<smile>', text)
    text = re.sub(sademoji, '<sadface>', text)
    text = re.sub(neutralemoji, '<neutralface>', text)
    text = re.sub(lolemoji, '<lolface>', text)

    # Remove non-alphanumeric and symbols
    text = re.sub(alphaPattern, ' ', text)

    # Adding space on either side of '/' to seperate words (After replacing URLS).
    text = re.sub(r'/', ' / ', text)
    return text


def preprocess_text2(text):
    sequence = loaded_tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=60)
    return padded_sequence

def predict_sentiment(text):
    preprocessed_text = preprocess_text2(text)
    # Ensure the input has three dimensions (batch_size, sequence_length, features)
    preprocessed_text = preprocessed_text.reshape(1, preprocessed_text.shape[1], 1)
    # Predict sentiment
    prediction = model.predict(preprocessed_text)
    return prediction

def analyze(text):
    if text >= 0.5:
        return 'Positive'
    elif text <= 0.3:
        return 'Negative'
    else:
        return 'Neutral'



## Streamlit starts here ##
st.title('Senty!')

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Welcome", "Instructions"])

with tab1:
    tab1.subheader("Welcome! Go through the instructions or analyze a paragraph.")


with tab2:
    st.markdown(
        """
        <div style="text-align: center; padding: 10px;">
            <h3> How To Use </h3>
        </div>
        <div style="text-align:left; padding:0px;">
        <li> Prepare your CSV file. Ensure the first column contains your desired text. <span><a href= https://support.microsoft.com/en-us/office/rearrange-the-order-of-columns-in-a-table-d1701654-fe43-4ae3-adbc-29ee03a97054 target='_blank'> How? </a></span></li>
        <li> Upload your CSV file using the Analyze CSV dropdown </li>
        <li> After a few seconds, your processed dataset should be ready for download</li>
        <li> Processing time depends on the size of your dataset </li>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_save():
    st.session_state.text =''

with st.expander('Analyze text'):
    text = st.text_area(label='Your text', key='text')
    if st.button('Analyze sentiment'):
        
        if text:
            text = preprocess_apply(text)
            text = predict_sentiment(text)

            sentiment = analyze(text)
            sentiment_result = analyze(text)
            #st.write(f"The sentiment is {sentiment}!")
            #st.write(f'<span style="color:{style_sentiment(sentiment_result)}">{sentiment_result}</span>', unsafe_allow_html=True)
            if sentiment_result == 'Negative':
                st.write(f'The text has a :red[{sentiment}] sentiment.')
            elif sentiment_result == 'Positive':
                st.write(f'The text has a :green[{sentiment}] sentiment.')
            else:
                st.write(f'The text is just :grey[{sentiment}].')
            #st.markdown(f'<div style="padding: 10px; border-radius: 5px; {style_sentiment(sentiment_result)}">{sentiment_result}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter some text to analyze.")


    st.button('Clear', on_click=get_save) #clear text input
        

with st.expander('Analyze CSV'):
      
    upl = st.file_uploader('Upload CSV file')   

        
df = None

with st.spinner():

    if upl:
        df = pd.read_csv(upl, encoding='unicode_escape')
        
        df['Sentiment'] = np.nan
            
    
        df['Processed'] = df.iloc[:,0].apply(preprocess_apply)
        df['Value'] = df['Processed'].apply(predict_sentiment)
        
        df['Value'] = df['Value'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
        df['Sentiment'] = df['Value'].apply(analyze)
                       

        st.markdown("""---""")
        st.subheader('Sample')
        
        # Display the updated DataFrame with sentiment predictions
        st.write('**Random sample of 10 rows with their sentiment predictions:**')
        
        st.write(df.sample(10))
    
    
        st.title('Dataset Summary')
        
        avg_sentiment = np.mean(df['Value'])
        
        count = df['Sentiment'].value_counts()
        
        st.write(f'Average Sentiment Score: {avg_sentiment[0]:.2f}')
        st.write(count)
        
        st.markdown("""---""")
    
        
        width = 5
        height = 5
        fig,ax = plt.subplots(figsize=(width,height), facecolor='none')
        ax.pie(count,labels=count.index,autopct='%1.1f%%',colors=['red', 'green', 'grey'])
        plt.title('Percentage of Sentiments')
        plt.style.use('dark_background')
        st.pyplot(fig)
        ### ###

    # else:
    #     st.write("Please upload a file")



# @st.cache_data
# def cacheDF(df):
#         return df.to_csv().encode('utf-8')

# csv = cacheDF(df)
# cached_csv = cacheDF(df)

if df is not None:
    st.markdown("""---""")
    st.write('Click the button below, your dataset is available for download.')
    # Displaying the download button with a dynamically generated file_name
    # Get the name of the uploaded file without the extension
    file_name = upl.name.split('.')[0] if upl else "sentiments"

    # Set the file_name dynamically
    file_name = f'{file_name}_processed.csv'

    # Display the download button
    if st.download_button(
        label='Download processed data',
        data=df.to_csv().encode("utf-8"),
        file_name=file_name,
        mime='text/csv'
    ):
        print("here")
        st.stop()
st.stop()
                



