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

#model = tf.keras.models.load_model(r'bestModel\bestmodel.h5')
model = tf.keras.models.load_model('bestmodel.h5')
#model = tf.keras.models.load_model(https://github.com/313Ade/Sentiment-Analysis-Project/blob/main/bestModel/bestmodel.h5)

#tokenizing and stuff

tokenizer = Tokenizer()
#with open(r'C:\Users\User\Desktop\SA\tokenizer.pkl', 'rb') as tokenizer_file:
with open('tokenizer.pkl', 'rb') as tokenizer_file:
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

#def analyze(x):
    if x>= 0.3:
        return 'Positive'
    elif x <= -0.3:
        return 'Negative'
    else:
        return 'Neutral'


## Streamlit starts here ##
st.title('Senty!')

# Insert containers separated into tabs:
tab1, tab2 = st.tabs(["Welcome", "Instructions"])

with tab1:
    tab1.write("Welcome to my Sentiment Analysis webapp powered by Streamlit! Please read the instructions.")


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


# with st.expander('Analyze CSV'):
upl = st.file_uploader('Upload CSV file')    

def analyze(text):
    if text >= 0.5:
        return 'Positive'
    elif text <= 0.3:
        return 'Negative'
    else:
        return 'Neutral'
        
df = None

if st.button("Analyze data"):

    with st.spinner():

        if upl:
            df = pd.read_csv(upl)
            #df['Processed_Text'] = df.iloc[:,0].apply(preprocess_apply)
            #input_data = np.array([processed_text]) 
            #input_data = np.array(df['Processed_Text']) #tuple index out of range >>#this is probably the wrong error
            
            #sentiment = model.predict(input_data)
            #prediction = predict_sentiment(input_data) 
            #input 0 of LSTM layer is incompatible with the layer, expected ndim3 but found ndim2 

            #df['Analysis'] = df['Reviews'].apply(preprocess_apply) #cant rememeber why I did this

            df['Sentiment'] = np.nan
                
        
            df['Processed'] = df.iloc[:,0].apply(preprocess_apply)
            df['Value'] = df['Processed'].apply(predict_sentiment)
            
            df['Value'] = df['Value'].apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)
            df['Sentiment'] = df['Value'].apply(analyze)
            #.apply(lambda x: x[0] if isinstance(x, np.ndarray) else x)

            #df['Processed'] = df['Processed'].apply(lambda x: [round(x[0], 2)] if isinstance(x, np.ndarray) else x)
                
            #df['Processed Text'] = df.iloc[:,0].apply(predict_sentiment)

            
            #st.write(input.shape)
            #input = input.reshape(6,100) #tried to reshape the array
            #sentiment = model.predict(input) 


            ### TextBlob ###

            #df['Processed_Text'] = df.iloc[:,0].apply(preprocess_apply)
            #df['Score'] = df['Processed_Text'].apply(score)
            #df['Analysis'] = df['Score'].apply(analyze)

            ###---###           

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
                



