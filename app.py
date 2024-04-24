 
import pickle
import streamlit as st
from keras.utils import pad_sequences

# loading the trained model
pickle_in = open('Classifier2.pkl', 'rb') 
pickle_in_tok = open('tokenizer2.pkl', 'rb') 
classifier = pickle.load(pickle_in)
tokenizer = pickle.load(pickle_in_tok)

@st.cache_data()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(text):
    
    new_text_seq = tokenizer.texts_to_sequences([text])
    new_text_padded = pad_sequences(new_text_seq, padding='post', maxlen=35)  # Use the max_len determined during training
    
    predictions = classifier.predict(new_text_padded)
    
     # Making predictions
    predicted_class_index = predictions.argmax(axis=-1)
    print(predicted_class_index)
    if predicted_class_index[0] == 0:
        pred = "Negative"
    elif predicted_class_index[0] == 1:
        pred = "Positive"
    else:
        pred = "Neutral"
        
    return pred
      
      
  
# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:Maroon;padding:1px"> 
    <h1 style ="color:white;text-align:center;">Streamlit Sentiment Classifier App</h1> 
    </div> 
    """
      
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
    # following lines create boxes in which user can enter data required to make prediction 

    text=st.text_area("ENTER A REVIEW")
   
    result =""
      
    # when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(text) 
        st.success('The sentiment is {}'.format(result))
        print(result)
     
if __name__=='__main__': 
    main()
