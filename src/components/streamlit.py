import streamlit as st
import os
import sys
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
#from src.logger import logging





st.title(':blue[Bike Sharing Demand Prediction]')
st.subheader(':green[Enter accurate information to predict bike sharing demand]' )


st.subheader(':red[Enter weather specific attribute]') 

#Attributes regarding weather
col1, col2, col3, col4,col5= st.columns(5)  

with col1:
  weathersit=st.number_input('Enter Weather situation',min_value=1, max_value=4)
  st.write('Weather situation:',weathersit)   

with col2:
  temp=st.number_input('Enter Temperature value',min_value=0.02, max_value=1.0)
  st.write('Temerature entered:',temp)

with col3:
  atemp= st.number_input('Enter absolute Temperature',min_value=0.0, max_value=1.0)
  st.write('Temerature entered:',atemp)

with col4:
  hum=st.number_input('Enter  absolute Humidity value',min_value=0.0, max_value=1.0)
  st.write('Humidity entered:',hum)

with col5:
  windspeed=st.number_input('Enter Windspeed value',min_value=0.0, max_value=0.86)
  st.write('Windspeed entered:',windspeed)


# Attributes regarding  
st.subheader(':red[Enter information about time of the year]') 
col1, col2, col3, col4= st.columns(4)




with col1:
  season= st.slider(':green[Season]',1,4)
  st.write('Season:', season)

with col2:
    yr= st.slider(':green[Select Year]',0, 1)
    st.write('Selected Year:', yr)

with col3:
  mnth= st.slider(':green[Select Month of year]',1,12)
  st.write('Selected Month:', mnth)    

with col4:
  hr= st.slider(':green[Select hour of the day]',1,23)
  st.write('Selected hour:', hr)





st.subheader(':red[Enter Specific Day attribute]') 


#Attributes regarding screen
col1, col2, col3, col4= st.columns(4)  


with col1:
  weekday= st.slider(':green[Select day of the week]',0, 6)
  st.write('WeekDay:',weekday)
   

with col2:
  holiday= st.slider(':green[Select holiday or not]',0, 1)
  st.write('Holiday:',holiday)

with col3:
  workingday= st.slider(':green[Select whether working day]',0, 1)
  st.write('Selected value :',workingday)
   





  
 

#Taking input and predicting using saved model 

if st.button('Predict'):
    dict={'season':season, 'yr':yr,'mnth':mnth,'hr':hr,'holiday':holiday,
      'weekday':weekday,'workingday':workingday,'weathersit':weathersit,'temp':temp,'atemp':atemp,'hum':hum,'windspeed':windspeed}
      
    #converting input to dataframe
    input_df=pd.DataFrame([dict])
    #logging.info('Input succesfully taken in streamlit and converted in dataframe')

    #Prepossesing
    train_df = pd.read_csv(os.path.join('artifacts','train.csv'))
    #logging.info('Reading training data completed')
  

    target_column_name = 'cnt'
    drop_columns = [target_column_name,'instant','dteday','casual','registered']

    input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
    target_feature_train_df=train_df[target_column_name]

    #Creating object 
    scaler = StandardScaler()

    ## Transformating using preprocessor obj
    input_feature_train_arr=scaler.fit_transform(input_feature_train_df)
    input_feature_test_arr=scaler.transform(input_df)

    #logging.info("Completed scaling datasets.")

    
    
  

    #Loading saved model
    model_path=os.path.join("artifacts","model.pkl")
  
    loaded_model = pickle.load(open(model_path,'rb'))
    #logging.info('Model loaded')
    y_preds=loaded_model.predict(input_df)
    st.subheader(":red[Expected Bike Sharing Demand]")
   

    # Define CSS style for red color
    red_color = "<style> .red-text { color: red; } </style>"

    # Display the output with red color
    st.markdown(red_color, unsafe_allow_html=True)
    st.subheader(str(round(y_preds[0], 0)))

  
    #logging.info("Project run successful")
 


























