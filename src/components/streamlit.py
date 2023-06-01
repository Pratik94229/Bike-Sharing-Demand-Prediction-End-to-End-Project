import streamlit as st
import pandas as pd
import pickle
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler




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
  st.write('[mobile weight:', hr)





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
    df=pd.DataFrame([dict])
    
    
  

    #Loading saved model
    loaded_model = pickle.load(open(r'C:\Users\prati\Desktop\Project\Bike sharing demand Prediction\artifacts\model.pkl','rb'))
    y_preds=loaded_model.predict(df)
    st.subheader("Expected Bike Sharing demand")
    st.write(round(y_preds[0],0))
 


























