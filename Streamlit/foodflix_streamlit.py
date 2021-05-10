import sys
sys.path.insert(0,'/fonctions')

from fonctions  import *
import streamlit as st

# Importing the libraries
import pandas as pd


#%%

# Importing the dataset
df = pd.read_csv('/home/apprenant/PycharmProjects/pythonProject/foodflix/File/02_Cleaned/food_facts_fr.csv')

#df = df.rename(columns={ df.columns[0]: "index" })
df.reset_index(inplace=True)
#%%

#df=df[['index','product_name','brands']]
#%%
df['content'] = df[['product_name','brands']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
df['content'].fillna('Null', inplace = True)


methode=['TfIdf','Countvectorizer']
st.sidebar.image('/home/apprenant/PycharmProjects/pythonProject/foodflix/File/03_image/Foodflix.png',output_format='PNG')
st.title('RECOMMANDATION')

choosen_method= st.sidebar.selectbox('Quelle méthode de recommandation: ',methode)

product = st.text_input('SAISISSEZ UN PRODUIT')
if choosen_method == 'TfIdf':

    found = recotfidf(product,df)
else:

    found = recocount(product,df)

results = {}
results = get_results(df, found)
if product:
    for _, el in enumerate(results):

        st.header(el[0])
        st.subheader(f"Marque : '{el[1]}'")
        col1, col2 = st.beta_columns(2)
        with col1:
            st.markdown(f"**_NUTRISCORE: {el[2]}_**")
            st.markdown(f"**_Allergènes: {el[3]}_**")

        with col2:
            st.text("Valeurs énergétiques :")
            st.dataframe(el[4])
        st.markdown("_______")

