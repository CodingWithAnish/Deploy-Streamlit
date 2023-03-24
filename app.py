import streamlit as st
import numpy as np
import pickle

with open('model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

ticket_sold = st.text_input("Number of tickets sold")
occu_perc = st.text_input("Occupancy percentage")
capacity = st.text_input("Total capacity")
show_time = st.text_input("Show time ")
ticket_price = st.text_input("Ticket price")
ticket_use = st.text_input("Ticket usage rate")

if st.button('Submit'):
    def lr_predict(ticket_sold, occu_perc, capacity, show_time, ticket_price, ticket_use):
        x = np.zeros(6)
        x[0] = ticket_sold
        x[1] = occu_perc
        x[2] = capacity
        x[3] = show_time
        x[4] = ticket_price
        x[5] = ticket_use
        return lr_model.predict([x])[0]

    output = pow(10, lr_predict(ticket_sold, occu_perc, capacity, show_time, ticket_price, ticket_use))

    st.write("Predicted revenue:", output)
