import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib

st.set_page_config(page_title="Welcome to Placement Prediction Project", layout="wide")
matplotlib.use("agg")

_lock = RendererAgg.lock


#st.write("# Welcome to Placement Prediction Project  ")

st.sidebar.success("Select a Distribution below.")

st.markdown(
    """
     # Welcome to Exploratory Data Analysis of Placement Prediction Project
"""
)


data = pd.read_csv("Data.csv")
data.dropna()

def show_data():
    st.write(data.head(10))

# Convert string to lower case
data["ssc_b"] = data["ssc_b"].str.lower()
data["hsc_b"] = data["hsc_b"].str.lower()
data["hsc_s"] = data["hsc_s"].str.lower()
data["degree_t"] = data["degree_t"].str.lower()
data["workex"] = data["workex"].str.lower()
data["specialisation"] = data["specialisation"].str.lower()

def show_distribution():
    #gender
    row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )

    with row3_1, _lock:
        # st.subheader("Gender Distribution")
        temp=data["gender"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="gender", data=data)
        ax.set_title("Gender Distribution")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]

        st.pyplot(plt)
        #plt.show()

    with row3_2, _lock:
    #SSc_Branch
        temp=data["ssc_b"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="ssc_b", data=data)
        ax.set_title("SSC Branch Distribution")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]
        st.pyplot(plt)

    st.write("")
    row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )
    with row4_1, _lock:
    #HSC Branch
        temp=data["hsc_b"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="hsc_b", data=data)
        ax.set_title("HSC Branch")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]

        # plt.show()
        st.pyplot(plt)


    #HSC Stream
    with row4_2, _lock:
        temp=data["hsc_s"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="hsc_s", data=data)
        ax.set_title("HSC Stream")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]

        # plt.show()
        st.pyplot(plt)

    st.write("")
    row5_space1, row5_1, row5_space2, row5_2, row5_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )
    with row5_1, _lock:

        #Degree Stream
        temp=data["degree_t"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="degree_t", data=data)
        ax.set_title("Degree")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]

        # plt.show()
        st.pyplot(plt)

    with row5_2, _lock:
        #Work Exp
        temp=data["workex"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="workex", data=data)
        ax.set_title("Work Experience")
        ax.set_ylabel('workex')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]

        # plt.show()
        st.pyplot(plt)

    st.write("")
    row6_space1, row6_1, row6_space2, row6_2, row6_space3 = st.columns(
        (0.1, 1, 0.1, 1, 0.1)
    )

    with row6_1, _lock:

        #specialisation
        temp=data["specialisation"].value_counts()
        x_labels=temp.index
        plt.figure(figsize=(8, 4))
        #ax = temp.plot(kind='bar', alpha=0.4)
        ax = sns.countplot(x="specialisation", data=data)
        ax.set_title("Specialisation Distribution")
        ax.set_ylabel('Count')
        ax.set_xticklabels(x_labels)

        rects = ax.patches
        labels = list(temp.values/temp.values.sum()*100)
        labels=[str(round(l,0))+'%' for l in labels]
        # for item in ax.get_xticklabels():
        #         item.set_rotation(90)
        plt.subplots_adjust()
        #plt.show()
        st.pyplot(plt)

def gender_status():
    st.markdown("## How many Male /Females got placed ")
    data.groupby(['gender', 'status']).size().unstack().plot(kind='bar', stacked=True)
    #plt.show()
    st.pyplot(plt)

def ssb_status():
    st.markdown("## How many Male /Females got placed ")
    data.groupby(['ssc_b', 'status']).size().unstack().plot(kind='bar', stacked=True)
    st.pyplot(plt)

def hsb_status():
    st.markdown("## How many Male /Females got placed ")
    data.groupby(['hsc_b', 'status']).size().unstack().plot(kind='bar', stacked=True)
    st.pyplot(plt)

def degree_status():
    st.markdown("## How many Male /Females got placed ")
    data.groupby(['degree_t','status']).size().unstack().plot(kind='bar',stacked=True)
    st.pyplot(plt)

def workex_status():
    st.markdown("## How many Male /Females got placed ")
    data.groupby(['workex','status']).size().unstack().plot(kind='bar',stacked=True)
    st.pyplot(plt)


page_names_to_funcs = {
    "Show Data": show_data,
    "Distributions Demo": show_distribution,
    "Gender Distribution vs Status": gender_status,
    "SSB Distribution vs Status": ssb_status,
    "HSB Distribution vs Status": hsb_status,
    "Degree Distribution vs Status": degree_status,
    "Workex Distribution vs Status": workex_status
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
