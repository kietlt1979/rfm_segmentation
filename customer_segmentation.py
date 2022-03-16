import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  
# import pandas_profiling as pp
# from scipy.stats import chi2_contingency
# from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import squarify
from datetime import datetime
import pickle
import streamlit as st

# 1. Read data
# data = pd.read_csv('OnlineRetail.csv', encoding='unicode_escape')

#--------------
# GUI
st.title("Data Science Project")
st.write("## Customer Segmentation")

# Upload file/ Read file
uploaded_file = st.file_uploader('Choose a file', type = ['csv'])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='unicode_escape')
    data.to_pickle("OnlineRetail_new.gzip", compression='gzip')
else:
    data = pd.read_pickle("OnlineRetail.gzip",compression='gzip')

# 2. Data pre-processing
# Xóa các hóa đơn bị hủy
data = data.loc[~data['InvoiceNo'].str.startswith('C',na=False)]
# Tạo cột "day" để phân tích RFM
string_to_date = lambda x : datetime.strptime(x, "%d-%m-%Y %H:%M").date()
# Convert InvoiceDate from object to datetime format
data['day'] = data['InvoiceDate'].apply(string_to_date)
data['day'] = data['day'].astype('datetime64[ns]')
# Drop NA values
data = data.dropna()
# Delete quantity < 0
data = data.loc[data.Quantity >0]
# Create gross_sales column
data['gross_sales'] = data.Quantity * data.UnitPrice
# RFM
# Convert string to date, get max date of dataframe
max_date = data['day'].max().date()

Recency = lambda x: (max_date - x.max().date()).days
Frequency = lambda x: len(x.unique())
Monetary = lambda x: round(sum(x), 2)

df_RFM = data.groupby('CustomerID').agg({'day': Recency,
                                        'InvoiceNo': Frequency,
                                        'gross_sales': Monetary})
# Rename the columns of DataFrame
df_RFM.columns = ['Recency', 'Frequency', 'Monetary']
# Descending Sorting
df_RFM = df_RFM.sort_values('Monetary', ascending = False)
# Create labels for Recency, Frequency, Monetary
r_labels = range(4, 0, -1) #số ngày tính từ lần cuối mua hàng lớn thì gán nhãn nhỏ, ngược lại thì nhãn lớn
f_labels = range(1, 5)
m_labels = range(1, 5)
# Assign these labels to 4 equal percentile groups
r_groups = pd.qcut(df_RFM['Recency'].rank(method='first'), q=4, labels=r_labels)
f_groups = pd.qcut(df_RFM['Frequency'].rank(method='first'), q=4, labels=f_labels)
m_groups = pd.qcut(df_RFM['Monetary'].rank(method='first'), q=4, labels=m_labels)
# Creat new columns R, F, M
df_RFM = df_RFM.assign(R = r_groups.values, F = f_groups.values, M=m_groups.values )
def join_rfm(x): return str(int(x['R'])) + str(int(x['F'])) + str(int(x['M']))
df_RFM['RFM_Segment'] = df_RFM.apply(join_rfm, axis=1)
# Calculate RFM score
df_RFM['RFM_score'] = df_RFM[['R', 'F', 'M']].sum(axis=1)


# 3. Build model
# C1: Định nghĩa nhóm khách hàng
def rfm_level(df):
    if (df['R'] == 4 and df['F'] ==4 and df['M'] == 4)  :
        return 'STARS'
    
    elif (df['R'] == 4 and df['F'] ==1 and df['M'] == 1):
        return 'NEW'
    
    else:     
        if df['M'] == 4:
            return 'BIG SPENDER'
        
        elif df['F'] == 4:
            return 'LOYAL'
        
        elif df['R'] == 4:
            return 'ACTIVE'
        
        elif df['R'] == 1:
            return 'LOST'
        
        elif df['M'] == 1:
            return 'LIGHT'
        
        return 'REGULARS'
# Create a new column_RFM_level
df_RFM['RFM_level'] = df_RFM.apply(rfm_level, axis = 1)
# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_agg = df_RFM.groupby('RFM_level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']}).round(0)

rfm_agg.columns = rfm_agg.columns.droplevel()
rfm_agg.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
rfm_agg['Percent'] = round((rfm_agg['Count']/rfm_agg.Count.sum())*100, 2)

# Reset the index
rfm_agg = rfm_agg.reset_index()

#5. Save models
# luu model classication
# pkl_filename = "ham_spam_model.pkl"  
# with open(pkl_filename, 'wb') as file:  
#     pickle.dump(model, file)
  
# luu model CountVectorizer (count)
# pkl_count = "count_model.pkl"  
# with open(pkl_count, 'wb') as file:  
#     pickle.dump(count, file)


#6. Load models 
# Đọc model
# import pickle
# with open(pkl_filename, 'rb') as file:  
#     ham_spam_model = pickle.load(file)
# # doc model count len
# with open(pkl_count, 'rb') as file:  
#     count_model = pickle.load(file)

# GUI
menu = ["Business Objective", "Data Explorer Analysis", "RFM method", "RFM-Kmeans"]

choice = st.sidebar.selectbox('Menu', menu)
if choice == "Business Objective" :
    st.subheader("Business Objective")
     
    st.write("""###### => Problem: Công ty X chủ yếu bán các sản phẩm là quà tặng dành cho những dịp đặc biệt. Nhiều khách hàng của công ty là khách hàng bán buôn.
###### => Requirement: Công ty X mong muốn có thể bán được nhiều sản phẩm hơn cũng như giới thiệu sản phẩm đến đúng đối tượng khách hàng, chăm sóc và làm hài lòng khách hàng.
""")
    st.image("customer_segmentation.png")
elif choice == "Data Explorer Analysis" :
    st.subheader("Data Explorer Analysis")
    st.write("#### 1. Some data")
    st.dataframe(data[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']].head(3))
    st.dataframe(data[['InvoiceNo', 'StockCode', 'Quantity', 'UnitPrice', 'CustomerID']].tail(3))
    
    st.write("#### 2. Visualize R, F, M")

    fig1 = plt.figure(figsize=(8,10))
    plt.subplot(3, 1, 1)
    sns.distplot(df_RFM['Recency'])# Plot distribution of R
    plt.subplot(3, 1, 2)
    sns.distplot(df_RFM['Frequency'])# Plot distribution of F
    plt.subplot(3, 1, 3)
    sns.distplot(df_RFM['Monetary'])
    st.pyplot(fig1) 
    
elif choice == 'RFM method':
    
    st.write("### RFM segmentation")
    st.write("##### Định nghĩa khách hàng:")
    st.write(""" Thuật toán này phân nhóm khách hàng dựa vào R,F,M:
    - Nhóm STARS: R, F, M = 4, 4, 4
    - Nhóm NEWS: R, F, M = 4, 1, 1
    - Nhóm BIG SPENDER: M = 4 (và không thuộc các nhóm trên)
    - Nhóm LOYAL: F = 4 (và không thuộc các nhóm trên)
    - Nhóm ACTIVE: R = 4 (và không thuộc các nhóm trên)
    - Nhóm LOST: R = 1 (và không thuộc các nhóm trên)
    - Nhóm LIGHT: M = 1 (và không thuộc các nhóm trên)
    - Nhóm bình thường: nhóm còn lại
    """)
    st.write("##### Average values for each RFM Level:")
    st.dataframe(rfm_agg)

    #Tree map.
    st.write("##### Tree map (RFM)")    
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(14, 10)
    colors_dict = {'ACTIVE':'yellow','BIG SPENDER':'royalblue', 'LIGHT':'cyan',
               'LOST':'red', 'LOYAL':'purple', 'POTENTIAL':'green', 'STARS':'gold'}
    squarify.plot(sizes=rfm_agg['Count'],
              text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
              color=colors_dict.values(),
              label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg.iloc[i])
                      for i in range(0, len(rfm_agg))], alpha=0.5 )
    plt.title("Customers Segments",fontsize=26,fontweight="bold")
    plt.axis('off')
    st.pyplot(fig)
    # Scatter Plot
    st.write("##### Scatter Plot (RFM)")
    # plt.clf()
    
    fig = px.scatter(rfm_agg, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="RFM_level",
           hover_name="RFM_level", size_max=100)
    st.plotly_chart(fig)

    # Scatter Plot
    st.write("##### 3D Scatter Plot (RFM)")
    fig = px.scatter_3d(df_RFM, x='Recency', y='Frequency', z='Monetary',
                    color = 'RFM_level', opacity=0.5,
                    color_discrete_map = colors_dict)
    fig.update_traces(marker=dict(size=5),
                  
                  selector=dict(mode='markers'))
    st.plotly_chart(fig)
    
    st.write("##### Summary: This model makes it easy to segment customers based on company definitions.")

elif choice == 'RFM-Kmeans':
    
    st.write("### RFM Kmeans")
    st.write("##### Thuật toán này phân nhóm khách hàng dựa vào RFM và phân nhóm tự động theo K-Means")

    # Code
    df_now = df_RFM[['Recency','Frequency','Monetary']]
    sse = {}
    for k in range(1, 20):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_now)
        sse[k] = kmeans.inertia_ # SSE to closest cluster centroid

    #Elbows.
    fig = plt.figure(figsize=(6,6))
    st.write("##### Elbows")    
    plt.clf()
    plt.title('The Elbow Method')
    plt.xlabel('k')
    plt.ylabel('SSE')
    sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
    st.pyplot(fig)
    # Build model with k=5
    k = st.slider("Chọn k", 2, 20, 1)
    submit = st.button("Make K-means with select k")
    if submit:
        model = KMeans(n_clusters=k, random_state=42)
        model.fit(df_now)
        df_now["Cluster"] = model.labels_
        # Calculate average values for each RFM_Level, and return a size of each segment 
        rfm_agg2 = df_now.groupby('Cluster').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'count']}).round(0)

        rfm_agg2.columns = rfm_agg2.columns.droplevel()
        rfm_agg2.columns = ['RecencyMean','FrequencyMean','MonetaryMean', 'Count']
        rfm_agg2['Percent'] = round((rfm_agg2['Count']/rfm_agg2.Count.sum())*100, 2)

        # Reset the index
        rfm_agg2 = rfm_agg2.reset_index()
        # Change thr Cluster Columns Datatype into discrete values
        rfm_agg2['Cluster'] = 'Cluster '+ rfm_agg2['Cluster'].astype('str')
        # Kết quả phân nhóm
        st.write("##### RFM-Kmeans Results:")
        st.dataframe(rfm_agg2)
        #Tree map.
        st.write("##### Tree map (RFM-Kmeans)")
        plt.clf()
        fig = plt.gcf()
        fig.set_size_inches(14, 10)    
        colors_dict2 = {'Cluster0':'yellow','Cluster1':'royalblue', 'Cluster2':'cyan',
                    'Cluster3':'red', 'Cluster4':'purple', 'Cluster5':'green', 'Cluster6':'gold'}

        squarify.plot(sizes=rfm_agg2['Count'],
                    text_kwargs={'fontsize':12,'weight':'bold', 'fontname':"sans serif"},
                    color=colors_dict2.values(),
                    label=['{} \n{:.0f} days \n{:.0f} orders \n{:.0f} $ \n{:.0f} customers ({}%)'.format(*rfm_agg2.iloc[i])
                            for i in range(0, len(rfm_agg2))], alpha=0.5 )


        plt.title("Customers Segments",fontsize=26,fontweight="bold")
        plt.axis('off')
        st.pyplot(fig)

        # Scatter Plot
        st.write("##### Scatter Plot (RFM-Kmeans)")
        # plt.clf()
        
        fig = px.scatter(rfm_agg2, x="RecencyMean", y="MonetaryMean", size="FrequencyMean", color="Cluster",
            hover_name="Cluster", size_max=100)
        st.plotly_chart(fig)

        # 3D Scatter Plot
        st.write("##### 3D Scatter Plot (RFM)")
        fig = px.scatter_3d(rfm_agg2, x='RecencyMean', y='FrequencyMean', z='MonetaryMean',
                        color = 'Cluster', opacity=0.3)
        fig.update_traces(marker=dict(size=20),
                    
                    selector=dict(mode='markers'))
        st.plotly_chart(fig)
        st.write("""##### Summary: 
        - Mô hình này khách hàng không cần phải xây dựng công thức định nghĩa nhóm khách hàng, mà dựa vào Elbows để phân nhóm.
        - Khi có kết quả phân cụm, cần phải giải thích từng cụm khách hàng theo các thuộc tính RFM ở bảng kết quả trên.""")