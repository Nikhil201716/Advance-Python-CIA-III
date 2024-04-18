import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv("Dataset - WomensClothingE-CommerceReviews.csv")
    return df

df = load_data()

# Text preprocessing and normalization
def preprocess_text(text):
    if isinstance(text, str):
        # Tokenization
        tokens = word_tokenize(text.lower())
        # Remove punctuation and stopwords
        tokens = [word for word in tokens if word.isalnum() and word not in set(stopwords.words('english'))]
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Join tokens back into a string
        return ' '.join(tokens)
    else:
        return ""

# Apply text preprocessing to the review text column
df['Cleaned Review Text'] = df['Review Text'].apply(preprocess_text)

# Define the 3D Plot Visualization tab/menu
def plot_3d_visualization():
    st.title("3D Plot Visualization")
    # Filter out NaN values if any
    df_filtered = df.dropna(subset=['Age', 'Rating', 'Positive Feedback Count'])
    # Create 3D scatter plot
    fig = px.scatter_3d(df_filtered, x='Age', y='Rating', z='Positive Feedback Count', color='Rating', size_max=10)
    # Update layout
    fig.update_layout(scene=dict(xaxis_title='Age', yaxis_title='Rating', zaxis_title='Positive Feedback Count'))
    # Display the plot
    st.plotly_chart(fig)

# Text similarity analysis
def text_similarity_analysis(division_name):
    st.title("Text Similarity Analysis")
    st.write(f"Similar reviews within the dataset for Division Name: {division_name}")
    # Filter dataset by division name
    division_df = df[df['Division Name'] == division_name]
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(division_df['Cleaned Review Text'])
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    # Display similar reviews
    for i in range(len(similarity_matrix)):
        similar_indices = similarity_matrix[i].argsort()[-2:-11:-1]
        st.write(f"Review {i}: {division_df.iloc[i]['Review Text']}")
        st.write("Similar Reviews:")
        for j in similar_indices:
            if j != i:
                st.write(f"- {division_df.iloc[j]['Review Text']}")

# Define the main function to create the Streamlit app
def main():
    st.sidebar.title("Navigation")
    tabs = ["3D Plot Visualization", "Text Similarity Analysis"]
    selected_tab = st.sidebar.radio("Select Tab", tabs)

    if selected_tab == "3D Plot Visualization":
        plot_3d_visualization()
    elif selected_tab == "Text Similarity Analysis":
        division_names = df['Division Name'].unique()
        division_name = st.selectbox("Select Division Name", division_names)
        if division_name:
            text_similarity_analysis(division_name)

if __name__ == "__main__":
    main()
