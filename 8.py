# Step 1: Import necessary libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 2: Sample document
sample_document = "Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering, and artificial intelligence concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data."

# Step 3: Tokenization
tokens = word_tokenize(sample_document)

# Step 4: POS Tagging
pos_tags = pos_tag(tokens)

# Step 5: Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# Step 6: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Step 7: Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

# Step 8: TF-IDF representation
corpus = [sample_document]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()

# Print the results
print("Tokenization:", tokens)
print("\nPOS Tagging:", pos_tags)
print("\nFiltered Tokens (Stop words removal):", filtered_tokens)
print("\nStemming:", stemmed_tokens)
print("\nLemmatization:", lemmatized_tokens)
print("\nTF-IDF representation:")
print(pd.DataFrame(X.toarray(), columns=terms))
