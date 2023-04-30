import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


def recom(books_user_likes):
	def get_title_from_index(index):
		return df[df.index == index]["Title"].values[0]

	def get_index_from_title(Title):
		if df[df.Title == Title]["index"].values[0]:
			return df[df.Title == Title]["index"].values[0]
		else:
			exit

	books = pd.read_csv(r"C:\\Users\\purvi\\Desktop\\Book\\Book-Recommendation-System\\Bookz.csv")
	books=books[:1000]
	df=books
	img=pd.read_csv(r"C:\\Users\\purvi\\Desktop\\Book\\Book-Recommendation-System\\Imagez.csv")
	# ratings = pd.read_csv("BX-Book-Ratings.csv", sep=";", error_bad_lines=False, encoding="latin-1")


	features = ['Title','Author','Publisher','ISBN']
	for feature in features:
		df[feature] = df[feature].fillna('')

	def combine_features(row):
		try:
			return row['Title'] +" "+row['Author']+" "+row['Publisher']
		except:
			print("Error:", row)

	df["combined_features"] = df.apply(combine_features,axis=1)

#Create count matrix from this new combined column
	cv = CountVectorizer()
	count_matrix = cv.fit_transform(df["combined_features"])

#Compute the Cosine Similarity based on the count_matrix
	cosine_sim = cosine_similarity(count_matrix) 

#Get index of this book from its title
	books_index = get_index_from_title(books_user_likes)
	similar_books = list(enumerate(cosine_sim[books_index]))

#Get a list of similar books in descending order of similarity score
	sorted_similar_books = sorted(similar_books,key=lambda x:x[1],reverse=True)

    
   

# titles of first 50 books
	l=[]
	t=[]
	i=0
	for element in sorted_similar_books:
			l.append(get_title_from_index(element[0]))
			t.append(get_index_from_title(l[i]))
			i=i+1
			if i>9:
				break

	output=l
	index=t

	imgg=[]
	year=[]
	author=[]
	final_list=[]
	for i in index:
		imgg.append(img["Image-URL-M"][i-1])
		year.append(books["Year"][i-1])
		author.append(books["Author"][i-1])
	for i in range(len(index)):
		temp=[]
		temp.append(output[i])
		temp.append(imgg[i])
		temp.append(year[i])
		temp.append(author[i])
		final_list.append(temp)
	return final_list




def bookdisp():
	books=pd.read_csv("Bookz.csv")
	img=pd.read_csv("Imagez.csv")

	title=[]
	imgg=[]
	year=[]
	author=[]
	finallist=[]

	r=np.random.randint(2,1000,10)

	for i in r:
		title.append(books["Title"][i-1])
		imgg.append(img["Image-URL-M"][i-1])
		year.append(books["Year"][i-1])
		author.append(books["Author"][i-1])

	for i in range(10):
		temp=[]
		temp.append(title[i])
		temp.append(imgg[i])
		temp.append(year[i])
		temp.append(author[i])
		finallist.append(temp)

	return finallist


def pointwise_rank(books_user_likes):
	def get_title_from_index(index):
		return df[df.index == index]["Title"].values[0]

	def get_index_from_title(Title):
		return df[df.Title == Title]["index"].values[0]

# Load dataset
	books = pd.read_csv(r"C:\\Users\\purvi\\Desktop\\Book\\Book-Recommendation-System\\Bookz.csv")
	books=books[:1000]
	df=books
	img=pd.read_csv(r"C:\\Users\\purvi\\Desktop\\Book\\Book-Recommendation-System\\Imagez.csv")
	ratings = pd.read_csv(r"C:\\Users\\purvi\\Desktop\\Book\\Book-Recommendation-System\\BX-Book-Ratings.csv", sep=";", encoding="latin-1")
	

# Remove books with less than 10 ratings
	data = pd.merge(books, ratings, on='ISBN')

	features = ['Title','Author','Publisher','ISBN','Book-Rating']
	for feature in features:
		df[feature] = df[feature].fillna('')

	def combine_features(row):
		try:
			return row['Title'] +" "+row['Author']+" "+row['Publisher']
		except:
			print("Error:", row)

	df["combined_features"] = df.apply(combine_features,axis=1)

# Perform content-based filtering
	cv = CountVectorizer()
	count_matrix = cv.fit_transform(data["combined_features"])

#Compute the Cosine Similarity based on the count_matrix
	cosine_sim = cosine_similarity(count_matrix) 

#Get index of this book from its title
	

	# Recommend similar books based on cosine similarity
	indices = pd.Series(data.index, index=data["combined_features"])
	X = count_matrix
	y = data['Book-Rating']

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Train XGBoost model
	dtrain = xgb.DMatrix(X_train, label=y_train)
	dtest = xgb.DMatrix(X_test)
	params = {'objective': 'reg:squarederror', 'eval_metric': 'rmse'}
	model = xgb.train(params, dtrain)
	y_pred = model.predict(dtest)
	print("RMSE: ", np.sqrt(np.mean((y_test - y_pred)**2)))

	books_index_l = get_index_from_title(books_user_likes)
	idx = indices[books_index_l]
	rate_pointwise = model.predict(xgb.DMatrix(count_matrix[idx]))[0]
	book_indices = np.argsort(-rate_pointwise)[:10]
	similar_books_ranking = list(enumerate(book_indices))
	sorted_similar_books_ranking = sorted(similar_books_ranking,key=lambda x:x[1],reverse=True)

	l=[]
	t=[]
	i=0
	for element in sorted_similar_books_ranking:
			l.append(get_title_from_index(element[0]))
			t.append(get_index_from_title(l[i]))
			i=i+1
			if i>9:
				break

	output=l
	index=t

	imgg=[]
	year=[]
	author=[]
	final_list=[]
	for i in index:
		imgg.append(img["Image-URL-M"][i-1])
		year.append(books["Year"][i-1])
		author.append(books["Author"][i-1])
	for i in range(len(index)):
		temp=[]
		temp.append(output[i])
		temp.append(imgg[i])
		temp.append(year[i])
		temp.append(author[i])
		final_list.append(temp)
	return final_list
	

	


	

