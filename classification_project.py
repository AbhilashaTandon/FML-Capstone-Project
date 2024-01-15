import random

random.seed(16333323)  # first thing is initialize random with n number

import pandas as pd

colors = ["#000000", "#006400", "#ff0000", "#ffd700", "#00ff00", "#e9967a", "#00ffff", "#0000ff", "#6495ed", "#ff1493"] #colors for visualizations

import numpy as np
from sklearn.metrics import classification_report #for seeing performance of model
import nltk #for analyzing artist names and song titles
import matplotlib.pyplot as plt

from nltk.corpus import stopwords

stops = set(
	stopwords.words("english")
) | set(["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must', "n't", 'need', 'sha', 'wo', 'would']) 
# stopwords are filler words like and, is, that, no semantic data
#we add contractions not in the original set

from sklearn.feature_extraction.text import TfidfVectorizer
#NLP technique to turn lists of strings into word occurence matrices
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.decomposition import TruncatedSVD
#To do PCA for sparse matrices

def word_vectors(train_data, test_data, num_components):  # turns lists of strings into tf-idf-vectors
	vectorizer = TfidfVectorizer(
		decode_error="replace", 
		strip_accents="unicode",
		tokenizer=nltk.word_tokenize, #splits text into tokens
		stop_words=stops, #ignores these
	)
	tf_idfs_train = vectorizer.fit_transform(train_data)

	vocab = vectorizer.vocabulary_ #all words in training

	vectorizer = TfidfVectorizer(
		decode_error="replace",
		strip_accents="unicode",
		tokenizer=nltk.word_tokenize,
		stop_words=stops,
		vocabulary=vocab, #this is the main difference, we only use words we found in training data to avoid leakage and keep everything consistent
	)
	tf_idfs_test = vectorizer.fit_transform(test_data)
	
	clf = TruncatedSVD(n_components=num_components) #PCA but for sparse matrices
	tf_idfs_train = clf.fit_transform(tf_idfs_train)
	tf_idfs_test = clf.transform(tf_idfs_test)
	
	#finds most relevant features of word occurences, makes word vectors with dimension num_components

	return tf_idfs_train, tf_idfs_test 

def setup(file_path, pca_components): #imports data and preprocesses it
	data = pd.read_csv(file_path, encoding='utf-8')
	
	keys = sorted([x for x in set(data["key"]) if isinstance(x, str)])
	modes = sorted([x for x in set(data["mode"]) if isinstance(x, str)])
	genres = sorted([x for x in set(data["music_genre"]) if isinstance(x, str)])
	# get lists of values for categorical variables
	
	data = data.dropna()
	data = data.drop(data[data.tempo == "?"].index)
	data = data.drop(data[data.duration_ms == -1].index)
	# drop bad values
	
	data['tempo'] = pd.to_numeric(data['tempo'])
	#make this into numbers
	
	for key in keys:
		if key != key: continue # if nan ignore
		data[key] = [int(x == key) for x in data["key"]]
		# add new column, one hot encoding
	
	for mode in modes:
		if mode != mode: continue  #if nan ignore
		data[mode] = [int(x == mode) for x in data["mode"]] 
		# add new column, one hot encoding

	# dont decode genre, we can deal with categorical outputs since classification problem

	from sklearn.model_selection import train_test_split
	
	X, y = data.drop(columns=["obtained_date", "key", "mode", "music_genre"]), data[["music_genre"]]
	
	from sklearn.preprocessing import StandardScaler

	scaler = StandardScaler()
	X.iloc[:,3:] = scaler.fit_transform(X.iloc[:,3:])
	#ignores instanceid, artist name, track name since these arent numerical
	#normalizes numerical data with z scoring
	
	x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y)
	#stratify makes sure test data has equal class distribution
	
	y_train = np.array(y_train).reshape(-1)
	y_test = np.array(y_test).reshape(-1)

	artists_train = x_train["artist_name"]
	artists_test = x_test["artist_name"]
	titles_train = x_train["track_name"]
	titles_test = x_test["track_name"]
	
	x_train = x_train.drop(columns=["artist_name", "track_name", "instance_id"])
	x_test = x_test.drop(columns=["artist_name", "track_name", "instance_id"])
	#remove non numerical data

	artists_train = np.array(artists_train)
	artists_test = np.array(artists_test)
	titles_train = np.array(titles_train)
	titles_test = np.array(titles_test)

	title_vector_train, title_vector_test = word_vectors(titles_train, titles_test, pca_components)

	x_train = np.concatenate((x_train, title_vector_train), axis=1)
	x_test = np.concatenate((x_test, title_vector_test), axis=1)

	artists_vector_train, artists_vector_test = word_vectors(artists_train, artists_test, pca_components)
	
	x_train = np.concatenate((x_train, artists_vector_train), axis=1)
	x_test = np.concatenate((x_test, artists_vector_test), axis=1)

	return (x_train, x_test, y_train, y_test, genres)


def run_model(model, params, x_train, y_train, x_test, y_test, genres, show_plot=False):
	clf = model(**params)
	clf.fit(x_train, y_train)
	y_hat = clf.predict(x_test)
	probs = clf.predict_proba(x_test) #vector of class probabilities

	print(model)
	model_score = clf.score(x_test, y_test)
	#accuracy score
	
	fprs, tprs, scores = [],[],[]
	#roc data and roc_auc scores
	
	for idx, genre in enumerate(genres): #finds all roc_auc scores with one versus rest approach
		y_hat_ovr = np.array([y[idx] for y in probs]) #uses probabilities for roc
		#print(y_hat_ovr[:100])
		y_test_ovr = np.array([y == genre for y in y_test])
		fpr, tpr, _ = roc_curve(y_test_ovr, y_hat_ovr)
		score = roc_auc_score(y_test_ovr, y_hat_ovr)
		fprs.append(fpr)
		tprs.append(tpr)
		scores.append(score)
	
	roc_auc = (np.mean(scores)) #weighted average slightly more accurate but all classes have almost exact same # data points
	print(classification_report(y_test, y_hat))
	
	print("ROC AUC Scores")
	for genre, score in zip(genres, scores):
		print("%s: %.3f" % (genre, score))
	
	if(show_plot): #shows plot of ROC Curves
		for (genre, fpr, tpr, score, color) in zip(genres, fprs, tprs, scores, colors):
			plt.plot(fpr, tpr, color=color, lw=2, label=f'%s: area = %.3f' % (genre, score))
			#plots roc curve for each genre
		plt.plot([0, 1], [0, 1], 'k--', lw=2)
		#plots dotted line in middle of roc chart
		plt.xlim([-0.05, 1.0])
		plt.ylim([0.0, 1.05])
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('%s ROC Curves' % model)
		plt.legend(loc="lower right")
		plt.savefig("C:\\Users\\abhil\\Documents\\School\\Spring 2023\\Machine Learning\\Capstone Project\\ROC AUC Curves.png",
                    format='png',dpi=300)
		plt.clf()
	
	return model_score, roc_auc #accuracy score and roc auc score

def hyper_tune(model, hypers, params, x_train, y_train, x_test, y_test):
	# find ideal hyper params, hypers are hyperparams to tune, params are params not tuned
	from sklearn.model_selection import GridSearchCV

	clf = model(**params)
	hyper_param_tuning = GridSearchCV(clf, hypers, verbose=3)
	hyper_param_tuning.fit(x_train, y_train)

	print(hyper_param_tuning.best_params_)

def show_scatterplot(x_data, y_data, file_path, plot_title, coloring, labels_):
	from matplotlib.lines import Line2D #for matplotlib aesthetics
	
	scatter = plt.scatter(x_data, y_data, c = coloring, alpha=0.5)
	
	plt.xlabel("Embedding 1")
	plt.ylabel("Embedding 2")
	plt.title(plot_title)
	handles_ = [Line2D([0], [0], marker='o', color='w', label=genre_,
                        markerfacecolor=color_, markersize=5) for color_, genre_ in zip(colors, labels_)]
	plt.legend(handles=handles_, labels=labels_, title="Genres", loc='upper center', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)
	plt.savefig(file_path, format='png',dpi=300)
	plt.clf()

def main():
	from sklearn.ensemble import HistGradientBoostingClassifier #classifier used
	from sklearn.manifold import TSNE #dim reduce method
	
	
	(x_train, x_test, y_train, y_test, genres ) = setup("C:\\Users\\abhil\\Documents\\School\\Spring 2023\\Machine Learning\\Capstone Project\\musicData.csv", pca_components=20)
	#gets preprocessed data
	
	from sklearn.decomposition import PCA
	
	clf = PCA(n_components = 2)
	clf.fit(x_train)
	pca_embedding = clf.transform(x_test)
	#does dim reduction on data
	
	coloring = [colors[genres.index(g)] for g in y_test]
	
	show_scatterplot(pca_embedding[:,0], pca_embedding[:,1], "C:\\Users\\abhil\\Documents\\School\\Spring 2023\\Machine Learning\\Capstone Project\\PCA Embedding.png", "PCA 2-D Embedding", coloring, genres)
	#pca visualization
	
	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
	
	clf = LDA(solver='svd', n_components = 2)
	clf.fit(x_train, y_train)
	lda_embedding = clf.transform(x_test)
	
	show_scatterplot(lda_embedding[:,0], lda_embedding[:,1], "C:\\Users\\abhil\\Documents\\School\\Spring 2023\\Machine Learning\\Capstone Project\\LDA Embedding.png", "LDA 2-D Embedding", coloring, genres)
	#lda visualization
	
	clf = TSNE(n_components = 2, perplexity=200,init='pca')
	tsne_embedding = clf.fit_transform(x_test)
	#does dim reduction on data
	
	show_scatterplot(tsne_embedding[:,0], tsne_embedding[:,1], "C:\\Users\\abhil\\Documents\\School\\Spring 2023\\Machine Learning\\Capstone Project\\TSNE Embedding.png", "TSNE 2-D Embedding", coloring, genres)
	
	m_score, r_score = run_model(HistGradientBoostingClassifier, {'loss': 'auto', 'learning_rate': 1e-1, 'max_leaf_nodes': 31, 'tol': 1e-7},x_train, y_train, x_test, y_test, genres, show_plot=True)
	#runs model and gets results, shows plot
		
	print(f'Accuracy: {m_score}, ROC_AUC Score: {r_score}')
	

if __name__ == "__main__": main()