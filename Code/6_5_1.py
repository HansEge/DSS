import pandas as pd
import itertools
import numpy as np
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt


def bestSubsetSoultion(X,y,numOfFeatures):
	models_best = pd.DataFrame(columns=["RSS", "model"])

	tic = time.time()

	for i in range(1, numOfFeatures):
		models_best.loc[i] = getBestModel(i, X, y)

	toc = time.time()

	print("Total elapsed time:", (toc - tic), "seconds.")

	return models_best

def getBestModel(k, X, y):
		results = []
		# Goes through all combinations of k number of features
		for combination in itertools.combinations(X.columns, k):
			results.append(subset_Process(combination, X, y))

		# Stores all combinations i a single dataframe
		models = pd.DataFrame(results)

		#Select the model with the lowest RSS
		best_model = models.loc[models['RSS'].argmin()]

		return best_model


def subset_Process(predictor, X, y):
	# sm.OLS is an estimator that performs linear regression on a model
	temp_model = sm.OLS(y, X[list(predictor)])
	model = temp_model.fit()

	#Then calculate the RSS for the chosen model, and return the model and its RSS together
	RSS = ((model.predict(X[list(predictor)]) - y) ** 2).sum()
	return {"model": model, "RSS": RSS}


def plotBestSubsetSolutions(best_Models):
	plt.figure(figsize=(15, 10))

	#set up subplot
	plt.subplot(2, 2, 1)

	plt.plot(best_Models["RSS"])
	plt.xlabel('Number of  Predictors')
	plt.ylabel('RSS')

	r2_adj = best_Models.apply(lambda row: row[1].rsquared_adj, axis=1)

	#setup subplot
	plt.subplot(2, 2, 2)

	plt.plot(r2_adj)
	plt.xlabel('Number of Predictors')
	plt.ylabel('Adjusted R^2')

	# We'll do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
	aic = best_Models.apply(lambda row: row[1].aic, axis=1)

	plt.subplot(2, 2, 3)
	plt.plot(aic)
	plt.xlabel('# Predictors')
	plt.ylabel('AIC')

	bic = best_Models.apply(lambda row: row[1].bic, axis=1)

	plt.subplot(2, 2, 4)
	plt.plot(bic)
	plt.xlabel('# Predictors')
	plt.ylabel('BIC')

	plt.show()


def main():
	# Read in the Hitters data-set
	df = pd.read_csv('Hitters.csv')
	print(df.head())

	# It can be seen from the Hitters data-set that the salary variable is missing for some of the player.
	numberOfMissingSalaryVar = df["Salary"].isnull().sum()
	print("Number players missing the salary variable", numberOfMissingSalaryVar)

	#As we want to predict a player's Salary on the basis of various statistics, we need to remove all players with missing variables.
	#The fuction dropna does this
	df_tidy = df.dropna().drop('Player', axis=1)
	print("Dimensions of tidy data:", df_tidy.shape)

	#Next we want to remove all the predicators witch are catagorical. We'll use the panda lib to generate dymme variables for them.
	dummy_vars = pd.get_dummies(df_tidy[['League', 'Division', 'NewLeague']])

	# y (salary) is the responds we want our features to valued upon
	y = df_tidy.Salary

	#X_temp is a temporary X without the catagory variabes and salary
	X_temp = df_tidy.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

	#X is the complete training data set where dummy vars is included
	X = pd.concat([X_temp, dummy_vars[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)
	print("Dimensions of tidy data:", X)

	best_models = bestSubsetSoultion(X,y,10)

	plotBestSubsetSolutions(best_models)





if __name__ == '__main__':
	main()