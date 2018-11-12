CSCI57300	Data Mining
Homework 3  Bayes Classifier
Author: Jingyi Hui
Data:	11/10/2018

-----------------------------------------
List of Documents:
1. README.txt
2. JH_Bayes_Model.py 		(Q1 FULL)
3. JH_Bayes_Test.py 		(Q2 FULL)
4. JH_Bayes_Validation.py	(Q3 FULL) 
5. JH_Naive_Bayes_Model.py	(Q1 NAIVE)
6. JH_Naive_Bayes_Test.py	(Q2 NAIVE)   
7. JH_Naive_Bayes_Validation.py	(Q3 NAIVE) 
4. JH_HW3_EvaluationReport.pdf	(3-fold validation report)

-----------------------------------------
To run the program with iris data:
1. Login to Tesla and copy all the files under a directory;
2. Make all the .py file executable, type:
	chmod +x *.py
3. To run the program: JH_Bayes_Model.py with a training dataset: train.csv, type:
	./JH_Bayes_Model.py train.csv	
4. To run the program: JH_Bayes_Test.py with a model file: model_bayes.csv and a testing set: test.csv, type:
	./JH_Bayes_Test.py model_bayes.csv test.csv
5. To run the program:  JH_Bayes_Validation.py with a data file: iris.txt.shuffled, type:
	./JH_Bayes_Validation.py iris.txt.shuffled
6. To run the program: JH_Naive_Bayes_Model.py with a training dataset: train.csv, type:
	./JH_Naive_Bayes_Model.py train.csv	
7. To run the program: JH_Naive_Bayes_Test.py with a model file: model_naive_bayes.csv and a testing set: test.csv, type:
	./JH_Naive_Bayes_Test.py model_naive_bayes.csv test.csv
8. To run the program:  JH_Naive_Bayes_Validation.py with a data file: iris.txt.shuffled, type:
	./JH_Naive_Bayes_Validation.py iris.txt.shuffled
9. For the convenience of verification, the console will print out all the original test data instances with truth labels and the prediction classes in all the testing and validation processes.
10. All the model file generated from training processes will be saved in csv files.

