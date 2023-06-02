project.tar: MD.mp4 Group_names.txt data/allUsers.lcl.csv data/info.txt docs/MD2.pdf docs/MD2.pptx docs/Report-DM-Project2.pdf src/dataLoader.py src/Bivariate.ipynb src/KNN.ipynb src/meta_learning.ipynb src/naiveBayes.ipynb src/nullables.ipynb src/univariate_analisis.ipynb src/SVM.ipynb src/decision_tree.ipynb
	tar -cvf project.tar MD.mp4 requirements.txt Group_names.txt data/allUsers.lcl.csv data/info.txt docs/MD2.pdf docs/MD2.pptx docs/Report-DM-Project2.pdf src/dataLoader.py src/Bivariate.ipynb src/KNN.ipynb src/meta_learning.ipynb src/naiveBayes.ipynb src/nullables.ipynb src/univariate_analisis.ipynb src/SVM.ipynb src/decision_tree.ipynb

clean:
	rm -f project.tar