# NLP_Exploration

I will be creating and adding custome implementations of various NLP related functions here
Once I build enough I may even publish PyPi
Below are links to any completed and working code

## TF_IDF: 
TF-IDF (Term Frequency - Inverse Document Frequency) is a tool used to assess the count of words in a document when compared against other documents in a set (corpus). 
Term Frequency is calculated as such:<br><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=TF(t)&space;=&space;\frac{\text{Number&space;of&space;times&space;t&space;appears&space;in&space;document}}{\text{Total&space;number&space;of&space;terms&space;in&space;the&space;document}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?TF(t)&space;=&space;\frac{\text{Number&space;of&space;times&space;t&space;appears&space;in&space;document}}{\text{Total&space;number&space;of&space;terms&space;in&space;the&space;document}}" title="TF(t) = \frac{\text{Number of times t appears in document}}{\text{Total number of terms in the document}}" /></a>
<br><br>
Inverse Document Frequency is calculated as such (for Sklearn implementation):<br><br>
<a href="https://www.codecogs.com/eqnedit.php?latex=IDF(t)&space;=&space;1&plus;\log_{e}\frac{1\text{&space;}&plus;\text{&space;Total&space;number&space;of&space;documents}}&space;{1&plus;\text{Count&space;of&space;documents&space;with&space;term&space;t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?IDF(t)&space;=&space;1&plus;\log_{e}\frac{1\text{&space;}&plus;\text{&space;Total&space;number&space;of&space;documents}}&space;{1&plus;\text{Count&space;of&space;documents&space;with&space;term&space;t}}" title="IDF(t) = 1+\log_{e}\frac{1\text{ }+\text{ Total number of documents}} {1+\text{Count of documents with term t}}" /></a>
<br><br>
### My implementation:
- [Python code](https://github.com/Zethtren/NLP_Exploration/blob/main/tf_idf.py)
- [Usage](https://github.com/Zethtren/NLP_Exploration/blob/main/tf_idf_usage.ipynb)
