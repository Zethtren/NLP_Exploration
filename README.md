# NLP_Exploration

I will be creating and adding custom implementations of various NLP related functions here.
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

The TF in my implementation will be returned as list of dictionaries. Each dictionary will contain each word that exists in all the documents. The TF values for each word will be assigned as the value for the corresponding word for each dictionary.<br>

The IDF in my implementation will be returned as a single dictionary. This dictionary will have the IDF value calculated as above for each word in all the documents.<br>

TF-IDF values will be obtained from multiplying each word in the TF dictionaries against their corresponding value in the IDF dictionary. This will return a list the of dictionaries identical in shape to the TF list of dictionaries with their values modified to correspond with TF-IDF expectations<br>

These values will then be normalized using L2 Normalization as this is what is done with sklearn.

### My implementation:
- [Python code](https://github.com/Zethtren/NLP_Exploration/blob/main/tf_idf.py)
- [Usage](https://github.com/Zethtren/NLP_Exploration/blob/main/tf_idf_usage.ipynb)

### Use case for TF-IDF:
TF-IDF is frequently used in NLP for analyzing the meaning of a sentence, identifying topics, and identifying unusual text behaviour. Since it captures the information about how frequently a word is used in one place versus another it can help categorize texts based on content. It is generally easier to use for modeling than a "bag-of-words" apporach as it actually contains useful numerical information about word usage. This makes it great for classification and searching for relevant texts.

