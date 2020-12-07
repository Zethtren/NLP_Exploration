import numpy as np

class TfIdfCustom:
    def __init__(self, corpus):
        """
        Creates a word dictionary, Term-Frequency list of dictionaries, Inverse-Document-Frequency dictionary,
        Term-Frequency Inverse-Document-Frequency list of dictionaries, and normalized version of that list as well

        Args:
            corpus:                 A list of strings that should represent the entire content of a document.
        
        Attributes:
            word_dict:              Dictionary map of words to integers
            tf_:                    Term-Frequency list of dictionaries
            idf_:                   Inverse-Document-Frequency dictionary
            tfidf_:                 Term-Frequency Inverse-Document-Frequency list of dictionaries
            tf_idf_normalized_:     Normalized Term-Frequency Inverse-Document-Frequency list of dictionaries
            
        Methods:
            print_word_map_results: Prints an easily legible view of word_dict
            print_tf_results:       Prints an easily legible view of tf_
            print_idf_results       Prints an easily legible view of idf_
            print_tfidf_results     Prints an easily legible view of tf_idf_normalized_

        Notes:
            This implimentation is meant to reflect Sklearns implementation
            
        """
        # Initialize corpus and word_dict
        self.corpus = corpus
        self.map_words()
        self.tf_custom()
        self.idf_custom()
        self.tf_idf_custom()
        self.normalized_array_tfidf()
        
    def map_words(self):
        """
        Method of TfIdfCustom custom class
        Uses corpus to instantiate a word dictionary.
        """
        
        # Establish an empty list for collecting words:
        word_list = []

        # Iterate through your corpus of documents
        for i in self.corpus:

            # Split each document in the corpus into a list of words 
            words = i.split(" ")

            # Add these values to your complete list
            for word in words:
                word_list.append(word)

        # Cast your list as a set to remove duplicates
        word_set = set(word_list)

        # Create an empty dictionay
        word_dict = dict()

        # Enumerate through your word set
        for i, word in enumerate(word_set):

            # Save the enumerated value "i" as the value for its corresponding key "word" into your dictionary
            word_dict[word] = i

        # Save your dictionary
        self.word_dict = word_dict

    def print_word_map_results(self):
        print("This is the resulting dictionary from the custom implementation of vectorizer.get_feature_names: ")
        print("===========================================================================================================")
        print(self.word_dict)
        print("")
        print("")

    def tf_custom(self):
        """
        Method of TfIdfCustom class
        Uses Corpus and and word_dict to instantiate a list of Term Frequency Dictionaries
        """
        corpus = self.corpus
        word_dict = self.word_dict
        # Create an empty list for storing the results 
        tf_results = [] 

        # Iterate through the documents
        for document in corpus:
            words = document.split(" ")

            # Create an empty dictionary for storing word counts
            count_dict = dict()

            # Fill that dictionary with "0" counts from all the words in your word dictionary
            for word in word_dict.keys():
                count_dict[word] = 0

            # Add a value to the count in the count dictionary if the word occurs in document
            for word in words:
                count_dict[word] += 1

            # Divide each value in the count dictionary by the length of the document
            # This completes Term-Frequency calculation for this document
            for word in count_dict:
                count_dict[word] /= len(words)
                
            # Append the results to a list and save the result for further use
            tf_results.append(count_dict)
        
        # Save the TF results
        self.tf_ = tf_results

    def print_tf_results(self):
        print("These are the results from Term Frequency calculations")
        print("===========================================================================================================")
        for item in self.tf_:
            print(item)
            print("")
        print("")
        print("")

    def idf_custom(self):
        """
        Method of TfIdfCustom class
        Uses Corpus and and word_dict to instantiate a single dictionary containing IDF multipliers
        """
        corpus = self.corpus
        word_dict = self.word_dict
        
        # Create an empty list of word sets
        word_sets = []
        
        # Create a word set for each document in the corpus
        for document in corpus:
            words = document.split(" ")
            word_sets.append(set(words))
            
        # Create a dictionary of "0"s to count how many documents a word appears in
        document_word_count = dict()
        
        for word in word_dict.keys():
            document_word_count[word] = 0

        # Store the count of documents the word appears into the dictionary by adding 1 for each occurence
        for word in word_dict.keys():
            for word_set in word_sets:
                if word in word_set:
                    document_word_count[word] += 1
                    
        # Create a dictionary for storing the idf calculations            
        idf_counts = dict()
        
        # Calculate IDF 
        for word in document_word_count:
            idf_counts[word] = 1 + np.log( (1+len(corpus)) / (1 + document_word_count[word]) )
        
        # Save IDF results
        self.idf_ = idf_counts

    def print_idf_results(self):
        print("This is the result from IDF calculations: ")
        print("===========================================================================================================")
        print(self.idf_)
        print("")
        print("")

    def tf_idf_custom(self):
        """
        Method of TfIdfCustom class
        Uses Term Frequency and Inverse Document Frequency to instantiate a dictionary of these values for the entire corpus
        """
        idf_counts = self.idf_
        tf_dicts = self.tf_
        
        # Apply the IDF multiplier to each occurence of
        for document_tf in tf_dicts:
            for word in document_tf.keys():
                document_tf[word] *= idf_counts[word]
                
        # Save TfIdf results
        self.tfidf_ = tf_dicts

    def normalized_array_tfidf(self):
        """
        Method of TfIdfCustom class
        Use Term-Frequency Inverse-Document-Frequency to establish a normalized TF-IDF result
        """
        tf_idf_dicts = self.tfidf_
        
        # Create an empty list to save noramlized values to.
        tf_idf_list = []
        
        # Iterate through non-normalized TF-IDF calculation
        for tf_idf in tf_idf_dicts:
            
            # Create a list from the values in each documents TfIdf calculation
            array = list(tf_idf.values())
            
            # Create an empty list as we want to ignore the words that don't occur
            working_list = []
            
            # Fill list with all values that have non-zero occurence
            for value in array:
                if value != 0:
                    working_list.append(value)
                    
            # Convert list to numpy array to collect normalization multiplier
            working_array = np.array(working_list)
            
            # Calculate normalization factor using L2 Norm standards
            normalizer = np.sqrt(np.dot(working_array, working_array))
            
            # Modify the value TfIdf value stored with each word in the document
            for word in tf_idf.keys():
                tf_idf[word] /= normalizer
                
            # Save the noramlized dictionaries back to our empty list 
            tf_idf_list.append(tf_idf)
            
        # Save normalized TfIdf results
        self.tf_idf_normalized_ = tf_idf_list

    def print_tfidf_results(self): 
        print("These are the final normalized TF_IDF results")
        print("===========================================================================================================")
        for result in self.tf_idf_normalized_:
            print(result)
            print("")
        print("")
        print("")
        
def tf_idf_custom_init(corpus):
    return TfIdfCustom(corpus)