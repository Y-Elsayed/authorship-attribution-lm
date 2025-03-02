from sklearn.model_selection import train_test_split
import spacy

class DataProcessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def __load_file(self, file_path):
        with open(file_path,"r",encoding="utf-8") as f:
            return f.read()

    def __file_sentences(self, file_text):
        paragraphs = [para.replace("\n", " ").strip() for para in file_text.split("\n\n") if para.strip()]
        tokenized_sentences = [[token.text for token in self.nlp(para)]for para in paragraphs]
        return tokenized_sentences
    
    def __split_data(self, sentences, test_size = 0.1, random_state=42):
        print("Splitting into training and development...")
        return train_test_split(sentences, test_size = test_size,random_state=random_state)
    
    # Need to tokenize
    def process_file(self,file_path):
        text = self.__load_file(file_path=file_path)
        sentences = self.__file_sentences(file_text=text)
        return sentences
    
    def process_split_file(self,file_path):
        text = self.__load_file(file_path=file_path)
        sentences = self.__file_sentences(file_text=text)
        return self.__split_data(sentences)