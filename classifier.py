import ngram_authorship_classifier as nac
import data_processor as dp
import argparse




def main():
    parser = argparse.ArgumentParser(description="Authorship attribution with n-grams")
    parser.add_argument("authorlist",help="Path to the authorlist file")
    parser.add_argument('-approach',choices=["generative","discriminative"],required=True,help="The approach to use for authorship attribution")
    parser.add_argument("-test",default=None,help="Path to the file to test the model")
    args = parser.parse_args()