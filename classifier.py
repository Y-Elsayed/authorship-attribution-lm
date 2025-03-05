from ngram_authorship_classifier import NgramAuthorshipClassifier
from hf_sequence_classifier import SequenceClassifier
from data_processor import DataProcessor
import argparse



def main():
    parser = argparse.ArgumentParser(description="Authorship attribution with n-grams")
    parser.add_argument("authorlist",help="Path to the authorlist file")
    parser.add_argument('-approach',choices=["generative","discriminative"],required=True,help="The approach to use for authorship attribution")
    parser.add_argument("-test",default=None,help="Path to the file to test the model")
    args = parser.parse_args()

    # Loading the authorlist
    with open(args.authorlist) as f:
        author_files = f.read().splitlines()

    # Processing the data
    data_proc = DataProcessor()
    authors_train_data = dict()
    authors_test_data = dict()
    # Automatically splitting the data if the test file is not provided otherwise the testset is None
    process_function = data_proc.process_file if args.test else data_proc.process_split_file
    for author_file in author_files:
        print(f"Processing data for author: {author_file}")
        trainset, testset = process_function(author_file)
        authors_train_data[author_file] = trainset
        if not args.test:
            authors_test_data[author_file] = testset 
    # If the test file is provided
    if args.test:
        test_data = data_proc.process_file(args.test)
    
    
    # Choosing the model according to the approach argument
    if args.approach == "generative":
        model = NgramAuthorshipClassifier(n=1, smoothing = 'lp')
    else:
        model = SequenceClassifier()

    # Training the model
    model.train(authors_train_data)

    # Testing the model
    if args.test:
        model.predict(test_data)
    else:
        model.evaluate_devset(authors_test_data, show_accuracy=True)

if __name__ == '__main__':
    main()