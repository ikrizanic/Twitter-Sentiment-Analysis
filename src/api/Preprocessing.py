import string
from re import sub
from nltk.tokenize import TweetTokenizer


class Preprocessing:
    """
    Methods:
        remove_usernames(string): returns input string without usernames (words starting with '@')
        remove_punctuation(string): returns input string without punctuation
        remove_links(string): returns input string without links (words starting with 'http' or 'https')
        tokenize(string): returns tokens of input string, using nltk.TweetTokenizer
        preprocessing_pipe(string, boolean): returns tokenized input with applying all of preprocessing steps

    """

    def __init__(self):
        self.tokenizer = TweetTokenizer()

    @staticmethod
    def remove_usernames(raw):
        return sub(r'@[^\s]*', '', raw)

    @staticmethod
    def remove_punctuation(raw):
        return raw.translate(str.maketrans('', '', string.punctuation))

    @staticmethod
    def remove_links(raw):
        return sub(r'https?://\S+', '', raw)

    def tokenize(self, raw):
        tokenized = self.tokenizer.tokenize(raw)
        return tokenized

    @staticmethod
    def to_lower(raw):
        return raw.lower()

    def preprocessing_pipeline(self, raw, group=False):
        """
        Parameters:
            raw: list of input strings or single string
            group: if True, works with list of input data, else expects single string input
        """
        if group:
            return [self.tokenize(self.to_lower(self.remove_punctuation(self.remove_usernames(
                self.remove_links(tweet))))) for tweet in raw]

        return self.tokenize(self.to_lower(self.remove_punctuation(self.remove_usernames(self.remove_links(raw)))))
