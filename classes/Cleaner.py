import re


class Cleaner:

    def __init__(self):
        """
        init re
        """
        self.eyes = '[8:=;]'
        self.nose = "['`\-]?"
        self.url = r'https?://[A-Za-z0-9./]+'
        self.user = r'@[A-Za-z0-9]+'
        self.hashtag = r'#[A-Za-z0-9]+'
        self.smile = r'' + self.eyes + self.nose + '[)d]+|[)d]+' + self.nose + self.eyes + '}'
        self.lol = r'' + self.eyes + self.nose + 'p+'
        self.sad = r'' + self.eyes + self.nose + '\(+|\)+' + self.nose + self.eyes
        self.neutral = r'' + self.eyes + self.nose + '[\/|l]'
        self.heart = r'<3'
        self.number = r'[-+]?[.\d]*[\d]+[:,.\d]*'
        self.repeat = r'([!?.]){2,}'
        self.elong = r'\b(\S*?)(.)\2{2,}\b'

    def cleane(self, input_str, debug=False):
        """
        säubert einen String und ersetzt bestimmte Wörter bzw. Zeichenketten mit Tokens des Embeddings

        :param input_str: String input
        :param debug: bool, falls mehr Infos geprinted werden solleb
        :return: gesäuberter String
        """
        if debug:
            print()
            print(input_str)
            print()
        input_str = input_str.replace("\\u2019", "'").replace("\\u002c", ",").replace(u"\ufffd", "?").lower()

        input_str = re.sub(self.url, "<URL>", input_str)
        input_str = re.sub(self.user, "<USER>", input_str)
        input_str = re.sub(self.smile, "<SMILE>", input_str)
        input_str = re.sub(self.lol, "<LOLFACE>", input_str)
        input_str = re.sub(self.sad, "<SADFACE>", input_str)

        input_str = re.sub(self.neutral, "<NEUTRALFACE>", input_str)
        input_str = re.sub(self.heart, "<HEART>", input_str)
        input_str = re.sub(self.number, "<NUMBER>", input_str)
        input_str = re.sub(self.repeat, r"\1 " + "<REPEAT>", input_str)
        input_str = re.sub(self.elong, r"\1\2 " + "<ELONG>", input_str)
        input_str = re.sub(self.hashtag, "<HASHTAG>", input_str)

        return input_str
