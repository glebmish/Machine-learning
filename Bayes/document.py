
class Document(object):

    def __init__(self, name, subject, content, is_spam=None):
        self.name = name
        self.subject = subject
        self.content = content
        self.is_spam = is_spam

    def __str__(self):
        return "Document [name={}, subject={}, content={}, is_spam={}]".format(self.name, self.subject, self.content, self.is_spam)

    def __repr__(self):
        return self.__str__()
